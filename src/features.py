from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import MINUTE_BARS_DDL, MINUTE_FEATURES_DDL
from utils import now_kst_text


FEATURE_COLUMNS = [
    "symbol",
    "ts_kst",
    "ret_1m",
    "ret_3m",
    "ret_5m",
    "ret_10m",
    "ret_20m",
    "ret_60m",
    "range_1m",
    "body",
    "upper_shadow",
    "lower_shadow",
    "vol_5m",
    "vol_20m",
    "vol_60m",
    "volume_1m",
    "volume_change_1m",
    "volume_ma_5m",
    "volume_ma_20m",
    "volume_z_20m",
    "volume_z_60m",
    "minutes_from_open",
    "minutes_to_close",
    "session_bucket",
    "return_from_open",
    "distance_from_day_high",
    "distance_from_day_low",
    "vwap",
    "vwap_distance",
    "created_at",
]


@dataclass
class FeatureBuildSummary:
    input_rows: int
    valid_rows: int
    feature_rows: int
    date_range: str
    output_parquet_path: Path
    sqlite_saved_rows: int


def load_minute_bars(
    *,
    db_path: Path,
    input_table: str,
    parquet_fallback: Path,
    symbol: str,
    source: str,
    logger: logging.Logger,
):
    import pandas as pd

    if source not in {"sqlite", "parquet"}:
        raise ValueError("source must be sqlite or parquet")

    if source == "sqlite":
        df = _load_sqlite_minute_bars(
            db_path=db_path,
            input_table=input_table,
            symbol=symbol,
            logger=logger,
        )
        if not df.empty:
            return df
        logger.warning("SQLite input is empty; trying parquet fallback path=%s", parquet_fallback)

    if parquet_fallback.exists():
        df = pd.read_parquet(parquet_fallback)
        if "symbol" in df.columns:
            df = df.loc[df["symbol"].astype(str) == symbol].copy()
        else:
            df = df.copy()
            df["symbol"] = symbol
        logger.info("loaded parquet fallback rows=%s path=%s", len(df), parquet_fallback)
        return df

    logger.warning("no input data found source=%s db=%s parquet=%s", source, db_path, parquet_fallback)
    return pd.DataFrame(columns=["symbol", "ts_kst", "open", "high", "low", "close", "volume"])


def build_minute_features(raw_df, *, logger: logging.Logger):
    import numpy as np
    import pandas as pd

    input_rows = len(raw_df)
    if raw_df.empty:
        logger.info("feature input empty")
        return _empty_feature_frame(pd), input_rows, 0

    df = raw_df.copy()
    if "datetime" in df.columns and "ts_kst" not in df.columns:
        df["ts_kst"] = df["datetime"].astype(str).map(_compact_datetime_to_text)

    required = ["symbol", "ts_kst", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df[required].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["symbol", "ts_kst", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["symbol", "ts_kst"], keep="last")
    df = df.loc[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ].copy()

    market_open = pd.to_datetime("09:00:00").time()
    market_close = pd.to_datetime("15:30:00").time()
    times = df["ts_kst"].dt.time
    df = df.loc[(times >= market_open) & (times <= market_close)].copy()

    if df.empty:
        logger.info("feature valid rows empty after cleaning")
        return _empty_feature_frame(pd), input_rows, 0

    df = df.sort_values(["symbol", "ts_kst"]).reset_index(drop=True)
    df["trade_date"] = df["ts_kst"].dt.date.astype(str)
    group = df.groupby(["symbol", "trade_date"], sort=False)

    features = df[["symbol", "ts_kst"]].copy()
    close_group = group["close"]
    ret_1m = close_group.pct_change(1)
    features["ret_1m"] = ret_1m
    for window in [3, 5, 10, 20, 60]:
        features[f"ret_{window}m"] = close_group.pct_change(window)

    features["range_1m"] = df["high"] / df["low"] - 1
    features["body"] = df["close"] / df["open"] - 1
    features["upper_shadow"] = df["high"] / df[["open", "close"]].max(axis=1) - 1
    features["lower_shadow"] = df[["open", "close"]].min(axis=1) / df["low"] - 1

    for window in [5, 20, 60]:
        features[f"vol_{window}m"] = ret_1m.groupby([df["symbol"], df["trade_date"]]).rolling(window).std().reset_index(level=[0, 1], drop=True)

    volume_group = group["volume"]
    features["volume_1m"] = df["volume"].astype("int64")
    features["volume_change_1m"] = volume_group.pct_change(1)
    features["volume_ma_5m"] = volume_group.rolling(5).mean().reset_index(level=[0, 1], drop=True)
    volume_ma_20 = volume_group.rolling(20).mean().reset_index(level=[0, 1], drop=True)
    volume_std_20 = volume_group.rolling(20).std().reset_index(level=[0, 1], drop=True)
    volume_ma_60 = volume_group.rolling(60).mean().reset_index(level=[0, 1], drop=True)
    volume_std_60 = volume_group.rolling(60).std().reset_index(level=[0, 1], drop=True)
    features["volume_ma_20m"] = volume_ma_20
    features["volume_z_20m"] = (df["volume"] - volume_ma_20) / volume_std_20
    features["volume_z_60m"] = (df["volume"] - volume_ma_60) / volume_std_60

    minute_of_day = df["ts_kst"].dt.hour * 60 + df["ts_kst"].dt.minute
    features["minutes_from_open"] = (minute_of_day - 9 * 60).astype("int64")
    features["minutes_to_close"] = (15 * 60 + 30 - minute_of_day).astype("int64")
    features["session_bucket"] = pd.cut(
        features["minutes_from_open"],
        bins=[-1, 120, 300, 390],
        labels=["open", "mid", "late"],
    ).astype("object")

    first_open = group["open"].transform("first")
    features["return_from_open"] = df["close"] / first_open - 1
    day_high = group["high"].cummax()
    day_low = group["low"].cummin()
    features["distance_from_day_high"] = df["close"] / day_high - 1
    features["distance_from_day_low"] = df["close"] / day_low - 1

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_value = (typical_price * df["volume"]).groupby([df["symbol"], df["trade_date"]]).cumsum()
    cum_volume = df["volume"].groupby([df["symbol"], df["trade_date"]]).cumsum()
    features["vwap"] = cum_value / cum_volume.replace(0, np.nan)
    features["vwap_distance"] = df["close"] / features["vwap"] - 1

    features["ts_kst"] = features["ts_kst"].dt.strftime("%Y-%m-%d %H:%M:%S")
    features["created_at"] = now_kst_text()
    features = features[FEATURE_COLUMNS]
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.sort_values(["symbol", "ts_kst"]).reset_index(drop=True)

    valid_rows = len(df)
    logger.info("feature rows built input=%s valid=%s features=%s", input_rows, valid_rows, len(features))
    return features, input_rows, valid_rows


def save_features(
    *,
    features_df,
    db_path: Path,
    output_parquet: Path,
    logger: logging.Logger,
) -> int:
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_parquet, index=False)
    logger.info("feature parquet saved path=%s rows=%s", output_parquet, len(features_df))

    with sqlite3.connect(db_path) as conn:
        conn.execute(MINUTE_FEATURES_DDL)
        if features_df.empty:
            conn.commit()
            logger.info("feature SQLite saved rows=0")
            return 0
        rows = [
            tuple(None if value != value else value for value in row)
            for row in features_df[FEATURE_COLUMNS].itertuples(index=False, name=None)
        ]
        cursor = conn.executemany(
            """
            INSERT OR REPLACE INTO minute_features (
                symbol, ts_kst, ret_1m, ret_3m, ret_5m, ret_10m, ret_20m, ret_60m,
                range_1m, body, upper_shadow, lower_shadow, vol_5m, vol_20m, vol_60m,
                volume_1m, volume_change_1m, volume_ma_5m, volume_ma_20m,
                volume_z_20m, volume_z_60m, minutes_from_open, minutes_to_close,
                session_bucket, return_from_open, distance_from_day_high,
                distance_from_day_low, vwap, vwap_distance, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    saved_rows = int(cursor.rowcount or 0)
    logger.info("feature SQLite saved rows=%s", saved_rows)
    return saved_rows


def build_and_save_features(
    *,
    db_path: Path,
    input_table: str,
    parquet_fallback: Path,
    output_parquet: Path,
    symbol: str,
    source: str,
    logger: logging.Logger,
) -> FeatureBuildSummary:
    raw_df = load_minute_bars(
        db_path=db_path,
        input_table=input_table,
        parquet_fallback=parquet_fallback,
        symbol=symbol,
        source=source,
        logger=logger,
    )
    features_df, input_rows, valid_rows = build_minute_features(raw_df, logger=logger)
    sqlite_saved_rows = save_features(
        features_df=features_df,
        db_path=db_path,
        output_parquet=output_parquet,
        logger=logger,
    )
    if features_df.empty:
        date_range = ""
    else:
        date_range = f"{features_df['ts_kst'].min()} ~ {features_df['ts_kst'].max()}"
    return FeatureBuildSummary(
        input_rows=input_rows,
        valid_rows=valid_rows,
        feature_rows=len(features_df),
        date_range=date_range,
        output_parquet_path=output_parquet,
        sqlite_saved_rows=sqlite_saved_rows,
    )


def _load_sqlite_minute_bars(
    *,
    db_path: Path,
    input_table: str,
    symbol: str,
    logger: logging.Logger,
):
    import pandas as pd

    allowed_tables = {"minute_bars", "realtime_minute_bars"}
    if input_table not in allowed_tables:
        raise ValueError(f"input_table must be one of {sorted(allowed_tables)}")

    with sqlite3.connect(db_path) as conn:
        conn.execute(MINUTE_BARS_DDL)
        conn.execute(MINUTE_FEATURES_DDL)
        conn.commit()
        df = pd.read_sql_query(
            f"""
            SELECT symbol, ts_kst, open, high, low, close, volume
            FROM {input_table}
            WHERE symbol = ?
            """,
            conn,
            params=(symbol,),
        )
    logger.info("loaded SQLite rows=%s table=%s symbol=%s", len(df), input_table, symbol)
    return df


def _empty_feature_frame(pd):
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _compact_datetime_to_text(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 14 and text[:14].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]} {text[8:10]}:{text[10:12]}:{text[12:14]}"
    return text
