from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from config import MINUTE_BARS_DDL, REALTIME_MINUTE_BARS_DDL


@dataclass
class CompareSummary:
    total_bars: int
    matched_close_count: int
    matched_volume_count: int
    close_mismatch_count: int
    volume_mismatch_count: int
    max_close_diff: int
    max_volume_diff: int
    csv_path: Path


def compare_realtime_vs_opt10080(
    *,
    db_path: Path,
    output_dir: Path,
    symbol: str,
    date: str,
    logger: logging.Logger,
) -> CompareSummary:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    day_start = f"{date} 00:00:00"
    day_end = f"{date} 23:59:59"

    with sqlite3.connect(db_path) as conn:
        conn.execute(MINUTE_BARS_DDL)
        conn.execute(REALTIME_MINUTE_BARS_DDL)
        conn.commit()
        realtime = pd.read_sql_query(
            """
            SELECT symbol, ts_kst, open, high, low, close, volume
            FROM realtime_minute_bars
            WHERE symbol = ? AND ts_kst BETWEEN ? AND ?
            """,
            conn,
            params=(symbol, day_start, day_end),
        )
        opt10080 = pd.read_sql_query(
            """
            SELECT symbol, ts_kst, open, high, low, close, volume
            FROM minute_bars
            WHERE symbol = ? AND ts_kst BETWEEN ? AND ?
            """,
            conn,
            params=(symbol, day_start, day_end),
        )

    logger.info(
        "compare rows loaded symbol=%s date=%s realtime=%s opt10080=%s",
        symbol,
        date,
        len(realtime),
        len(opt10080),
    )

    merged = realtime.merge(
        opt10080,
        on=["symbol", "ts_kst"],
        how="outer",
        suffixes=("_realtime", "_opt10080"),
        indicator=True,
    ).sort_values("ts_kst")

    for col in ["open", "high", "low", "close", "volume"]:
        merged[f"{col}_diff"] = merged[f"{col}_realtime"] - merged[f"{col}_opt10080"]

    merged["match_close"] = merged["close_diff"] == 0
    merged["match_volume"] = merged["volume_diff"] == 0

    csv_path = output_dir / f"realtime_vs_opt10080_{date.replace('-', '')}.csv"
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")

    total_bars = int(len(merged))
    matched_close_count = int(merged["match_close"].fillna(False).sum())
    matched_volume_count = int(merged["match_volume"].fillna(False).sum())
    close_mismatch_count = total_bars - matched_close_count
    volume_mismatch_count = total_bars - matched_volume_count
    max_close_diff = _max_abs_int(merged["close_diff"])
    max_volume_diff = _max_abs_int(merged["volume_diff"])

    logger.info(
        "compare summary total_bars=%s matched_close=%s matched_volume=%s "
        "close_mismatch=%s volume_mismatch=%s max_close_diff=%s max_volume_diff=%s csv_path=%s",
        total_bars,
        matched_close_count,
        matched_volume_count,
        close_mismatch_count,
        volume_mismatch_count,
        max_close_diff,
        max_volume_diff,
        csv_path,
    )

    return CompareSummary(
        total_bars=total_bars,
        matched_close_count=matched_close_count,
        matched_volume_count=matched_volume_count,
        close_mismatch_count=close_mismatch_count,
        volume_mismatch_count=volume_mismatch_count,
        max_close_diff=max_close_diff,
        max_volume_diff=max_volume_diff,
        csv_path=csv_path,
    )


def _max_abs_int(series) -> int:
    clean = series.dropna()
    if clean.empty:
        return 0
    return int(clean.abs().max())
