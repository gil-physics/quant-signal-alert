"""Step 5: build 3-minute / 5-minute bars from 1-minute bars.

Buckets are aligned to the session open (09:00):
  3m: 09:00-09:03, 09:03-09:06, ...   (3 one-minute bars each)
  5m: 09:00-09:05, 09:05-09:10, ...   (5 one-minute bars each)

Aggregation per bucket:
  open=first 1m open, high=max high, low=min low, close=last 1m close, volume=sum.

A bucket is kept only when it is COMPLETE (contains all ``interval_minutes``
one-minute bars). The trailing partial bucket (e.g. the lone 15:30 closing-auction
bar) is therefore dropped — "완성되지 않은 마지막 봉은 학습에 쓰지 않는다".
"""

from __future__ import annotations

import sqlite3

from utils import now_kst_text

_OUT_COLUMNS = ["symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume"]
_INSERT_COLUMNS = _OUT_COLUMNS + ["source", "created_at"]


def resample_ohlcv(df_1m, interval_minutes: int, *, session_open: str = "09:00:00"):
    """Resample a 1-minute OHLCV DataFrame into ``interval_minutes`` bars.

    df_1m needs columns: symbol, bar_start ("YYYY-MM-DD HH:MM:SS"), open, high,
    low, close, volume. Works across multiple symbols and days at once. Returns a
    DataFrame with columns symbol, bar_start, bar_end, open, high, low, close, volume.
    """
    import pandas as pd

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    if df_1m is None or len(df_1m) == 0:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    df = df_1m.copy()
    df["_ts"] = pd.to_datetime(df["bar_start"])
    df["_date"] = df["_ts"].dt.strftime("%Y-%m-%d")
    df["_open_dt"] = pd.to_datetime(df["_date"] + " " + session_open)
    df["_minutes"] = ((df["_ts"] - df["_open_dt"]).dt.total_seconds() // 60).astype("int64")

    df = df[df["_minutes"] >= 0].copy()  # ignore anything before the session open
    if df.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    bucket_offset = (df["_minutes"] // interval_minutes) * interval_minutes
    df["_bucket_start"] = df["_open_dt"] + pd.to_timedelta(bucket_offset, unit="m")
    df = df.sort_values(["symbol", "_ts"])

    grouped = df.groupby(["symbol", "_bucket_start"], sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n=("close", "size"),
    ).reset_index()

    grouped = grouped[grouped["n"] == interval_minutes].copy()  # complete buckets only
    if grouped.empty:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    grouped["bar_start"] = grouped["_bucket_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    grouped["bar_end"] = (
        grouped["_bucket_start"] + pd.to_timedelta(interval_minutes, unit="m")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ("open", "high", "low", "close", "volume"):
        grouped[col] = grouped[col].astype("int64")

    return grouped[_OUT_COLUMNS].reset_index(drop=True)


def store_bars(conn: sqlite3.Connection, df, *, table: str, source: str, logger=None) -> dict:
    """Insert resampled bars into ``table`` (idempotent on the primary key)."""
    created_at = now_kst_text()
    rows = [
        (
            r.symbol,
            r.bar_start,
            r.bar_end,
            int(r.open),
            int(r.high),
            int(r.low),
            int(r.close),
            int(r.volume),
            source,
            created_at,
        )
        for r in df.itertuples(index=False)
    ]
    placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
    sql = f"INSERT OR IGNORE INTO {table} ({', '.join(_INSERT_COLUMNS)}) VALUES ({placeholders})"
    before = conn.total_changes
    conn.executemany(sql, rows)
    conn.commit()
    inserted = conn.total_changes - before
    result = {
        "table": table,
        "source": source,
        "total": len(rows),
        "inserted": inserted,
        "skipped_existing": len(rows) - inserted,
    }
    if logger:
        logger.info("store_bars table=%s source=%s total=%s inserted=%s skipped=%s",
                    table, source, result["total"], result["inserted"], result["skipped_existing"])
    return result
