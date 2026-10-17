"""Step 6: data-quality checks over bars_1m / bars_3m / bars_5m.

Per (symbol, date, timeframe) the report records: actual bar count vs a data-driven
expected range, missing/range deviation, duplicate bar_start, OHLC relation errors,
price<=0, negative volume, bar_end interval errors, out-of-session rows,
source-mixing (rows under unexpected sources), and abnormal spikes / range outliers.

Status policy (step6.md #9/#10 + completion criteria):
  FAIL  duplicate, OHLC error, price<=0, volume<0, bar_end error, out-of-session,
        source-mixing  (hard structural errors — must not enter training)
  WARN  missing bars / actual outside expected range, abnormal spikes, range outliers
  OK    otherwise

Step 6 only reports; it never deletes or edits data.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Sequence, Tuple

DEFAULT_SPIKE_PCT = 0.10   # |close_t/close_{t-1} - 1| within a day
DEFAULT_RANGE_PCT = 0.10   # (high - low) / low
DEFAULT_RANGE_TOL = 0.10   # acceptable band around the symbol's median daily count

# Report column order (CSV).
REPORT_COLUMNS = [
    "symbol", "date", "timeframe", "source", "expected", "actual", "missing",
    "duplicate", "ohlc_error", "price_zero", "volume_negative", "bar_end_error",
    "out_of_session", "source_mixing", "spike", "range_outlier", "status",
]


def compute_day_metrics(
    group,
    *,
    session_window: Tuple[str, str],
    spike_pct: float,
    range_pct: float,
    source_mixing_count: int,
    expected_lo: int,
    expected_hi: int,
) -> Dict:
    """Pure per-day metric computation on a DataFrame group.

    The group must have columns: bar_start, open, high, low, close, volume,
    and _exp_end (the expected bar_end), sorted by bar_start.
    """
    g = group
    actual = len(g)
    # Defensive: within one (symbol, source) the PK (symbol, bar_start, source)
    # guarantees unique bar_start, so this is normally 0. Kept as a safety net and
    # so the check is exercised on raw (pre-insert) frames in tests.
    duplicate = int(g["bar_start"].duplicated().sum())
    ohlc_error = int(
        (
            (g["high"] < g["low"])
            | (g["high"] < g["open"])
            | (g["high"] < g["close"])
            | (g["low"] > g["open"])
            | (g["low"] > g["close"])
        ).sum()
    )
    price_zero = int((g[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    volume_negative = int((g["volume"] < 0).sum())
    bar_end_error = int((g["bar_end"] != g["_exp_end"]).sum())
    times = g["bar_start"].str.slice(11)
    out_of_session = int(((times < session_window[0]) | (times > session_window[1])).sum())

    prev_close = g["close"].shift(1)
    ret = (g["close"] / prev_close - 1).abs()
    spike = int((ret > spike_pct).sum())
    low_safe = g["low"].where(g["low"] > 0)
    rng = (g["high"] - g["low"]) / low_safe
    range_outlier = int((rng > range_pct).sum())

    missing = max(0, expected_lo - actual)

    fail = (
        duplicate > 0
        or ohlc_error > 0
        or price_zero > 0
        or volume_negative > 0
        or bar_end_error > 0
        or out_of_session > 0
        or source_mixing_count > 0
    )
    warn = (actual < expected_lo or actual > expected_hi) or spike > 0 or range_outlier > 0
    status = "FAIL" if fail else ("WARN" if warn else "OK")

    return {
        "actual": actual,
        "missing": missing,
        "duplicate": duplicate,
        "ohlc_error": ohlc_error,
        "price_zero": price_zero,
        "volume_negative": volume_negative,
        "bar_end_error": bar_end_error,
        "out_of_session": out_of_session,
        "source_mixing": int(source_mixing_count),
        "spike": spike,
        "range_outlier": range_outlier,
        "status": status,
    }


def check_table(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    table: str,
    timeframe: str,
    interval_minutes: int,
    expected_source: str,
    allowed_sources: Sequence[str],
    session_window: Tuple[str, str] = ("09:00:00", "15:30:00"),
    spike_pct: float = DEFAULT_SPIKE_PCT,
    range_pct: float = DEFAULT_RANGE_PCT,
    range_tol: float = DEFAULT_RANGE_TOL,
) -> List[Dict]:
    """Per-day quality rows for one (symbol, table). Checks only ``expected_source``;
    source-mixing is detected against ``allowed_sources`` over the whole table."""
    import pandas as pd

    df = pd.read_sql_query(
        f"SELECT bar_start, bar_end, open, high, low, close, volume FROM {table} "
        "WHERE symbol = ? AND source = ? ORDER BY bar_start",
        conn,
        params=(symbol, expected_source),
    )

    placeholders = ", ".join("?" for _ in allowed_sources)
    mix = pd.read_sql_query(
        f"SELECT substr(bar_start,1,10) AS d, COUNT(*) AS n FROM {table} "
        f"WHERE symbol = ? AND source NOT IN ({placeholders}) GROUP BY d",
        conn,
        params=(symbol, *allowed_sources),
    )
    mix_by_date = dict(zip(mix["d"].tolist(), mix["n"].tolist()))

    base = {"symbol": symbol, "timeframe": timeframe, "source": expected_source}

    if df.empty:
        # No model rows at all. Report a single FAIL row; flag any stray sources.
        rows = []
        if mix_by_date:
            for date, n in sorted(mix_by_date.items()):
                rows.append({**base, "date": date, "expected": "0", "actual": 0, "missing": 0,
                             "duplicate": 0, "ohlc_error": 0, "price_zero": 0, "volume_negative": 0,
                             "bar_end_error": 0, "out_of_session": 0, "source_mixing": int(n),
                             "spike": 0, "range_outlier": 0, "status": "FAIL"})
        else:
            # No data at all = the extreme of "missing" -> WARN per step6.md #9 (not a
            # data-corruption FAIL). Surfaced for review/exclusion before training.
            rows.append({**base, "date": "-", "expected": "-", "actual": 0, "missing": 0,
                         "duplicate": 0, "ohlc_error": 0, "price_zero": 0, "volume_negative": 0,
                         "bar_end_error": 0, "out_of_session": 0, "source_mixing": 0,
                         "spike": 0, "range_outlier": 0, "status": "WARN"})
        return rows

    df["_ts"] = pd.to_datetime(df["bar_start"])
    df["_exp_end"] = (df["_ts"] + pd.to_timedelta(interval_minutes, unit="m")).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["date"] = df["bar_start"].str.slice(0, 10)

    daily_counts = df.groupby("date").size()
    median = float(daily_counts.median())
    expected_lo = int(round(median * (1 - range_tol)))
    expected_hi = int(round(median * (1 + range_tol)))
    expected_label = f"{expected_lo}-{expected_hi}"

    rows = []
    seen_dates = set()
    for date, g in df.groupby("date", sort=True):
        seen_dates.add(date)
        metrics = compute_day_metrics(
            g,
            session_window=session_window,
            spike_pct=spike_pct,
            range_pct=range_pct,
            source_mixing_count=mix_by_date.get(date, 0),
            expected_lo=expected_lo,
            expected_hi=expected_hi,
        )
        rows.append({**base, "date": date, "expected": expected_label, **metrics})

    # dates that ONLY have stray-source rows (no expected-source bars)
    for date, n in sorted(mix_by_date.items()):
        if date not in seen_dates:
            rows.append({**base, "date": date, "expected": expected_label, "actual": 0, "missing": expected_lo,
                         "duplicate": 0, "ohlc_error": 0, "price_zero": 0, "volume_negative": 0,
                         "bar_end_error": 0, "out_of_session": 0, "source_mixing": int(n),
                         "spike": 0, "range_outlier": 0, "status": "FAIL"})
    return rows
