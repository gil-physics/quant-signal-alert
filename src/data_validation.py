"""Step 4: validate the model-source (al_regular) bars in bars_1m.

Produces an anomaly report per symbol WITHOUT modifying or dropping any data.
Whether to exclude flagged bars from training is decided later (Step 6).

Checks (all computed in SQL, stdlib only):
  structural (errors): NULL columns, OHLC inconsistency, out-of-session bars,
                       bar_end != bar_start + 1 minute, duplicate (symbol, bar_start)
  review (not errors): per-minute close-to-close spikes and intrabar range
                       outliers above configurable thresholds (within a trade
                       day, so overnight gaps are never counted)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_SESSION_WINDOW = ("09:00:00", "15:30:00")
DEFAULT_SPIKE_PCT = 0.10   # |close_t / close_{t-1} - 1| above this is flagged
DEFAULT_RANGE_PCT = 0.10   # (high - low) / low above this is flagged


@dataclass
class ValidationResult:
    symbol: str
    source: str
    total_rows: int
    distinct_dates: int
    date_min: Optional[str]
    date_max: Optional[str]
    # structural error counts
    null_rows: int
    ohlc_violations: int
    out_of_window: int
    bad_bar_end: int
    duplicate_bar_start: int
    # review-level counts + observed extremes
    spike_count: int
    range_count: int
    max_abs_ret_pct: Optional[float]
    max_range_pct: Optional[float]
    spike_examples: List[tuple] = field(default_factory=list)
    range_examples: List[tuple] = field(default_factory=list)

    @property
    def structural_total(self) -> int:
        return (
            self.null_rows
            + self.ohlc_violations
            + self.out_of_window
            + self.bad_bar_end
            + self.duplicate_bar_start
        )

    @property
    def severity(self) -> str:
        if self.total_rows == 0:
            return "MISSING"
        if self.structural_total > 0:
            return "SEVERE"
        if self.spike_count + self.range_count > 0:
            return "REVIEW"
        return "OK"


def _scalar(conn, sql, params):
    return conn.execute(sql, params).fetchone()[0]


def validate_source(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    source: str,
    session_window: Tuple[str, str] = DEFAULT_SESSION_WINDOW,
    spike_pct: float = DEFAULT_SPIKE_PCT,
    range_pct: float = DEFAULT_RANGE_PCT,
    examples: int = 5,
    table: str = "bars_1m",
) -> ValidationResult:
    p = (symbol, source)
    win_lo, win_hi = session_window

    total = _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=?", p)
    dmin, dmax, ndays = conn.execute(
        f"SELECT MIN(bar_start), MAX(bar_start), COUNT(DISTINCT substr(bar_start,1,10)) "
        f"FROM {table} WHERE symbol=? AND source=?",
        p,
    ).fetchone()

    null_rows = _scalar(
        conn,
        f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=? AND ("
        "bar_start IS NULL OR bar_end IS NULL OR open IS NULL OR high IS NULL "
        "OR low IS NULL OR close IS NULL OR volume IS NULL)",
        p,
    )
    ohlc_violations = _scalar(
        conn,
        f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=? AND ("
        "high < low OR high < open OR high < close OR low > open OR low > close "
        "OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR volume < 0)",
        p,
    )
    out_of_window = _scalar(
        conn,
        f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=? "
        "AND substr(bar_start,12) NOT BETWEEN ? AND ?",
        (symbol, source, win_lo, win_hi),
    )
    bad_bar_end = _scalar(
        conn,
        f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=? "
        "AND bar_end != strftime('%Y-%m-%d %H:%M:%S', datetime(bar_start, '+1 minute'))",
        p,
    )
    duplicate_bar_start = _scalar(
        conn,
        f"SELECT COUNT(*) FROM (SELECT bar_start FROM {table} WHERE symbol=? AND source=? "
        "GROUP BY bar_start HAVING COUNT(*) > 1)",
        p,
    )

    # close-to-close return, partitioned by trade day so overnight gaps don't count
    ret_cte = (
        f"WITH d AS (SELECT bar_start, close, "
        f"LAG(close) OVER (PARTITION BY substr(bar_start,1,10) ORDER BY bar_start) AS prev_close "
        f"FROM {table} WHERE symbol=? AND source=?) "
    )
    spike_count = _scalar(
        conn,
        ret_cte + "SELECT COUNT(*) FROM d WHERE prev_close IS NOT NULL AND prev_close > 0 "
        "AND ABS(CAST(close AS REAL)/prev_close - 1.0) > ?",
        (symbol, source, spike_pct),
    )
    max_abs_ret = _scalar(
        conn,
        ret_cte + "SELECT MAX(ABS(CAST(close AS REAL)/prev_close - 1.0)) FROM d "
        "WHERE prev_close IS NOT NULL AND prev_close > 0",
        p,
    )
    spike_examples = conn.execute(
        ret_cte + "SELECT bar_start, prev_close, close, "
        "ROUND(CAST(close AS REAL)/prev_close - 1.0, 5) AS ret FROM d "
        "WHERE prev_close IS NOT NULL AND prev_close > 0 "
        "AND ABS(CAST(close AS REAL)/prev_close - 1.0) > ? "
        "ORDER BY ABS(CAST(close AS REAL)/prev_close - 1.0) DESC LIMIT ?",
        (symbol, source, spike_pct, examples),
    ).fetchall()

    range_count = _scalar(
        conn,
        f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=? AND low > 0 "
        "AND CAST(high - low AS REAL)/low > ?",
        (symbol, source, range_pct),
    )
    max_range = _scalar(
        conn,
        f"SELECT MAX(CAST(high - low AS REAL)/low) FROM {table} "
        "WHERE symbol=? AND source=? AND low > 0",
        p,
    )
    range_examples = conn.execute(
        f"SELECT bar_start, low, high, ROUND(CAST(high - low AS REAL)/low, 5) AS rng "
        f"FROM {table} WHERE symbol=? AND source=? AND low > 0 "
        "AND CAST(high - low AS REAL)/low > ? ORDER BY CAST(high - low AS REAL)/low DESC LIMIT ?",
        (symbol, source, range_pct, examples),
    ).fetchall()

    return ValidationResult(
        symbol=symbol,
        source=source,
        total_rows=total,
        distinct_dates=ndays or 0,
        date_min=dmin,
        date_max=dmax,
        null_rows=null_rows,
        ohlc_violations=ohlc_violations,
        out_of_window=out_of_window,
        bad_bar_end=bad_bar_end,
        duplicate_bar_start=duplicate_bar_start,
        spike_count=spike_count,
        range_count=range_count,
        max_abs_ret_pct=round(max_abs_ret * 100, 4) if max_abs_ret is not None else None,
        max_range_pct=round(max_range * 100, 4) if max_range is not None else None,
        spike_examples=spike_examples,
        range_examples=range_examples,
    )
