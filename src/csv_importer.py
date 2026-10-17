"""Step 2: import raw minute CSV files into the unified, symbol-keyed bar tables.

Source CSV columns: datetime,open,high,low,close,volume,value
  datetime  YYYYMMDDHHMMSS -> bar_start "YYYY-MM-DD HH:MM:SS", bar_end = +1 minute
  open/high/low/close/volume -> integers
  value     -> dropped (not in the bars schema; empty in the current feed)
  symbol    -> supplied by caller (from filename/CLI)
  source    -> supplied by caller (e.g. csv_regular, csv_integrated)
  created_at-> import time (KST)

Imports are idempotent: re-running INSERTs OR IGNOREs on the primary key.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from config import BARS_1M_DDL, BARS_3M_DDL, BARS_5M_DDL
from utils import normalize_ts_kst, now_kst_text

ISO_FMT = "%Y-%m-%d %H:%M:%S"
REQUIRED_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
_INSERT_COLUMNS = (
    "symbol",
    "bar_start",
    "bar_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "created_at",
)


@dataclass
class ImportSummary:
    csv_path: str
    symbol: str
    source: str
    table: str
    read_rows: int
    inserted: int
    skipped_existing: int
    skipped_invalid: int
    filtered_out: int
    bar_start_min: Optional[str]
    bar_start_max: Optional[str]


def create_bars_tables(conn: sqlite3.Connection) -> None:
    """Create bars_1m / bars_3m / bars_5m if they do not exist."""
    for ddl in (BARS_1M_DDL, BARS_3M_DDL, BARS_5M_DDL):
        conn.execute(ddl)
    conn.commit()


def _bar_end(bar_start: str) -> str:
    return (datetime.strptime(bar_start, ISO_FMT) + timedelta(minutes=1)).strftime(ISO_FMT)


def _ingest(
    conn: sqlite3.Connection,
    records,
    *,
    symbol: str,
    source: str,
    table: str,
    session_window: Optional[Tuple[str, str]],
    date_range: Optional[Tuple[Optional[str], Optional[str]]],
    origin: str,
    logger,
) -> ImportSummary:
    """Shared core: convert raw records -> bar rows, filter, insert OR IGNORE."""
    if not symbol:
        raise ValueError("symbol is required")
    if not source:
        raise ValueError("source is required")

    created_at = now_kst_text()
    read_rows = 0
    skipped_invalid = 0
    filtered_out = 0
    rows = []

    for row in records:
        read_rows += 1
        try:
            bar_start = normalize_ts_kst(row["datetime"])
            if session_window is not None:
                time_of_day = bar_start[11:]
                if not (session_window[0] <= time_of_day <= session_window[1]):
                    filtered_out += 1
                    continue
            if date_range is not None:
                day = bar_start[:10]
                lo, hi = date_range
                if (lo and day < lo) or (hi and day > hi):
                    filtered_out += 1
                    continue
            bar_end = _bar_end(bar_start)
            values = (
                symbol,
                bar_start,
                bar_end,
                int(row["open"]),
                int(row["high"]),
                int(row["low"]),
                int(row["close"]),
                int(row["volume"]),
                source,
                created_at,
            )
        except (ValueError, KeyError, TypeError):
            skipped_invalid += 1
            if logger:
                logger.warning("skip invalid row #%s in %s: %r", read_rows, origin, row)
            continue
        rows.append(values)

    placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
    sql = f"INSERT OR IGNORE INTO {table} ({', '.join(_INSERT_COLUMNS)}) VALUES ({placeholders})"
    before = conn.total_changes
    conn.executemany(sql, rows)
    conn.commit()
    inserted = conn.total_changes - before
    skipped_existing = len(rows) - inserted

    bar_starts = [r[1] for r in rows]
    summary = ImportSummary(
        csv_path=origin,
        symbol=symbol,
        source=source,
        table=table,
        read_rows=read_rows,
        inserted=inserted,
        skipped_existing=skipped_existing,
        skipped_invalid=skipped_invalid,
        filtered_out=filtered_out,
        bar_start_min=min(bar_starts) if bar_starts else None,
        bar_start_max=max(bar_starts) if bar_starts else None,
    )
    if logger:
        logger.info(
            "import done origin=%s symbol=%s source=%s table=%s read=%s inserted=%s "
            "skipped_existing=%s skipped_invalid=%s filtered_out=%s range=%s..%s",
            origin,
            symbol,
            source,
            table,
            summary.read_rows,
            summary.inserted,
            summary.skipped_existing,
            summary.skipped_invalid,
            summary.filtered_out,
            summary.bar_start_min,
            summary.bar_start_max,
        )
    return summary


def import_csv_to_bars(
    conn: sqlite3.Connection,
    *,
    csv_path,
    symbol: str,
    source: str,
    table: str = "bars_1m",
    session_window: Optional[Tuple[str, str]] = None,
    date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    logger=None,
) -> ImportSummary:
    """Import one CSV file into ``table`` for ``symbol`` tagged with ``source``.

    If ``session_window`` is given as ("HH:MM:SS", "HH:MM:SS"), only rows whose
    time-of-day falls within that inclusive window are kept (used to derive the
    regular-session series from the full integrated file).

    If ``date_range`` is given as ("YYYY-MM-DD" | None, "YYYY-MM-DD" | None),
    only rows whose date falls within that inclusive range are kept. Either
    bound may be None to leave that side open.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")
        return _ingest(
            conn,
            reader,
            symbol=symbol,
            source=source,
            table=table,
            session_window=session_window,
            date_range=date_range,
            origin=path.name,
            logger=logger,
        )


def import_records_to_bars(
    conn: sqlite3.Connection,
    *,
    records,
    symbol: str,
    source: str,
    origin: str = "records",
    table: str = "bars_1m",
    session_window: Optional[Tuple[str, str]] = None,
    date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    logger=None,
) -> ImportSummary:
    """Import an iterable of record mappings (datetime/open/high/low/close/volume)
    into ``table``. Same filtering/idempotency as the CSV path; used by the direct
    Kiwoom-to-DB collector so no CSV file is involved."""
    return _ingest(
        conn,
        records,
        symbol=symbol,
        source=source,
        table=table,
        session_window=session_window,
        date_range=date_range,
        origin=origin,
        logger=logger,
    )
