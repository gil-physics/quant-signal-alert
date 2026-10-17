"""Step 3: multi-symbol historical 1-minute collector.

`collect_1m(symbol, ...)` collects one symbol's historical 1-minute bars from the
minute_data CSVs and stores them into the standard SQLite table ``bars_1m`` using
the source classification from config.yaml:

    raw_source        (al_raw)         full integrated file, 08:00-19:59
    model_source      (al_regular)     integrated file filtered to the regular session
    comparison_source (single_regular) single-exchange regular file (comparison)

Imports are idempotent on (symbol, bar_start, source): re-collecting never
duplicates rows.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import DB_PATH, MINUTE_DATA_DIR, minute_csv_path
from csv_importer import (
    ImportSummary,
    create_bars_tables,
    import_csv_to_bars,
    import_records_to_bars,
)
from settings import load_data_config, load_session_config


@dataclass
class CollectSummary:
    symbol: str
    inserted_total: int
    per_source: Dict[str, ImportSummary] = field(default_factory=dict)
    missing_files: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if at least one source file was found and imported."""
        return bool(self.per_source)


def collect_1m(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
    db_path=DB_PATH,
    data_dir=None,
    logger=None,
) -> CollectSummary:
    """Collect one symbol's historical 1-minute bars into bars_1m.

    start_date / end_date are inclusive "YYYY-MM-DD" bounds (either may be None).
    Missing CSV files are recorded (not fatal) so the caller can report them.
    """
    if not symbol:
        raise ValueError("symbol is required")

    data_cfg = load_data_config()
    session = load_session_config()
    window = (session.train_start, session.train_end)
    base_dir = Path(data_dir) if data_dir else MINUTE_DATA_DIR

    al_file = minute_csv_path(symbol, integrated=True, data_dir=base_dir)
    single_file = minute_csv_path(symbol, integrated=False, data_dir=base_dir)
    date_range = (start_date, end_date) if (start_date or end_date) else None

    # (csv_path, source, session_window) — al_raw full, al_regular filtered, single_regular as-is.
    jobs = [
        (al_file, data_cfg.raw_source, None),
        (al_file, data_cfg.model_source, window),
        (single_file, data_cfg.comparison_source, None),
    ]

    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(db_path)
    try:
        create_bars_tables(conn)
        per_source: Dict[str, ImportSummary] = {}
        missing: List[str] = []
        for csv_path, source, win in jobs:
            if not csv_path.exists():
                if csv_path.name not in missing:
                    missing.append(csv_path.name)
                if logger:
                    logger.warning("missing file symbol=%s source=%s path=%s", symbol, source, csv_path)
                continue
            per_source[source] = import_csv_to_bars(
                conn,
                csv_path=csv_path,
                symbol=symbol,
                source=source,
                session_window=win,
                date_range=date_range,
                logger=logger,
            )
    finally:
        if own_conn:
            conn.close()

    inserted_total = sum(s.inserted for s in per_source.values())
    return CollectSummary(
        symbol=symbol,
        inserted_total=inserted_total,
        per_source=per_source,
        missing_files=missing,
    )


def store_minute_records(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    al_records=None,
    single_records=None,
    date_range=None,
    data_cfg=None,
    session=None,
    logger=None,
) -> CollectSummary:
    """Store already-fetched minute records (e.g. from Kiwoom) directly into bars_1m.

    al_records      -> raw_source (full) AND model_source (session-filtered)
    single_records  -> comparison_source

    Each record is a mapping with keys datetime/open/high/low/close/volume. No CSV
    is involved. Idempotent on (symbol, bar_start, source).
    """
    if not symbol:
        raise ValueError("symbol is required")
    data_cfg = data_cfg or load_data_config()
    session = session or load_session_config()
    window = (session.train_start, session.train_end)

    create_bars_tables(conn)
    per_source: Dict[str, ImportSummary] = {}

    if al_records is not None:
        al_records = list(al_records)  # iterated twice (raw + regular)
        per_source[data_cfg.raw_source] = import_records_to_bars(
            conn,
            records=al_records,
            symbol=symbol,
            source=data_cfg.raw_source,
            origin=f"kiwoom:{symbol}_AL",
            date_range=date_range,
            logger=logger,
        )
        per_source[data_cfg.model_source] = import_records_to_bars(
            conn,
            records=al_records,
            symbol=symbol,
            source=data_cfg.model_source,
            origin=f"kiwoom:{symbol}_AL",
            session_window=window,
            date_range=date_range,
            logger=logger,
        )

    if single_records is not None:
        per_source[data_cfg.comparison_source] = import_records_to_bars(
            conn,
            records=list(single_records),
            symbol=symbol,
            source=data_cfg.comparison_source,
            origin=f"kiwoom:{symbol}",
            date_range=date_range,
            logger=logger,
        )

    inserted_total = sum(s.inserted for s in per_source.values())
    return CollectSummary(
        symbol=symbol, inserted_total=inserted_total, per_source=per_source, missing_files=[]
    )


def symbol_coverage(
    conn: sqlite3.Connection,
    symbol: str,
    sources: List[str],
    *,
    model_source: str,
    table: str = "bars_1m",
) -> dict:
    """Coverage row for the missing report: per-source counts + model-source dates."""
    counts = {
        src: conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE symbol=? AND source=?", (symbol, src)
        ).fetchone()[0]
        for src in sources
    }
    dmin, dmax, ndays = conn.execute(
        f"SELECT MIN(substr(bar_start,1,10)), MAX(substr(bar_start,1,10)), "
        f"COUNT(DISTINCT substr(bar_start,1,10)) FROM {table} WHERE symbol=? AND source=?",
        (symbol, model_source),
    ).fetchone()
    model_rows = counts.get(model_source, 0)
    total = sum(counts.values())
    if total == 0:
        status = "MISSING"
    elif model_rows == 0:
        status = "NO_MODEL_ROWS"
    else:
        status = "OK"
    return {
        "symbol": symbol,
        "status": status,
        "total_rows": total,
        "model_rows": model_rows,
        "model_distinct_dates": ndays or 0,
        "date_min": dmin,
        "date_max": dmax,
        "counts": counts,
    }
