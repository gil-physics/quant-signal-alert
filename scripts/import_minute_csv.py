from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import (
    AL_RAW_SOURCE,
    AL_REGULAR_SOURCE,
    DB_PATH,
    IMPORT_LOG_PATH,
    MINUTE_DATA_DIR,
    SINGLE_REGULAR_SOURCE,
    ensure_directories,
    minute_csv_path,
)
from csv_importer import create_bars_tables, import_csv_to_bars
from settings import default_symbol, load_session_config
from utils import setup_logger

JOB_NAMES = ["al_raw", "al_regular", "single_regular"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import minute CSVs from minute_data into the unified bars_1m table. "
        "The integrated-quote (AL) file is the basis: al_raw (full), al_regular "
        "(AL filtered to the regular session), and single_regular (single-exchange, comparison)."
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol code. Defaults to the first active symbol in config/symbols_test.yaml.",
    )
    parser.add_argument(
        "--jobs",
        default="all",
        help="Comma-separated subset of {al_raw,al_regular,single_regular} or 'all' (default).",
    )
    parser.add_argument("--csv", default=None, help="Explicit CSV path (overrides --jobs).")
    parser.add_argument(
        "--source", default=None, help="Source tag for an explicit --csv import (default al_raw)."
    )
    parser.add_argument(
        "--data-dir",
        default=str(MINUTE_DATA_DIR),
        help="Directory holding the minute CSV files.",
    )
    parser.add_argument("--table", default="bars_1m", help="Target table (default bars_1m).")
    return parser.parse_args()


def _resolve_jobs(args, symbol: str):
    """Return a list of (csv_path, source, session_window)."""
    if args.csv:
        return [(Path(args.csv), args.source or AL_RAW_SOURCE, None)]

    data_dir = Path(args.data_dir)
    al_file = minute_csv_path(symbol, integrated=True, data_dir=data_dir)
    single_file = minute_csv_path(symbol, integrated=False, data_dir=data_dir)

    session = load_session_config()
    window = (session.train_start, session.train_end)

    catalogue = {
        "al_raw": (al_file, AL_RAW_SOURCE, None),
        "al_regular": (al_file, AL_REGULAR_SOURCE, window),
        "single_regular": (single_file, SINGLE_REGULAR_SOURCE, None),
    }

    selected = JOB_NAMES if args.jobs.strip() == "all" else [j.strip() for j in args.jobs.split(",")]
    unknown = [j for j in selected if j not in catalogue]
    if unknown:
        raise SystemExit(f"unknown --jobs values: {unknown}; choose from {JOB_NAMES} or 'all'")
    return [catalogue[j] for j in selected]


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(IMPORT_LOG_PATH, name="import")
    symbol = args.symbol or default_symbol()

    conn = sqlite3.connect(DB_PATH)
    try:
        create_bars_tables(conn)
        for csv_path, source, window in _resolve_jobs(args, symbol):
            if not csv_path.exists():
                print(f"[SKIP] missing file: {csv_path}")
                logger.warning("missing csv path=%s symbol=%s source=%s", csv_path, symbol, source)
                continue
            summary = import_csv_to_bars(
                conn,
                csv_path=csv_path,
                symbol=symbol,
                source=source,
                table=args.table,
                session_window=window,
                logger=logger,
            )
            win = f" window={window[0]}..{window[1]}" if window else ""
            print(
                f"[OK] {csv_path.name} -> {summary.table} | source={summary.source}{win} "
                f"read={summary.read_rows} inserted={summary.inserted} "
                f"skipped_existing={summary.skipped_existing} skipped_invalid={summary.skipped_invalid} "
                f"filtered_out={summary.filtered_out} range={summary.bar_start_min}..{summary.bar_start_max}"
            )

        print(f"\n-- SELECT symbol, source, COUNT(*) FROM {args.table} GROUP BY symbol, source --")
        for row in conn.execute(
            f"SELECT symbol, source, COUNT(*) FROM {args.table} GROUP BY symbol, source ORDER BY symbol, source"
        ):
            print(f"  symbol={row[0]} source={row[1]} count={row[2]}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
