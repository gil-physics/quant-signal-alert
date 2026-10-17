from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    DB_PATH,
    FEATURE_LOG_PATH,
    ensure_directories,
    feature_parquet_path,
    raw_parquet_path,
)
from features import build_and_save_features
from settings import default_symbol
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model features from 1-minute OHLCV bars.")
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol code. Defaults to the first active symbol in config/symbols_test.yaml.",
    )
    parser.add_argument("--source", choices=["sqlite", "parquet"], default="sqlite")
    parser.add_argument("--input-table", default="minute_bars")
    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Output parquet path. Defaults to a per-symbol path under data/processed/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(FEATURE_LOG_PATH, name="features")
    symbol = args.symbol or default_symbol()
    output_parquet = Path(args.output_parquet) if args.output_parquet else feature_parquet_path(symbol)
    if not output_parquet.is_absolute():
        output_parquet = ROOT_DIR / output_parquet

    summary = build_and_save_features(
        db_path=DB_PATH,
        input_table=args.input_table,
        parquet_fallback=raw_parquet_path(symbol),
        output_parquet=output_parquet,
        symbol=symbol,
        source=args.source,
        logger=logger,
    )

    print(f"input rows: {summary.input_rows}")
    print(f"valid rows: {summary.valid_rows}")
    print(f"feature rows: {summary.feature_rows}")
    print(f"date range: {summary.date_range}")
    print(f"output parquet path: {summary.output_parquet_path}")
    print(f"SQLite saved rows: {summary.sqlite_saved_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
