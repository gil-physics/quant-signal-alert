from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PyQt5.QtWidgets import QApplication

from config import (
    DB_PATH,
    DEFAULT_MAX_PAGES,
    DEFAULT_REQUEST_DELAY_SEC,
    DEFAULT_SCREEN_NO,
    DEFAULT_TICK,
    LOG_PATH,
    ensure_directories,
    raw_parquet_path,
)
from kiwoom_client import KiwoomClient
from minute_collector import MinuteBarCollector
from settings import default_symbol
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 1-minute OHLCV bars with Kiwoom opt10080."
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol code. Defaults to the first active symbol in config/symbols_test.yaml.",
    )
    parser.add_argument("--tick", default=DEFAULT_TICK)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--request-delay-sec", type=float, default=DEFAULT_REQUEST_DELAY_SEC)
    parser.add_argument("--screen-no", default=DEFAULT_SCREEN_NO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(LOG_PATH)
    symbol = args.symbol or default_symbol()

    if platform.architecture()[0] != "32bit":
        message = "Kiwoom OpenAPI+ requires 32-bit Python."
        logger.error(message)
        raise SystemExit(message)

    app = QApplication.instance() or QApplication(sys.argv)
    client = KiwoomClient(logger=logger, screen_no=args.screen_no)

    if not client.login():
        print("[FAIL] Login failed")
        return 1
    print("[OK] Login success")

    print(f"[OK] Requested opt10080: symbol={symbol}, tick={args.tick}")
    raw_rows = client.request_minute_bars(
        symbol=symbol,
        tick=str(args.tick),
        max_pages=args.max_pages,
        request_delay_sec=args.request_delay_sec,
    )
    print(f"[OK] Received rows: {len(raw_rows)}")

    collector = MinuteBarCollector(
        db_path=DB_PATH,
        parquet_path=raw_parquet_path(symbol),
        logger=logger,
    )
    collector.initialize_storage()
    summary = collector.collect_to_storage(symbol=symbol, raw_rows=raw_rows)

    print(f"[OK] Valid rows: {summary.valid_rows}")
    print(f"[OK] SQLite inserted: {summary.sqlite_inserted}")
    print(f"[OK] SQLite duplicate skipped: {summary.sqlite_skipped_duplicates}")
    print(f"[OK] Parquet saved: {summary.parquet_path}")

    logger.info("final parquet path=%s", summary.parquet_path)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
