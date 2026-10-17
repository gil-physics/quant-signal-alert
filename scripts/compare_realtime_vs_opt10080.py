from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import COMPARE_LOG_PATH, DB_PATH, PROCESSED_DIR, ensure_directories
from realtime_compare import compare_realtime_vs_opt10080
from settings import default_symbol
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare realtime minute bars with opt10080 minute bars.")
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol code. Defaults to the first active symbol in config/symbols_test.yaml.",
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(COMPARE_LOG_PATH, name="compare")
    symbol = args.symbol or default_symbol()
    summary = compare_realtime_vs_opt10080(
        db_path=DB_PATH,
        output_dir=PROCESSED_DIR,
        symbol=symbol,
        date=args.date,
        logger=logger,
    )

    print(f"total_bars={summary.total_bars}")
    print(f"matched_close_count={summary.matched_close_count}")
    print(f"matched_volume_count={summary.matched_volume_count}")
    print(f"close_mismatch_count={summary.close_mismatch_count}")
    print(f"volume_mismatch_count={summary.volume_mismatch_count}")
    print(f"max_close_diff={summary.max_close_diff}")
    print(f"max_volume_diff={summary.max_volume_diff}")
    print(f"csv_path={summary.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
