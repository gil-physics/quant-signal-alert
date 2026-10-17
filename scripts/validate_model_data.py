from __future__ import annotations

import argparse
import csv as csv_mod
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

from config import DB_PATH, PROCESSED_DIR, VALIDATE_LOG_PATH, ensure_directories
from data_validation import DEFAULT_RANGE_PCT, DEFAULT_SPIKE_PCT, validate_source
from settings import load_data_config, load_session_config
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: validate the model-source (al_regular) bars in bars_1m and write an anomaly report."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = all symbols present for the source.")
    parser.add_argument("--source", default=None, help="Source to validate; default = config.data.model_source.")
    parser.add_argument("--spike-pct", type=float, default=DEFAULT_SPIKE_PCT, help="Per-minute close-change flag threshold (fraction).")
    parser.add_argument("--range-pct", type=float, default=DEFAULT_RANGE_PCT, help="Intrabar (high-low)/low flag threshold (fraction).")
    parser.add_argument("--examples", type=int, default=5, help="Worst-N example rows per outlier kind.")
    parser.add_argument("--report-dir", default=str(PROCESSED_DIR), help="Where to write the report CSVs.")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(VALIDATE_LOG_PATH, name="validate")

    source = args.source or load_data_config().model_source
    session = load_session_config()
    window = (session.train_start, session.train_end)

    conn = sqlite3.connect(args.db)
    try:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM bars_1m WHERE source=? ORDER BY symbol", (source,)
                )
            ]

        print(f"[{now_kst_text()}] validating source={source} window={window[0]}..{window[1]} symbols={symbols}")
        logger.info("validate start source=%s symbols=%s spike=%s range=%s", source, symbols, args.spike_pct, args.range_pct)

        results = [
            validate_source(
                conn,
                sym,
                source=source,
                session_window=window,
                spike_pct=args.spike_pct,
                range_pct=args.range_pct,
                examples=args.examples,
            )
            for sym in symbols
        ]
    finally:
        conn.close()

    stamp = now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
    report_dir = Path(args.report_dir)
    summary_path = report_dir / f"al_regular_validation_{stamp}.csv"
    details_path = report_dir / f"al_regular_validation_details_{stamp}.csv"

    summary_fields = [
        "symbol", "source", "severity", "total_rows", "distinct_dates", "date_min", "date_max",
        "null_rows", "ohlc_violations", "out_of_window", "bad_bar_end", "duplicate_bar_start",
        "spike_count", "range_count", "max_abs_ret_pct", "max_range_pct",
    ]
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv_mod.writer(fh)
        w.writerow(summary_fields)
        for r in results:
            w.writerow([
                r.symbol, r.source, r.severity, r.total_rows, r.distinct_dates, r.date_min, r.date_max,
                r.null_rows, r.ohlc_violations, r.out_of_window, r.bad_bar_end, r.duplicate_bar_start,
                r.spike_count, r.range_count, r.max_abs_ret_pct, r.max_range_pct,
            ])

    with open(details_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv_mod.writer(fh)
        w.writerow(["symbol", "kind", "bar_start", "ref_value", "value", "metric_pct"])
        for r in results:
            for bar_start, prev_close, close, ret in r.spike_examples:
                w.writerow([r.symbol, "spike", bar_start, prev_close, close, round(ret * 100, 3)])
            for bar_start, low, high, rng in r.range_examples:
                w.writerow([r.symbol, "range", bar_start, low, high, round(rng * 100, 3)])

    print(f"\nsymbol     severity  rows     dates  spikes range  max_ret%  max_range%")
    severe = 0
    for r in results:
        if r.severity in ("SEVERE", "MISSING"):
            severe += 1
        print(
            f"  {r.symbol:<8} {r.severity:<8} {r.total_rows:>7}  {r.distinct_dates:>4}  "
            f"{r.spike_count:>5}  {r.range_count:>4}  {str(r.max_abs_ret_pct):>8}  {str(r.max_range_pct):>9}"
        )
    print(f"\nsummary -> {summary_path}")
    print(f"details -> {details_path}")
    print(f"[{now_kst_text()}] {len(results)} symbols validated, {severe} SEVERE/MISSING")
    logger.info("validate end symbols=%s severe=%s summary=%s", len(results), severe, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
