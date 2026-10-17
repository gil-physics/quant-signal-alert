from __future__ import annotations

import argparse
import csv as csv_mod
import sqlite3
import sys
from collections import Counter
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
    PROCESSED_DIR,
    QUALITY_LOG_PATH,
    SINGLE_REGULAR_SOURCE,
    ensure_directories,
    resampled_source,
)
from quality import (
    DEFAULT_RANGE_PCT,
    DEFAULT_RANGE_TOL,
    DEFAULT_SPIKE_PCT,
    REPORT_COLUMNS,
    check_table,
)
from settings import load_active_symbols, load_data_config, load_session_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 6: data-quality report over bars_1m/3m/5m for active symbols."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--spike-pct", type=float, default=DEFAULT_SPIKE_PCT)
    parser.add_argument("--range-pct", type=float, default=DEFAULT_RANGE_PCT)
    parser.add_argument("--range-tol", type=float, default=DEFAULT_RANGE_TOL, help="Band around the median daily count.")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(QUALITY_LOG_PATH, name="quality")

    data_cfg = load_data_config()
    session = load_session_config()
    window = (session.train_start, session.train_end)
    model_source = data_cfg.model_source
    resampled = resampled_source(model_source)

    # (timeframe, table, interval, expected_source, allowed_sources)
    timeframes = [
        ("1m", "bars_1m", 1, model_source, [AL_RAW_SOURCE, AL_REGULAR_SOURCE, SINGLE_REGULAR_SOURCE]),
        ("3m", "bars_3m", 3, resampled, [resampled]),
        ("5m", "bars_5m", 5, resampled, [resampled]),
    ]

    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    print(f"[{now_kst_text()}] quality check symbols={symbols} window={window[0]}..{window[1]}")
    print(f"  1m source={model_source}  3m/5m source={resampled}")
    logger.info("quality start symbols=%s model_source=%s resampled=%s", symbols, model_source, resampled)

    conn = sqlite3.connect(args.db)
    rows = []
    try:
        for symbol in symbols:
            for timeframe, table, interval, expected_source, allowed in timeframes:
                rows.extend(
                    check_table(
                        conn,
                        symbol,
                        table=table,
                        timeframe=timeframe,
                        interval_minutes=interval,
                        expected_source=expected_source,
                        allowed_sources=allowed,
                        session_window=window,
                        spike_pct=args.spike_pct,
                        range_pct=args.range_pct,
                        range_tol=args.range_tol,
                    )
                )
    finally:
        conn.close()

    report_path = (
        Path(args.report_path)
        if args.report_path
        else PROCESSED_DIR
        / ("quality_report_" + now_kst_text().replace(":", "").replace(" ", "_").replace("-", "") + ".csv")
    )
    with open(report_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in REPORT_COLUMNS})

    # summary: status counts per timeframe + worst offenders
    by_tf = {}
    for r in rows:
        by_tf.setdefault(r["timeframe"], Counter())[r["status"]] += 1
    print("\ntimeframe  OK     WARN   FAIL")
    total = Counter()
    for tf in ("1m", "3m", "5m"):
        c = by_tf.get(tf, Counter())
        total.update(c)
        print(f"  {tf:<8} {c.get('OK',0):>5}  {c.get('WARN',0):>5}  {c.get('FAIL',0):>5}")
    print(f"  {'TOTAL':<8} {total.get('OK',0):>5}  {total.get('WARN',0):>5}  {total.get('FAIL',0):>5}")

    fails = [r for r in rows if r["status"] == "FAIL"]
    if fails:
        print(f"\nFAIL rows ({len(fails)}) — first 10:")
        for r in fails[:10]:
            print(f"  {r['symbol']} {r['date']} {r['timeframe']} dup={r['duplicate']} ohlc={r['ohlc_error']} "
                  f"px0={r['price_zero']} volneg={r['volume_negative']} barend={r['bar_end_error']} "
                  f"oos={r['out_of_session']} mix={r['source_mixing']}")

    print(f"\nreport -> {report_path}")
    print(f"[{now_kst_text()}] {len(rows)} rows | OK={total.get('OK',0)} WARN={total.get('WARN',0)} FAIL={total.get('FAIL',0)}")
    logger.info("quality end rows=%s ok=%s warn=%s fail=%s report=%s",
                len(rows), total.get("OK", 0), total.get("WARN", 0), total.get("FAIL", 0), report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
