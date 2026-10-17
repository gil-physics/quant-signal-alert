from __future__ import annotations

import argparse
import csv as csv_mod
import os
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

from config import COLLECT_LOG_PATH, DB_PATH, MINUTE_DATA_DIR, PROCESSED_DIR, ensure_directories
from historical_collector import collect_1m, symbol_coverage
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical 1-minute bars for all active symbols into bars_1m."
    )
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD lower bound.")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD upper bound.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated codes to override the active-symbol list (testing).",
    )
    parser.add_argument("--data-dir", default=str(MINUTE_DATA_DIR), help="minute_data directory.")
    parser.add_argument("--report-path", default=None, help="Where to write the missing report CSV.")
    return parser.parse_args()


def _stamp() -> str:
    return f"[{now_kst_text()}]"


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(COLLECT_LOG_PATH, name="collect_1m")
    data_cfg = load_data_config()

    if args.symbols:
        raw_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbols = [normalize_symbol_code(s) for s in raw_symbols]
        for orig, norm in zip(raw_symbols, symbols):
            if orig != norm:
                print(f"{_stamp()} WARN normalized symbol code {orig!r} -> {norm!r}")
                logger.warning("normalized symbol code %s -> %s", orig, norm)
    else:
        symbols = load_active_symbols()

    print(f"{_stamp()} active symbols: {symbols} (model_source={data_cfg.model_source})")
    logger.info("run start symbols=%s model_source=%s", symbols, data_cfg.model_source)

    conn = sqlite3.connect(DB_PATH)
    results = []
    try:
        for symbol in symbols:
            print(f"{_stamp()} START collect {symbol}")
            logger.info("START collect %s", symbol)
            try:
                summary = collect_1m(
                    symbol,
                    args.start_date,
                    args.end_date,
                    conn=conn,
                    data_dir=Path(args.data_dir),
                    logger=logger,
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures
                print(f"{_stamp()} FAIL collect {symbol}: {exc}")
                logger.exception("FAIL collect %s: %s", symbol, exc)
                continue

            breakdown = " ".join(f"{src}={s.inserted}" for src, s in summary.per_source.items())
            print(
                f"{_stamp()} DONE collect {symbol} rows={summary.inserted_total} ({breakdown})"
                + (f" missing={summary.missing_files}" if summary.missing_files else "")
            )
            logger.info(
                "DONE collect %s rows=%s breakdown=%s missing=%s",
                symbol,
                summary.inserted_total,
                breakdown,
                summary.missing_files,
            )
            results.append(summary)

        # --- missing report over the active/selected symbols ---
        sources = [data_cfg.raw_source, data_cfg.model_source, data_cfg.comparison_source]
        report_rows = [
            symbol_coverage(conn, sym, sources, model_source=data_cfg.model_source) for sym in symbols
        ]
    finally:
        conn.close()

    report_path = (
        Path(args.report_path)
        if args.report_path
        else PROCESSED_DIR / f"collect_missing_report_{now_kst_text().replace(':', '').replace(' ', '_').replace('-', '')}_{os.getpid()}.csv"
    )
    fields = ["symbol", "status", "total_rows", "model_rows", "model_distinct_dates", "date_min", "date_max"] + sources
    with open(report_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv_mod.writer(fh)
        writer.writerow(fields)
        for r in report_rows:
            writer.writerow(
                [r["symbol"], r["status"], r["total_rows"], r["model_rows"], r["model_distinct_dates"], r["date_min"], r["date_max"]]
                + [r["counts"].get(src, 0) for src in sources]
            )

    print(f"\n{_stamp()} missing report -> {report_path}")
    print("symbol     status         model_rows  dates  range")
    incomplete = 0
    for r in report_rows:
        if r["status"] != "OK":
            incomplete += 1
        print(
            f"  {r['symbol']:<8} {r['status']:<13} {r['model_rows']:>9}  {r['model_distinct_dates']:>4}  "
            f"{r['date_min']}..{r['date_max']}"
        )
    print(f"{_stamp()} done: {len(results)} collected, {incomplete} incomplete/missing of {len(symbols)} symbols")
    logger.info("run end collected=%s incomplete=%s report=%s", len(results), incomplete, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
