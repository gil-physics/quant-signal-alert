"""Step 3 (option B): collect Kiwoom opt10080 1-minute bars DIRECTLY into bars_1m.

No CSV is written. This reuses the proven downloader in
C:\\OpenAPI\\download_kiwoom_minute.py and stores results straight into the
SQLite table bars_1m using the config.yaml source classification:

    <symbol>_AL  (integrated quote) -> al_raw (full) + al_regular (regular session)
    <symbol>     (single exchange)  -> single_regular (comparison)

Run with 32-bit Python on Windows (Kiwoom OpenAPI+ is a 32-bit ActiveX control)
and after the Kiwoom login window appears:

    ..\\Python311-32\\python.exe scripts\\collect_kiwoom_to_bars.py --symbols 373220,028260
    ..\\Python311-32\\python.exe scripts\\collect_kiwoom_to_bars.py              # all active symbols
"""

from __future__ import annotations

import argparse
import platform
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]          # ...\quant
REPO_ROOT = ROOT_DIR.parent                              # ...\OpenAPI (holds download_kiwoom_minute.py)
SRC_DIR = ROOT_DIR / "src"
for p in (str(SRC_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import COLLECT_LOG_PATH, DB_PATH, PROCESSED_DIR, ensure_directories
from historical_collector import store_minute_records, symbol_coverage
from settings import load_active_symbols, load_data_config, load_session_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Kiwoom opt10080 1-minute bars directly into bars_1m (no CSV)."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--tick-range", default="1", help="opt10080 tick range (default 1 minute).")
    parser.add_argument("--adjusted", default="1", choices=["0", "1"], help="1 applies adjusted price.")
    parser.add_argument("--al-suffix", default="_AL", help="Suffix for the integrated-quote code.")
    parser.add_argument("--screen-no", default="1000")
    parser.add_argument("--delay-sec", type=float, default=3.6, help="Delay between TR requests.")
    parser.add_argument("--max-pages", type=int, default=None, help="Debug page cap; omit for all.")
    parser.add_argument("--from-date", default=None, help="Optional YYYYMMDD lower bound.")
    parser.add_argument("--to-date", default=None, help="Optional YYYYMMDD upper bound.")
    parser.add_argument("--no-single", action="store_true", help="Skip the single-exchange (comparison) fetch.")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite path (default project bars DB).")
    return parser.parse_args()


def _stamp() -> str:
    return f"[{now_kst_text()}]"


def _bar_to_record(bar) -> dict:
    return {
        "datetime": bar.datetime,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def _fetch(downloader, code, args):
    return downloader.download(
        code=code,
        tick_range=args.tick_range,
        adjusted=args.adjusted,
        screen_no=args.screen_no,
        delay_sec=args.delay_sec,
        max_pages=args.max_pages,
        from_date=args.from_date,
        to_date=args.to_date,
    )


def main() -> int:
    args = parse_args()

    if platform.architecture()[0] != "32bit":
        raise SystemExit(
            "Kiwoom OpenAPI+ requires 32-bit Python.\n"
            "  ..\\Python311-32\\python.exe scripts\\collect_kiwoom_to_bars.py"
        )

    # Imported here so the rest of the module stays importable without the ActiveX control.
    from PyQt5.QtWidgets import QApplication
    from download_kiwoom_minute import KiwoomMinuteDownloader

    ensure_directories()
    logger = setup_logger(COLLECT_LOG_PATH, name="collect_1m")
    data_cfg = load_data_config()
    session = load_session_config()

    if args.symbols:
        raw_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbols = [normalize_symbol_code(s) for s in raw_symbols]
        for orig, norm in zip(raw_symbols, symbols):
            if orig != norm:
                print(f"{_stamp()} WARN normalized symbol code {orig!r} -> {norm!r}")
                logger.warning("normalized symbol code %s -> %s", orig, norm)
    else:
        symbols = load_active_symbols()

    print(f"{_stamp()} kiwoom->bars_1m symbols={symbols} model_source={data_cfg.model_source}")
    logger.info("kiwoom run start symbols=%s db=%s", symbols, args.db)

    app = QApplication.instance() or QApplication(sys.argv)
    downloader = KiwoomMinuteDownloader()
    downloader.login()
    print(f"{_stamp()} Kiwoom login OK")

    conn = sqlite3.connect(args.db)
    results = []
    try:
        for symbol in symbols:
            print(f"{_stamp()} START collect {symbol}")
            logger.info("START collect %s", symbol)
            try:
                al_code = f"{symbol}{args.al_suffix}"
                al_records = [_bar_to_record(b) for b in _fetch(downloader, al_code, args)]
                single_records = None
                if not args.no_single:
                    single_records = [_bar_to_record(b) for b in _fetch(downloader, symbol, args)]

                summary = store_minute_records(
                    conn,
                    symbol,
                    al_records=al_records,
                    single_records=single_records,
                    data_cfg=data_cfg,
                    session=session,
                    logger=logger,
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures
                print(f"{_stamp()} FAIL collect {symbol}: {exc}")
                logger.exception("FAIL collect %s: %s", symbol, exc)
                continue

            breakdown = " ".join(f"{src}={s.inserted}" for src, s in summary.per_source.items())
            print(f"{_stamp()} DONE collect {symbol} rows={summary.inserted_total} ({breakdown})")
            logger.info("DONE collect %s rows=%s breakdown=%s", symbol, summary.inserted_total, breakdown)
            results.append(summary)

        sources = [data_cfg.raw_source, data_cfg.model_source, data_cfg.comparison_source]
        report_rows = [
            symbol_coverage(conn, sym, sources, model_source=data_cfg.model_source) for sym in symbols
        ]
    finally:
        conn.close()
        app.quit()

    report_path = PROCESSED_DIR / (
        "collect_kiwoom_report_"
        + now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
        + ".csv"
    )
    import csv as csv_mod

    fields = ["symbol", "status", "total_rows", "model_rows", "model_distinct_dates", "date_min", "date_max"] + sources
    with open(report_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv_mod.writer(fh)
        writer.writerow(fields)
        for r in report_rows:
            writer.writerow(
                [r["symbol"], r["status"], r["total_rows"], r["model_rows"], r["model_distinct_dates"], r["date_min"], r["date_max"]]
                + [r["counts"].get(src, 0) for src in sources]
            )

    print(f"\n{_stamp()} report -> {report_path}")
    for r in report_rows:
        print(f"  {r['symbol']:<8} {r['status']:<13} model_rows={r['model_rows']} dates={r['model_distinct_dates']} {r['date_min']}..{r['date_max']}")
    print(f"{_stamp()} done: {len(results)}/{len(symbols)} symbols collected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
