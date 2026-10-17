"""Replace one trade date in bars_1m from a fresh Kiwoom opt10080 request.

Use this when symbols were collected at different request times and the newest
date is inconsistent. The script fetches ONLY the requested YYYYMMDD date, backs
up existing rows for that date, deletes those rows, then inserts the fresh rows.

It does not mix al_raw into model data:
  <symbol>_AL -> al_raw + al_regular (regular-session filtered)
  <symbol>    -> single_regular unless --no-single is used

Run with 32-bit Python after Kiwoom login is available:
  ..\Python311-32\python.exe scripts\replace_kiwoom_date_to_bars.py --date 20260622 --execute

Default is dry-run: fetch and report only; no DB writes.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
for p in (str(SRC_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import COLLECT_LOG_PATH, DB_PATH, PROCESSED_DIR, ensure_directories
from historical_collector import store_minute_records
from settings import load_active_symbols, load_data_config, load_session_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one YYYYMMDD from Kiwoom and replace that date in bars_1m."
    )
    parser.add_argument("--date", required=True, help="Target date, YYYYMMDD or YYYY-MM-DD.")
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--execute", action="store_true", help="Actually delete/reinsert DB rows. Default is dry-run.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow replacing with zero fetched rows for a source.")
    parser.add_argument("--no-single", action="store_true", help="Skip single_regular comparison source.")
    parser.add_argument("--tick-range", default="1")
    parser.add_argument("--adjusted", default="1", choices=["0", "1"])
    parser.add_argument("--al-suffix", default="_AL")
    parser.add_argument("--screen-no", default="1000")
    parser.add_argument("--delay-sec", type=float, default=3.6)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def _stamp() -> str:
    return f"[{now_kst_text()}]"


def _normalize_date(value: str) -> tuple[str, str]:
    digits = value.replace("-", "")
    if len(digits) != 8 or not digits.isdigit():
        raise SystemExit("--date must be YYYYMMDD or YYYY-MM-DD")
    return digits, f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def _bar_to_record(bar) -> dict:
    return {
        "datetime": bar.datetime,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def _fetch(downloader, *, code: str, args, yyyymmdd: str):
    rows = downloader.download(
        code=code,
        tick_range=args.tick_range,
        adjusted=args.adjusted,
        screen_no=args.screen_no,
        delay_sec=args.delay_sec,
        max_pages=args.max_pages,
        from_date=yyyymmdd,
        to_date=yyyymmdd,
    )
    return [_bar_to_record(b) for b in rows if b.datetime and b.datetime[:8] == yyyymmdd]


def _backup_rows(conn: sqlite3.Connection, *, symbol: str, sources: list[str], date_iso: str, backup_path: Path) -> int:
    placeholders = ",".join("?" for _ in sources)
    rows = conn.execute(
        f"SELECT symbol, bar_start, bar_end, open, high, low, close, volume, source, created_at "
        f"FROM bars_1m WHERE symbol=? AND substr(bar_start,1,10)=? AND source IN ({placeholders}) "
        f"ORDER BY source, bar_start",
        (symbol, date_iso, *sources),
    ).fetchall()
    exists = backup_path.exists()
    with open(backup_path, "a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume", "source", "created_at"])
        writer.writerows(rows)
    return len(rows)


def _delete_rows(conn: sqlite3.Connection, *, symbol: str, sources: list[str], date_iso: str) -> int:
    placeholders = ",".join("?" for _ in sources)
    before = conn.total_changes
    conn.execute(
        f"DELETE FROM bars_1m WHERE symbol=? AND substr(bar_start,1,10)=? AND source IN ({placeholders})",
        (symbol, date_iso, *sources),
    )
    conn.commit()
    return conn.total_changes - before


def _count_date(conn: sqlite3.Connection, *, symbol: str, source: str, date_iso: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM bars_1m WHERE symbol=? AND source=? AND substr(bar_start,1,10)=?",
        (symbol, source, date_iso),
    ).fetchone()[0]


def main() -> int:
    args = parse_args()
    yyyymmdd, date_iso = _normalize_date(args.date)

    if platform.architecture()[0] != "32bit":
        raise SystemExit(
            "Kiwoom OpenAPI+ requires 32-bit Python.\n"
            "  ..\\Python311-32\\python.exe scripts\\replace_kiwoom_date_to_bars.py --date "
            f"{yyyymmdd} --execute"
        )

    from PyQt5.QtWidgets import QApplication
    from download_kiwoom_minute import KiwoomMinuteDownloader

    ensure_directories()
    logger = setup_logger(COLLECT_LOG_PATH, name="collect_1m")
    data_cfg = load_data_config()
    session = load_session_config()

    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    sources = [data_cfg.raw_source, data_cfg.model_source]
    if not args.no_single:
        sources.append(data_cfg.comparison_source)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"{_stamp()} replace date={date_iso} mode={mode} symbols={symbols}")
    print(f"  sources={sources}")
    logger.info("replace date start date=%s mode=%s symbols=%s", date_iso, mode, symbols)

    app = QApplication.instance() or QApplication(sys.argv)
    downloader = KiwoomMinuteDownloader()
    downloader.login()
    print(f"{_stamp()} Kiwoom login OK")

    conn = sqlite3.connect(args.db)
    backup_path = PROCESSED_DIR / (
        "bars_1m_backup_before_replace_"
        + date_iso.replace("-", "")
        + "_"
        + now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
        + ".csv"
    )
    report_rows = []
    try:
        for symbol in symbols:
            print(f"\n{_stamp()} FETCH {symbol} date={date_iso}")
            al_code = f"{symbol}{args.al_suffix}"
            al_records = _fetch(downloader, code=al_code, args=args, yyyymmdd=yyyymmdd)
            single_records = None if args.no_single else _fetch(downloader, code=symbol, args=args, yyyymmdd=yyyymmdd)

            fetched_raw = len(al_records)
            fetched_single = len(single_records) if single_records is not None else 0
            if not args.allow_empty and (fetched_raw == 0 or (single_records is not None and fetched_single == 0)):
                print(f"{_stamp()} SKIP {symbol}: empty fetch al={fetched_raw} single={fetched_single}")
                logger.warning("replace date skip empty symbol=%s al=%s single=%s", symbol, fetched_raw, fetched_single)
                report_rows.append([symbol, "SKIP_EMPTY", fetched_raw, fetched_single, *([0] * len(sources)), 0, 0, 0])
                continue

            existing = {src: _count_date(conn, symbol=symbol, source=src, date_iso=date_iso) for src in sources}
            print(f"  fetched: al_records={fetched_raw} single_records={fetched_single} existing={existing}")

            if not args.execute:
                report_rows.append([symbol, "DRY_RUN", fetched_raw, fetched_single, *[existing.get(src, 0) for src in sources], 0, 0, 0])
                continue

            backed_up = _backup_rows(conn, symbol=symbol, sources=sources, date_iso=date_iso, backup_path=backup_path)
            deleted = _delete_rows(conn, symbol=symbol, sources=sources, date_iso=date_iso)
            summary = store_minute_records(
                conn,
                symbol,
                al_records=al_records,
                single_records=single_records,
                date_range=(date_iso, date_iso),
                data_cfg=data_cfg,
                session=session,
                logger=logger,
            )
            inserted = {src: s.inserted for src, s in summary.per_source.items()}
            final = {src: _count_date(conn, symbol=symbol, source=src, date_iso=date_iso) for src in sources}
            print(f"  backed_up={backed_up} deleted={deleted} inserted={inserted} final={final}")
            report_rows.append([
                symbol,
                "REPLACED",
                fetched_raw,
                fetched_single,
                *[existing.get(src, 0) for src in sources],
                backed_up,
                deleted,
                sum(inserted.values()),
            ])
    finally:
        conn.close()
        app.quit()

    report_path = PROCESSED_DIR / (
        "replace_kiwoom_date_report_"
        + date_iso.replace("-", "")
        + "_"
        + now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
        + ".csv"
    )
    fields = ["symbol", "status", "fetched_al", "fetched_single"] + [f"existing_{s}" for s in sources] + ["backed_up", "deleted", "inserted"]
    with open(report_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        writer.writerows(report_rows)

    print(f"\n{_stamp()} report -> {report_path}")
    if args.execute:
        print(f"{_stamp()} backup -> {backup_path}")
        print("NOTE: bars_3m/5m, features, labels, rule_signals, and backtests should be rebuilt after replacement.")
    else:
        print("DRY-RUN only. Re-run with --execute to replace DB rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
