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

from bars_reader import load_bars
from config import (
    BARS_TABLES,
    DB_PATH,
    RESAMPLE_LOG_PATH,
    ensure_directories,
    resampled_source,
)
from csv_importer import create_bars_tables
from resampler import resample_ohlcv, store_bars
from settings import load_data_config, load_session_config, normalize_symbol_code
from utils import now_kst_text, setup_logger

# interval (minutes) -> bars table name
BARS_TABLES_BY_INTERVAL = {1: BARS_TABLES["1m"], 3: BARS_TABLES["3m"], 5: BARS_TABLES["5m"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: resample 1-minute bars into 3m/5m and store into bars_3m/bars_5m."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = all symbols present for the base source.")
    parser.add_argument("--source", default=None, help="Base 1m source to resample; default = config.data.model_source (al_regular).")
    parser.add_argument("--intervals", default="3,5", help="Comma-separated minute intervals (default 3,5).")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(RESAMPLE_LOG_PATH, name="resample")

    base_source = args.source or load_data_config().model_source
    out_source = resampled_source(base_source)
    session = load_session_config()
    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]
    for iv in intervals:
        if iv not in BARS_TABLES_BY_INTERVAL:
            raise SystemExit(f"no bars table for interval {iv}m; known: {sorted(BARS_TABLES_BY_INTERVAL)}")

    conn = sqlite3.connect(args.db)
    summary_rows = []
    try:
        create_bars_tables(conn)
        if args.symbols:
            raw = [s.strip() for s in args.symbols.split(",") if s.strip()]
            symbols = [normalize_symbol_code(s) for s in raw]
            for o, n in zip(raw, symbols):
                if o != n:
                    print(f"[{now_kst_text()}] WARN normalized symbol code {o!r} -> {n!r}")
        else:
            symbols = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM bars_1m WHERE source=? ORDER BY symbol", (base_source,)
                )
            ]

        print(f"[{now_kst_text()}] resample base_source={base_source} -> source={out_source} intervals={intervals} symbols={symbols}")
        logger.info("resample start base_source=%s out_source=%s intervals=%s symbols=%s", base_source, out_source, intervals, symbols)

        for symbol in symbols:
            df_1m = load_bars(symbol, source=base_source, conn=conn)
            n_1m = len(df_1m)
            row = {"symbol": symbol, "n_1m": n_1m}
            if n_1m == 0:
                print(f"[{now_kst_text()}] SKIP {symbol}: no {base_source} bars")
                logger.warning("skip symbol=%s no base bars source=%s", symbol, base_source)
                summary_rows.append(row)
                continue

            days = df_1m["bar_start"].str.slice(0, 10).nunique()
            for iv in intervals:
                table = BARS_TABLES_BY_INTERVAL[iv]
                resampled = resample_ohlcv(df_1m, iv, session_open=session.train_start)
                res = store_bars(conn, resampled, table=table, source=out_source, logger=logger)
                per_day = round(len(resampled) / days, 1) if days else 0
                row[f"n_{iv}m"] = len(resampled)
                row[f"inserted_{iv}m"] = res["inserted"]
                row[f"per_day_{iv}m"] = per_day
            print(
                f"[{now_kst_text()}] DONE {symbol} days={days} 1m={n_1m} "
                + " ".join(f"{iv}m={row.get(f'n_{iv}m')}(~{row.get(f'per_day_{iv}m')}/day)" for iv in intervals)
            )
            summary_rows.append(row)
    finally:
        conn.close()

    print(f"\nsymbol     1m/day    " + "  ".join(f"{iv}m/day" for iv in intervals))
    for row in summary_rows:
        if "n_1m" in row and row["n_1m"]:
            print(
                f"  {row['symbol']:<8} "
                + "  ".join(str(row.get(f"per_day_{iv}m", "-")).rjust(7) for iv in intervals)
            )
    print(f"[{now_kst_text()}] resampled {sum(1 for r in summary_rows if r.get('n_1m'))} symbols")
    logger.info("resample end symbols=%s", len(summary_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
