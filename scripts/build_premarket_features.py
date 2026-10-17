"""v3.0: build the per-trading-date premarket (08:00-09:00) feature table.

Reads premarket + regular-session-tail bars from ``bars_1m`` (default source
``al_raw``, which spans 08:00-19:59 and so carries both the premarket window and the
regular close used for the overnight-gap base), builds the daily premarket features
(``src/premarket_features.py``), and writes them to
``data/processed/premarket_features_daily.csv`` + the ``premarket_features_daily``
DB table. One row per trading date.

The premarket window closes before the regular open, so every value is causal for the
same-day regular-session decision rows it is later joined onto by the risk model.

Research / feature-engineering only -- no orders, no Kiwoom, no SendOrder.

    ..\\Python311-64\\python.exe scripts\\build_premarket_features.py --code 000660 --source al_raw
"""

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

import pandas as pd

from config import DB_PATH, PROCESSED_DIR, ensure_directories
from premarket_features import (
    PREMARKET_END,
    PREMARKET_START,
    build_daily_premarket_features,
    pm_target_columns,
    pm_universe_columns,
)
from settings import (leader_symbols, load_active_symbols, load_data_config,
                      normalize_symbol_code, realtime_default_symbol)
from utils import now_kst_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3.0: build daily premarket context features.")
    parser.add_argument("--code", default=None, help="Target symbol; default = realtime default (config).")
    parser.add_argument("--source", default=None, help="bars_1m source for premarket (default = raw_source, e.g. al_raw).")
    parser.add_argument("--symbols", default=None, help="Comma-separated universe; default = active symbols.")
    parser.add_argument("--premarket-start", default=PREMARKET_START)
    parser.add_argument("--premarket-end", default=PREMARKET_END)
    parser.add_argument("--with-anchors", action="store_true",
                        help="Also emit anchor columns (config relative.leaders) for v3.0c/v3.0d. "
                             "Default off -- v3.0a/v3.0b need only target + universe.")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def load_bars(conn, source, symbols):
    """Premarket window (08:00-09:00 by bar_end) + the regular-session tail (15:00-15:30
    by bar_start, for the previous-close base). Time-of-day strings compare lexically
    because they are fixed-width HH:MM:SS."""
    placeholders = ",".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, bar_start, bar_end, open, high, low, close, volume
        FROM bars_1m
        WHERE source = ?
          AND symbol IN ({placeholders})
          AND (
                (substr(bar_end, 12, 8) >= '08:00:00' AND substr(bar_end, 12, 8) < '09:00:00')
             OR (substr(bar_start, 12, 8) >= '15:00:00' AND substr(bar_start, 12, 8) <= '15:30:00')
          )
    """
    return pd.read_sql_query(query, conn, params=[source, *symbols])


def main() -> int:
    args = parse_args()
    ensure_directories()

    target = normalize_symbol_code(args.code) if args.code else realtime_default_symbol()
    source = args.source or load_data_config().raw_source
    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()
    if target not in symbols:
        symbols = [target] + symbols
    anchors = []
    if args.with_anchors:
        anchors = [code for _lab, code in leader_symbols() if code in symbols and code != target]

    print(f"[{now_kst_text()}] premarket features target={target} source={source} "
          f"universe={len(symbols)} anchors={anchors} window={args.premarket_start}-{args.premarket_end}")

    conn = sqlite3.connect(args.db)
    try:
        bars = load_bars(conn, source, symbols)
        if bars.empty:
            raise SystemExit(f"no bars for source={source!r} symbols={symbols} -- import al_raw bars first.")
        out = build_daily_premarket_features(
            bars,
            symbols=symbols,
            target_symbol=target,
            anchors=anchors,
            universe_symbols=symbols,
            premarket_start=args.premarket_start,
            premarket_end=args.premarket_end,
        )
        out.insert(2, "source", source)
        parquet_path = PROCESSED_DIR / "premarket_features_daily.parquet"
        csv_path = PROCESSED_DIR / "premarket_features_daily.csv"
        out.to_parquet(parquet_path, index=False)
        out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out.to_sql("premarket_features_daily", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    has_data_col = f"pm_{target}_has_data"
    n_days = len(out)
    n_with = int(out[has_data_col].fillna(0).sum()) if has_data_col in out.columns else 0
    n_ret = int(out[f"pm_{target}_ret"].notna().sum()) if f"pm_{target}_ret" in out.columns else 0
    print(f"  rows(days)={n_days} target_has_premarket={n_with} target_pm_ret_defined={n_ret} "
          f"(missing prev-close on the first day per symbol is expected)")
    print(f"  target model features ({len(pm_target_columns(target))}): {pm_target_columns(target)}")
    print(f"  universe features ({len(pm_universe_columns(target))}): present="
          f"{all(c in out.columns for c in pm_universe_columns(target))}")
    print(f"[{now_kst_text()}] premarket_features_daily -> db + {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
