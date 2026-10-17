"""Live smoke-test helper: inspect realtime_minute_bars (steps 2 & 6).

Verifies that the 32-bit collector is populating the bridge table:
  * step 2 - new 1-minute bars keep accumulating (per-symbol counts + freshness)
  * step 6 - multiple symbols are ALIGNED on the same ts_kst (a symbol x minute grid)

Read-only; safe to run repeatedly (or with --watch) while the collector runs.
Run (cwd = quant, either runtime):
    ..\\Python311-64\\python.exe scripts\\check_realtime_bars.py --watch 30
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
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

from config import DB_PATH, connect
from settings import load_active_symbols


def _load(db_path, day):
    conn = connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT symbol, ts_kst, open, high, low, close, volume, created_at "
            "FROM realtime_minute_bars WHERE substr(ts_kst,1,10)=? ORDER BY ts_kst, symbol",
            conn, params=(day,))
    finally:
        conn.close()


def report(db_path, day, last_n):
    df = _load(db_path, day)
    print(f"\n=== realtime_minute_bars  date={day}  rows={len(df)}  "
          f"checked_at={datetime.now().strftime('%H:%M:%S')} ===")
    if df.empty:
        print("  (no rows yet — is the 32-bit collector running and the market open?)")
        return

    # step 2: per-symbol accumulation + freshness
    print("  per-symbol: bars / first / last / latest close")
    for sym, g in df.groupby("symbol"):
        print(f"    {sym}  bars={len(g):>3}  {g['ts_kst'].iloc[0][11:16]}..{g['ts_kst'].iloc[-1][11:16]}  "
              f"close={g['close'].iloc[-1]:,.0f}")
    latest_created = pd.to_datetime(df["created_at"]).max()
    age = (datetime.now() - latest_created.to_pydatetime()).total_seconds()
    print(f"  newest bar written {age:.0f}s ago "
          f"({'OK, fresh' if age < 120 else 'STALE — collector may be stopped/closed'})")

    # step 6: same-timestamp alignment across symbols (last_n minutes x symbols grid)
    minutes = sorted(df["ts_kst"].unique())[-last_n:]
    syms = sorted(df["symbol"].unique())
    grid = df[df["ts_kst"].isin(minutes)].pivot_table(
        index="ts_kst", columns="symbol", values="close", aggfunc="last")
    print(f"\n  alignment grid (last {len(minutes)} minutes x {len(syms)} symbols; ✓=bar present):")
    print("    minute   " + " ".join(f"{s[-4:]:>5}" for s in syms))
    for m in minutes:
        row = grid.loc[m] if m in grid.index else None
        marks = " ".join((f"{'   ✓ ':>5}" if (row is not None and pd.notna(row.get(s))) else f"{'   . ':>5}") for s in syms)
        present = int(grid.loc[m].notna().sum()) if m in grid.index else 0
        print(f"    {m[11:16]}   {marks}   [{present}/{len(syms)} aligned]")
    full = sum(1 for m in minutes if m in grid.index and grid.loc[m].notna().sum() == len(syms))
    print(f"  fully-aligned minutes: {full}/{len(minutes)}  (all {len(syms)} symbols share the ts_kst)")


def main():
    p = argparse.ArgumentParser(description="Inspect realtime_minute_bars (live smoke test).")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest date present, else today).")
    p.add_argument("--last-n", type=int, default=8, help="minutes to show in the alignment grid.")
    p.add_argument("--watch", type=float, default=0.0, help="repeat every N seconds (0 = once).")
    p.add_argument("--db", default=str(DB_PATH))
    args = p.parse_args()

    print(f"active symbols configured: {len(load_active_symbols())}")
    day = args.date
    if not day:
        conn = connect(args.db)
        try:
            day = conn.execute("SELECT max(substr(ts_kst,1,10)) FROM realtime_minute_bars").fetchone()[0]
        finally:
            conn.close()
        day = day or datetime.now().strftime("%Y-%m-%d")

    if args.watch > 0:
        try:
            while True:
                report(args.db, day, args.last_n)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        report(args.db, day, args.last_n)


if __name__ == "__main__":
    main()
