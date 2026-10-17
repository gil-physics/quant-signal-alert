"""Phase 1 / v1.1 parity check: served REAL cross-symbol features == offline truth.

Builds the cross-symbol serving features for one target symbol (default 000660)
from a multi-day, multi-symbol ``bars_1m`` window via
``realtime_features.build_serving_features_cross`` and compares the 8 Step-9
cross-sectional columns against the offline ``features_relative`` table for the
same (symbol, decision_time). They MUST match on full-universe minutes because
both call the same ``relative_features.build_relative_features`` over the same
symbols. Partial-universe minutes (a peer missing/warming up) are reported, not
failed. Read-only.

Run (cwd = quant, 64-bit):
    ..\\Python311-64\\python.exe scripts\\check_cross_symbol_parity.py --date 2026-06-19
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

import numpy as np
import pandas as pd

from config import DB_PATH
from realtime_features import REL_COLUMNS, build_serving_features_cross
from settings import (load_active_symbols, load_data_config, normalize_symbol_code,
                      realtime_default_symbol)

BAR_COLUMNS = ["symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume"]
HISTORY_DAYS = 8
RANK_COLS = ["active_symbol_count", "rank_ret_5m_1m", "rank_ret_20m_1m", "rank_volume_z_20m_1m"]


def _load_window(conn, symbols, source, date):
    days = pd.read_sql_query(
        "SELECT DISTINCT substr(bar_start,1,10) d FROM bars_1m WHERE source=? "
        "AND substr(bar_start,1,10) < ? ORDER BY d", conn, params=(source, date))
    keep = days["d"].tolist()[-HISTORY_DAYS:]
    lo = keep[0] if keep else date
    placeholders = ",".join("?" * len(symbols))
    return pd.read_sql_query(
        f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol IN ({placeholders}) "
        "AND substr(bar_start,1,10) BETWEEN ? AND ? ORDER BY bar_start",
        conn, params=(source, *symbols, lo, date))


def main():
    p = argparse.ArgumentParser(description="v1.1 cross-symbol serving parity check.")
    p.add_argument("--code", default=None, help="target symbol (default: realtime default).")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest bars_1m day).")
    p.add_argument("--source", default=None)
    args = p.parse_args()

    symbols = [normalize_symbol_code(s) for s in load_active_symbols()]
    target = normalize_symbol_code(args.code or realtime_default_symbol())
    source = args.source or load_data_config().model_source

    conn = sqlite3.connect(DB_PATH)
    try:
        date = args.date or conn.execute(
            "SELECT max(substr(bar_start,1,10)) FROM bars_1m WHERE source=? AND symbol=?",
            (source, target)).fetchone()[0]
        window = _load_window(conn, symbols, source, date)
        offline = pd.read_sql_query(
            f"SELECT decision_time, {', '.join(REL_COLUMNS)} FROM features_relative "
            "WHERE symbol=? AND substr(decision_time,1,10)=? ORDER BY decision_time",
            conn, params=(target, date))
    finally:
        conn.close()

    print(f"parity {target} {date}  source={source}  peers={len(symbols)}  "
          f"window_rows={len(window)}  offline_rows={len(offline)}")
    serv = build_serving_features_cross(window, target=target, source=source,
                                        active_count=len(symbols))
    serv = serv[serv["decision_time"].str.slice(0, 10) == date]
    if serv.empty or offline.empty:
        raise SystemExit("no rows to compare (empty serving or offline frame).")

    m = serv[["decision_time"] + REL_COLUMNS].merge(
        offline, on="decision_time", suffixes=("_srv", "_off"), validate="one_to_one")
    full = m["active_symbol_count_srv"] == len(symbols)
    print(f"  compared {len(m)} shared minutes; full-universe (all {len(symbols)}): {int(full.sum())}; "
          f"partial: {int((~full).sum())}")

    ok = True
    for c in REL_COLUMNS:
        a, b = m.loc[full, f"{c}_srv"], m.loc[full, f"{c}_off"]
        if c in RANK_COLS:
            mism = int((a.astype("int64") != b.astype("int64")).sum())
            status = "OK" if mism == 0 else f"MISMATCH×{mism}"
            ok &= mism == 0
            print(f"    {c:32s} {status}")
        else:
            diff = (a.astype(float) - b.astype(float)).abs()
            bad = int((~np.isclose(a.astype(float), b.astype(float), atol=1e-6, equal_nan=True)).sum())
            status = "OK" if bad == 0 else f"MISMATCH×{bad}"
            ok &= bad == 0
            print(f"    {c:32s} {status}  max|Δ|={diff.max():.3e}")
    print("\n  RESULT:", "PARITY OK (full-universe minutes match offline)" if ok else "PARITY FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
