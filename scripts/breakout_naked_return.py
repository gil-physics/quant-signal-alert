"""Reconcile the 'chart looks profitable' impression with the backtest.

Takes the PURE 5MA up-cross (every cross, no v0 / cross-symbol filter, no daily
top-k, no cooldown) and measures, entering at the cross bar's close:

  * NAKED forward return (no stop) at +10/+30/+60 min and EOD -- mean, median,
    and % positive. This is what eyeballing a chart actually measures.
  * the +3% / -1.5% first-touch bracket outcome and realised P&L (gross + net).

Run for both the intraday (time-of-day) and daily 5MA. Crosses are computed from
features_merged; forward outcomes from bars_1m al_regular. Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\breakout_naked_return.py
"""

from __future__ import annotations

import sqlite3
import sys
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

import numpy as np
import pandas as pd

from bracket_eval import day_arrays, prepare_entry, simulate_bracket
from rule_signal_ma import breakout_flags
from settings import load_active_symbols, load_data_config

TARGET, STOP, COST = 0.03, 0.015, 0.002
MIN_TIME, MAX_TIME = "09:10:00", "15:00:00"
HORIZONS = (10, 30, 60)


def _forward(arr, dec_dt, close_t):
    rec = prepare_entry(arr, dec_dt, close_t, delay_bars=0)
    if rec is None:
        return None
    ep, h, l, c, off = rec
    out = {}
    for n in HORIZONS:
        w = c[off <= n]
        out[f"r{n}"] = float(w[-1] / ep - 1) if len(w) else np.nan
    out["r_eod"] = float(c[-1] / ep - 1)
    br = simulate_bracket(ep, h, l, c, off, target_pct=TARGET, stop_pct=STOP, time_exit_min=float("inf"))
    out["bk_outcome"] = br["outcome"]
    out["bk_real"] = br["realized_return"]
    return out


def main():
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        fm = pd.read_sql_query(
            "SELECT symbol, decision_time, close, ma5_daily, ma5_intraday "
            "FROM features_merged WHERE source = ?", conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()
    for f in (fm, bars):
        f.drop(f.index[~f["symbol"].isin(symbols)], inplace=True)
        f.reset_index(drop=True, inplace=True)
    by_day = day_arrays(bars)

    print(f"active symbols: {len(symbols)}  model_source: {model_source}  "
          f"entry=cross-bar close (delay 0)  window {MIN_TIME}-{MAX_TIME}")

    rows = []
    for kind, col in (("intraday(1m)", "ma5_intraday"), ("daily", "ma5_daily")):
        d, up, _ = breakout_flags(fm, col)
        d = d.assign(_up=up)
        d["_t"] = d["decision_time"].str.slice(11, 19)
        cross = d[(d["_up"]) & (d["_t"] >= MIN_TIME) & (d["_t"] < MAX_TIME)]
        cross = cross[["symbol", "_date", "decision_time", "close"]]

        recs = []
        for sym, date, dtime, close_t in cross.itertuples(index=False, name=None):
            arr = by_day.get((sym, date))
            if arr is None:
                continue
            fwd = _forward(arr, pd.Timestamp(dtime), float(close_t))
            if fwd is not None:
                recs.append(fwd)
        ev = pd.DataFrame(recs)
        n = len(ev)

        print(f"\n===== {kind} 5MA up-cross — {n} events =====")
        print("  NAKED forward return (no stop), entry at cross close:")
        print(f"    {'horizon':<8}{'mean%':>9}{'median%':>9}{'%positive':>11}")
        for h in [f"r{n_}" for n_ in HORIZONS] + ["r_eod"]:
            s = ev[h].dropna()
            label = h.replace("r_", "").replace("r", "+") + ("m" if h != "r_eod" else "")
            print(f"    {label:<8}{s.mean()*100:>9.3f}{s.median()*100:>9.3f}{(s > 0).mean()*100:>11.1f}")
        tgt = (ev["bk_outcome"] == "target").mean()
        stp = (ev["bk_outcome"] == "stop").mean()
        gross = ev["bk_real"].mean()
        print(f"  +{TARGET*100:.0f}%/-{STOP*100:.1f}% bracket: target_first={tgt*100:.1f}%  "
              f"stop_first={stp*100:.1f}%  gross/trade={gross*100:.3f}%  "
              f"net/trade(@{COST*100:.1f}%)={ (gross-COST)*100:.3f}%")

        rows.append({"signal": kind, "n": n,
                     "naked_r30_mean": ev["r30"].mean(), "naked_r30_pos": (ev["r30"] > 0).mean(),
                     "naked_eod_mean": ev["r_eod"].mean(), "naked_eod_pos": (ev["r_eod"] > 0).mean(),
                     "bk_target": tgt, "bk_stop": stp,
                     "bk_gross": gross, "bk_net": gross - COST})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT_DIR / "data" / "processed" / f"breakout_naked_return_{ts}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {path}")


if __name__ == "__main__":
    main()
