"""Does the 1-minute chart's 5-period MA breakout (a fast scalp) have an edge?

This tests the ACTUAL signal the user watches: on the 1-minute chart, price
crossing above the rolling 5-bar moving average (mean of the last 5 one-minute
closes) -- entered at the cross bar's close, managed as a small/fast scalp.

  * NAKED forward return (no stop) at +1/+2/+5/+10/+30 min -- mean, % positive.
  * small target/stop first-touch brackets with a 30-minute time cap, gross + net
    of a 0.2% round-trip cost.

Every up-cross is counted (no filter), within 09:10-15:00. The 5-bar MA is
computed within each trading day (no cross-day leak); the cross at a bar is known
at that bar's close, so entry-at-close is causal. Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\ma5min_scalp_eval.py
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

from bracket_eval import simulate_bracket
from settings import load_active_symbols, load_data_config

MA_WINDOW = 5
COST = 0.002
TIME_CAP = 30.0          # minutes (fast scalp)
MIN_TIME, MAX_TIME = "09:10:00", "15:00:00"
NAKED_HORIZONS = (1, 2, 5, 10, 30)
# (target, stop) small/fast scalp brackets
BRACKETS = [(0.005, 0.003), (0.005, 0.005), (0.010, 0.005), (0.010, 0.010)]


def main():
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()
    bars = bars[bars["symbol"].isin(symbols)].copy()
    bars["_date"] = bars["bar_start"].str.slice(0, 10)
    bars["_t"] = bars["bar_start"].str.slice(11, 19)
    bars = bars.sort_values(["symbol", "_date", "bar_start"]).reset_index(drop=True)

    naked = {h: [] for h in NAKED_HORIZONS}
    naked_eod = []
    brk = {b: [] for b in BRACKETS}        # realized returns
    brk_out = {b: [] for b in BRACKETS}     # outcomes
    n_cross = 0

    for (sym, date), g in bars.groupby(["symbol", "_date"], sort=False):
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        tod = g["_t"].to_numpy()
        if len(close) <= MA_WINDOW:
            continue
        ma = pd.Series(close).rolling(MA_WINDOW).mean().to_numpy()
        # up-cross at i (>=1): prev below/at MA, current above MA
        prev_below = close[:-1] <= ma[:-1]
        cur_above = close[1:] > ma[1:]
        valid = ~np.isnan(ma[:-1]) & ~np.isnan(ma[1:])
        cross = np.nonzero(prev_below & cur_above & valid)[0] + 1

        for i in cross:
            if not (MIN_TIME <= tod[i] < MAX_TIME):
                continue
            entry = close[i]
            if i + 1 >= len(close):
                continue
            post_c = close[i + 1:]
            post_h = high[i + 1:]
            post_l = low[i + 1:]
            offsets = np.arange(1, len(post_c) + 1, dtype=float)
            n_cross += 1
            for h in NAKED_HORIZONS:
                w = post_c[offsets <= h]
                naked[h].append(w[-1] / entry - 1 if len(w) else np.nan)
            naked_eod.append(post_c[-1] / entry - 1)
            for (tgt, stp) in BRACKETS:
                r = simulate_bracket(entry, post_h, post_l, post_c, offsets,
                                     target_pct=tgt, stop_pct=stp, time_exit_min=TIME_CAP)
                if r is None:
                    brk[(tgt, stp)].append(np.nan)
                    brk_out[(tgt, stp)].append("none")
                else:
                    brk[(tgt, stp)].append(r["realized_return"])
                    brk_out[(tgt, stp)].append(r["outcome"])

    print(f"active symbols: {len(symbols)}  model_source: {model_source}")
    print(f"1m chart 5-bar MA up-crosses (09:10-15:00): {n_cross} events  "
          f"(entry at cross close, cost {COST*100:.1f}% round-trip)\n")

    print("NAKED forward return after the cross (no stop):")
    print(f"  {'horizon':<8}{'mean%':>9}{'median%':>9}{'%positive':>11}")
    for h in NAKED_HORIZONS:
        s = pd.Series(naked[h]).dropna()
        print(f"  +{h:<7}{s.mean()*100:>9.4f}{s.median()*100:>9.4f}{(s > 0).mean()*100:>11.1f}")
    se = pd.Series(naked_eod).dropna()
    print(f"  {'eod':<8}{se.mean()*100:>9.4f}{se.median()*100:>9.4f}{(se > 0).mean()*100:>11.1f}")

    print(f"\nSmall/fast brackets (30-min cap), break-even win needs to beat cost:")
    print(f"  {'target/stop':<14}{'tgt%':>7}{'stop%':>7}{'time%':>7}{'win%':>7}{'aW%':>7}{'aL%':>7}"
          f"{'gross%':>8}{'net%':>8}{'expR_net':>9}")
    rows = []
    for (tgt, stp) in BRACKETS:
        r = pd.Series(brk[(tgt, stp)]).dropna()
        out = pd.Series(brk_out[(tgt, stp)])
        wins, losses = r[r > 0], r[r < 0]
        gross = r.mean()
        s = {
            "target": tgt, "stop": stp, "n": int(len(r)),
            "target_first": float((out == "target").mean()),
            "stop_first": float((out == "stop").mean()),
            "time_exit": float((out == "time").mean()),
            "win_rate": float((r > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "gross": float(gross), "net": float(gross - COST), "expR_net": float((gross - COST) / stp),
        }
        rows.append(s)
        print(f"  +{tgt*100:.1f}/-{stp*100:.1f}{'':<7}{s['target_first']*100:>7.1f}{s['stop_first']*100:>7.1f}"
              f"{s['time_exit']*100:>7.1f}{s['win_rate']*100:>7.1f}{s['avg_win']*100:>7.2f}{s['avg_loss']*100:>7.2f}"
              f"{s['gross']*100:>8.3f}{s['net']*100:>8.3f}{s['expR_net']:>9.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT_DIR / "data" / "processed" / f"ma5min_scalp_eval_{ts}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {path}")


if __name__ == "__main__":
    main()
