"""Realistic online top1 evaluation of v2_daily.

Corrects the look-ahead in daily top-k selection (which picks the day's best in
hindsight) by committing to the FIRST qualifying signal each day. Then:
  1. online vs hindsight top1 (quantifies the look-ahead premium)
  2. monthly walk-forward (a fixed rule -> every month is out-of-sample)
  3. transaction cost / slippage sensitivity (round-trip, deducted per trade)

Execution: entry_delay_3m, +3% / -1.5% first-touch bracket held to EOD. Selection
is causal; labels are not used. Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\online_top1_eval.py
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
from daily_topk_backtest import (
    ThresholdConfig,
    apply_base_filters,
    apply_long_filter,
    quality_exclusion_pairs,
    select_daily_topk,
)
from online_select import select_online_top1
from rule_signal import build_rule_signals
from rule_signal_ma import breakout_flags
from settings import load_active_symbols, load_data_config

TARGET, STOP, DELAY = 0.03, 0.015, 3
TRAIN_END = "2026-02-28"
VAL_START, VAL_END = "2026-03-01", "2026-06-19"
# round-trip cost scenarios (commission + KOSPI sell tax ~0.15% + slippage)
COSTS = [0.000, 0.001, 0.002, 0.003]
DEFAULT_COST = 0.002


def _latest_quality_report():
    reports = sorted((ROOT_DIR / "data" / "processed").glob("quality_report_*.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _candidates():
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        rel = pd.read_sql_query("SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,))
        ma = pd.read_sql_query(
            "SELECT symbol, decision_time, ma5_daily FROM features_merged WHERE source = ?",
            conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()
    for f in (rel, ma, bars):
        f.drop(f.index[~f["symbol"].isin(symbols)], inplace=True)
        f.reset_index(drop=True, inplace=True)

    v0 = build_rule_signals(rel)
    d = v0.merge(ma, on=["symbol", "decision_time"], how="left", validate="one_to_one")
    d, up, down = breakout_flags(d, "ma5_daily")
    d["is_up_candidate"] = up.astype(int)
    d["is_down_candidate"] = down.astype(int)
    d["date"] = d["decision_time"].str.slice(0, 10)
    d["time"] = d["decision_time"].str.slice(11, 19)
    d["score_margin"] = d["up_score"] - d["down_score"]
    d["_decision_dt"] = pd.to_datetime(d["decision_time"])

    fail_pairs, outlier_pairs = quality_exclusion_pairs(pd.read_csv(_latest_quality_report()))
    base, _ = apply_base_filters(d, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)
    cfg = ThresholdConfig("v2_daily", up_threshold=4.0, down_risk_limit=1.0, score_margin=3.0)
    candidates, _ = apply_long_filter(base, cfg)
    return candidates, day_arrays(bars), symbols, model_source


def _evaluate(selected, by_day):
    rows = []
    cols = selected[["symbol", "date", "_decision_dt", "close"]]
    for sym, date, dec_dt, close_t in cols.itertuples(index=False, name=None):
        arr = by_day.get((sym, date))
        if arr is None:
            continue
        rec = prepare_entry(arr, dec_dt, float(close_t), delay_bars=DELAY)
        if rec is None:
            continue
        ep, h, l, c, off = rec
        br = simulate_bracket(ep, h, l, c, off, target_pct=TARGET, stop_pct=STOP, time_exit_min=float("inf"))
        if br is None:
            continue
        rows.append({"symbol": sym, "date": date, "month": date[:7],
                     "outcome": br["outcome"], "realized": br["realized_return"],
                     "mfe": br["mfe"], "mae": br["mae"], "eod": float(c[-1] / ep - 1)})
    return pd.DataFrame(rows)


def _summ(df, cost=DEFAULT_COST):
    if df is None or len(df) == 0:
        return {k: 0.0 for k in ("n", "target", "stop", "gross", "net", "expR_net", "mae", "eod", "conc")}
    g = df["realized"].mean()
    return {
        "n": len(df),
        "target": float((df["outcome"] == "target").mean()),
        "stop": float((df["outcome"] == "stop").mean()),
        "gross": float(g),
        "net": float(g - cost),
        "expR_net": float((g - cost) / STOP),
        "mae": float(df["mae"].mean()),
        "eod": float(df["eod"].mean() - cost),
        "conc": float(df["symbol"].value_counts().max() / len(df)),
    }


def _mask(df, split):
    if split == "train":
        return df["date"] <= TRAIN_END
    if split == "validation":
        return (df["date"] >= VAL_START) & (df["date"] <= VAL_END)
    return pd.Series(True, index=df.index)


def main():
    candidates, by_day, symbols, model_source = _candidates()
    print(f"active symbols: {len(symbols)}  model_source: {model_source}  "
          f"entry_delay_3m, +{TARGET*100:.0f}%/-{STOP*100:.1f}% bracket, default cost={DEFAULT_COST*100:.1f}% round-trip")

    hind_sel, _ = select_daily_topk(candidates, top_k=1, cooldown_minutes=60)
    online_sel = select_online_top1(candidates)
    hind = _evaluate(hind_sel, by_day)
    online = _evaluate(online_sel, by_day)

    out_rows = []
    print("\n=== A. ONLINE (causal, first-of-day) vs HINDSIGHT (day's best) top1 — net of "
          f"{DEFAULT_COST*100:.1f}% ===")
    print(f"  {'method/split':<24}{'n':>5}{'tgt%':>7}{'stop%':>7}{'gross%':>8}{'net%':>8}"
          f"{'expR_net':>9}{'MAE%':>7}{'eod%':>8}{'conc%':>6}")
    for name, df in (("hindsight", hind), ("online", online)):
        for split in ("full", "train", "validation"):
            s = _summ(df[_mask(df, split)])
            print(f"  {name+':'+split:<24}{int(s['n']):>5}{s['target']*100:>7.1f}{s['stop']*100:>7.1f}"
                  f"{s['gross']*100:>8.3f}{s['net']*100:>8.3f}{s['expR_net']:>9.3f}{s['mae']*100:>7.2f}"
                  f"{s['eod']*100:>8.3f}{s['conc']*100:>6.0f}")
            out_rows.append({"section": "A", "method": name, "split": split, "cost": DEFAULT_COST, **s})

    print("\n=== B. COST SENSITIVITY (online top1, full period) ===")
    print(f"  {'round_trip_cost':<18}{'net_mean%':>10}{'expR_net':>10}{'net_eod%':>10}")
    for c in COSTS:
        s = _summ(online, cost=c)
        print(f"  {c*100:>6.2f}%{'':<11}{s['net']*100:>10.3f}{s['expR_net']:>10.3f}{s['eod']*100:>10.3f}")
        out_rows.append({"section": "B", "method": "online", "split": "full", "cost": c, **s})

    print(f"\n=== C. MONTHLY WALK-FORWARD (online top1, net of {DEFAULT_COST*100:.1f}%) ===")
    print(f"  {'month':<10}{'n':>4}{'tgt%':>7}{'stop%':>7}{'gross%':>8}{'net%':>8}{'eod%':>8}")
    for month, g in online.groupby("month"):
        s = _summ(g)
        print(f"  {month:<10}{int(s['n']):>4}{s['target']*100:>7.1f}{s['stop']*100:>7.1f}"
              f"{s['gross']*100:>8.3f}{s['net']*100:>8.3f}{s['eod']*100:>8.3f}")
        out_rows.append({"section": "C", "method": "online", "split": month, "cost": DEFAULT_COST, **s})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT_DIR / "data" / "processed" / f"online_top1_eval_{ts}.csv"
    pd.DataFrame(out_rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {path}")


if __name__ == "__main__":
    main()
