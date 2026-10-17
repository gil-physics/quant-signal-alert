"""Evaluate the winning rule version (v2_daily) under the step13 execution model.

  - Selection: v0 + daily-MA breakout FILTER (v2_daily) = up_cross AND v0 score gate.
  - Entry: compared at delay 0 (immediate) vs delay 3m (entry_delay_3m).
  - Exit: +3% / -1.5% first-touch bracket, held to EOD if neither fires.
  - Reported per split (full/train/validation) and per daily rank (all/top1/top2).
  - Step 6: loosen the selection score threshold to try to recover the hit-rate.

Selection uses only pre-decision info; labels are not used here at all (the bracket
outcome is recomputed from bars). Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\evaluate_v2_daily.py
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
    join_signals_labels,
    quality_exclusion_pairs,
    select_daily_topk,
)
from rule_signal import build_rule_signals
from rule_signal_ma import breakout_flags
from settings import load_active_symbols, load_data_config

TARGET, STOP = 0.03, 0.015
TRAIN_END = "2026-02-28"
VAL_START, VAL_END = "2026-03-01", "2026-06-19"
SPLITS = ["full", "train", "validation"]
TOP_K = 2

# step6: (label, up_threshold, down_risk_limit, score_margin) — first row = fixed v2_daily baseline
LOOSEN_CFGS = [
    ("u4_d1_m3 (baseline)", 4.0, 1.0, 3.0),
    ("u3_d1_m2", 3.0, 1.0, 2.0),
    ("u3_d2_m2", 3.0, 2.0, 2.0),
    ("u2_d2_m1", 2.0, 2.0, 1.0),
]


def _latest_quality_report():
    reports = sorted((ROOT_DIR / "data" / "processed").glob("quality_report_*.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _load():
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        rel = pd.read_sql_query("SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,))
        ma = pd.read_sql_query(
            "SELECT symbol, decision_time, ma5_daily FROM features_merged WHERE source = ?",
            conn, params=(model_source,))
        labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()
    for f in (rel, ma, labels, bars):
        f.drop(f.index[~f["symbol"].isin(symbols)], inplace=True)
        f.reset_index(drop=True, inplace=True)
    return rel, ma, labels, bars, symbols, model_source


def _cross_gated_base(rel, ma, labels, symbols, fail_pairs, outlier_pairs):
    """v0 scores with is_up/down_candidate REPLACED by the daily-MA cross, run through
    join + base filters. apply_long_filter(cfg) then adds the score gate -> v2_daily."""
    v0 = build_rule_signals(rel)
    d = v0.merge(ma, on=["symbol", "decision_time"], how="left", validate="one_to_one")
    d, up, down = breakout_flags(d, "ma5_daily")
    d["is_up_candidate"] = up.astype(int)
    d["is_down_candidate"] = down.astype(int)
    joined = join_signals_labels(d, labels, symbols=symbols)
    base, _ = apply_base_filters(joined, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)
    return base


def _split_mask(df, split):
    if split == "train":
        return df["date"] <= TRAIN_END
    if split == "validation":
        return (df["date"] >= VAL_START) & (df["date"] <= VAL_END)
    return pd.Series(True, index=df.index)


def _evaluate(selected, by_day, delay):
    rows = []
    cols = selected[["symbol", "date", "_decision_dt", "close", "daily_rank"]]
    for sym, date, dec_dt, close_t, rank in cols.itertuples(index=False, name=None):
        arr = by_day.get((sym, date))
        if arr is None:
            continue
        rec = prepare_entry(arr, dec_dt, float(close_t), delay_bars=delay)
        if rec is None:
            continue
        ep, h, l, c, off = rec
        br = simulate_bracket(ep, h, l, c, off, target_pct=TARGET, stop_pct=STOP, time_exit_min=float("inf"))
        if br is None:
            continue
        br.update({"symbol": sym, "date": date, "daily_rank": int(rank),
                   "eod_return": float(c[-1] / ep - 1)})
        rows.append(br)
    return pd.DataFrame(rows)


def _summ(df):
    if df is None or len(df) == 0:
        return {k: 0.0 for k in ("n", "target", "stop", "noevent", "win", "avg_win",
                                 "avg_loss", "realized", "expR", "mfe", "mae", "eod", "conc")}
    n = len(df)
    realized = df["realized_return"].to_numpy()
    wins, losses = realized[realized > 0], realized[realized < 0]
    return {
        "n": n,
        "target": float((df["outcome"] == "target").mean()),
        "stop": float((df["outcome"] == "stop").mean()),
        "noevent": float((df["outcome"] == "time").mean()),
        "win": float((realized > 0).mean()),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "realized": float(realized.mean()),
        "expR": float(realized.mean() / STOP),
        "mfe": float(df["mfe"].mean()),
        "mae": float(df["mae"].mean()),
        "eod": float(df["eod_return"].mean()),
        "conc": float(df["symbol"].value_counts().max() / n),
    }


def _print_header():
    print(f"  {'scope':<22}{'n':>5}{'tgt':>6}{'stop':>6}{'win':>6}{'aW':>6}{'aL':>7}"
          f"{'real':>7}{'expR':>7}{'MFE':>6}{'MAE':>7}{'eod':>7}{'conc':>6}")


def _print_row(label, s):
    print(f"  {label:<22}{int(s['n']):>5}{s['target']*100:>6.1f}{s['stop']*100:>6.1f}{s['win']*100:>6.1f}"
          f"{s['avg_win']*100:>6.2f}{s['avg_loss']*100:>7.2f}{s['realized']*100:>7.3f}{s['expR']:>7.3f}"
          f"{s['mfe']*100:>6.2f}{s['mae']*100:>7.2f}{s['eod']*100:>7.3f}{s['conc']*100:>6.0f}")


def main():
    rel, ma, labels, bars, symbols, model_source = _load()
    fail_pairs, outlier_pairs = quality_exclusion_pairs(pd.read_csv(_latest_quality_report()))
    print(f"active symbols: {len(symbols)}  model_source: {model_source}  "
          f"bracket: +{TARGET*100:.0f}% / -{STOP*100:.1f}%")
    print("building v0 + daily-MA cross gate (slow step) ...")
    base = _cross_gated_base(rel, ma, labels, symbols, fail_pairs, outlier_pairs)
    by_day = day_arrays(bars)

    out_rows = []

    # ---- Steps 1-5: fixed v2_daily (conservative), immediate vs delay 3m ----
    cfg = ThresholdConfig("v2_daily", up_threshold=4.0, down_risk_limit=1.0, score_margin=3.0)
    candidates, _ = apply_long_filter(base, cfg)
    print("\n################  v2_daily (conservative)  —  IMMEDIATE vs ENTRY_DELAY_3M  ################")
    for split in SPLITS:
        sub = candidates[_split_mask(candidates, split)].reset_index(drop=True)
        selected, _ = select_daily_topk(sub, top_k=TOP_K, cooldown_minutes=60)
        print(f"\n=== {split.upper()} (selected={len(selected)}) ===")
        for delay in (0, 3):
            ev = _evaluate(selected, by_day, delay)
            tag = "immediate" if delay == 0 else "delay_3m"
            _print_header() if delay == 0 else None
            for scope, mask in (("all", None), ("top1", ev.get("daily_rank") == 1),
                                ("top2", ev.get("daily_rank") == 2)):
                d = ev if mask is None else ev[mask]
                s = _summ(d)
                _print_row(f"{tag}:{scope}", s)
                out_rows.append({"section": "v2_daily", "split": split, "entry": tag, "scope": scope, **s})

    # ---- Step 6: loosen the selection score threshold (entry_delay_3m) ----
    print("\n################  STEP 6: LOOSEN SELECTION THRESHOLD (entry_delay_3m, all top-k)  ################")
    for split in SPLITS:
        print(f"\n=== {split.upper()} ===")
        _print_header()
        for label, up_t, down_l, margin in LOOSEN_CFGS:
            cfg = ThresholdConfig(label, up_threshold=up_t, down_risk_limit=down_l, score_margin=margin)
            cand, _ = apply_long_filter(base, cfg)
            sub = cand[_split_mask(cand, split)].reset_index(drop=True)
            selected, _ = select_daily_topk(sub, top_k=TOP_K, cooldown_minutes=60)
            ev = _evaluate(selected, by_day, 3)
            s = _summ(ev)
            _print_row(label, s)
            out_rows.append({"section": "loosen", "split": split, "entry": "delay_3m",
                             "scope": label, **s})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT_DIR / "data" / "processed" / f"v2_daily_eval_{ts}.csv"
    pd.DataFrame(out_rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {path}")


if __name__ == "__main__":
    main()
