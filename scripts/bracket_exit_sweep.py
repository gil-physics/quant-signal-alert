"""Exit-mechanics sweep on a fixed, reasonable selection baseline.

Selection knobs are HELD at a sensible baseline (not tuned): up_threshold and
score_margin are fixed; down_limit and top_k are exposed as args (the retained
selection knobs). The sweep then varies only the EXIT mechanics:

    target    x  stop  x  time_exit  x  entry_delay

For each combination it reports realised P&L statistics from a first-touch
bracket (see src/bracket_eval.py). Labels are not used; outcomes are recomputed
from the actual entry price.

Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\bracket_exit_sweep.py
    ..\\Python311-64\\python.exe scripts\\bracket_exit_sweep.py --down-limit 2 --top-k 1
"""

from __future__ import annotations

import argparse
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

import pandas as pd

from bracket_eval import day_arrays, prepare_entry, simulate_bracket, summarize_bracket
from daily_topk_backtest import (
    ThresholdConfig,
    apply_base_filters,
    apply_long_filter,
    join_signals_labels,
    quality_exclusion_pairs,
    select_daily_topk,
)
from settings import load_active_symbols, load_data_config

# --- exit grid (the analysis spec) ---------------------------------------
TARGETS = [0.010, 0.015, 0.020, 0.030, 0.050]
STOPS = [0.007, 0.010, 0.015, 0.020]
TIME_EXITS = [("10m", 10.0), ("20m", 20.0), ("30m", 30.0),
              ("60m", 60.0), ("120m", 120.0), ("eod", float("inf"))]
DELAYS = [0, 3, 5]


def _latest_quality_report():
    reports = sorted(
        (ROOT_DIR / "data" / "processed").glob("quality_report_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return reports[0] if reports else None


def _selected_baseline(down_limit, top_k, up_threshold, score_margin):
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        rule_signals = pd.read_sql_query(
            "SELECT * FROM rule_signals WHERE source = ?", conn, params=(model_source,))
        labels = pd.read_sql_query(
            "SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()

    for frame in (rule_signals, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    joined = join_signals_labels(rule_signals, labels, symbols=symbols)
    qr_path = _latest_quality_report()
    fail_pairs, outlier_pairs = quality_exclusion_pairs(pd.read_csv(qr_path))
    base, _ = apply_base_filters(joined, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)

    cfg = ThresholdConfig("baseline", up_threshold=up_threshold,
                          down_risk_limit=down_limit, score_margin=score_margin)
    candidates, long_stats = apply_long_filter(base, cfg)
    selected, _ = select_daily_topk(candidates, top_k=top_k, cooldown_minutes=60)
    return selected, day_arrays(bars), symbols, model_source, qr_path, long_stats


def _prepare_per_delay(selected, by_day):
    """For each delay, a list of (entry_price, highs, lows, closes, offsets)."""
    prepared = {d: [] for d in DELAYS}
    for _, row in selected.iterrows():
        arr = by_day.get((row["symbol"], row["date"]))
        if arr is None:
            continue
        for d in DELAYS:
            rec = prepare_entry(arr, row["_decision_dt"], float(row["close"]), delay_bars=d)
            if rec is not None:
                prepared[d].append(rec)
    return prepared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down-limit", type=float, default=1.0,
                    help="down_score ceiling (retained knob; baseline 1)")
    ap.add_argument("--top-k", type=int, default=2, help="daily top-k (retained knob; baseline 2)")
    ap.add_argument("--up-threshold", type=float, default=4.0, help="held baseline (non-binding for top-k)")
    ap.add_argument("--score-margin", type=float, default=3.0, help="held baseline (non-binding for top-k)")
    args = ap.parse_args()

    selected, by_day, symbols, model_source, qr_path, long_stats = _selected_baseline(
        args.down_limit, args.top_k, args.up_threshold, args.score_margin)

    print(f"active symbols: {len(symbols)}  model_source: {model_source}")
    print(f"quality report: {Path(qr_path).name if qr_path else '(none)'}")
    print(f"baseline selection: up>={args.up_threshold} down<={args.down_limit} "
          f"margin>={args.score_margin} top_k={args.top_k}  "
          f"candidates={long_stats['after_score_filter']}  selected={len(selected)}")

    prepared = _prepare_per_delay(selected, by_day)

    rows = []
    for delay in DELAYS:
        recs = prepared[delay]
        for target in TARGETS:
            for stop in STOPS:
                for label, tmin in TIME_EXITS:
                    results = [
                        r for r in (
                            simulate_bracket(ep, h, l, c, off,
                                             target_pct=target, stop_pct=stop, time_exit_min=tmin)
                            for (ep, h, l, c, off) in recs
                        ) if r is not None
                    ]
                    s = summarize_bracket(results, stop_pct=stop)
                    rows.append({
                        "entry_delay": f"{delay}m",
                        "target_pct": target,
                        "stop_pct": stop,
                        "time_exit": label,
                        **s,
                    })

    out = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT_DIR / "data" / "processed" / f"bracket_exit_sweep_{ts}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    def fmt(df):
        for _, r in df.iterrows():
            print(f"  delay{r['entry_delay']:<3} T+{r['target_pct']*100:>4.1f}% "
                  f"S-{r['stop_pct']*100:>4.1f}% exit={r['time_exit']:<4} "
                  f"n={int(r['n']):>4} win={r['win_rate']*100:>5.1f}% "
                  f"avgW={r['avg_win']*100:>5.2f}% avgL={r['avg_loss']*100:>6.2f}% "
                  f"real={r['realized_return']*100:>6.3f}% expR={r['expectancy_r']:>6.3f} "
                  f"tgt={r['target_first_rate']*100:>4.1f}% stop={r['stop_first_rate']*100:>4.1f}% "
                  f"time={r['time_exit_rate']*100:>4.1f}% hold={r['avg_hold_min']:>5.1f}m")

    print("\n=== TOP 15 by expectancy_r (realised return per unit of stop risk) ===")
    fmt(out.sort_values("expectancy_r", ascending=False).head(15))
    print("\n=== TOP 10 by realized_return ===")
    fmt(out.sort_values("realized_return", ascending=False).head(10))
    print(f"\nfull grid ({len(out)} rows) -> {path}")


if __name__ == "__main__":
    main()
