"""Compare rule versions: v0 vs MA-breakout variants, through the Step 12 backtest.

Versions (see src/rule_signal_ma.py):
  v0            existing cross-symbol rule (reference)
  ma_only_*     MA breakout alone
  v1_*          v0 + MA breakout (additive score)
  v2_*          v0 with a MA-breakout filter (drop non-breakout candidates)
  (* = daily | intraday)

Evaluation follows the step13 baseline: conservative threshold, top1 primary /
top2 secondary, UP2/UP3 emphasised (not just UP5), risk metrics, and a
train/validation split. Labels are used only for evaluation, never for selection.

Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\compare_rule_versions.py
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

import pandas as pd

from daily_topk_backtest import join_signals_labels, quality_exclusion_pairs, run_backtests, threshold_config_by_name
from rule_signal import build_rule_signals
from rule_signal_ma import build_ma_rule_signals
from settings import load_active_symbols, load_data_config

TRAIN_END = "2026-02-28"
VAL_START, VAL_END = "2026-03-01", "2026-06-19"
SPLITS = ["full", "train", "validation"]

REPORT_COLS = [
    "rule_version", "split", "up_cross_n", "down_cross_n",
    "signal_count", "avg_signals_per_day",
    "hit_up2_rate", "hit_up3_rate", "hit_up5_rate",
    "average_mfe", "average_mae", "stop_first_rate",
    "return_30m", "return_60m", "return_eod", "symbol_concentration",
    "top1_hit_up3_rate", "top2_hit_up3_rate",
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
            "SELECT symbol, decision_time, ma5_daily, ma5_intraday FROM features_merged WHERE source = ?",
            conn, params=(model_source,))
        labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()
    for frame in (rel, ma, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return rel, ma, labels, bars, symbols, model_source


def _split_mask(joined, split):
    if split == "train":
        return joined["date"] <= TRAIN_END
    if split == "validation":
        return (joined["date"] >= VAL_START) & (joined["date"] <= VAL_END)
    return pd.Series(True, index=joined.index)


def main():
    rel, ma, labels, bars, symbols, model_source = _load()
    cfg = threshold_config_by_name("conservative")
    fail_pairs, outlier_pairs = quality_exclusion_pairs(pd.read_csv(_latest_quality_report()))

    print(f"active symbols: {len(symbols)}  model_source: {model_source}")
    print("building v0 (this is the slow step) ...")
    v0 = build_rule_signals(rel)

    versions = [("v0", v0)]
    for mode in ("ma_only", "v1", "v2"):
        for kind in ("daily", "intraday"):
            versions.append((f"{mode}_{kind}",
                             build_ma_rule_signals(rel, ma, mode=mode, ma_kind=kind, base=v0)))

    rows = []
    for name, sig in versions:
        up_n = int(sig["up_cross"].sum()) if "up_cross" in sig.columns else 0
        down_n = int(sig["down_cross"].sum()) if "down_cross" in sig.columns else 0
        joined = join_signals_labels(sig, labels, symbols=symbols)
        for split in SPLITS:
            sub = joined[_split_mask(joined, split)].reset_index(drop=True)
            if sub.empty:
                continue
            _, _, tsumm = run_backtests(sub, bars, threshold_configs=[cfg],
                                        fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)
            r = tsumm.iloc[0].to_dict()
            r.update({"rule_version": name, "split": split, "up_cross_n": up_n, "down_cross_n": down_n})
            rows.append(r)

    out = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT_DIR / "data" / "processed" / f"rule_version_comparison_{ts}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    report = out[REPORT_COLS].copy()
    for split in SPLITS:
        print(f"\n=== {split.upper()} (conservative; rates in %, n=signal_count) ===")
        print(f"  {'version':<16}{'n':>5}{'/day':>6}{'hit2':>7}{'hit3':>7}{'hit5':>6}"
              f"{'MFE':>7}{'MAE':>7}{'stop':>7}{'r30':>7}{'r60':>7}{'eod':>7}{'conc':>6}")
        for _, r in report[report["split"] == split].iterrows():
            print(f"  {r['rule_version']:<16}{int(r['signal_count']):>5}{r['avg_signals_per_day']:>6.2f}"
                  f"{r['hit_up2_rate']*100:>7.1f}{r['hit_up3_rate']*100:>7.1f}{r['hit_up5_rate']*100:>6.1f}"
                  f"{r['average_mfe']*100:>7.2f}{r['average_mae']*100:>7.2f}{r['stop_first_rate']*100:>7.1f}"
                  f"{r['return_30m']*100:>7.2f}{r['return_60m']*100:>7.2f}{r['return_eod']*100:>7.2f}"
                  f"{r['symbol_concentration']*100:>6.0f}")

    print(f"\nfull summary ({len(out)} rows, all columns) -> {path}")


if __name__ == "__main__":
    main()
