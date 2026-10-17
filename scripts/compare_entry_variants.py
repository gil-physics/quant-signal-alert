"""Compare entry/selection variants against the conservative top1/top2 baseline.

The conservative threshold's daily top1/top2 is the FIXED baseline. Four variants
are evaluated on the active symbols (currently 15; expands to 20 as data arrives):

  baseline         conservative top-k, enter at decision close.
  entry_delay_3m   identical selection, but enter 3 minutes after the decision close.
  upper_shadow     candidate pool also requires upper_shadow_1m <= 0.5% (weak filter),
                   then re-select top-k.
  preopen_penalty  pre-open weakness is a SOFT score penalty (not a hard filter) that
                   re-ranks the daily top-k.

Selection and the pre-open penalty use only pre-decision information; labels are
never used for selection. Evaluation (MFE/MAE/hits/stop_first, target +5% / stop
-1.5%) is recomputed from each variant's actual entry price, so all four variants
share one evaluation method and differ only in selection or entry timing.

Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\compare_entry_variants.py
"""

from __future__ import annotations

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

from daily_topk_backtest import (
    apply_base_filters,
    apply_long_filter,
    join_signals_labels,
    quality_exclusion_pairs,
    threshold_config_by_name,
)
from entry_variants import (
    ENTRY_DELAY_BARS,
    PREOPEN_PENALTY,
    UPPER_SHADOW_LIMIT,
    add_selection_margin,
    bars_by_day,
    build_preopen_context,
    evaluate_variant,
    select_topk,
    summarize,
)
from settings import load_active_symbols, load_data_config

BASELINE_THRESHOLD = "conservative"
TOP_K = 2
COOLDOWN_MINUTES = 60
OUT_PATH = ROOT_DIR / "data" / "processed" / "entry_variant_summary.csv"


def _latest_quality_report():
    reports = sorted(
        (ROOT_DIR / "data" / "processed").glob("quality_report_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return reports[0] if reports else None


def _load_base():
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        rule_signals = pd.read_sql_query(
            "SELECT * FROM rule_signals WHERE source = ?", conn, params=(model_source,))
        labels = pd.read_sql_query(
            "SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume "
            "FROM bars_1m WHERE source = ?", conn, params=(model_source,))
        extras = pd.read_sql_query(
            "SELECT symbol, decision_time, upper_shadow_1m "
            "FROM features_merged WHERE source = ?", conn, params=(model_source,))
        raw_bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume "
            "FROM bars_1m WHERE source = 'al_raw'", conn)
    finally:
        conn.close()

    for frame in (rule_signals, labels, bars, extras, raw_bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    joined = join_signals_labels(rule_signals, labels, symbols=symbols)
    joined = joined.merge(extras, on=["symbol", "decision_time"], how="left", validate="one_to_one")

    quality_report = _latest_quality_report()
    qr = pd.read_csv(quality_report)
    fail_pairs, outlier_pairs = quality_exclusion_pairs(qr)
    base, _ = apply_base_filters(joined, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)

    preopen = build_preopen_context(raw_bars)
    base = base.merge(
        preopen[["symbol", "date", "preopen_ok"]],
        on=["symbol", "date"], how="left", validate="many_to_one")

    by_day = bars_by_day(bars)
    return base, by_day, symbols, model_source, quality_report, preopen


def _scoped_rows(evaluated, variant):
    """All / top1 / top2 summary rows for one evaluated variant frame."""
    rows = []
    for scope, frame in (
        ("all", evaluated),
        ("top1", evaluated[evaluated.get("daily_rank") == 1] if len(evaluated) else evaluated),
        ("top2", evaluated[evaluated.get("daily_rank") == 2] if len(evaluated) else evaluated),
    ):
        s = summarize(frame, name=variant)
        s = {"variant": variant, "scope": scope, **{k: v for k, v in s.items() if k != "variant"}}
        rows.append(s)
    return rows


def _selection_keys(selected):
    if selected is None or len(selected) == 0:
        return set()
    return set(zip(selected["symbol"], selected["decision_time"]))


def _fmt(s):
    return (
        f"n={s['n']:>4} hit2={s['hit2']*100:>5.2f}% hit3={s['hit3']*100:>5.2f}% "
        f"hit5={s['hit5']*100:>5.2f}% mfe={s['mfe']*100:>5.2f}% mae={s['mae']*100:>6.2f}% "
        f"stop={s['stop_first']*100:>5.2f}% r30={s['ret30']*100:>6.2f}% "
        f"r60={s['ret60']*100:>6.2f}% r120={s['ret120']*100:>6.2f}% eod={s['eod']*100:>6.2f}% "
        f"conc={s['symbol_concentration']*100:>5.1f}%"
    )


def main():
    base, by_day, symbols, model_source, quality_report, preopen = _load_base()
    cfg = threshold_config_by_name(BASELINE_THRESHOLD)
    candidates, long_stats = apply_long_filter(base, cfg)

    preopen_ok_count = int(candidates["preopen_ok"].fillna(False).astype(bool).sum())
    upper_known = int(candidates["upper_shadow_1m"].notna().sum())

    print(f"active symbols: {len(symbols)}  model_source: {model_source}")
    print(f"quality report: {Path(quality_report).name if quality_report else '(none)'}")
    print(
        f"conservative candidates: {len(candidates)}  "
        f"preopen_ok: {preopen_ok_count}  upper_shadow_1m known: {upper_known}")

    # --- selection per variant ---------------------------------------------
    baseline_sel = select_topk(
        add_selection_margin(candidates, preopen_penalty=0.0),
        top_k=TOP_K, cooldown_minutes=COOLDOWN_MINUTES)

    upper_pool = candidates[candidates["upper_shadow_1m"] <= UPPER_SHADOW_LIMIT].reset_index(drop=True)
    upper_sel = select_topk(
        add_selection_margin(upper_pool, preopen_penalty=0.0),
        top_k=TOP_K, cooldown_minutes=COOLDOWN_MINUTES)

    preopen_sel = select_topk(
        add_selection_margin(candidates, preopen_penalty=PREOPEN_PENALTY),
        top_k=TOP_K, cooldown_minutes=COOLDOWN_MINUTES)

    # --- evaluation ---------------------------------------------------------
    variants = [
        ("baseline", baseline_sel, 0),
        (f"entry_delay_{ENTRY_DELAY_BARS}m", baseline_sel, ENTRY_DELAY_BARS),
        ("upper_shadow", upper_sel, 0),
        ("preopen_penalty", preopen_sel, 0),
    ]

    base_keys = _selection_keys(baseline_sel)
    summary_rows = []
    print("\n=== ENTRY/SELECTION VARIANTS vs CONSERVATIVE BASELINE ===")
    for name, selected, delay in variants:
        evaluated = evaluate_variant(selected, by_day, delay_bars=delay)
        diff = len(_selection_keys(selected) ^ base_keys) // 1  # symmetric-difference size
        rows = _scoped_rows(evaluated, name)
        summary_rows.extend(rows)
        all_row = rows[0]
        print(f"{name:<18} sel_diff_vs_baseline={diff:>3}  {_fmt(all_row)}")

    out = pd.DataFrame(summary_rows)
    out.insert(0, "model_source", model_source)
    out.insert(1, "n_symbols", len(symbols))
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {OUT_PATH}")
    print("\n(top1/top2 breakdown is in the CSV; sel_diff is symmetric set difference "
          "of selected (symbol, decision_time) vs baseline.)")


if __name__ == "__main__":
    main()
