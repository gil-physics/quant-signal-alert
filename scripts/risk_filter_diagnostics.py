"""Risk-filter diagnostics: what separates winners from stop-outs at entry time.

Each signal is classified by first-touch of (+target, -stop) held to EOD:
    A = hit target first (success)   B = stopped first   C = no event
Then, for a set of entry-time features, we compare A/B/C distributions and the
A/B rate across per-feature quantile bins -- to surface a discriminating risk
filter. The selection threshold is held at a reasonable baseline (not tuned).

Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\risk_filter_diagnostics.py
    ..\\Python311-64\\python.exe scripts\\risk_filter_diagnostics.py --scope selected --target 0.03 --stop 0.015
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

import numpy as np
import pandas as pd

from bracket_eval import classify_outcome, day_arrays, prepare_entry
from daily_topk_backtest import (
    ThresholdConfig,
    apply_base_filters,
    apply_long_filter,
    join_signals_labels,
    quality_exclusion_pairs,
    select_daily_topk,
)
from entry_variants import build_preopen_context
from settings import load_active_symbols, load_data_config

# entry-time features to diagnose, and where each one lives
RULE_FEATURES = ["ret_20m_1m", "volume_z_20m_1m", "rank_ret_20m_1m",
                 "relative_ret_20m_to_universe", "universe_breadth"]
MERGED_FEATURES = ["ret_60m_1m", "vwap_distance_1m", "upper_shadow_1m", "range_1m",
                   "distance_from_day_high_1m", "distance_from_day_low_1m"]
PREOPEN_FEATURES = ["preopen_ret", "preopen_vwap_distance"]
FEATURES = RULE_FEATURES + MERGED_FEATURES + PREOPEN_FEATURES


def _latest_quality_report():
    reports = sorted(
        (ROOT_DIR / "data" / "processed").glob("quality_report_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return reports[0] if reports else None


def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _auc(a, b):
    """P(feature(A) > feature(B)); 0.5 = no separation, >0.5 = A higher."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = _rankdata(np.concatenate([a, b]))
    ra = r[:len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b)))


def _cohens_d(a, b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else float("nan")


def _load_signals(scope, down_limit, top_k, up_threshold, score_margin):
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
        extras = pd.read_sql_query(
            "SELECT symbol, decision_time, " + ", ".join(MERGED_FEATURES)
            + " FROM features_merged WHERE source = ?", conn, params=(model_source,))
        raw_bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume "
            "FROM bars_1m WHERE source = 'al_raw'", conn)
    finally:
        conn.close()

    for frame in (rule_signals, labels, bars, extras, raw_bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    joined = join_signals_labels(rule_signals, labels, symbols=symbols)
    # join_signals_labels keeps only SELECTION_COLUMNS; re-attach the rule features
    # it dropped (relative columns) from rule_signals.
    extra_rule = [c for c in RULE_FEATURES if c not in joined.columns]
    if extra_rule:
        joined = joined.merge(
            rule_signals[["symbol", "decision_time", *extra_rule]],
            on=["symbol", "decision_time"], how="left", validate="one_to_one")
    joined = joined.merge(extras, on=["symbol", "decision_time"], how="left", validate="one_to_one")

    preopen = build_preopen_context(raw_bars)
    joined = joined.merge(preopen[["symbol", "date", *PREOPEN_FEATURES]],
                          on=["symbol", "date"], how="left", validate="many_to_one")

    qr_path = _latest_quality_report()
    fail_pairs, outlier_pairs = quality_exclusion_pairs(pd.read_csv(qr_path))
    base, _ = apply_base_filters(joined, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)

    cfg = ThresholdConfig("baseline", up_threshold=up_threshold,
                          down_risk_limit=down_limit, score_margin=score_margin)
    candidates, long_stats = apply_long_filter(base, cfg)
    if scope == "selected":
        candidates, _ = select_daily_topk(candidates, top_k=top_k, cooldown_minutes=60)
    return candidates.reset_index(drop=True), day_arrays(bars), symbols, model_source, qr_path


def _classify_frame(cand, by_day, target, stop):
    groups, realized = [], []
    keep = np.zeros(len(cand), dtype=bool)
    cols = cand[["symbol", "date", "_decision_dt", "close"]]
    for pos, (sym, date, dec_dt, close_t) in enumerate(cols.itertuples(index=False, name=None)):
        arr = by_day.get((sym, date))
        if arr is None:
            continue
        rec = prepare_entry(arr, dec_dt, float(close_t), delay_bars=0)
        if rec is None:
            continue
        res = classify_outcome(*rec, target_pct=target, stop_pct=stop)
        if res is None:
            continue
        keep[pos] = True
        groups.append(res[0])
        realized.append(res[1]["realized_return"])
    out = cand.loc[keep].copy()
    out["group"] = groups
    out["realized_return"] = realized
    return out


def _group_stats(df):
    rows = []
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    for feat in FEATURES:
        for grp in ("A", "B", "C", "ALL"):
            s = df[feat] if grp == "ALL" else df.loc[df["group"] == grp, feat]
            s = s.dropna()
            row = {"feature": feat, "group": grp, "n": int(len(s)),
                   "mean": float(s.mean()) if len(s) else float("nan"),
                   "std": float(s.std()) if len(s) else float("nan")}
            for q in qs:
                row[f"p{int(q*100)}"] = float(s.quantile(q)) if len(s) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _discrimination(df):
    rows = []
    a_all = df["group"] == "A"
    b_all = df["group"] == "B"
    for feat in FEATURES:
        a = df.loc[a_all, feat].to_numpy(dtype=float)
        b = df.loc[b_all, feat].to_numpy(dtype=float)
        c = df.loc[df["group"] == "C", feat].to_numpy(dtype=float)
        auc = _auc(a, b)
        rows.append({
            "feature": feat,
            "n_A": int(np.sum(~np.isnan(a))), "n_B": int(np.sum(~np.isnan(b))),
            "n_C": int(np.sum(~np.isnan(c))),
            "mean_A": float(np.nanmean(a)) if a.size else float("nan"),
            "mean_B": float(np.nanmean(b)) if b.size else float("nan"),
            "mean_C": float(np.nanmean(c)) if c.size else float("nan"),
            "diff_A_minus_B": (float(np.nanmean(a) - np.nanmean(b))
                               if a.size and b.size else float("nan")),
            "cohens_d_AB": _cohens_d(a, b),
            "auc_AB": auc,
            "auc_strength": abs(auc - 0.5) if auc == auc else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("auc_strength", ascending=False).reset_index(drop=True)


def _quantile_bins(df, n_bins):
    rows = []
    for feat in FEATURES:
        sub = df[[feat, "group", "realized_return"]].dropna(subset=[feat])
        if len(sub) < n_bins:
            continue
        try:
            binned, edges = pd.qcut(sub[feat], n_bins, retbins=True, duplicates="drop")
        except ValueError:
            continue
        sub = sub.assign(_bin=binned)
        for i, (interval, g) in enumerate(sub.groupby("_bin", observed=True), start=1):
            n = len(g)
            rows.append({
                "feature": feat, "bin": i,
                "edge_lo": float(interval.left), "edge_hi": float(interval.right),
                "n": int(n), "feat_mean": float(g[feat].mean()),
                "A_rate": float((g["group"] == "A").mean()),
                "B_rate": float((g["group"] == "B").mean()),
                "C_rate": float((g["group"] == "C").mean()),
                "realized_mean": float(g["realized_return"].mean()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["pool", "selected"], default="pool",
                    help="pool = long-candidate universe (default, more samples); "
                         "selected = daily top-k baseline")
    ap.add_argument("--target", type=float, default=0.03, help="success target (default +3%)")
    ap.add_argument("--stop", type=float, default=0.015, help="stop (default -1.5%)")
    ap.add_argument("--down-limit", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--up-threshold", type=float, default=4.0)
    ap.add_argument("--score-margin", type=float, default=3.0)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    cand, by_day, symbols, model_source, qr_path = _load_signals(
        args.scope, args.down_limit, args.top_k, args.up_threshold, args.score_margin)
    df = _classify_frame(cand, by_day, args.target, args.stop)

    counts = df["group"].value_counts()
    nA, nB, nC = int(counts.get("A", 0)), int(counts.get("B", 0)), int(counts.get("C", 0))
    print(f"active symbols: {len(symbols)}  model_source: {model_source}  "
          f"quality: {Path(qr_path).name if qr_path else '(none)'}")
    print(f"scope={args.scope}  partition target=+{args.target*100:.1f}% stop=-{args.stop*100:.1f}%")
    print(f"signals classified: {len(df)}  A(hit target)={nA}  B(stopped)={nB}  C(no event)={nC}")
    if len(df):
        print(f"base rates: A={nA/len(df)*100:.1f}%  B={nB/len(df)*100:.1f}%  C={nC/len(df)*100:.1f}%")

    gstats = _group_stats(df)
    disc = _discrimination(df)
    qbins = _quantile_bins(df, args.bins)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    proc = ROOT_DIR / "data" / "processed"
    gstats.to_csv(proc / f"risk_diag_group_stats_{ts}.csv", index=False, encoding="utf-8-sig")
    disc.to_csv(proc / f"risk_diag_discrimination_{ts}.csv", index=False, encoding="utf-8-sig")
    qbins.to_csv(proc / f"risk_diag_quantile_bins_{ts}.csv", index=False, encoding="utf-8-sig")

    print("\n=== FEATURE DISCRIMINATION (A=success vs B=stop), ranked by separation ===")
    print("  AUC>0.5: winners have HIGHER feature; AUC<0.5: winners LOWER. |AUC-0.5| = strength.")
    for _, r in disc.iterrows():
        print(f"  {r['feature']:<30} AUC_AB={r['auc_AB']:.3f}  d={r['cohens_d_AB']:+.3f}  "
              f"meanA={r['mean_A']:+.4f} meanB={r['mean_B']:+.4f} meanC={r['mean_C']:+.4f}")

    print(f"\n=== STOP-RATE (B) BY QUINTILE for top-{min(5, len(disc))} discriminating features ===")
    for feat in disc["feature"].head(5):
        fb = qbins[qbins["feature"] == feat]
        if fb.empty:
            continue
        trend = "  ".join(f"q{int(r['bin'])}={r['B_rate']*100:.0f}%" for _, r in fb.iterrows())
        arate = "  ".join(f"q{int(r['bin'])}={r['A_rate']*100:.0f}%" for _, r in fb.iterrows())
        print(f"  {feat}\n      B(stop): {trend}\n      A(win) : {arate}")

    print(f"\nCSVs -> {proc}\\risk_diag_(group_stats|discrimination|quantile_bins)_{ts}.csv")


if __name__ == "__main__":
    main()
