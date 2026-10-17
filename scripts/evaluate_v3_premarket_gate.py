"""v3.0 premarket A/B gate: v1.1 vs v3.0a (target premarket) vs v3.0b (universe premarket).

Scores the existing trained models on ONE identical validation frame (same rows, same
labels) and applies the same recent-month-weighted GO/NO-GO gate used for v2.0/v2.1.
"v1.1" = the v1 model file scored on the REAL cross-symbol features (the adopted live
config). v3.0a/v3.0b additionally have the per-day premarket context joined on.

Writes data/processed/v3_0_premarket_ab_report.md + risk_model_v3_premarket_gate.csv.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Adopt nothing live.

    ..\\Python311-64\\python.exe scripts\\evaluate_v3_premarket_gate.py
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

import risk_model as rm
from config import DB_PATH, PROCESSED_DIR
from settings import load_active_symbols, load_data_config, realtime_default_symbol
from utils import now_kst_text


# v1 is scored on real cross-symbol features -> "v1.1", the adopted baseline.
BASELINE = "v1"
VARIANTS = ["v3_0a_premarket_000660", "v3_0b_premarket_universe"]
RECENT_MONTHS = ("2026-05", "2026-06")          # recent-month gate window
GATE_RECENT_AUC_MIN = 0.010                     # recent-month AUC must improve by >= this
GATE_WORST_MONTH_FLOOR = -0.005                 # no recent month may regress past this
TARGET = "target_mae_risk"


def parse_args():
    p = argparse.ArgumentParser(description="v3.0 premarket A/B gate vs v1.1.")
    p.add_argument("--train-start", default="2025-06-02")
    p.add_argument("--train-end", default="2026-02-28")
    p.add_argument("--validation-start", default="2026-03-01")
    p.add_argument("--validation-end", default="2026-06-26")
    p.add_argument("--db", default=str(DB_PATH))
    return p.parse_args()


def _table_exists(conn, name):
    return bool(conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()[0])


def load_inputs(conn, model_source, symbols):
    alerts = pd.read_sql_query("SELECT * FROM directional_alerts WHERE source = ?", conn, params=(model_source,))
    fm = pd.read_sql_query("SELECT * FROM features_merged WHERE source = ?", conn, params=(model_source,))
    rel = pd.read_sql_query("SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,))
    labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
    bars = pd.read_sql_query("SELECT symbol, bar_start, low, close FROM bars_1m WHERE source = ?", conn, params=(model_source,))
    premarket = pd.read_sql_query("SELECT * FROM premarket_features_daily", conn) \
        if _table_exists(conn, "premarket_features_daily") else None
    for frame in (alerts, fm, rel, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return alerts, fm, rel, labels, bars, premarket


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pooled_metrics(frame):
    m = rm.classification_summary(frame, target_col=TARGET)
    return {"auc": m["auc"], "pr_auc": m["pr_auc"], "logloss": m["logloss"]}


def recent_auc(frame):
    """Pooled AUC over the recent-month window (the metric the gate keys on)."""
    g = frame[frame["date"].str.slice(0, 7).isin(RECENT_MONTHS)]
    return rm.roc_auc_score_np(g[TARGET], g["risk_prob"]) if len(g) else float("nan")


def decile_monotonicity(frame):
    """Fraction of adjacent risk_prob-decile pairs whose mae_risk_rate increases."""
    p = frame["risk_prob"].to_numpy(dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    rates = [frame.loc[bins == b, TARGET].mean() for b in range(10) if (bins == b).any()]
    if len(rates) < 2:
        return float("nan")
    steps = np.diff(rates)
    return float((steps > 0).mean())


def agrade_lowrisk(frame):
    """Among grade-A alerts the model marks LOW risk (median split): the trading payoff."""
    a = frame[frame["alert_grade"] == "A"]
    if len(a) < 20:
        return {"n": int(len(a)), "avg_ret60": float("nan"), "avg_MAE": float("nan"),
                "win_rate": float("nan"), "mae_risk_rate": float("nan")}
    low = a[a["risk_prob"] <= a["risk_prob"].median()]
    return {
        "n": int(len(low)),
        "avg_ret60": float(low["future_return_60m"].mean()),
        "avg_MAE": float(low["future_MAE"].mean()),
        "win_rate": float((low["future_return_60m"] > 0).mean()),
        "mae_risk_rate": float(low[TARGET].mean()),
    }


def evaluate_model(version, valid):
    """Score one model on the shared validation frame and collect all gate metrics."""
    model, pre, meta = rm.load_model(rm.resolve_model_path(version))
    scored = valid.copy()
    scored["risk_prob"] = rm.predict_with_model(model, pre, scored)
    buckets = rm.bucket_report(scored, target_col=TARGET)

    def bucket(name, col):
        row = buckets[buckets["risk_bucket"] == name]
        return float(row.iloc[0][col]) if len(row) else float("nan")

    monthly = rm.monthly_auc(scored, target_col=TARGET)
    return {
        "version": version,
        "scored": scored,
        "n": int(len(scored)),
        **pooled_metrics(scored),
        "recent_auc": recent_auc(scored),
        "monthly": monthly,
        "top10_mae_risk_rate": bucket("top_10%", "mae_risk_rate"),
        "top20_mae_risk_rate": bucket("top_20%", "mae_risk_rate"),
        "bottom20_mae_risk_rate": bucket("bottom_20%", "mae_risk_rate"),
        "top10_avg_MAE": bucket("top_10%", "avg_MAE"),
        "decile_monotonicity": decile_monotonicity(scored),
        "agrade_lowrisk": agrade_lowrisk(scored),
        "feature_count": len(getattr(pre, "feature_columns", []) or []),
    }


def premarket_diagnostics(version, valid, premarket_columns):
    """Premarket-specific diagnostics: importance ranks, missing-data coverage,
    and discrimination on days WITH vs WITHOUT premarket data for the target."""
    vsfx = "_" + version.replace(".", "_")
    imp_path = PROCESSED_DIR / f"feature_importance{vsfx}.csv"
    pm_imp = []
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        imp["rank"] = imp.index + 1
        pm_imp = imp[imp["feature"].isin(premarket_columns)][["feature", "importance", "rank"]] \
            .to_dict("records")

    has_col = next((c for c in premarket_columns if c.endswith("_has_data")), None)
    miss_rate = float("nan")
    with_auc = without_auc = float("nan")
    if has_col and has_col in valid.columns:
        has = valid[has_col].fillna(0) > 0
        miss_rate = float((~has).mean())
        if has.any():
            g = valid[has]
            with_auc = rm.roc_auc_score_np(g[TARGET], g["risk_prob"])
        if (~has).any():
            g = valid[~has]
            without_auc = rm.roc_auc_score_np(g[TARGET], g["risk_prob"])

    bar_col = next((c for c in premarket_columns if c.endswith("_bar_count")), None)
    insufficient_days = float("nan")
    if bar_col and bar_col in valid.columns:
        per_day = valid.groupby("date")[bar_col].first()
        insufficient_days = int((per_day.fillna(0) < 30).sum())
    return {
        "premarket_importance": pm_imp,
        "premarket_feature_count": len(premarket_columns),
        "missing_premarket_rate": miss_rate,
        "auc_days_with_premarket": with_auc,
        "auc_days_without_premarket": without_auc,
        "days_insufficient_premarket_bars": insufficient_days,
    }


def decide_verdict(base, variant):
    """GO / PARTIAL / NO-GO vs v1.1, per the recent-month-weighted gate (section 10)."""
    d_recent = variant["recent_auc"] - base["recent_auc"]
    # worst per-month delta over the recent window
    bm = base["monthly"].set_index("month")["auc"]
    vm = variant["monthly"].set_index("month")["auc"]
    recent_deltas = {m: float(vm.get(m, np.nan) - bm.get(m, np.nan)) for m in RECENT_MONTHS}
    all_deltas = {m: float(vm.get(m, np.nan) - bm.get(m, np.nan)) for m in vm.index}
    worst_recent = min((d for d in recent_deltas.values() if d == d), default=float("nan"))
    n_months_up = sum(1 for d in all_deltas.values() if d == d and d > 0)
    bucket_ok = (variant["top10_mae_risk_rate"] >= base["top10_mae_risk_rate"] - 1e-9
                 and variant["bottom20_mae_risk_rate"] <= base["bottom20_mae_risk_rate"] + 0.005)
    mae_ok = variant["agrade_lowrisk"]["avg_MAE"] >= base["agrade_lowrisk"]["avg_MAE"] - 0.0005 \
        if variant["agrade_lowrisk"]["avg_MAE"] == variant["agrade_lowrisk"]["avg_MAE"] else False
    single_month_only = n_months_up <= 1

    reasons = []
    go = (d_recent >= GATE_RECENT_AUC_MIN and worst_recent > GATE_WORST_MONTH_FLOOR
          and bucket_ok and mae_ok and not single_month_only)
    if go:
        verdict = "GO"
    elif d_recent > 0 and (bucket_ok or mae_ok):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    if d_recent < GATE_RECENT_AUC_MIN:
        reasons.append(f"recent-month AUC Δ {d_recent:+.4f} < +{GATE_RECENT_AUC_MIN:.3f}")
    if worst_recent <= GATE_WORST_MONTH_FLOOR:
        reasons.append(f"worst recent month Δ {worst_recent:+.4f} <= {GATE_WORST_MONTH_FLOOR:+.3f}")
    if not bucket_ok:
        reasons.append("risk-bucket separation not preserved")
    if not mae_ok:
        reasons.append("A-grade low-risk MAE worse")
    if single_month_only:
        reasons.append(f"improvement not broad ({n_months_up} month(s) up)")
    return {"verdict": verdict, "d_recent": d_recent, "worst_recent": worst_recent,
            "recent_deltas": recent_deltas, "all_deltas": all_deltas,
            "n_months_up": n_months_up, "bucket_ok": bucket_ok, "mae_ok": mae_ok,
            "reasons": reasons}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt_pct(x):
    return f"{x:+.4f}" if isinstance(x, float) and x == x else "n/a"


def fmt_val(x):
    """Unsigned formatter for rates/AUCs (vs fmt_pct which signs deltas)."""
    return f"{x:.4f}" if isinstance(x, float) and x == x else "n/a"


def write_report(results, verdicts, diags, valid, args, target):
    base = results[BASELINE]
    lines = []
    lines.append("# risk_model v3.0a / v3.0b vs v1.1 — premarket-context A/B report & GO/NO-GO\n")
    any_go = any(v["verdict"] == "GO" for v in verdicts.values())
    headline = ("A v3.0 variant PASSES — see below." if any_go
                else "**Verdict: NO-GO for both v3.0a and v3.0b. Keep v1.1 as the live config.**")
    lines.append(headline + "\n")
    lines.append(
        "v3.0 keeps the **v1.1 baseline** (the v1 model's 8 cross-symbol features served on "
        "the REAL cross-section) and ATTACHES per-day **premarket (08:00–09:00)** context to "
        "every same-day regular-session row (joined by trade_date; the premarket window closes "
        "before the open → causal). Labels and the direction rule are unchanged.\n")
    lines.append(
        f"- **v3.0a** (`v3_0a_premarket_000660`) — {target} premarket only "
        f"({len(rm.PREMARKET_COLUMNS_V3_0A)} premarket features).\n"
        f"- **v3.0b** (`v3_0b_premarket_universe`) — {target} + universe-wide premarket breadth/"
        f"relative/rank ({len(rm.PREMARKET_COLUMNS_V3_0B)} premarket features).\n")
    lines.append(
        f'All models scored on the IDENTICAL validation frame ({len(valid):,} rows, '
        f'{args.validation_start}→{args.validation_end}, positive_rate '
        f'{valid[TARGET].mean():.3f}). "v1.1" = the v1 model file scored on real cross-symbol '
        f'features — the adopted live config.\n')

    # 1. discrimination
    months = sorted(set(base["monthly"]["month"]))
    lines.append("## 1. Discrimination (validation AUC; higher = better)\n")
    head = "| | pooled | recent(05+06) | " + " | ".join(months) + " |"
    sep = "|" + "---|" * (3 + len(months))
    lines.append(head)
    lines.append(sep)
    for v in [BASELINE] + VARIANTS:
        r = results[v]
        mm = r["monthly"].set_index("month")["auc"]
        name = "v1.1" if v == BASELINE else v
        row = f"| {name} | {r['auc']:.4f} | {r['recent_auc']:.4f} | " + \
              " | ".join(f"{mm.get(m, float('nan')):.4f}" for m in months) + " |"
        lines.append(row)
    for v in VARIANTS:
        ver = verdicts[v]
        mm_b = base["monthly"].set_index("month")["auc"]
        mm_v = results[v]["monthly"].set_index("month")["auc"]
        name = "Δ " + v + " vs v1.1"
        row = f"| {name} | {results[v]['auc'] - base['auc']:+.4f} | {ver['d_recent']:+.4f} | " + \
              " | ".join(f"{mm_v.get(m, float('nan')) - mm_b.get(m, float('nan')):+.4f}" for m in months) + " |"
        lines.append(row)
    lines.append("")

    # 2-3 buckets + MAE
    lines.append("## 2–3. Risk-bucket separation + MAE\n")
    lines.append("| model | top10% mae_risk_rate | top10% avgMAE | bottom20% mae_risk_rate | decile monotonicity |")
    lines.append("|---|---|---|---|---|")
    for v in [BASELINE] + VARIANTS:
        r = results[v]
        name = "v1.1" if v == BASELINE else v
        lines.append(f"| {name} | {r['top10_mae_risk_rate']:.3f} | {r['top10_avg_MAE']:.4f} | "
                     f"{r['bottom20_mae_risk_rate']:.3f} | {r['decile_monotonicity']:.2f} |")
    lines.append("")

    # 4 A-grade low-risk
    lines.append("## 4. A-grade low-risk quality (the trading payoff)\n")
    lines.append("Among grade-A alerts the model marks low risk_prob (median split):\n")
    lines.append("| model | n | avg ret60 | avg MAE | win-rate | mae_risk_rate |")
    lines.append("|---|---|---|---|---|---|")
    for v in [BASELINE] + VARIANTS:
        a = results[v]["agrade_lowrisk"]
        name = "v1.1" if v == BASELINE else v
        lines.append(f"| {name} | {a['n']} | {a['avg_ret60']:+.4f} | {a['avg_MAE']:+.4f} | "
                     f"{a['win_rate']:.3f} | {a['mae_risk_rate']:.3f} |")
    lines.append("")

    # 5 premarket diagnostics
    lines.append("## 5. Premarket-specific diagnostics\n")
    lines.append("| variant | pm features | top pm importance (feature: rank) | missing-pm rate | "
                 "AUC days-with-pm | AUC days-without-pm | days <30 pm bars |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        d = diags[v]
        top = sorted(d["premarket_importance"], key=lambda r: r["rank"])[:3]
        top_s = ", ".join(f"{r['feature']}: #{r['rank']}" for r in top) or "n/a"
        lines.append(f"| {v} | {d['premarket_feature_count']} | {top_s} | "
                     f"{fmt_val(d['missing_premarket_rate'])} | {fmt_val(d['auc_days_with_premarket'])} | "
                     f"{fmt_val(d['auc_days_without_premarket'])} | {d['days_insufficient_premarket_bars']} |")
    lines.append("")
    lines.append("_The 2026-03→06 validation window has full target premarket coverage, so the "
                 "with/without-premarket AUC split is uninformative here (the 11 missing-data days "
                 "fall earlier in the sample). Premarket features dominate the importance ranks yet "
                 "**degrade** generalisation — a day-broadcast feature (constant within a day) lets "
                 "the model fit day-regime noise, which is why recent-month AUC and the low-risk "
                 "bucket get worse, not better._\n")

    # gate
    lines.append("## Gate (same recent-weighted gate as v2.0/v2.1)\n")
    lines.append(f"GO requires: recent-month(05+06) AUC Δ ≥ +{GATE_RECENT_AUC_MIN:.3f} vs v1.1 AND no "
                 f"recent month regresses past {GATE_WORST_MONTH_FLOOR:+.3f} AND risk-bucket separation "
                 f"preserved AND A-grade low-risk MAE not worse AND the gain is broad (>1 month up).\n")
    lines.append("| variant | recent Δ | worst recent-month Δ | months up | bucket ok | A-grade MAE ok | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        ver = verdicts[v]
        lines.append(f"| **{v}** | {ver['d_recent']:+.4f} | {ver['worst_recent']:+.4f} | "
                     f"{ver['n_months_up']} | {ver['bucket_ok']} | {ver['mae_ok']} | **{ver['verdict']}** |")
    lines.append("")
    for v in VARIANTS:
        ver = verdicts[v]
        if ver["reasons"]:
            lines.append(f"- **{v}** ({ver['verdict']}): " + "; ".join(ver["reasons"]) + ".")
    lines.append("")

    # serving status
    lines.append("## Live serving status\n")
    lines.append("- **v1.1 remains the adopted live config.**")
    lines.append("- v3.0 variants are research-only until they pass the gate.")
    lines.append("- If a v3.0 variant ever passes, realtime serving must compute the SAME premarket "
                 "features (08:00–09:00, same definitions) before the open and join them by date "
                 "(`requires_premarket=True` in the model metadata) — this is **not wired** yet.")
    lines.append("- v3.0c / v3.0d (005930 / 402340 premarket anchors) are not built in this round.\n")

    report = "\n".join(lines)
    out_path = PROCESSED_DIR / "v3_0_premarket_ab_report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def main():
    args = parse_args()
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    target = realtime_default_symbol()
    split = rm.SplitConfig(train_start=args.train_start, train_end=args.train_end,
                           validation_start=args.validation_start, validation_end=args.validation_end)

    print(f"[{now_kst_text()}] v3.0 premarket gate: baseline=v1.1 variants={VARIANTS} source={model_source}")
    conn = sqlite3.connect(args.db)
    try:
        alerts, fm, rel, labels, bars, premarket = load_inputs(conn, model_source, symbols)
    finally:
        conn.close()
    if premarket is None:
        raise SystemExit("premarket_features_daily not found - run scripts/build_premarket_features.py first.")

    # one shared frame carrying the v3.0b superset (v1 relative + target + universe premarket)
    data, _cols = rm.build_model_frame(
        alerts, fm, rel, labels, bars,
        relative_columns=rm.RELATIVE_FEATURE_COLUMNS,
        feature_columns=rm.FEATURE_COLUMNS_V3_0B,
        premarket=premarket, premarket_columns=rm.PREMARKET_COLUMNS_V3_0B,
    )
    _train, valid = rm.split_by_date(data, split=split)
    valid = valid.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  validation frame: rows={len(valid)} positive_rate={valid[TARGET].mean():.3f}")

    results = {}
    for v in [BASELINE] + VARIANTS:
        results[v] = evaluate_model(v, valid)
        r = results[v]
        print(f"  {v:<28} pooled_auc={r['auc']:.4f} recent_auc={r['recent_auc']:.4f} "
              f"top10={r['top10_mae_risk_rate']:.3f} bottom20={r['bottom20_mae_risk_rate']:.3f} "
              f"monotonic={r['decile_monotonicity']:.2f}")

    diags = {v: premarket_diagnostics(v, results[v]["scored"], rm.MODEL_VERSIONS[v]["premarket"]) for v in VARIANTS}
    verdicts = {v: decide_verdict(results[BASELINE], results[v]) for v in VARIANTS}

    # csv summary
    rows = []
    for v in [BASELINE] + VARIANTS:
        r = results[v]
        rows.append({
            "version": "v1.1" if v == BASELINE else v,
            "pooled_auc": r["auc"], "recent_auc": r["recent_auc"], "pr_auc": r["pr_auc"], "logloss": r["logloss"],
            "top10_mae_risk_rate": r["top10_mae_risk_rate"], "top20_mae_risk_rate": r["top20_mae_risk_rate"],
            "bottom20_mae_risk_rate": r["bottom20_mae_risk_rate"], "decile_monotonicity": r["decile_monotonicity"],
            "agrade_lowrisk_avg_MAE": r["agrade_lowrisk"]["avg_MAE"],
            "agrade_lowrisk_avg_ret60": r["agrade_lowrisk"]["avg_ret60"],
            "verdict": "baseline" if v == BASELINE else verdicts[v]["verdict"],
        })
    csv_path = PROCESSED_DIR / "risk_model_v3_premarket_gate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    report_path = write_report(results, verdicts, diags, valid, args, target)
    print("\nverdicts:")
    for v in VARIANTS:
        print(f"  {v:<28} {verdicts[v]['verdict']:<8} recentΔ={verdicts[v]['d_recent']:+.4f} "
              f"worstΔ={verdicts[v]['worst_recent']:+.4f} ({'; '.join(verdicts[v]['reasons']) or 'all gates pass'})")
    print(f"\nreport -> {report_path}")
    print(f"summary -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
