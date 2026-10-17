"""Step 16/17: A/B gate -- TCN challenger vs LightGBM champion, per lineage.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Adopt nothing live:
whatever the verdict, v1.1 remains the served model until a GO verdict here AND the
Phase-2 serving-parity work are both done.

Same recent-weighted gate math as v2.x/v3.0 (decide_verdict functionally identical to
scripts/evaluate_v3_premarket_gate.py; Step 17 only added the additive ``mae_margin``
kwarg whose default reproduces the v3 hardcode). The champion side is rescored
in-process (LightGBM model file on the shared frame); the TCN side is read from the
predictions parquet(s) written by scripts/train_tcn_risk_model.py, so this script
never imports torch and runs on any 64-bit environment.

Lineages (--tcn-version, see GATE_LINEAGES):
  v4_tcn      Step 16: rest-of-day target_mae_risk, single split, 5-seed ensemble,
              champion = v1.1 (v1 model on the real cross-symbol frame).
  v4_tcn_60m  Step 17: fixed-horizon target_mae_risk_60m, expanding walk-forward folds
              (risk_model.WALK_FORWARD_FOLDS_60M), single pre-registered seed per fold,
              champion = v1_60m fold models. The four fold validation months (each
              genuinely out-of-sample) are STITCHED into one walk-forward frame and the
              unchanged gate math runs on it; per-fold deltas are reported alongside.
              Main population = decision time-of-day <= HORIZON_60M_MAIN_CUTOFF; later
              rows are a late-session diagnostic (truncated-horizon label), never gated.

Run:
  ..\\Python311-64\\python.exe scripts\\evaluate_tcn_gate.py
  ..\\Python311-64\\python.exe scripts\\evaluate_tcn_gate.py --tcn-version v4_tcn_60m
  ..\\Python311-64\\python.exe scripts\\evaluate_tcn_gate.py ^
      --ablation-predictions ablA_no_static=risk_model_predictions_v4_tcn_ablA.parquet ^
      --ablation-predictions ablB_seq120=risk_model_predictions_v4_tcn_ablB.parquet
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
import tcn_model as tm   # torch-free import: tcn_model only imports torch lazily
from config import DB_PATH, PROCESSED_DIR
from settings import load_active_symbols, load_data_config
from utils import now_kst_text


# v1 is scored on real cross-symbol features -> "v1.1", the adopted baseline.
BASELINE = "v1"
CHALLENGER = "v4_tcn"
RECENT_MONTHS = ("2026-05", "2026-06")          # recent-month gate window
GATE_RECENT_AUC_MIN = 0.010                     # recent-month AUC must improve by >= this
GATE_WORST_MONTH_FLOOR = -0.005                 # no recent month may regress past this
TARGET = "target_mae_risk"

# Step 17: per-lineage gate configuration. The v4_tcn entry reproduces the Step 16
# behavior exactly (its mae_margin 0.0005 = the historical decide_verdict hardcode).
GATE_LINEAGES = {
    "v4_tcn": {
        "target": TARGET,
        "mae_col": "future_MAE",
        "champion": BASELINE,               # scored on the real cross-symbol frame = v1.1
        "champion_label": "v1.1",
        "horizon_minutes": None,
        "cutoff": None,
        "mae_margin": 0.0005,
        "folds": None,
        "predictions": "risk_model_predictions_v4_tcn.parquet",
        "report": "tcn_vs_lgbm_ab_report.md",
        "csv": "risk_model_tcn_gate.csv",
    },
    "v4_tcn_60m": {
        "target": "target_mae_risk_60m",
        "mae_col": "future_MAE_60m",
        "champion": "v1_60m",
        "champion_label": "v1_60m",
        "horizon_minutes": 60,
        "cutoff": rm.HORIZON_60M_MAIN_CUTOFF,
        # USER-decided (2026-07-12, from the survey's dispersion section): the A-grade
        # low-risk MAE margin is scale-matched to the tighter 60m label -- 60m/EOD
        # validation future_MAE std ratio 0.930%/1.578% ~= 0.59 -> 0.0005 * 0.59 ~= 0.0003.
        "mae_margin": 0.0003,
        "folds": rm.WALK_FORWARD_FOLDS_60M,
        "predictions": "risk_model_predictions_v4_tcn_60m_fold{fold}.parquet",
        "report": "tcn_vs_lgbm_60m_ab_report.md",
        "csv": "risk_model_tcn_60m_gate.csv",
    },
    # Step 18: the 1D-CNN challenger gates against the SAME champion/folds/margin as
    # the 60m TCN lineage; primary read (CNN s42 vs TCN s42) is done by passing the
    # TCN fold parquets via --ablation-predictions (they satisfy the target seal).
    "v5_cnn_60m": {
        "target": "target_mae_risk_60m",
        "mae_col": "future_MAE_60m",
        "champion": "v1_60m",
        "champion_label": "v1_60m",
        "horizon_minutes": 60,
        "cutoff": rm.HORIZON_60M_MAIN_CUTOFF,
        "mae_margin": 0.0003,
        "folds": rm.WALK_FORWARD_FOLDS_60M,
        "predictions": "risk_model_predictions_v5_cnn_60m_fold{fold}.parquet",
        "report": "cnn_vs_lgbm_60m_ab_report.md",
        "csv": "risk_model_cnn_60m_gate.csv",
    },
    # Step 19: the multi-scale CNN tie-breaker gates against the SAME champion/folds/
    # margin; primary read (v6 s42 vs TCN s42 AND vs CNN s42) is done by passing both
    # incumbents' fold parquets via --ablation-predictions (8 slots).
    "v6_mscnn_60m": {
        "target": "target_mae_risk_60m",
        "mae_col": "future_MAE_60m",
        "champion": "v1_60m",
        "champion_label": "v1_60m",
        "horizon_minutes": 60,
        "cutoff": rm.HORIZON_60M_MAIN_CUTOFF,
        "mae_margin": 0.0003,
        "folds": rm.WALK_FORWARD_FOLDS_60M,
        "predictions": "risk_model_predictions_v6_mscnn_60m_fold{fold}.parquet",
        "report": "mscnn_vs_lgbm_60m_ab_report.md",
        "csv": "risk_model_mscnn_60m_gate.csv",
    },
}


_DATE_DEFAULTS = ("2025-06-02", "2026-02-28", "2026-03-01", "2026-06-26")


def parse_args():
    p = argparse.ArgumentParser(description="TCN challenger A/B gate vs the LightGBM champion.")
    p.add_argument("--tcn-version", default="v4_tcn", choices=list(GATE_LINEAGES),
                   help="lineage to gate (see GATE_LINEAGES).")
    p.add_argument("--train-start", default=_DATE_DEFAULTS[0])
    p.add_argument("--train-end", default=_DATE_DEFAULTS[1])
    p.add_argument("--validation-start", default=_DATE_DEFAULTS[2])
    p.add_argument("--validation-end", default=_DATE_DEFAULTS[3])
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--tcn-predictions", default=None,
                   help="parquet written by train_tcn_risk_model.py (risk_prob = seed "
                        "ensemble). Default: the lineage's canonical name; for fold "
                        "lineages this is a TEMPLATE containing {fold}.")
    p.add_argument("--seed-prob-prefix", default="risk_prob_seed",
                   help="per-seed probability column prefix inside the predictions parquet.")
    p.add_argument("--ablation-predictions", action="append", default=[],
                   metavar="NAME=PATH",
                   help="optional diagnostic parquet(s), e.g. ablA_no_static=risk_model_"
                        "predictions_v4_tcn_ablA.parquet (relative to data/processed). "
                        "Reported as diagnostics only, never gated. Repeatable. For fold "
                        "lineages the parquet's sidecar must declare its fold.")
    args = p.parse_args()
    lineage = GATE_LINEAGES[args.tcn_version]
    if lineage["folds"] is not None:
        got = (args.train_start, args.train_end, args.validation_start, args.validation_end)
        if got != _DATE_DEFAULTS:
            p.error(f"{args.tcn_version!r} takes its dates from "
                    "risk_model.WALK_FORWARD_FOLDS_60M; do not pass explicit date args.")
        if args.tcn_predictions is not None and "{fold}" not in args.tcn_predictions:
            p.error("--tcn-predictions for a fold lineage must be a template "
                    "containing {fold}.")
    if args.tcn_predictions is None:
        args.tcn_predictions = str(PROCESSED_DIR / lineage["predictions"])
    return args


def load_inputs(conn, model_source, symbols):
    alerts = pd.read_sql_query("SELECT * FROM directional_alerts WHERE source = ?", conn, params=(model_source,))
    fm = pd.read_sql_query("SELECT * FROM features_merged WHERE source = ?", conn, params=(model_source,))
    rel = pd.read_sql_query("SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,))
    labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
    bars = pd.read_sql_query("SELECT symbol, bar_start, low, close FROM bars_1m WHERE source = ?", conn, params=(model_source,))
    for frame in (alerts, fm, rel, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return alerts, fm, rel, labels, bars


def _assert_full_timestamps(values, where):
    """decision_time must carry a date component ('YYYY-MM-DD HH:MM[:SS]') -- a bare
    time-of-day would cross-match identical clock times on different days."""
    s = values.astype(str)
    ok = ((s.str.len() >= 16) & (s.str.slice(4, 5) == "-")
          & (s.str.slice(10, 11).isin([" ", "T"])) & (s.str.slice(13, 14) == ":"))
    if not bool(ok.all()):
        raise SystemExit(f"{where}: decision_time is not a full timestamp, "
                         f"e.g. {s[~ok].head(3).tolist()}")


# --------------------------------------------------------------------------- #
# Metrics (math verbatim from evaluate_v3_premarket_gate.py; Step 17 threaded the
# additive target=/mae_col= keywords whose defaults reproduce the v3 behavior)
# --------------------------------------------------------------------------- #
def pooled_metrics(frame, *, target=TARGET):
    m = rm.classification_summary(frame, target_col=target)
    return {"auc": m["auc"], "pr_auc": m["pr_auc"], "logloss": m["logloss"]}


def recent_auc(frame, prob_col="risk_prob", *, target=TARGET):
    """Pooled AUC over the recent-month window (the metric the gate keys on)."""
    g = frame[frame["date"].str.slice(0, 7).isin(RECENT_MONTHS)]
    return rm.roc_auc_score_np(g[target], g[prob_col]) if len(g) else float("nan")


def decile_monotonicity(frame, *, target=TARGET):
    """Fraction of adjacent risk_prob-decile pairs whose mae_risk_rate increases."""
    p = frame["risk_prob"].to_numpy(dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    rates = [frame.loc[bins == b, target].mean() for b in range(10) if (bins == b).any()]
    if len(rates) < 2:
        return float("nan")
    steps = np.diff(rates)
    return float((steps > 0).mean())


def agrade_lowrisk(frame, *, target=TARGET, mae_col="future_MAE"):
    """Among grade-A alerts the model marks LOW risk (median split): the trading payoff."""
    a = frame[frame["alert_grade"] == "A"]
    if len(a) < 20:
        return {"n": int(len(a)), "avg_ret60": float("nan"), "avg_MAE": float("nan"),
                "win_rate": float("nan"), "mae_risk_rate": float("nan"), "n_agrade": int(len(a))}
    low = a[a["risk_prob"] <= a["risk_prob"].median()]
    return {
        "n": int(len(low)),
        "avg_ret60": float(low["future_return_60m"].mean()),
        "avg_MAE": float(low[mae_col].mean()),
        "win_rate": float((low["future_return_60m"] > 0).mean()),
        "mae_risk_rate": float(low[target].mean()),
        "n_agrade": int(len(a)),
    }


def collect_metrics(version, scored, *, target=TARGET, mae_col="future_MAE"):
    """All gate metrics for one scored frame (risk_prob column already attached)."""
    buckets = rm.bucket_report(scored, target_col=target, mae_col=mae_col)

    def bucket(name, col):
        row = buckets[buckets["risk_bucket"] == name]
        return float(row.iloc[0][col]) if len(row) else float("nan")

    monthly = rm.monthly_auc(scored, target_col=target)
    return {
        "version": version,
        "scored": scored,
        "n": int(len(scored)),
        **pooled_metrics(scored, target=target),
        "recent_auc": recent_auc(scored, target=target),
        "monthly": monthly,
        "top10_mae_risk_rate": bucket("top_10%", "mae_risk_rate"),
        "top20_mae_risk_rate": bucket("top_20%", "mae_risk_rate"),
        "bottom20_mae_risk_rate": bucket("bottom_20%", "mae_risk_rate"),
        "top10_avg_MAE": bucket("top_10%", "avg_MAE"),
        "decile_monotonicity": decile_monotonicity(scored, target=target),
        "agrade_lowrisk": agrade_lowrisk(scored, target=target, mae_col=mae_col),
    }


def evaluate_champion(valid):
    """Score v1.1 (= the frozen v1 model on the real cross-symbol frame) in-process."""
    model, pre, _meta = rm.load_model(rm.resolve_model_path(BASELINE))
    scored = valid.copy()
    scored["risk_prob"] = rm.predict_with_model(model, pre, scored)
    return collect_metrics(BASELINE, scored)


def champion_fold_model_path(version, fold):
    """train_risk_model.py --fold k writes <name>_fold<k>_lgbm.txt next to the
    registry filename."""
    p = rm.resolve_model_path(version)
    return p.with_name(p.name.replace("_lgbm.txt", f"_fold{int(fold)}_lgbm.txt"))


def load_challenger_predictions(path, valid, seed_prefix, *, target=TARGET):
    """Attach the parquet's ensemble + per-seed probabilities to the shared frame.

    Hard 1:1 coverage contract: every validation row must be present exactly once
    with a non-null score (the trainer pads short windows instead of dropping rows,
    so any mismatch means the frames diverged -- refuse to gate on it).
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"TCN predictions not found: {path} - run scripts/train_tcn_risk_model.py first.")
    preds = pd.read_parquet(path)
    preds = preds[preds["split"] == "validation"].reset_index(drop=True)
    _assert_full_timestamps(preds["decision_time"], where=str(path.name))
    _assert_full_timestamps(valid["decision_time"], where="validation frame")

    seed_cols = sorted([c for c in preds.columns if c.startswith(seed_prefix)],
                       key=lambda c: int(c[len(seed_prefix):]))
    keep = ["symbol", "decision_time", "risk_prob", target] + seed_cols
    merged = valid.drop(columns=["risk_prob"], errors="ignore").merge(
        preds[keep], on=["symbol", "decision_time"], how="inner",
        validate="one_to_one", suffixes=("", "_tcn"))
    if len(merged) != len(valid) or len(merged) != len(preds):
        raise SystemExit(
            f"row alignment failed: validation_frame={len(valid)} tcn_validation_rows={len(preds)} "
            f"inner_join={len(merged)} - the frames must be identical 1:1.")
    if not bool((merged[target] == merged[f"{target}_tcn"]).all()):
        raise SystemExit("label mismatch between the shared frame and the TCN parquet "
                         f"({target} differs) - rebuild the predictions.")
    if not bool(merged["risk_prob"].notna().all()):
        raise SystemExit("TCN risk_prob has NaNs - the trainer must score every row (padding, not dropping).")
    # Enforce (not just document) the pre-registered mean-of-seeds rule: the gated
    # risk_prob MUST equal the mean of the per-seed columns, so a best-seed pick or
    # any post-hoc blend can never reach decide_verdict.
    tm.assert_ensemble_consistency(merged, seed_cols, prob_col="risk_prob")
    merged = merged.drop(columns=[f"{target}_tcn"])
    return merged, seed_cols


def load_ablation_predictions(spec, valid):
    """NAME=PATH diagnostic parquet -> (name, metrics dict). Never gated."""
    if "=" not in spec:
        raise SystemExit(f"--ablation-predictions expects NAME=PATH, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw)
    if not path.exists():
        path = PROCESSED_DIR / raw
    if not path.exists():
        raise SystemExit(f"ablation predictions not found: {raw}")
    preds = pd.read_parquet(path)
    preds = preds[preds["split"] == "validation"].reset_index(drop=True)
    _assert_full_timestamps(preds["decision_time"], where=str(path.name))
    merged = valid.drop(columns=["risk_prob"], errors="ignore").merge(
        preds[["symbol", "decision_time", "risk_prob"]], on=["symbol", "decision_time"],
        how="inner", validate="one_to_one")
    if len(merged) != len(valid):
        raise SystemExit(f"ablation {name}: row alignment failed "
                         f"({len(merged)} joined vs {len(valid)} validation rows).")
    return name, collect_metrics(name, merged)


# --------------------------------------------------------------------------- #
# Diagnostics (informational only -- the gate math below stays verbatim)
# --------------------------------------------------------------------------- #
def seed_dispersion(base, scored, seed_cols, *, target=TARGET):
    rows = []
    for c in seed_cols:
        seed = c.split("seed")[-1]
        pooled = rm.roc_auc_score_np(scored[target], scored[c])
        rec = recent_auc(scored, prob_col=c, target=target)
        rows.append({"seed": seed, "pooled_auc": pooled, "recent_auc": rec,
                     "d_recent": rec - base["recent_auc"]})
    df = pd.DataFrame(rows)
    d = df["d_recent"].to_numpy(dtype=float)
    return {
        "per_seed": df,
        "n_seeds": len(df),
        "majority_sign_up": int((d > 0).sum()),
        "recent_auc_std": float(df["recent_auc"].std(ddof=0)) if len(df) else float("nan"),
        "median_seed_d_recent": float(np.median(d)) if len(df) else float("nan"),
    }


def calibration_diagnostics(scored, *, target=TARGET):
    """Decile reliability (mean predicted vs observed positive rate) + Brier + ties."""
    p = scored["risk_prob"].to_numpy(dtype=float)
    y = scored[target].astype(int).to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    rel = []
    for b in range(10):
        m = bins == b
        if m.any():
            rel.append({"decile": b + 1, "n": int(m.sum()),
                        "mean_pred": float(p[m].mean()), "obs_rate": float(y[m].mean())})
    return {
        "reliability": pd.DataFrame(rel),
        "brier": float(((p - y) ** 2).mean()),
        "unique_score_ratio": float(len(np.unique(p)) / max(len(p), 1)),
        "score_deciles": [float(v) for v in edges],
    }


# --------------------------------------------------------------------------- #
# Gate (math verbatim from evaluate_v3_premarket_gate.py; the additive mae_margin
# kwarg's default 0.0005 IS the v3 hardcode -- tests pin functional equivalence)
# --------------------------------------------------------------------------- #
def decide_verdict(base, variant, *, mae_margin=0.0005):
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
    mae_ok = variant["agrade_lowrisk"]["avg_MAE"] >= base["agrade_lowrisk"]["avg_MAE"] - mae_margin \
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
def fmt_val(x):
    return f"{x:.4f}" if isinstance(x, float) and x == x else "n/a"


def write_report(base, tcn, verdict, seeds_diag, calib, ablations, valid, args):
    lines = []
    lines.append("# risk_model v4_tcn (TCN seed ensemble) vs v1.1 — A/B report & GO/NO-GO\n")
    headline = ("**Verdict: GO — subject to the Phase-2 serving-parity work before any live wiring.**"
                if verdict["verdict"] == "GO"
                else f"**Verdict: {verdict['verdict']}. Keep v1.1 as the live config.**")
    lines.append(headline + "\n")
    if tcn.get("warning"):
        lines.append(f"> ⚠ **{tcn['warning']}**\n")
    lines.append(
        "v4_tcn is an ARCHITECTURE challenger: same rows (directional-alert candidates), same "
        f"`{TARGET}` label, same date split as v1.1 — but instead of the 49 flat aggregated "
        "features it consumes a causal window of the last "
        f"{tcn['seq_len']} one-minute bars' per-bar features "
        f"({tcn['n_seq_channels']} channels incl. time-of-day + pad mask) through dilated causal "
        "convolutions (Bai et al. TCN), fusing the v1 static features (5 alert + 8 cross-symbol) "
        "in the head. decision_time = the CONFIRMED feature bar's bar_end; the window's newest "
        "element is that bar and the label's forward window starts at bar_start >= decision_time, "
        "so window and label never overlap.\n")
    lines.append(
        f"**Pre-registered seed rule** (fixed in src/tcn_model.py BEFORE any validation scoring): "
        f"seeds {tcn['seeds']}, submission = arithmetic MEAN of the per-seed probabilities. "
        "No seed selection of any kind; per-seed dispersion is reported as a diagnostic below.\n")
    lines.append(
        f'Both models scored on the IDENTICAL validation frame ({len(valid):,} rows, '
        f'{args.validation_start}→{args.validation_end}, positive_rate '
        f'{valid[TARGET].mean():.3f}). "v1.1" = the v1 model file scored on real cross-symbol '
        'features — the adopted live config.\n')

    # 1. discrimination
    months = sorted(set(base["monthly"]["month"]))
    lines.append("## 1. Discrimination (validation AUC; higher = better)\n")
    lines.append("| | pooled | recent(05+06) | " + " | ".join(months) + " |")
    lines.append("|" + "---|" * (3 + len(months)))
    for name, r in (("v1.1", base), ("v4_tcn (ensemble)", tcn["metrics"])):
        mm = r["monthly"].set_index("month")["auc"]
        lines.append(f"| {name} | {r['auc']:.4f} | {r['recent_auc']:.4f} | "
                     + " | ".join(f"{mm.get(m, float('nan')):.4f}" for m in months) + " |")
    mm_b = base["monthly"].set_index("month")["auc"]
    mm_v = tcn["metrics"]["monthly"].set_index("month")["auc"]
    lines.append(f"| Δ v4_tcn vs v1.1 | {tcn['metrics']['auc'] - base['auc']:+.4f} | "
                 f"{verdict['d_recent']:+.4f} | "
                 + " | ".join(f"{mm_v.get(m, float('nan')) - mm_b.get(m, float('nan')):+.4f}"
                              for m in months) + " |")
    lines.append("")

    # 2-3 buckets + MAE
    lines.append("## 2–3. Risk-bucket separation + MAE\n")
    lines.append("| model | top10% mae_risk_rate | top10% avgMAE | bottom20% mae_risk_rate | decile monotonicity |")
    lines.append("|---|---|---|---|---|")
    for name, r in (("v1.1", base), ("v4_tcn (ensemble)", tcn["metrics"])):
        lines.append(f"| {name} | {r['top10_mae_risk_rate']:.3f} | {r['top10_avg_MAE']:.4f} | "
                     f"{r['bottom20_mae_risk_rate']:.3f} | {r['decile_monotonicity']:.2f} |")
    lines.append("")

    # 4 A-grade low-risk
    lines.append("## 4. A-grade low-risk quality (the trading payoff)\n")
    lines.append("Among grade-A alerts the model marks low risk_prob (each model's OWN median split; "
                 "`n` asymmetry vs half of n_A signals heavy score ties):\n")
    lines.append("| model | n_A | n (low half) | avg ret60 | avg MAE | win-rate | mae_risk_rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in (("v1.1", base), ("v4_tcn (ensemble)", tcn["metrics"])):
        a = r["agrade_lowrisk"]
        lines.append(f"| {name} | {a.get('n_agrade', 'n/a')} | {a['n']} | {a['avg_ret60']:+.4f} | "
                     f"{a['avg_MAE']:+.4f} | {a['win_rate']:.3f} | {a['mae_risk_rate']:.3f} |")
    lines.append("")

    # 5 TCN diagnostics
    lines.append("## 5. TCN-specific diagnostics (informational — the gate math is unchanged)\n")
    lines.append("### Seed dispersion (pre-registered ensemble; no seed was selected)\n")
    lines.append("| seed | pooled AUC | recent AUC | Δ recent vs v1.1 |")
    lines.append("|---|---|---|---|")
    for row in seeds_diag["per_seed"].to_dict("records"):
        lines.append(f"| {row['seed']} | {fmt_val(row['pooled_auc'])} | {fmt_val(row['recent_auc'])} | "
                     f"{row['d_recent']:+.4f} |")
    frag = (verdict["d_recent"] >= GATE_RECENT_AUC_MIN
            and seeds_diag["median_seed_d_recent"] < 0)
    lines.append("")
    lines.append(f"- seed majority sign: **{seeds_diag['majority_sign_up']}/{seeds_diag['n_seeds']} "
                 "seeds have Δ recent > 0**")
    lines.append(f"- recent-AUC std across seeds: {fmt_val(seeds_diag['recent_auc_std'])}"
                 + (" — exceeds half the +0.010 gate margin; treat a marginal pass as seed noise"
                    if seeds_diag["recent_auc_std"] == seeds_diag["recent_auc_std"]
                    and seeds_diag["recent_auc_std"] > 0.005 else ""))
    lines.append(f"- median-seed Δ recent: {seeds_diag['median_seed_d_recent']:+.4f}"
                 + (" — **FRAGILITY: the ensemble clears the gate but the median seed does not**"
                    if frag else ""))
    lines.append("")

    lines.append("### Calibration (reported, never gated — both models are class-weighted and "
                 "emit uncalibrated probabilities)\n")
    lines.append("| model | brier | logloss | unique-score ratio | score deciles (q0..q10) |")
    lines.append("|---|---|---|---|---|")
    for name, c, r in (("v1.1", calib["base"], base), ("v4_tcn", calib["tcn"], tcn["metrics"])):
        qs = " ".join(f"{v:.2f}" for v in c["score_deciles"])
        lines.append(f"| {name} | {c['brier']:.4f} | {r['logloss']:.4f} | "
                     f"{c['unique_score_ratio']:.3f} | {qs} |")
    lines.append("")
    lines.append("Reliability by own-score decile (mean predicted prob vs observed positive rate):\n")
    lines.append("| decile | v1.1 pred | v1.1 obs | v4_tcn pred | v4_tcn obs |")
    lines.append("|---|---|---|---|---|")
    rb = calib["base"]["reliability"].set_index("decile")
    rt = calib["tcn"]["reliability"].set_index("decile")
    for d in range(1, 11):
        b_pred = rb["mean_pred"].get(d, float("nan")); b_obs = rb["obs_rate"].get(d, float("nan"))
        t_pred = rt["mean_pred"].get(d, float("nan")); t_obs = rt["obs_rate"].get(d, float("nan"))
        lines.append(f"| {d} | {fmt_val(float(b_pred))} | {fmt_val(float(b_obs))} | "
                     f"{fmt_val(float(t_pred))} | {fmt_val(float(t_obs))} |")
    lines.append("")
    lines.append("_The serving bands RISK_HIGH=0.60 / RISK_MED=0.45 are tuned to the LightGBM "
                 "score distribution and do NOT transfer to the TCN (see the score deciles above); "
                 "on a GO, Phase 2 re-derives TCN thresholds by matching v1.1's band occupancy "
                 "on this validation frame._\n")

    if ablations:
        lines.append("### Ablations (single seed, diagnostic only — never gated)\n")
        lines.append("| ablation | pooled AUC | recent AUC | Δ recent vs v1.1 | top10% rate | bottom20% rate | monotonicity |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, r in ablations:
            lines.append(f"| {name} | {fmt_val(r['auc'])} | {fmt_val(r['recent_auc'])} | "
                         f"{r['recent_auc'] - base['recent_auc']:+.4f} | "
                         f"{fmt_val(r['top10_mae_risk_rate'])} | {fmt_val(r['bottom20_mae_risk_rate'])} | "
                         f"{fmt_val(r['decile_monotonicity'])} |")
        lines.append("")
        lines.append("_ablA (no static head) isolates how much of the TCN's discrimination comes "
                     "from the sequence itself vs riding the v1 static features; ablB (seq_len 120) "
                     "checks whether the 60-bar window is the bottleneck._\n")

    # gate
    lines.append("## Gate (same recent-weighted gate as v2.x/v3.0)\n")
    lines.append(f"GO requires: recent-month(05+06) AUC Δ ≥ +{GATE_RECENT_AUC_MIN:.3f} vs v1.1 AND no "
                 f"recent month regresses past {GATE_WORST_MONTH_FLOOR:+.3f} AND risk-bucket separation "
                 "preserved AND A-grade low-risk MAE not worse AND the gain is broad (>1 month up).\n")
    lines.append("| variant | recent Δ | worst recent-month Δ | months up | bucket ok | A-grade MAE ok | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| **v4_tcn** | {verdict['d_recent']:+.4f} | {verdict['worst_recent']:+.4f} | "
                 f"{verdict['n_months_up']} | {verdict['bucket_ok']} | {verdict['mae_ok']} | "
                 f"**{verdict['verdict']}** |")
    lines.append("")
    if verdict["reasons"]:
        lines.append(f"- **v4_tcn** ({verdict['verdict']}): " + "; ".join(verdict["reasons"]) + ".")
    lines.append("")

    # serving status
    lines.append("## Live serving status\n")
    lines.append("- **v1.1 remains the adopted live config.**")
    lines.append("- The TCN challenger is research-only until it passes the gate; the "
                 "configuration evaluated here is documented above and NOT wired live.")
    if verdict["verdict"] == "GO":
        lines.append("- GO does not mean live: Phase 2 (sequence-serving builder parallel to "
                     "score_serving, a score-parity replay check, and band re-thresholding by "
                     "v1.1-occupancy matching) must pass before any wiring.")
    lines.append("")

    report = "\n".join(lines)
    out_path = PROCESSED_DIR / "tcn_vs_lgbm_ab_report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def tcn_data_config_from_meta(predictions_path, seeds):
    """Read seq_len / channel count / submission flag / frame fingerprint (json only,
    no torch). Provenance is read from the sidecar of the GATED PARQUET
    itself (<predictions_path>.meta.json), never from the default model artifacts, so
    a tuned/relocated parquet cannot borrow the submission's provenance."""
    import json

    fallback = {"seq_len": tm.SEQ_LEN, "n_channels": tm.TCNConfig().in_channels,
                "submission_run": None, "seeds": list(seeds),
                "validation_frame_fingerprint": None,
                "model_version": None, "target": None, "fold": None}
    meta_path = Path(str(predictions_path) + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "seq_len": int(meta.get("seq_len", fallback["seq_len"])),
            "n_channels": int(meta.get("n_seq_channels", fallback["n_channels"])),
            "submission_run": meta.get("submission_run"),
            "seeds": meta.get("seeds", list(seeds)),
            "validation_frame_fingerprint": meta.get("validation_frame_fingerprint"),
            "model_version": meta.get("model_version"),
            "target": meta.get("target"),
            "fold": meta.get("fold"),
        }
    return fallback


def submission_warning(seeds, provenance, *, expected_seeds=None):
    """Non-empty when the gated parquet is NOT the pre-registered submission.

    Fails CLOSED: a missing sidecar (submission_run is None) or any non-True value
    warns, and the seed columns present in the parquet must match the version's
    pre-registered seeds (default: the v4_tcn 5-seed protocol)."""
    expected = tuple(tm.SUBMISSION_SEEDS if expected_seeds is None else expected_seeds)
    submission_run = provenance.get("submission_run")
    meta_seeds = provenance.get("seeds")
    ok = (submission_run is True
          and tuple(seeds) == expected
          and (meta_seeds is None or tuple(meta_seeds) == expected))
    if ok:
        return ""
    return (f"DIAGNOSTIC predictions: parquet seed columns={tuple(seeds)}, sidecar "
            f"submission_run={submission_run}, sidecar seeds={meta_seeds} "
            f"(pre-registered submission = {expected}, submission_run=True). The "
            "verdict below is a pipeline check, NOT an adoption decision — rerun the full "
            "pre-registered training with the default config before treating any GO as real.")


def run_single_split_gate(args):
    """The Step 16 v4_tcn gate flow (behavior unchanged)."""
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    split = rm.SplitConfig(train_start=args.train_start, train_end=args.train_end,
                           validation_start=args.validation_start, validation_end=args.validation_end)

    print(f"[{now_kst_text()}] v4_tcn gate: baseline=v1.1 challenger={CHALLENGER} source={model_source}")
    conn = sqlite3.connect(args.db)
    try:
        alerts, fm, rel, labels, bars = load_inputs(conn, model_source, symbols)
    finally:
        conn.close()

    # the shared frame is the v1 configuration (real cross-symbol relatives)
    data, _cols = rm.build_model_frame(alerts, fm, rel, labels, bars,
                                       relative_columns=rm.RELATIVE_FEATURE_COLUMNS,
                                       feature_columns=rm.FEATURE_COLUMNS)
    _train, valid = rm.split_by_date(data, split=split)
    valid = valid.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  validation frame: rows={len(valid)} positive_rate={valid[TARGET].mean():.3f}")

    base = evaluate_champion(valid)
    print(f"  v1.1     pooled_auc={base['auc']:.4f} recent_auc={base['recent_auc']:.4f} "
          f"top10={base['top10_mae_risk_rate']:.3f} bottom20={base['bottom20_mae_risk_rate']:.3f}")

    scored, seed_cols = load_challenger_predictions(args.tcn_predictions, valid, args.seed_prob_prefix)
    tcn_metrics = collect_metrics(CHALLENGER, scored)
    print(f"  v4_tcn   pooled_auc={tcn_metrics['auc']:.4f} recent_auc={tcn_metrics['recent_auc']:.4f} "
          f"top10={tcn_metrics['top10_mae_risk_rate']:.3f} bottom20={tcn_metrics['bottom20_mae_risk_rate']:.3f} "
          f"seeds={len(seed_cols)}")

    verdict = decide_verdict(base, tcn_metrics)
    seeds_diag = seed_dispersion(base, scored, seed_cols)
    calib = {"base": calibration_diagnostics(base["scored"]),
             "tcn": calibration_diagnostics(scored)}
    ablations = [load_ablation_predictions(spec, valid) for spec in args.ablation_predictions]

    # provenance + warning FIRST so every artifact (csv/report/console) carries it
    seeds = tuple(int(c.split("seed")[-1]) for c in seed_cols)
    provenance = tcn_data_config_from_meta(args.tcn_predictions, seeds)
    # hard cross-lineage seal: a 60m parquet carries BOTH label families, so gating it
    # here would silently compare on the wrong label instead of failing the merge.
    if provenance.get("target") not in (None, TARGET):
        raise SystemExit(f"{Path(args.tcn_predictions).name}: sidecar target="
                         f"{provenance['target']!r} != this gate's {TARGET!r} -- "
                         "cross-lineage parquet refused (use --tcn-version).")
    warning = submission_warning(seeds, provenance)
    # feature-snapshot check: if features_merged/relative were rebuilt between training
    # and gating, the champion is rescored on a different snapshot than the challenger's
    # probabilities embed (keys+labels can't detect it -- values can).
    gate_fp = tm.frame_fingerprint(valid, list(rm.RELATIVE_FEATURE_COLUMNS)
                                   + list(rm.ALERT_FEATURE_COLUMNS))
    train_fp = provenance.get("validation_frame_fingerprint")
    if train_fp and train_fp != gate_fp:
        warning = (warning + " | " if warning else "") + (
            "FEATURE SNAPSHOT DRIFT: the validation feature fingerprint differs from the "
            "trainer's — features_merged/relative were rebuilt after training, so the "
            "champion and challenger are scored on different feature snapshots. Retrain.")
    if warning:
        print(f"\n  *** {warning}")

    # csv summary (v1.1 / v4_tcn / per-seed + ablation diagnostic rows)
    rows = []
    diag_flag = "diagnostic" if warning else "submission"

    def csv_row(version, r, verdict_label):
        return {
            "version": version,
            "pooled_auc": r["auc"], "recent_auc": r["recent_auc"],
            "pr_auc": r.get("pr_auc"), "logloss": r.get("logloss"),
            "top10_mae_risk_rate": r["top10_mae_risk_rate"],
            "top20_mae_risk_rate": r.get("top20_mae_risk_rate"),
            "bottom20_mae_risk_rate": r["bottom20_mae_risk_rate"],
            "decile_monotonicity": r["decile_monotonicity"],
            "agrade_lowrisk_avg_MAE": r["agrade_lowrisk"]["avg_MAE"],
            "agrade_lowrisk_avg_ret60": r["agrade_lowrisk"]["avg_ret60"],
            "verdict": verdict_label,
            "run_kind": diag_flag,
        }

    rows.append(csv_row("v1.1", base, "baseline"))
    tcn_metrics["seed_cols"] = seed_cols
    rows.append(csv_row("v4_tcn", tcn_metrics, verdict["verdict"]))
    for srow in seeds_diag["per_seed"].to_dict("records"):
        rows.append({"version": f"v4_tcn_seed{srow['seed']}", "pooled_auc": srow["pooled_auc"],
                     "recent_auc": srow["recent_auc"], "verdict": "seed-diagnostic",
                     "run_kind": diag_flag})
    for name, r in ablations:
        rows.append(csv_row(f"v4_tcn_{name}", r, "diagnostic"))
    csv_path = PROCESSED_DIR / "risk_model_tcn_gate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    tcn_payload = {
        "metrics": tcn_metrics,
        "seeds": seeds,
        "seq_len": provenance["seq_len"],
        "n_seq_channels": provenance["n_channels"],
        "warning": warning,
    }
    report_path = write_report(base, tcn_payload, verdict, seeds_diag, calib, ablations, valid, args)

    print("\nverdict:")
    print(f"  v4_tcn {verdict['verdict']:<8} recentΔ={verdict['d_recent']:+.4f} "
          f"worstΔ={verdict['worst_recent']:+.4f} "
          f"({'; '.join(verdict['reasons']) or 'all gates pass'})")
    print(f"  seed majority sign: {seeds_diag['majority_sign_up']}/{seeds_diag['n_seeds']} up; "
          f"recent-AUC seed std={seeds_diag['recent_auc_std']:.4f}")
    print(f"\nreport -> {report_path}")
    print(f"summary -> {csv_path}")
    return 0


# --------------------------------------------------------------------------- #
# Step 17: walk-forward (fold) gate for the 60m lineage
# --------------------------------------------------------------------------- #
def _join_warn(warning, extra):
    return (warning + " | " if warning else "") + extra


def load_fold_challenger(args, lineage, fold, fold_valid):
    """One fold's challenger parquet -> (merged frame, seed_cols, warning).

    Hard cross-lineage seals (SystemExit): the 60m parquet carries BOTH label
    families, so a target/fold mix-up would silently mis-gate instead of failing the
    merge -- refuse it outright. Soft (fail-closed warning): submission provenance,
    model_version, feature-snapshot fingerprint."""
    path = Path(str(args.tcn_predictions).format(fold=fold))
    if not path.is_absolute() and not path.exists():
        path = PROCESSED_DIR / path   # same relative-name convenience as ablations
    merged, seed_cols = load_challenger_predictions(path, fold_valid, args.seed_prob_prefix,
                                                    target=lineage["target"])
    seeds = tuple(int(c.split("seed")[-1]) for c in seed_cols)
    provenance = tcn_data_config_from_meta(path, seeds)
    if provenance.get("target") is not None and provenance["target"] != lineage["target"]:
        raise SystemExit(f"{path.name}: sidecar target={provenance['target']!r} != lineage "
                         f"target {lineage['target']!r} -- cross-lineage parquet refused.")
    if provenance.get("fold") is not None and int(provenance["fold"]) != int(fold):
        raise SystemExit(f"{path.name}: sidecar fold={provenance['fold']} != gate fold {fold} "
                         "-- wrong fold's parquet wired in.")
    expected_seeds = tm.tcn_version_config(args.tcn_version)["seeds"]
    warning = submission_warning(seeds, provenance, expected_seeds=expected_seeds)
    if provenance.get("model_version") not in (None, args.tcn_version):
        warning = _join_warn(warning, f"sidecar model_version={provenance['model_version']!r} "
                                      f"!= {args.tcn_version!r}")
    gate_fp = tm.frame_fingerprint(fold_valid, list(rm.RELATIVE_FEATURE_COLUMNS)
                                   + list(rm.ALERT_FEATURE_COLUMNS))
    train_fp = provenance.get("validation_frame_fingerprint")
    if train_fp and train_fp != gate_fp:
        warning = _join_warn(warning, (
            "FEATURE SNAPSHOT DRIFT: the validation feature fingerprint differs from the "
            "trainer's — features_merged/relative were rebuilt after training, so the "
            "champion and challenger are scored on different feature snapshots. Retrain."))
    return merged, seed_cols, warning


def score_fold_champion(lineage, fold, merged):
    """Attach the fold champion's probabilities as champ_prob (in-process rescoring),
    with a champion-side provenance seal."""
    champ_path = champion_fold_model_path(lineage["champion"], fold)
    if not champ_path.exists():
        raise SystemExit(f"champion model not found: {champ_path} - run "
                         f"scripts\\train_risk_model.py --risk-model-version "
                         f"{lineage['champion']} --target {lineage['target']} "
                         f"--fold {fold} first.")
    model, pre, meta = rm.load_model(champ_path)
    for key, expected in (("model_version", lineage["champion"]),
                          ("target", lineage["target"]), ("fold", fold)):
        got = (meta or {}).get(key)
        if got is not None and got != expected:
            raise SystemExit(f"{champ_path.name}: metadata {key}={got!r} != {expected!r} "
                             "-- wrong champion artifact.")
    merged = merged.copy()
    merged["champ_prob"] = rm.predict_with_model(model, pre, merged)
    return merged


def load_ablation_60m(spec, fold_valids, lineage):
    """NAME=PATH diagnostic parquet for the fold lineage. The sidecar must declare its
    fold; metrics are computed on that fold's MAIN population only. Never gated."""
    import json

    if "=" not in spec:
        raise SystemExit(f"--ablation-predictions expects NAME=PATH, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw)
    if not path.exists():
        path = PROCESSED_DIR / raw
    if not path.exists():
        raise SystemExit(f"ablation predictions not found: {raw}")
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"ablation {name}: sidecar {meta_path.name} missing - fold "
                         "lineages need it to know which fold the parquet belongs to.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fold = meta.get("fold")
    if fold is None or int(fold) not in fold_valids:
        raise SystemExit(f"ablation {name}: sidecar fold={fold!r} is not a known fold.")
    if meta.get("target") not in (None, lineage["target"]):
        raise SystemExit(f"ablation {name}: sidecar target={meta.get('target')!r} != "
                         f"{lineage['target']!r} -- cross-lineage parquet refused.")
    fold = int(fold)
    fold_valid = fold_valids[fold]
    preds = pd.read_parquet(path)
    preds = preds[preds["split"] == "validation"].reset_index(drop=True)
    _assert_full_timestamps(preds["decision_time"], where=str(path.name))
    merged = fold_valid.drop(columns=["risk_prob"], errors="ignore").merge(
        preds[["symbol", "decision_time", "risk_prob"]], on=["symbol", "decision_time"],
        how="inner", validate="one_to_one")
    if len(merged) != len(fold_valid):
        raise SystemExit(f"ablation {name}: row alignment failed "
                         f"({len(merged)} joined vs {len(fold_valid)} fold-validation rows).")
    main = merged[merged["time"] <= lineage["cutoff"]]
    y = main[lineage["target"]].astype(int).to_numpy()
    return {"name": name, "fold": fold, "n": int(len(main)),
            "auc": rm.roc_auc_score_np(y, main["risk_prob"].to_numpy(dtype=float))}


def write_report_60m(base, tcn, verdict, fold_rows, late_diag, calib, ablations,
                     lineage, args):
    target, mae_col = lineage["target"], lineage["mae_col"]
    lines = []
    lines.append(f"# risk_model {args.tcn_version} (TCN, walk-forward) vs "
                 f"{lineage['champion_label']} — A/B report & GO/NO-GO\n")
    headline = ("**Verdict: GO — subject to the Phase-2 serving-parity work before any live wiring.**"
                if verdict["verdict"] == "GO"
                else f"**Verdict: {verdict['verdict']}. Keep v1.1 as the live config.**")
    lines.append(headline + "\n")
    if tcn.get("warning"):
        lines.append(f"> ⚠ **{tcn['warning']}**\n")
    lines.append(
        f"Step 17 lineage: fixed-horizon label `{target}` = next-60-minute window "
        f"min/max (main population = decision time-of-day <= {lineage['cutoff']}, i.e. "
        "always exactly 60 minutes observed; later rows are the late-session diagnostic "
        "below, never gated). Champion = "
        f"`{lineage['champion']}` (v1 features + minutes_since_open); challenger = the "
        "TCN with its time-of-day channel KEPT — both models hold the clock, so the "
        "gate measures architecture value on top of clock access.\n")
    lines.append(
        "**Walk-forward protocol (user-decided):** expanding train windows, one "
        "validation month per fold, ONLY the periods differ between folds; the TCN's "
        f"pre-registered rule is seed(s) {tcn['seeds']} per fold (no seed selection). "
        "Pooled/recent metrics below STITCH the four out-of-sample months — every row "
        "is scored by a model that never saw its month.\n")

    lines.append("## 0. Per-fold out-of-sample AUC (the regime-robustness read)\n")
    lines.append("| fold | train window | val month | n (main) | pos_rate | "
                 f"{lineage['champion_label']} AUC | {args.tcn_version} AUC | Δ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in fold_rows:
        lines.append(f"| {r['fold']} | {r['train_start']}→{r['train_end']} | {r['month']} | "
                     f"{r['n']:,} | {r['pos_rate']:.3f} | {r['champ_auc']:.4f} | "
                     f"{r['chall_auc']:.4f} | {r['chall_auc'] - r['champ_auc']:+.4f} |")
    lines.append("")

    months = sorted(set(base["monthly"]["month"]))
    lines.append("## 1. Discrimination (stitched walk-forward validation AUC)\n")
    lines.append("| | pooled | recent(05+06) | " + " | ".join(months) + " |")
    lines.append("|" + "---|" * (3 + len(months)))
    for name, r in ((lineage["champion_label"], base), (f"{args.tcn_version}", tcn["metrics"])):
        mm = r["monthly"].set_index("month")["auc"]
        lines.append(f"| {name} | {r['auc']:.4f} | {r['recent_auc']:.4f} | "
                     + " | ".join(f"{mm.get(m, float('nan')):.4f}" for m in months) + " |")
    mm_b = base["monthly"].set_index("month")["auc"]
    mm_v = tcn["metrics"]["monthly"].set_index("month")["auc"]
    lines.append(f"| Δ | {tcn['metrics']['auc'] - base['auc']:+.4f} | "
                 f"{verdict['d_recent']:+.4f} | "
                 + " | ".join(f"{mm_v.get(m, float('nan')) - mm_b.get(m, float('nan')):+.4f}"
                              for m in months) + " |")
    lines.append("")

    lines.append("## 2–3. Risk-bucket separation + MAE (stitched)\n")
    lines.append(f"| model | top10% rate | top10% avg {mae_col} | bottom20% rate | decile monotonicity |")
    lines.append("|---|---|---|---|---|")
    for name, r in ((lineage["champion_label"], base), (args.tcn_version, tcn["metrics"])):
        lines.append(f"| {name} | {r['top10_mae_risk_rate']:.3f} | {r['top10_avg_MAE']:.4f} | "
                     f"{r['bottom20_mae_risk_rate']:.3f} | {r['decile_monotonicity']:.2f} |")
    lines.append("")

    lines.append("## 4. A-grade low-risk quality (the trading payoff)\n")
    lines.append(f"MAE margin for the gate's mae_ok check: **{lineage['mae_margin']}** "
                 "(user-decided; scale-matched to the tighter 60m label — 60m/EOD "
                 "future_MAE std ratio ≈ 0.59 × the legacy 0.0005).\n")
    lines.append(f"| model | n_A | n (low half) | avg ret60 | avg {mae_col} | win-rate | risk-rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in ((lineage["champion_label"], base), (args.tcn_version, tcn["metrics"])):
        a = r["agrade_lowrisk"]
        lines.append(f"| {name} | {a.get('n_agrade', 'n/a')} | {a['n']} | {a['avg_ret60']:+.4f} | "
                     f"{a['avg_MAE']:+.4f} | {a['win_rate']:.3f} | {a['mae_risk_rate']:.3f} |")
    lines.append("")

    lines.append("## 5. Late-session diagnostic (time > cutoff; truncated-horizon label — never gated)\n")
    if late_diag:
        lines.append("| scope | n | pos_rate | " + lineage["champion_label"] + " AUC | "
                     + args.tcn_version + " AUC |")
        lines.append("|---|---|---|---|---|")
        for r in late_diag:
            lines.append(f"| {r['scope']} | {r['n']:,} | {r['pos_rate']:.3f} | "
                         f"{fmt_val(r['champ_auc'])} | {fmt_val(r['chall_auc'])} |")
    else:
        lines.append("(no late-session rows)")
    lines.append("")

    lines.append("## 6. Calibration (reported, never gated)\n")
    lines.append("| model | brier | logloss | unique-score ratio |")
    lines.append("|---|---|---|---|")
    for name, c, r in ((lineage["champion_label"], calib["base"], base),
                       (args.tcn_version, calib["tcn"], tcn["metrics"])):
        lines.append(f"| {name} | {c['brier']:.4f} | {r['logloss']:.4f} | "
                     f"{c['unique_score_ratio']:.3f} |")
    lines.append("")

    if ablations:
        lines.append("## 7. Ablations (single fold each, diagnostic only — never gated)\n")
        lines.append("| ablation | fold | n (main) | AUC | champion same-fold AUC | Δ |")
        lines.append("|---|---|---|---|---|---|")
        champ_by_fold = {r["fold"]: r["champ_auc"] for r in fold_rows}
        for r in ablations:
            champ = champ_by_fold.get(r["fold"], float("nan"))
            lines.append(f"| {r['name']} | {r['fold']} | {r['n']:,} | {fmt_val(r['auc'])} | "
                         f"{fmt_val(champ)} | {r['auc'] - champ:+.4f} |")
        lines.append("")

    lines.append("## Gate (same recent-weighted math as v2.x/v3.0; walk-forward stitched)\n")
    lines.append(f"GO requires: recent-month(05+06) AUC Δ ≥ +{GATE_RECENT_AUC_MIN:.3f} vs "
                 f"{lineage['champion_label']} AND no recent month regresses past "
                 f"{GATE_WORST_MONTH_FLOOR:+.3f} AND risk-bucket separation preserved AND "
                 f"A-grade low-risk MAE not worse than -{lineage['mae_margin']} AND the gain "
                 "is broad (>1 month up).\n")
    lines.append("| variant | recent Δ | worst recent-month Δ | months up | bucket ok | A-grade MAE ok | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| **{args.tcn_version}** | {verdict['d_recent']:+.4f} | "
                 f"{verdict['worst_recent']:+.4f} | {verdict['n_months_up']} | "
                 f"{verdict['bucket_ok']} | {verdict['mae_ok']} | **{verdict['verdict']}** |")
    lines.append("")
    if verdict["reasons"]:
        lines.append(f"- **{args.tcn_version}** ({verdict['verdict']}): "
                     + "; ".join(verdict["reasons"]) + ".")
    lines.append("")
    lines.append("## Live serving status\n")
    lines.append("- **v1.1 remains the adopted live config** (the 60m lineage is research-only; "
                 "even its champion v1_60m is not wired live).")
    if verdict["verdict"] == "GO":
        lines.append("- GO does not mean live: Phase 2 (sequence serving, parity replay, band "
                     "re-thresholding) must pass before any wiring.")
    lines.append("")

    out_path = PROCESSED_DIR / lineage["report"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_walk_forward_gate(args, lineage):
    """Step 17: per-fold champion rescore + challenger parquet merge -> stitched
    walk-forward frame -> unchanged gate math (lineage mae_margin)."""
    target, mae_col, cutoff = lineage["target"], lineage["mae_col"], lineage["cutoff"]
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    print(f"[{now_kst_text()}] {args.tcn_version} walk-forward gate: "
          f"champion={lineage['champion']} source={model_source} "
          f"folds={len(lineage['folds'])} mae_margin={lineage['mae_margin']}")

    conn = sqlite3.connect(args.db)
    try:
        alerts, fm, rel, labels, bars = load_inputs(conn, model_source, symbols)
    finally:
        conn.close()
    data, _cols = rm.build_model_frame(alerts, fm, rel, labels, bars,
                                       relative_columns=rm.RELATIVE_FEATURE_COLUMNS,
                                       feature_columns=rm.FEATURE_COLUMNS_60M,
                                       horizon_minutes=lineage["horizon_minutes"])

    fold_valids, fold_rows, mains, lates = {}, [], [], []
    warning = ""
    seed_cols_ref = None
    for k, fsplit in enumerate(lineage["folds"], start=1):
        _train, fold_valid = rm.split_by_date(data, split=fsplit)
        fold_valid = fold_valid.dropna(subset=[target]).reset_index(drop=True)
        if fold_valid.empty:
            raise SystemExit(f"fold {k}: empty validation frame "
                             f"({fsplit.validation_start}..{fsplit.validation_end})")
        fold_valids[k] = fold_valid
        merged, seed_cols, fold_warn = load_fold_challenger(args, lineage, k, fold_valid)
        if seed_cols_ref is None:
            seed_cols_ref = seed_cols
        elif seed_cols != seed_cols_ref:
            raise SystemExit(f"fold {k}: seed columns {seed_cols} differ from fold 1's "
                             f"{seed_cols_ref}; the folds must share one protocol.")
        if fold_warn:
            warning = _join_warn(warning, f"fold{k}: {fold_warn}")
        merged = score_fold_champion(lineage, k, merged)
        merged["fold"] = k
        is_main = merged["time"] <= cutoff
        main, late = merged[is_main], merged[~is_main]
        y = main[target].astype(int).to_numpy()
        champ_auc = rm.roc_auc_score_np(y, main["champ_prob"].to_numpy(dtype=float))
        chall_auc = rm.roc_auc_score_np(y, main["risk_prob"].to_numpy(dtype=float))
        fold_rows.append({"fold": k, "train_start": fsplit.train_start,
                          "train_end": fsplit.train_end,
                          "month": fsplit.validation_start[:7], "n": int(len(main)),
                          "pos_rate": float(main[target].mean()),
                          "champ_auc": champ_auc, "chall_auc": chall_auc})
        mains.append(main)
        lates.append(late)
        print(f"  fold {k} [{fsplit.validation_start[:7]}] n_main={len(main):,} "
              f"pos={main[target].mean():.3f} champ={champ_auc:.4f} "
              f"chall={chall_auc:.4f} Δ={chall_auc - champ_auc:+.4f} late={len(late):,}")

    stitched = pd.concat(mains, ignore_index=True)
    base_scored = stitched.copy()
    base_scored["risk_prob"] = base_scored["champ_prob"]
    base = collect_metrics(lineage["champion_label"], base_scored, target=target, mae_col=mae_col)
    tcn_metrics = collect_metrics(args.tcn_version, stitched, target=target, mae_col=mae_col)
    verdict = decide_verdict(base, tcn_metrics, mae_margin=lineage["mae_margin"])
    calib = {"base": calibration_diagnostics(base_scored, target=target),
             "tcn": calibration_diagnostics(stitched, target=target)}

    late_all = pd.concat(lates, ignore_index=True) if lates else pd.DataFrame()
    late_diag = []
    if len(late_all):
        def _late_row(scope, g):
            y = g[target].astype(int).to_numpy()
            return {"scope": scope, "n": int(len(g)), "pos_rate": float(g[target].mean()),
                    "champ_auc": rm.roc_auc_score_np(y, g["champ_prob"].to_numpy(dtype=float)),
                    "chall_auc": rm.roc_auc_score_np(y, g["risk_prob"].to_numpy(dtype=float))}
        late_diag.append(_late_row("pooled", late_all))
        for k, g in late_all.groupby("fold"):
            late_diag.append(_late_row(f"fold {k}", g))

    ablations = [load_ablation_60m(spec, fold_valids, lineage)
                 for spec in args.ablation_predictions]
    if warning:
        print(f"\n  *** {warning}")

    diag_flag = "diagnostic" if warning else "submission"
    rows = [{"version": lineage["champion_label"], "pooled_auc": base["auc"],
             "recent_auc": base["recent_auc"], "pr_auc": base.get("pr_auc"),
             "logloss": base.get("logloss"),
             "top10_mae_risk_rate": base["top10_mae_risk_rate"],
             "bottom20_mae_risk_rate": base["bottom20_mae_risk_rate"],
             "decile_monotonicity": base["decile_monotonicity"],
             "agrade_lowrisk_avg_MAE": base["agrade_lowrisk"]["avg_MAE"],
             "verdict": "baseline", "run_kind": diag_flag},
            {"version": args.tcn_version, "pooled_auc": tcn_metrics["auc"],
             "recent_auc": tcn_metrics["recent_auc"], "pr_auc": tcn_metrics.get("pr_auc"),
             "logloss": tcn_metrics.get("logloss"),
             "top10_mae_risk_rate": tcn_metrics["top10_mae_risk_rate"],
             "bottom20_mae_risk_rate": tcn_metrics["bottom20_mae_risk_rate"],
             "decile_monotonicity": tcn_metrics["decile_monotonicity"],
             "agrade_lowrisk_avg_MAE": tcn_metrics["agrade_lowrisk"]["avg_MAE"],
             "verdict": verdict["verdict"], "run_kind": diag_flag}]
    for r in fold_rows:
        rows.append({"version": f"{args.tcn_version}_fold{r['fold']}",
                     "pooled_auc": r["chall_auc"], "recent_auc": None,
                     "champion_auc": r["champ_auc"], "month": r["month"],
                     "verdict": "fold-diagnostic", "run_kind": diag_flag})
    for r in late_diag:
        rows.append({"version": f"late_{r['scope'].replace(' ', '')}",
                     "pooled_auc": r["chall_auc"], "champion_auc": r["champ_auc"],
                     "verdict": "late-diagnostic", "run_kind": diag_flag})
    for r in ablations:
        rows.append({"version": f"{args.tcn_version}_{r['name']}", "pooled_auc": r["auc"],
                     "month": f"fold{r['fold']}", "verdict": "diagnostic",
                     "run_kind": diag_flag})
    csv_path = PROCESSED_DIR / lineage["csv"]
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    tcn_payload = {"metrics": tcn_metrics,
                   "seeds": tuple(int(c.split("seed")[-1]) for c in seed_cols_ref),
                   "warning": warning}
    report_path = write_report_60m(base, tcn_payload, verdict, fold_rows, late_diag,
                                   calib, ablations, lineage, args)

    print("\nverdict:")
    print(f"  {args.tcn_version} {verdict['verdict']:<8} recentΔ={verdict['d_recent']:+.4f} "
          f"worstΔ={verdict['worst_recent']:+.4f} "
          f"({'; '.join(verdict['reasons']) or 'all gates pass'})")
    print(f"\nreport -> {report_path}")
    print(f"summary -> {csv_path}")
    return 0


def main():
    args = parse_args()
    lineage = GATE_LINEAGES[args.tcn_version]
    if lineage["folds"] is not None:
        return run_walk_forward_gate(args, lineage)
    return run_single_split_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
