"""Step 20: pre-registered PRODUCTION selection between the TCN and CNN candidates.

Research / evaluation only -- no orders, no Kiwoom, no live wiring. This script only
READS the three production prediction parquets (champion v1_60m_prod + challengers
v4_tcn_60m_prod / v5_cnn_60m_prod), scores the July sanity window and prints/writes
the selection verdict. Serving remains Phase-2 work regardless of the verdict.

Selection rule -- PRE-REGISTERED 2026-07-16, before any July row was scored
(the window is ~8 trading days, small enough to eyeball-hack; hence a fixed rule):
  1. Window: rm.PRODUCTION_SPLIT_60M validation days, MAIN population only
     (time <= 14:30 -- every row has the full 60m forward window).
  2. Score: the ensemble risk_prob, ASSERTED to be the mean of the 5 pre-registered
     seeds (tm.assert_ensemble_consistency; a best-seed pick can never sneak in).
  3. Sanity floor (each candidate): ensemble AUC >= the v1_60m_prod champion's AUC
     on the identical window. Both below -> NO-SHIP (July regime investigation).
  4. Pick: the higher-AUC candidate if |dAUC| >= 0.005. Below that margin the July
     window cannot separate them (noise), so the tie goes to the CNN -- fixed on
     the FROZEN walk-forward evidence: CNN 5-seed ensemble recent 0.6971 > TCN
     0.6962, CNN best on the June drawdown fold (0.6846 vs 0.6789 s42), CNN best
     late-session diagnostics. This paragraph is the tie-break's entire authority;
     July numbers never re-litigate it.

Run (after both server trainings are copied back):
  ..\\Python311-64\\python.exe scripts\\select_prod_60m.py
"""

from __future__ import annotations

import json
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

from config import PROCESSED_DIR
import risk_model as rm
import tcn_model as tm
from utils import now_kst_text

CHAMPION = "v1_60m_prod"
CANDIDATES = ("v4_tcn_60m_prod", "v5_cnn_60m_prod")
SELECTION_MARGIN = 0.005          # |dAUC| below this = July cannot separate them
TIE_BREAK = "v5_cnn_60m_prod"     # frozen walk-forward evidence (docstring above)
TARGET = "target_mae_risk_60m"
MAE_COL = "future_MAE_60m"

REPORT_PATH = PROCESSED_DIR / "prod_60m_selection_report.md"
CSV_PATH = PROCESSED_DIR / "risk_model_prod_60m_selection.csv"


def _load_predictions(version):
    path = PROCESSED_DIR / f"risk_model_predictions_{version}.parquet"
    if not path.exists():
        raise SystemExit(f"missing predictions parquet: {path.name} - train {version} first.")
    df = pd.read_parquet(path)
    if TARGET not in df.columns:
        raise SystemExit(f"{path.name}: no {TARGET} column - wrong lineage parquet.")
    return df, path


def _sidecar(path):
    meta_path = Path(str(path) + ".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def _seal_challenger(version, meta):
    """Hard provenance seals for a candidate parquet (SystemExit on mismatch)."""
    split = rm.PRODUCTION_SPLIT_60M
    checks = {
        "model_version": (meta.get("model_version"), version),
        "target": (meta.get("target"), TARGET),
        "validation_start": (meta.get("validation_start"), split.validation_start),
        "validation_end": (meta.get("validation_end"), split.validation_end),
        "train_end": (meta.get("train_end"), split.train_end),
    }
    for key, (got, want) in checks.items():
        if got is not None and got != want:
            raise SystemExit(f"{version}: sidecar {key}={got!r} != {want!r} - "
                             "wrong artifact wired in.")
    warn = ""
    if meta.get("submission_run") is False:
        warn = (f"{version}: sidecar submission_run=False (diagnostic training run) - "
                "verdict below is a pipeline check, NOT a shippable selection.")
    seeds = tuple(meta.get("seeds") or ())
    if seeds and seeds != tuple(tm.tcn_version_config(version)["seeds"]):
        raise SystemExit(f"{version}: sidecar seeds={seeds} != pre-registered "
                         f"{tuple(tm.tcn_version_config(version)['seeds'])}.")
    return warn


def _main_window(df):
    split = rm.PRODUCTION_SPLIT_60M
    d = df[df["split"] == "validation"].copy()
    d["date"] = d["decision_time"].astype(str).str.slice(0, 10)
    d = d[(d["date"] >= split.validation_start) & (d["date"] <= split.validation_end)]
    late = d[d["time"].astype(str) > rm.HORIZON_60M_MAIN_CUTOFF]
    main = d[d["time"].astype(str) <= rm.HORIZON_60M_MAIN_CUTOFF]
    main = main[main[TARGET].notna()]
    return main.reset_index(drop=True), late[late[TARGET].notna()].reset_index(drop=True)


def _auc(frame, prob_col="risk_prob"):
    return rm.roc_auc_score_np(frame[TARGET].astype(int).to_numpy(),
                               frame[prob_col].to_numpy(dtype=float))


def _metrics(main, late, prob_col="risk_prob"):
    m = rm.classification_summary(main, target_col=TARGET, prob_col=prob_col)
    buckets = rm.bucket_report(main, target_col=TARGET, prob_col=prob_col,
                               mae_col=MAE_COL).set_index("risk_bucket")
    per_day = pd.Series({
        day: (_auc(g, prob_col) if g[TARGET].nunique() > 1 else float("nan"))
        for day, g in main.groupby("date")})
    y = main[TARGET].astype(float).to_numpy()
    return {"auc": float(m["auc"]), "pr_auc": float(m["pr_auc"]),
            "brier": float(((main[prob_col].to_numpy(dtype=float) - y) ** 2).mean()),
            "top10_rate": float(buckets.loc["top_10%", "mae_risk_rate"]),
            "bottom20_rate": float(buckets.loc["bottom_20%", "mae_risk_rate"]),
            "late_auc": (_auc(late, prob_col) if len(late) and late[TARGET].nunique() > 1
                         else float("nan")),
            "per_day": per_day}


def main() -> int:
    split = rm.PRODUCTION_SPLIT_60M
    print(f"[{now_kst_text()}] Step 20 production selection - window "
          f"{split.validation_start}..{split.validation_end} main cutoff "
          f"{rm.HORIZON_60M_MAIN_CUTOFF} margin {SELECTION_MARGIN} tie->{TIE_BREAK}")

    frames, metas, warns = {}, {}, []
    for version in (CHAMPION,) + CANDIDATES:
        df, path = _load_predictions(version)
        frames[version] = df
        if version != CHAMPION:
            meta = _sidecar(path)
            metas[version] = meta
            w = _seal_challenger(version, meta)
            if w:
                warns.append(w)

    mains, lates = {}, {}
    for version, df in frames.items():
        mains[version], lates[version] = _main_window(df)
        if mains[version].empty:
            raise SystemExit(f"{version}: no MAIN rows in the July window - "
                             "was the derived pipeline/frame cache rebuilt through "
                             f"{split.validation_end}?")

    # identical-frame seal: all three must score the exact same decision rows
    base_keys = set(map(tuple, mains[CHAMPION][["symbol", "decision_time"]].to_numpy()))
    for version in CANDIDATES:
        keys = set(map(tuple, mains[version][["symbol", "decision_time"]].to_numpy()))
        if keys != base_keys:
            raise SystemExit(f"{version}: validation rows differ from {CHAMPION}'s "
                             f"({len(keys ^ base_keys)} mismatched) - the models were "
                             "trained on different data snapshots; rebuild + retrain.")

    # the pre-registered mean-of-seeds rule, enforced not documented
    for version in CANDIDATES:
        seed_cols = [c for c in mains[version].columns if c.startswith("risk_prob_seed")]
        tm.assert_ensemble_consistency(mains[version], seed_cols)

    results = {v: _metrics(mains[v], lates[v]) for v in frames}
    for version in CANDIDATES:
        results[version]["per_seed_auc"] = {
            c.replace("risk_prob_", ""): _auc(mains[version], c)
            for c in sorted(mains[version].columns) if c.startswith("risk_prob_seed")}

    champ_auc = results[CHAMPION]["auc"]
    tcn_auc = results["v4_tcn_60m_prod"]["auc"]
    cnn_auc = results["v5_cnn_60m_prod"]["auc"]
    floor_ok = {v: results[v]["auc"] >= champ_auc for v in CANDIDATES}
    d = cnn_auc - tcn_auc

    if not any(floor_ok.values()):
        selected, reason = None, (f"NO-SHIP: both candidates below the champion floor "
                                  f"({champ_auc:.4f}) on the July window - investigate "
                                  "the July regime before shipping anything.")
    elif abs(d) >= SELECTION_MARGIN:
        selected = "v5_cnn_60m_prod" if d > 0 else "v4_tcn_60m_prod"
        reason = (f"|dAUC| = {abs(d):.4f} >= {SELECTION_MARGIN} - July separates the "
                  f"candidates; higher AUC wins.")
        if not floor_ok[selected]:
            other = next(v for v in CANDIDATES if v != selected)
            selected = other if floor_ok[other] else None
            reason += (" (winner was below the champion floor -> fell back to the "
                       "other candidate)" if selected else
                       " (winner below the champion floor and no fallback)")
    else:
        selected = TIE_BREAK
        reason = (f"|dAUC| = {abs(d):.4f} < {SELECTION_MARGIN} - July cannot separate "
                  f"them; pre-registered tie-break -> {TIE_BREAK} (frozen walk-forward "
                  "evidence, see the docstring).")
        if not floor_ok[selected]:
            other = next(v for v in CANDIDATES if v != selected)
            selected = other if floor_ok[other] else None
            reason += (" (tie-break pick below the champion floor -> the other "
                       "candidate)" if selected else " (both floors failed)")

    # ---------------- console + artifacts ----------------
    rows = []
    for version in (CHAMPION,) + CANDIDATES:
        r = results[version]
        rows.append({"version": version, "n_main": len(mains[version]),
                     "auc": r["auc"], "pr_auc": r["pr_auc"], "brier": r["brier"],
                     "top10_mae_risk_rate": r["top10_rate"],
                     "bottom20_mae_risk_rate": r["bottom20_rate"],
                     "late_auc": r["late_auc"],
                     "role": "champion-floor" if version == CHAMPION else
                             ("SELECTED" if version == selected else "candidate")})
    table = pd.DataFrame(rows)
    print("\n", table.to_string(index=False), sep="")
    for version in CANDIDATES:
        seeds = results[version]["per_seed_auc"]
        print(f"  {version} per-seed AUC: "
              + "  ".join(f"{k.replace('seed', 's')}={v:.4f}" for k, v in seeds.items()))
    print("\nper-day AUC (ensemble):")
    per_day = pd.DataFrame({v: results[v]["per_day"] for v in frames})
    print(per_day.round(4).to_string())
    for w in warns:
        print(f"\n  *** {w}")
    print(f"\nverdict: {'SELECT ' + selected if selected else 'NO-SHIP'}")
    print(f"  {reason}")

    lines = [
        "# Step 20: production 60m candidate selection (pre-registered rule)", "",
        f"Generated {now_kst_text()}. Window {split.validation_start}.."
        f"{split.validation_end} main (<= {rm.HORIZON_60M_MAIN_CUTOFF}), "
        f"n={len(mains[CHAMPION]):,}. Margin {SELECTION_MARGIN}; tie -> {TIE_BREAK} "
        "(frozen walk-forward evidence). Research-only: serving stays Phase-2 work.", "",
        "```", table.to_string(index=False), "```", "",
        "## Per-day AUC", "", "```", per_day.round(4).to_string(), "```", "",
        "## Verdict", "",
        f"**{'SELECT ' + selected if selected else 'NO-SHIP'}** - {reason}", "",
    ]
    if warns:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warns] + [""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nreport -> {REPORT_PATH}\nsummary -> {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
