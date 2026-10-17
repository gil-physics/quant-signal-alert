"""Step 14: train Model 1 risk model on directional-alert candidates."""

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

import pandas as pd

from config import DB_PATH, PROCESSED_DIR, ensure_directories
from risk_model import (
    HORIZON_60M_MAIN_CUTOFF,
    PRODUCTION_SPLIT_60M,
    MODEL_KINDS,
    MODEL_VERSIONS,
    WALK_FORWARD_FOLDS_60M,
    SplitConfig,
    walk_forward_fold,
    available_feature_columns,
    bucket_report,
    build_model_frame,
    classification_summary,
    existing_risk_grade_report,
    fit_binary_model,
    model_feature_importance,
    model_version_config,
    monthly_auc,
    predict_with_model,
    resolve_model_kind,
    roc_auc_score_np,
    resolve_model_path,
    save_model,
    split_by_date,
)
from settings import load_active_symbols, load_data_config
from utils import now_kst_text


# Step 15/17 mutual pairing: a restricted target and its version are mutually required,
# so the v1 baseline risk_model_lgbm.txt can never be overwritten by an up-risk or
# 60m-lineage model (and vice versa). Version-keyed on purpose -- an endswith("_60m")
# trigger would misfire on the unrelated legacy target_down2_60m.
REQUIRED_TARGET_BY_VERSION = {"v1_short": "target_up_risk", "v1_60m": "target_mae_risk_60m",
                              "v1_60m_prod": "target_mae_risk_60m"}
RESTRICTED_TARGETS = set(REQUIRED_TARGET_BY_VERSION.values())

# --fold sets all four dates from risk_model.WALK_FORWARD_FOLDS_60M; explicit date args
# alongside it are refused (checked against these parser defaults).
_DATE_DEFAULTS = ("2025-06-02", "2026-02-28", "2026-03-01", "2026-06-30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 14: train Model 1 risk model.")
    parser.add_argument("--target", default="target_mae_risk",
                        choices=["target_mae_risk", "target_up_fail", "target_down2_60m",
                                 "target_up_risk", "target_mae_risk_60m"])
    parser.add_argument("--train-start", default="2025-06-02")
    parser.add_argument("--train-end", default="2026-02-28")
    parser.add_argument("--validation-start", default="2026-03-01")
    parser.add_argument("--validation-end", default="2026-06-30")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--risk-model-version", default="v1", choices=list(MODEL_VERSIONS),
                        help="v1 = 8 cross-symbol features (baseline); v2.0 = + richer universe "
                             "features (rejected); v2.1a/v2.1b = v1 + one leader's lead-lag (rejected); "
                             "v3_0a_premarket_000660 = v1 + target premarket context; "
                             "v3_0b_premarket_universe = v1 + target + universe premarket context; "
                             "v1_short = v1 features trained on target_up_risk (short-adverse "
                             "mirror, separate model file); "
                             "v1_60m = v1 features + minutes_since_open trained on the "
                             "fixed-horizon target_mae_risk_60m (Step 17; fold-based, pass --fold). "
                             "Selects the feature set + default model path. v3.0 needs "
                             "premarket_features_daily (build_premarket_features.py).")
    parser.add_argument("--fold", type=int, default=None,
                        choices=range(1, len(WALK_FORWARD_FOLDS_60M) + 1),
                        help="Step 17 walk-forward fold (v1_60m only; REQUIRED for it). Sets "
                             "all four split dates from risk_model.WALK_FORWARD_FOLDS_60M and "
                             "suffixes every artifact with _fold<k> so folds never clobber "
                             "each other.")
    parser.add_argument("--model-path", default=None,
                        help="override the version's default model path.")
    parser.add_argument("--model", default="auto", choices=list(MODEL_KINDS),
                        help="Backend: auto (lightgbm>sklearn>numpy), lightgbm, "
                             "sklearn_hgb, sklearn_logistic, or numpy.")
    parser.add_argument("--no-lightgbm", action="store_true",
                        help="Deprecated alias: force the built-in numpy logistic fallback.")
    args = parser.parse_args()
    required = REQUIRED_TARGET_BY_VERSION.get(args.risk_model_version)
    if (required is not None and args.target != required) or \
            (required is None and args.target in RESTRICTED_TARGETS):
        parser.error(
            f"restricted pairing: version {args.risk_model_version!r} requires target "
            f"{required!r} and restricted targets {sorted(RESTRICTED_TARGETS)} may only "
            f"train their paired version (got target={args.target!r}); this prevents "
            "overwriting a frozen artifact with a differently-labelled model."
        )
    if args.fold is not None:
        if args.risk_model_version != "v1_60m":
            parser.error("--fold is only valid with --risk-model-version v1_60m")
        got = (args.train_start, args.train_end, args.validation_start, args.validation_end)
        if got != _DATE_DEFAULTS:
            parser.error("--fold sets all four split dates from WALK_FORWARD_FOLDS_60M; "
                         "do not pass explicit date args with it.")
        f = walk_forward_fold(args.fold)
        args.train_start, args.train_end = f.train_start, f.train_end
        args.validation_start, args.validation_end = f.validation_start, f.validation_end
    elif args.risk_model_version == "v1_60m":
        parser.error("v1_60m is fold-based (Step 17 user-decided walk-forward protocol): "
                     f"pass --fold 1..{len(WALK_FORWARD_FOLDS_60M)}.")
    elif args.risk_model_version == "v1_60m_prod":
        # Step 20: the production split is registry-pinned -- dates come from
        # rm.PRODUCTION_SPLIT_60M, never from the CLI (a mistyped window could
        # silently train into the sanity-check days).
        got = (args.train_start, args.train_end, args.validation_start, args.validation_end)
        if got != _DATE_DEFAULTS:
            parser.error("v1_60m_prod dates are pinned by risk_model.PRODUCTION_SPLIT_60M; "
                         "do not pass explicit date args.")
        s = PRODUCTION_SPLIT_60M
        args.train_start, args.train_end = s.train_start, s.train_end
        args.validation_start, args.validation_end = s.validation_start, s.validation_end
    return args


def _table_exists(conn, name):
    return bool(conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()[0])


def _load_inputs(conn, model_source, symbols):
    if not _table_exists(conn, "directional_alerts"):
        raise SystemExit("directional_alerts table not found - run scripts/online_alert_backtest.py first.")

    alerts = pd.read_sql_query("SELECT * FROM directional_alerts WHERE source = ?", conn, params=(model_source,))
    features_merged = pd.read_sql_query(
        "SELECT * FROM features_merged WHERE source = ?",
        conn,
        params=(model_source,),
    )
    features_relative = pd.read_sql_query(
        "SELECT * FROM features_relative WHERE source = ?",
        conn,
        params=(model_source,),
    )
    labels = pd.read_sql_query(
        "SELECT * FROM labels WHERE source = ?",
        conn,
        params=(model_source,),
    )
    bars = pd.read_sql_query(
        "SELECT symbol, bar_start, low, high, close FROM bars_1m WHERE source = ?",
        conn,
        params=(model_source,),
    )

    for frame in (alerts, features_merged, features_relative, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return alerts, features_merged, features_relative, labels, bars


def _load_premarket(conn):
    """The per-day premarket feature table (one row per trade_date; v3.0 models)."""
    if not _table_exists(conn, "premarket_features_daily"):
        raise SystemExit(
            "premarket_features_daily table not found - run scripts/build_premarket_features.py first "
            "(required for v3.0 premarket models)."
        )
    return pd.read_sql_query("SELECT * FROM premarket_features_daily", conn)


def _summary_rows(train, valid, target, model_type, feature_cols, metrics, deciles, existing):
    top10 = deciles[deciles["risk_bucket"] == "top_10%"].iloc[0].to_dict() if len(deciles) else {}
    bottom20 = deciles[deciles["risk_bucket"] == "bottom_20%"].iloc[0].to_dict() if len(deciles) else {}
    risk_grade = existing[existing["group"] == "existing_RISK_grade"].iloc[0].to_dict() if len(existing) else {}
    return pd.DataFrame([{
        "created_at": now_kst_text(),
        "model_type": model_type,
        "target": target,
        "feature_count": len(feature_cols),
        "train_start": train["date"].min() if len(train) else "",
        "train_end": train["date"].max() if len(train) else "",
        "validation_start": valid["date"].min() if len(valid) else "",
        "validation_end": valid["date"].max() if len(valid) else "",
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_positive_rate": float(train[target].mean()) if len(train) else 0.0,
        "validation_positive_rate": float(valid[target].mean()) if len(valid) else 0.0,
        "auc": metrics["auc"],
        "pr_auc": metrics["pr_auc"],
        "logloss": metrics["logloss"],
        "precision_at_top10": metrics["precision_at_top10"],
        "recall_at_top10": metrics["recall_at_top10"],
        "top10_mae_risk_rate": top10.get("mae_risk_rate", 0.0),
        "bottom20_mae_risk_rate": bottom20.get("mae_risk_rate", 0.0),
        "existing_RISK_mae_risk_rate": risk_grade.get("mae_risk_rate", 0.0),
        "top10_vs_existing_RISK_lift": (
            top10.get("mae_risk_rate", 0.0) - risk_grade.get("mae_risk_rate", 0.0)
        ),
    }])


def main() -> int:
    args = parse_args()
    ensure_directories()
    (ROOT_DIR / "models").mkdir(parents=True, exist_ok=True)

    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    split = SplitConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
    )

    version = args.risk_model_version
    cfg = model_version_config(version)
    model_kind = "numpy" if args.no_lightgbm else args.model
    resolved_kind = resolve_model_kind(model_kind)
    print(f"[{now_kst_text()}] train risk model version={version} target={args.target} source={model_source} "
          f"symbols={len(symbols)} model={model_kind}->{resolved_kind} relative_features={len(cfg['relative'])}")
    premarket_cols = cfg.get("premarket") or []
    conn = sqlite3.connect(args.db)
    try:
        alerts, fm, rel, labels, bars = _load_inputs(conn, model_source, symbols)
        premarket = _load_premarket(conn) if premarket_cols else None
    finally:
        conn.close()

    print(f"  inputs: alerts={len(alerts)} features_merged={len(fm)} features_relative={len(rel)} labels={len(labels)}"
          + (f" premarket_days={len(premarket)} premarket_features={len(premarket_cols)}" if premarket_cols else ""))
    # Step 17: the 60m lineage is version-keyed (NOT endswith("_60m") -- that would
    # misfire on the legacy target_down2_60m).
    horizon_minutes = 60 if version in ("v1_60m", "v1_60m_prod") else None
    data, feature_cols = build_model_frame(alerts, fm, rel, labels, bars,
                                           relative_columns=cfg["relative"], feature_columns=cfg["features"],
                                           premarket=premarket, premarket_columns=premarket_cols,
                                           horizon_minutes=horizon_minutes)
    feature_cols = available_feature_columns(data, cfg["features"])
    train, valid = split_by_date(data, split=split)
    train = train.dropna(subset=[args.target]).reset_index(drop=True)
    valid = valid.dropna(subset=[args.target]).reset_index(drop=True)
    late = None
    if horizon_minutes is not None:
        # Main population = decisions at or before the cutoff (always exactly 60m of
        # forward window). Later decisions carry a truncated-horizon label: they are
        # excluded from training and from every reported metric, but still scored into
        # the predictions parquet as the late-session DIAGNOSTIC rows.
        train = train[train["time"] <= HORIZON_60M_MAIN_CUTOFF].reset_index(drop=True)
        late = valid[valid["time"] > HORIZON_60M_MAIN_CUTOFF].reset_index(drop=True)
        valid = valid[valid["time"] <= HORIZON_60M_MAIN_CUTOFF].reset_index(drop=True)
        print(f"  60m main population (time <= {HORIZON_60M_MAIN_CUTOFF}): "
              f"train={len(train)} val={len(valid)} late_val={len(late)} fold={args.fold}")
    if train.empty or valid.empty:
        raise SystemExit(f"empty train/validation split: train={len(train)} validation={len(valid)}")

    print(
        f"  model_frame={len(data)} features={len(feature_cols)} "
        f"train={len(train)} val={len(valid)} "
        f"positive(train)={train[args.target].mean():.2%} positive(val)={valid[args.target].mean():.2%}"
    )

    model, pre = fit_binary_model(
        train,
        valid,
        feature_cols,
        args.target,
        model_kind=model_kind,
    )
    model_type = getattr(model, "model_type", type(model).__name__)
    best_iter = getattr(model, "best_iteration_used", None)
    if best_iter:
        print(f"  lightgbm early-stopped at {best_iter} trees (chronological train-tail holdout)")

    train = train.copy()
    valid = valid.copy()
    train["risk_prob"] = predict_with_model(model, pre, train)
    valid["risk_prob"] = predict_with_model(model, pre, valid)
    train["split"] = "train"
    valid["split"] = "validation"
    if late is not None and len(late):
        late = late.copy()
        late["risk_prob"] = predict_with_model(model, pre, late)
        late["split"] = "validation"   # distinguished by time > cutoff, not by split

    mae_col = "future_MAE_60m" if horizon_minutes is not None else "future_MAE"
    metrics = classification_summary(valid, target_col=args.target)
    deciles = bucket_report(valid, target_col=args.target, mae_col=mae_col)
    existing = existing_risk_grade_report(valid, target_col=args.target, mae_col=mae_col)
    summary = _summary_rows(train, valid, args.target, model_type, feature_cols, metrics, deciles, existing)

    importance = model_feature_importance(model, feature_cols)

    predictions_cols = [
        "split", "symbol", "decision_time", "date", "time", "source",
        "alert_type", "alert_grade", "risk_prob",
        "target_mae_risk", "target_up_fail", "target_down2_60m", "target_up_risk",
        "future_MAE", "future_return_60m", "future_MFE",
        # Step 17 60m lineage (existence-filtered: absent on legacy versions)
        "target_mae_risk_60m", "target_up_risk_60m", "future_MAE_60m", "future_MFE_60m",
    ]
    frames = [train, valid] + ([late] if late is not None and len(late) else [])
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions[[c for c in predictions_cols if c in predictions.columns]]

    # version-suffix non-v1 artifacts so the v1 baseline CSVs/tables are never clobbered;
    # fold runs additionally get _fold<k> so folds never clobber each other.
    vsfx = "" if version == "v1" else "_" + version.replace(".", "_")
    if args.fold is not None:
        vsfx += f"_fold{args.fold}"
    summary_path = PROCESSED_DIR / f"risk_model_summary{vsfx}.csv"
    decile_path = PROCESSED_DIR / f"risk_prob_decile_report{vsfx}.csv"
    importance_path = PROCESSED_DIR / f"feature_importance{vsfx}.csv"
    predictions_path = PROCESSED_DIR / f"risk_model_predictions{vsfx}.parquet"
    monthly_path = PROCESSED_DIR / f"risk_prob_monthly_auc{vsfx}.csv"
    model_path = Path(args.model_path) if args.model_path else resolve_model_path(version)
    if args.fold is not None and not args.model_path:
        model_path = model_path.with_name(
            model_path.name.replace("_lgbm.txt", f"_fold{args.fold}_lgbm.txt"))

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    deciles.to_csv(decile_path, index=False, encoding="utf-8-sig")
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    predictions.to_parquet(predictions_path, index=False)
    metadata = {
        "target": args.target,
        "model_source": model_source,
        "model_version": version,
        "requires_real_relative_features": cfg["requires_cross_symbol"],
        "requires_premarket": cfg.get("requires_premarket", False),
        "premarket_columns": premarket_cols,
        "feature_columns": feature_cols,
        "train_rows": len(train),
        "validation_rows": len(valid),
    }
    if horizon_minutes is not None:
        metadata.update({
            "fold": args.fold,
            "horizon_minutes": horizon_minutes,
            "main_cutoff": HORIZON_60M_MAIN_CUTOFF,
            "train_start": args.train_start, "train_end": args.train_end,
            "validation_start": args.validation_start, "validation_end": args.validation_end,
        })
    save_model(
        model,
        model_path,
        preprocessor=pre,
        metadata=metadata,
    )

    conn = sqlite3.connect(args.db)
    try:
        predictions.to_sql(f"risk_model_predictions{vsfx}", conn, if_exists="replace", index=False)
        summary.to_sql(f"risk_model_summary{vsfx}", conn, if_exists="replace", index=False)
        deciles.to_sql(f"risk_prob_decile_report{vsfx}", conn, if_exists="replace", index=False)
        importance.to_sql(f"feature_importance{vsfx}", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print("\nvalidation metrics:")
    print(summary.to_string(index=False))
    print("\nrisk_prob buckets:")
    print(deciles.to_string(index=False))
    print("\nexisting RISK grade:")
    print(existing.to_string(index=False))
    if late is not None and len(late):
        late_auc = roc_auc_score_np(late[args.target].astype(int).to_numpy(),
                                    late["risk_prob"].to_numpy(dtype=float))
        print(f"\nlate-session diagnostic (time > {HORIZON_60M_MAIN_CUTOFF}; truncated-horizon "
              "label, excluded from every metric above):")
        print(f"  n={len(late)} positive_rate={late[args.target].mean():.2%} auc={late_auc:.4f}")
    monthly = monthly_auc(valid, target_col=args.target)
    if len(monthly):
        monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        print("\nvalidation AUC by month (regime-decay check; report the most recent, not the pooled AUC):")
        print(monthly.to_string(index=False))
    print("\ntop feature importance:")
    print(importance.head(20).to_string(index=False))
    print(f"\npredictions -> db risk_model_predictions + {predictions_path}")
    print(f"summary -> {summary_path}")
    print(f"deciles -> {decile_path}")
    print(f"feature importance -> {importance_path}")
    print(f"model -> {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
