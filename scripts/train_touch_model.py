"""Touch-prediction trainer: B2 LightGBM champion + CNN/TCN sequence challengers.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Trains the
pre-registered touch study (src/touch_model.py): label = P2r +0.5% close-touch in
10 minutes at carry-variant MA5x20 crosses confirmed in the regular session.

Modes (--mode):
  pretrain (default)  pooled model per fold; the FT symbol's tail-block rows are
                      EXCLUDED from training (champion and challengers alike) so a
                      later fine-tune has an uncontaminated early-stopping set.
                      With --seeds (comma list) every seed is trained per fold and
                      the predictions parquet carries prob = the seed-mean ensemble
                      plus prob_seed{s} columns (the Step-22 grade-map source runs
                      WITHOUT --ft-symbol: FT was not adopted, so the OOF population
                      matches the prod training population).
  finetune            warm-start from the pooled fold artifact; train on the FT
                      symbol's non-tail train rows only (lr x0.1, few epochs); early
                      stopping on the reserved tail block (its only ever use).
  scratch             FT-symbol-only model from scratch (small-data control arm).
  prod                Step 22 production cut (registry touch_cnn_prod): the single
                      registry-pinned split (train_start fixed, train_end = the
                      latest label-complete trading day, resolved at run time and
                      recorded in every sidecar; NO validation window), 5-seed loop,
                      and the user-decided TWO-STAGE refit per seed -- stage-1 runs
                      the normal day-boundary-holdout early stopping ONLY to pick
                      best_epoch, stage-2 retrains from scratch on train+holdout for
                      exactly best_epoch epochs replaying stage-1's lr trajectory,
                      and only the stage-2 weights are saved. No predictions parquet
                      (nothing is held out); the frozen grade map comes from the
                      5-seed OOF fold parquets instead (scripts/build_touch_grade_map.py).

Baselines: every predictions parquet also carries prob_b0 (train per-direction
constant) and prob_b1 (train band x direction x early lookup) fitted on the SAME
pooled train rows, so the gate is parquet-only.

Run:
  ..\\Python311-64\\python.exe scripts\\train_touch_model.py --model lgbm --ft-symbol <code>
  ..\\Python311-64\\python.exe scripts\\train_touch_model.py --model cnn --ft-symbol <code> --device cuda:0
  ..\\Python311-64\\python.exe scripts\\train_touch_model.py --model cnn --mode finetune --ft-symbol <code> --device cuda:0
  5-seed OOF (grade-map source):
         ... --model cnn --seeds "42,43,44,45,46" --device cuda:0
  prod:  ... --model cnn --mode prod --device cuda:0
  smoke: ... --model cnn --fold 1 --max-epochs 2 --out-suffix _smoke --ft-symbol <code>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
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
import tcn_data
import tcn_model as tm
import touch_model as tmm
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from settings import load_active_symbols, load_data_config
from utils import now_kst_text

MODELS_DIR = ROOT_DIR / "models"
MODEL_KEYS = {"lgbm": "touch_lgbm", "cnn": "touch_cnn", "tcn": "touch_tcn"}
# pre-registered LightGBM mini-grid (selected on fold-1 pooled train's own
# day-boundary holdout ONLY, then frozen for every fold)
LGBM_GRID = ((31, 20), (31, 60), (63, 20), (63, 60))
LGBM_BASE = dict(n_estimators=600, learning_rate=0.05, subsample=0.9,
                 subsample_freq=1, colsample_bytree=0.9, class_weight="balanced")
LGBM_PARAMS_PATH = MODELS_DIR / "touch_model_lgbm_params.json"
FT_LR_SCALE = 0.1
FT_MAX_EPOCHS = 5


def parse_args():
    p = argparse.ArgumentParser(description="Touch-model trainer (champion + challengers).")
    p.add_argument("--model", required=True, choices=list(MODEL_KEYS))
    p.add_argument("--mode", default="pretrain",
                   choices=["pretrain", "finetune", "scratch", "prod"])
    p.add_argument("--fold", type=int, default=None,
                   choices=range(1, len(tmm.TOUCH_FOLDS) + 1),
                   help="one fold only (default: all folds).")
    p.add_argument("--ft-symbol", default="",
                   help="fine-tune symbol code (pre-registered: the main symbol). "
                        "Given on EVERY run of the fold protocol so the tail-block "
                        "exclusion is uniform across champion and challengers; the "
                        "Step-22 5-seed OOF and prod runs OMIT it (FT not adopted).")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default="",
                   help="comma list for multi-seed runs (sequence models only), e.g. "
                        "\"42,43,44,45,46\"; overrides --seed. prod defaults to the "
                        "registry seed set.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="sequence models only; default = the registry value (40).")
    p.add_argument("--out-suffix", default="",
                   help="artifact namespace for diagnostic/smoke runs; the canonical "
                        "(empty) suffix is RESERVED for default-knob runs.")
    args = p.parse_args()
    if args.mode in ("finetune", "scratch"):
        if not args.ft_symbol:
            p.error(f"--mode {args.mode} requires --ft-symbol.")
        if args.model == "lgbm":
            p.error("finetune/scratch apply to the sequence challengers only "
                    "(the champion stays pooled).")
        if args.seeds:
            p.error("finetune/scratch are single-seed protocols (--seed).")
    if args.mode == "prod":
        if args.model != "cnn":
            p.error("--mode prod is registered for the gate winner only "
                    "(touch_cnn_prod); pass --model cnn.")
        if args.fold is not None:
            p.error("--mode prod uses the single registry-pinned production split; "
                    "--fold does not apply.")
        if args.ft_symbol:
            p.error("--mode prod has no FT-tail exclusion (FT was not adopted); "
                    "drop --ft-symbol.")
    if args.seeds and args.model == "lgbm":
        p.error("--seeds applies to the sequence models only.")
    if args.out_suffix and not args.out_suffix.startswith("_"):
        p.error("--out-suffix must start with '_'.")
    return args


def resolve_seeds(args, cfg):
    if args.seeds:
        try:
            seeds = tuple(int(s) for s in args.seeds.split(","))
        except ValueError:
            raise SystemExit(f"--seeds must be a comma list of ints, got {args.seeds!r}")
    elif args.mode == "prod":
        seeds = tuple(cfg["seeds"])
    else:
        seeds = (args.seed,)
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds in {seeds}")
    return seeds


def _default_knobs(args, cfg, seeds) -> bool:
    default_epochs = cfg.get("max_epochs", 40)   # lgbm has no epoch knob
    canonical_seed_sets = ((42,), tuple(tmm.TOUCH_SUBMISSION_SEEDS))
    return ((args.max_epochs is None or args.max_epochs == default_epochs)
            and seeds in canonical_seed_sets and args.db == str(DB_PATH))


def load_inputs(args, symbols, need_sequences):
    conn = sqlite3.connect(args.db)
    try:
        mismatch = conn.execute(
            "SELECT COUNT(*) FROM bars_1m a JOIN bars_1m b ON a.symbol=b.symbol AND "
            "a.bar_start=b.bar_start WHERE a.source='al_raw' AND b.source='al_regular' "
            "AND (a.open!=b.open OR a.high!=b.high OR a.low!=b.low OR a.close!=b.close)"
        ).fetchone()[0]
        if mismatch:
            raise SystemExit(f"al_raw vs al_regular price mismatch on {mismatch} "
                             "overlapping bars -- the label (al_raw) and the features "
                             "(al_regular) would disagree; rebuild the imports first.")
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close FROM bars_1m "
            "WHERE source = 'al_raw'", conn)
        model_source = load_data_config().model_source
        merged_table_cols = [c for c in tmm.MERGED_FEATURES_V1
                             if c not in tmm.DERIVED_MERGED_FLAGS]
        fm = pd.read_sql_query(
            "SELECT symbol, decision_time, "
            + ", ".join(f'"{c}"' for c in merged_table_cols)
            + " FROM features_merged WHERE source = ?", conn, params=(model_source,))
        rel = pd.read_sql_query(
            "SELECT symbol, decision_time, "
            + ", ".join(f'"{c}"' for c in tmm.RELATIVE_FEATURES_V1)
            + " FROM features_relative WHERE source = ?", conn, params=(model_source,))
        features_1m = None
        if need_sequences:
            seq_cols = ", ".join(f'"{c}"' for c in tcn_data.SEQ_FEATURE_COLUMNS_V4)
            features_1m = pd.read_sql_query(
                f"SELECT symbol, decision_time, {seq_cols} FROM features_1m "
                "WHERE source = ?", conn, params=(model_source,))
    finally:
        conn.close()
    frames = [bars, fm, rel] + ([features_1m] if features_1m is not None else [])
    for frame in frames:
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return bars, fm, rel, features_1m


def _fit_lgbm(train, features, params, seed):
    """LightGBM with day-boundary ES holdout; returns (clf, best_iter, holdout_auc)."""
    import lightgbm as lgb

    fit_pos, eval_pos = tm._day_boundary_holdout(train, 0.15)
    if fit_pos is None:
        fit_pos, eval_pos = np.arange(len(train)), None
    x = train[features].astype(float).to_numpy()
    y = train[tmm.TOUCH_TARGET].astype(int).to_numpy()
    clf = lgb.LGBMClassifier(random_state=int(seed), n_jobs=-1, verbose=-1,
                             **{**LGBM_BASE, **params})
    if eval_pos is not None:
        clf.fit(x[fit_pos], y[fit_pos], eval_set=[(x[eval_pos], y[eval_pos])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iter = int(clf.best_iteration_ or clf.n_estimators)
        hold_auc = rm.roc_auc_score_np(
            y[eval_pos], clf.predict_proba(x[eval_pos])[:, 1])
    else:
        clf.fit(x, y)
        best_iter, hold_auc = int(clf.n_estimators), float("nan")
    return clf, best_iter, hold_auc


def select_lgbm_params(train, features, seed):
    """Pre-registered 4-combo grid on the FOLD-1 pooled train's internal holdout."""
    rows = []
    for leaves, min_child in LGBM_GRID:
        params = {"num_leaves": leaves, "min_child_samples": min_child}
        _clf, best_iter, auc = _fit_lgbm(train, features, params, seed)
        rows.append({"num_leaves": leaves, "min_child_samples": min_child,
                     "holdout_auc": auc, "best_iter": best_iter})
        print(f"    grid leaves={leaves} min_child={min_child}: "
              f"holdout_auc={auc:.4f} best_iter={best_iter}")
    best = max(rows, key=lambda r: (r["holdout_auc"], -r["num_leaves"]))
    return {"num_leaves": best["num_leaves"],
            "min_child_samples": best["min_child_samples"]}, rows


def artifact_names(model, mode, ft_symbol, out_suffix, fold, seed, *, n_seeds=1):
    if mode == "prod":                     # single split, weights only (no parquet)
        return MODELS_DIR / f"touch_model_{model}_prod{out_suffix}_seed{seed}.pt", None
    # multi-seed runs get their own namespace (risk-lineage "_5seed" convention) so
    # the canonical single-seed gate artifacts can never be clobbered.
    seed_tag = f"_{n_seeds}seed" if n_seeds > 1 else ""
    mode_sfx = {"pretrain": "", "finetune": f"_ft_{ft_symbol}",
                "scratch": f"_scratch_{ft_symbol}"}[mode]
    base = f"touch_model_{model}{seed_tag}{mode_sfx}{out_suffix}_fold{fold}"
    model_name = base + (".txt" if model == "lgbm" else f"_seed{seed}.pt")
    pred_name = (f"touch_predictions_{model}{seed_tag}{mode_sfx}{out_suffix}"
                 f"_fold{fold}.parquet")
    return MODELS_DIR / model_name, PROCESSED_DIR / pred_name


def write_predictions(pred_path, valid, prob, b0, b1, meta, *, extra=None):
    out = valid[["symbol", "date", "time", "month", "bar_start", "decision_time",
                 "direction", "band", "early_flag", "cluster_first", "dist_pct",
                 "dist_atr", tmm.TOUCH_TARGET, tmm.TOUCH_TARGET_AUX,
                 "w10_mae", "w10_endpoint_ret"]].copy()
    out["prob"] = prob
    out["prob_b0"] = b0
    out["prob_b1"] = b1
    for name, values in (extra or {}).items():     # e.g. prob_seed{s} columns
        out[name] = values
    out["split"] = "validation"
    out.to_parquet(pred_path, index=False)
    Path(str(pred_path) + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt_auc(value):
    return f"{value:.4f}" if value is not None else "n/a"


def run_prod(args, cfg, seeds, symbols, deterministic, events, stats,
             static_cols, bar_index, seq_config, canonical) -> int:
    """Step 22 production cut: registry-pinned split with train_end resolved at run
    time, per-seed TWO-STAGE refit (user decision), weights-only artifacts.

    stage-1 = the normal day-boundary-holdout early-stopping run, used ONLY to pick
    best_epoch (its weights are discarded); stage-2 retrains from scratch on
    train+holdout for exactly best_epoch epochs, replaying stage-1's per-epoch lr
    trajectory (the plateau scheduler cannot run without a holdout). Only the
    stage-2 weights are saved. No predictions parquet: nothing is held out, and the
    grade map is built from the 5-seed OOF fold parquets, never from prod outputs."""
    split = cfg["split"]
    resolved_end = str(events["date"].max())
    train = events[(events["date"] >= split.train_start)
                   & (events["date"] <= resolved_end)].reset_index(drop=True)
    if train.empty:
        raise SystemExit("prod: empty train population")
    print(f"\n[prod] train={len(train):,} rows {split.train_start}..{resolved_end} "
          f"(train_end = latest label-complete day, resolved from the event "
          f"population; no validation window) "
          f"pos={train[tmm.TOUCH_TARGET].mean():.4f}")
    tcn_config = tm.TCNConfig(
        in_channels=seq_config.n_channels, channels=int(cfg["channels"]),
        levels=int(cfg["levels"]), kernel_size=int(cfg["kernel_size"]),
        static_dim=len(static_cols), arch=cfg["arch"])
    seq_pre = tcn_data.SequencePreprocessor(
        list(seq_config.seq_feature_columns)).fit(
        bar_index.bar_mat_raw, dates=bar_index.dates, train_end=resolved_end)
    static_pre = rm.FeaturePreprocessor(static_cols).fit(train)
    stage1_config = tm.TrainConfig(max_epochs=args.max_epochs, lr=float(cfg["lr"]),
                                   holdout_by_day=True)
    for sd in seeds:
        t0 = time.monotonic()
        _state1, hist1 = tm.fit_sequence_model(
            train, bar_index, seq_pre, static_pre, tmm.TOUCH_TARGET,
            seq_config=seq_config, tcn_config=tcn_config,
            train_config=stage1_config, seed=sd, device=args.device,
            deterministic=deterministic)
        best_epoch = int(hist1["best_epoch"])
        if best_epoch < 1:
            raise SystemExit(f"prod seed {sd}: stage-1 yielded best_epoch={best_epoch}")
        # the lr USED during epoch e: epoch 1 runs at the config lr; the history logs
        # the lr AFTER the plateau scheduler stepped, i.e. entry e = epoch e+1's lr.
        used_lr = [float(stage1_config.lr)] + [
            float(e["lr"]) for e in hist1["epochs"][:best_epoch - 1]]
        stage2_config = tm.TrainConfig(max_epochs=best_epoch, lr=float(cfg["lr"]),
                                       holdout_frac=0.0, holdout_by_day=True)
        state2, hist2 = tm.fit_sequence_model(
            train, bar_index, seq_pre, static_pre, tmm.TOUCH_TARGET,
            seq_config=seq_config, tcn_config=tcn_config,
            train_config=stage2_config, seed=sd, device=args.device,
            deterministic=deterministic, lr_schedule=used_lr)
        model_path, _ = artifact_names(args.model, "prod", "", args.out_suffix,
                                       None, sd)
        meta = {
            "version": "touch_cnn_prod", "mode": "prod",
            "seed": sd, "seeds": list(seeds),
            "target": tmm.TOUCH_TARGET, "threshold": tmm.TOUCH_THRESHOLD,
            "window_min": tmm.TOUCH_WINDOW_MIN,
            "canonical_run": bool(canonical),
            "device": args.device,
            "train_start": split.train_start, "train_end": resolved_end,
            "train_end_resolved_at_runtime": True,
            "validation_start": None, "validation_end": None,
            "train_rows": int(len(train)),
            "train_base_rate": float(train[tmm.TOUCH_TARGET].mean()),
            "population_stats": stats,
            "two_stage_refit": True,
            "stage1_best_epoch": best_epoch,
            "stage1_holdout_auc": hist1.get("holdout_auc_best"),
            "stage1_holdout_rows": hist1.get("holdout_rows"),
            "refit_rows": hist2.get("fit_rows"),
            "lr_schedule": used_lr,
            # frozen at train time so serving can detect a universe change (the
            # symbol one-hots make static_dim depend on the active set).
            "symbols": sorted(symbols),
            "static_feature_columns": list(static_cols),
        }
        tm.save_tcn(model_path, state2, seq_pre=seq_pre, static_pre=static_pre,
                    seq_config=seq_config, tcn_config=tcn_config,
                    train_config=stage2_config, history=hist2,
                    metadata={**meta, "fit_rows": hist2.get("fit_rows"),
                              "static_features": "STATIC_FEATURES_V1",
                              "n_static": len(static_cols)})
        print(f"  seed {sd}: stage1 best_epoch={best_epoch} "
              f"holdout_auc={_fmt_auc(hist1.get('holdout_auc_best'))} "
              f"({hist1.get('holdout_rows') or 0:,} rows) -> refit "
              f"{best_epoch} epochs on {hist2.get('fit_rows'):,} rows, "
              f"{time.monotonic() - t0:,.0f}s -> {model_path.name}")
    print(f"\nprod artifacts: models/touch_model_cnn_prod{args.out_suffix}"
          f"_seed{{s}}.pt (+ .pt.meta.json) for seeds {list(seeds)}")
    print("next: copy the .pt + .pt.meta.json pairs back to the local models/ dir; "
          "the grade map comes from the 5-seed OOF parquets "
          "(scripts\\build_touch_grade_map.py).")
    return 0


def main() -> int:
    args = parse_args()
    version = "touch_cnn_prod" if args.mode == "prod" else MODEL_KEYS[args.model]
    cfg = tmm.touch_version_config(version)
    is_seq = cfg["kind"] == "sequence"
    seeds = resolve_seeds(args, cfg)
    if args.max_epochs is None:
        args.max_epochs = cfg.get("max_epochs", 40)
    canonical = _default_knobs(args, cfg, seeds)
    if not canonical and not args.out_suffix:
        raise SystemExit(
            "non-default knobs (smoke/diagnostic run) must use a distinct "
            "--out-suffix so the pre-registered artifacts can never be clobbered "
            f"(max_epochs={args.max_epochs}, seeds={list(seeds)}).")
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_active_symbols()
    if args.ft_symbol and args.ft_symbol not in symbols:
        raise SystemExit(f"--ft-symbol {args.ft_symbol!r} is not an active symbol.")
    device = tm.resolve_device(args.device) if is_seq else None
    deterministic = {"auto": None, "on": True, "off": False}[args.deterministic]

    folds_note = ("(prod: single registry split)" if args.mode == "prod"
                  else [args.fold] if args.fold
                  else list(range(1, len(tmm.TOUCH_FOLDS) + 1)))
    print(f"[{now_kst_text()}] touch trainer model={version} mode={args.mode} "
          f"ft_symbol={args.ft_symbol or '(none)'} folds={folds_note} "
          f"seeds={list(seeds)} device={device} suffix={args.out_suffix or '(canonical)'}")

    bars, fm, rel, features_1m = load_inputs(args, symbols, need_sequences=is_seq)
    print(f"  inputs: al_raw bars={len(bars):,} merged={len(fm):,} relative={len(rel):,}"
          + (f" features_1m={len(features_1m):,}" if features_1m is not None else ""))
    events, stats = tmm.build_touch_events(bars)
    events, n_uncovered = tmm.filter_feature_covered(events, fm)
    stats["n_uncovered_dropped"] = n_uncovered
    stats["n_events"] = int(len(events))
    print(f"  events: {stats['n_events']:,} "
          f"(eligible crosses={stats['n_crosses_eligible']:,}, premarket excluded="
          f"{stats['n_premarket_excluded']:,}, incomplete dropped="
          f"{stats['n_incomplete_dropped']:,}, no-feature-coverage dropped="
          f"{n_uncovered:,} — features_1m warmup starts ~10:01) "
          f"base_rate={events[tmm.TOUCH_TARGET].mean():.4f}")
    events = tmm.join_event_features(events, fm, rel, symbols)

    static_cols = tmm.static_features_v1(symbols)
    b2_cols = tmm.b2_features_v1(symbols)
    bar_index = seq_config = None
    if is_seq:
        seq_config = tcn_data.SequenceConfig(seq_len=int(cfg["seq_len"]))
        bar_index = tcn_data.build_bar_index(features_1m, symbols=symbols,
                                             seq_config=seq_config)

    if args.mode == "prod":
        return run_prod(args, cfg, seeds, symbols, deterministic, events, stats,
                        static_cols, bar_index, seq_config, canonical)

    folds = [args.fold] if args.fold else list(range(1, len(tmm.TOUCH_FOLDS) + 1))
    for fold in folds:
        split = tmm.touch_fold(fold)
        train = events[(events["date"] >= split.train_start)
                       & (events["date"] <= split.train_end)].reset_index(drop=True)
        valid = events[(events["date"] >= split.validation_start)
                       & (events["date"] <= split.validation_end)].reset_index(drop=True)
        tail_dates = tmm.ft_tail_dates(events, split) if args.ft_symbol else set()
        pooled_train, ft_es = tmm.split_ft_rows(train, args.ft_symbol, tail_dates)
        if pooled_train.empty or valid.empty:
            raise SystemExit(f"fold {fold}: empty split (train={len(pooled_train)}, "
                             f"valid={len(valid)})")
        b0 = tmm.fit_b0(pooled_train)
        b1 = tmm.fit_b1(pooled_train)
        print(f"\n[fold {fold}] train={len(train):,} (pooled={len(pooled_train):,}, "
              f"ft_tail={len(ft_es):,}/{int(ft_es[tmm.TOUCH_TARGET].sum()) if len(ft_es) else 0}+) "
              f"valid={len(valid):,} pos(train)={pooled_train[tmm.TOUCH_TARGET].mean():.4f} "
              f"pos(valid)={valid[tmm.TOUCH_TARGET].mean():.4f}")

        model_path, pred_path = artifact_names(args.model, args.mode, args.ft_symbol,
                                               args.out_suffix, fold, seeds[0],
                                               n_seeds=len(seeds))
        meta = {
            "version": version, "mode": args.mode, "fold": fold,
            "target": tmm.TOUCH_TARGET, "threshold": tmm.TOUCH_THRESHOLD,
            "window_min": tmm.TOUCH_WINDOW_MIN,
            "ft_symbol": args.ft_symbol or None,
            "ft_tail_dates": sorted(tail_dates),
            "ft_tail_rows": int(len(ft_es)),
            "ft_tail_positives": int(ft_es[tmm.TOUCH_TARGET].sum()) if len(ft_es) else 0,
            "canonical_run": bool(canonical and args.mode == "pretrain"),
            "seed": seeds[0] if len(seeds) == 1 else None,
            "seeds": list(seeds),
            "device": str(device) if device else "cpu",
            "train_start": split.train_start, "train_end": split.train_end,
            "validation_start": split.validation_start,
            "validation_end": split.validation_end,
            "train_rows": int(len(pooled_train)), "validation_rows": int(len(valid)),
            "train_base_rate": float(pooled_train[tmm.TOUCH_TARGET].mean()),
            "population_stats": stats,
        }

        t0 = time.monotonic()
        extra_cols = None                    # prob_seed{s} columns (multi-seed only)
        if args.model == "lgbm":
            if LGBM_PARAMS_PATH.exists():
                chosen = json.loads(LGBM_PARAMS_PATH.read_text(encoding="utf-8"))["params"]
            else:
                if fold != 1:
                    raise SystemExit("LightGBM params not frozen yet -- run fold 1 first "
                                     "(the pre-registered grid selects on fold 1 only).")
                print("  selecting LightGBM params on fold-1 internal holdout:")
                chosen, grid_rows = select_lgbm_params(pooled_train, b2_cols, args.seed)
                LGBM_PARAMS_PATH.write_text(json.dumps(
                    {"params": chosen, "grid": grid_rows, "base": LGBM_BASE,
                     "selected_on": "fold1 pooled-train day-boundary holdout"},
                    indent=2), encoding="utf-8")
                print(f"  frozen params -> {LGBM_PARAMS_PATH.name}: {chosen}")
            clf, best_iter, hold_auc = _fit_lgbm(pooled_train, b2_cols, chosen, args.seed)
            prob = clf.predict_proba(valid[b2_cols].astype(float).to_numpy())[:, 1]
            clf.booster_.save_model(str(model_path))
            meta.update({"lgbm_params": {**LGBM_BASE, **chosen},
                         "best_iteration": best_iter, "holdout_auc": hold_auc,
                         "features": "B2_FEATURES_V1", "n_features": len(b2_cols)})
        else:
            tcn_config = tm.TCNConfig(
                in_channels=seq_config.n_channels, channels=int(cfg["channels"]),
                levels=int(cfg["levels"]), kernel_size=int(cfg["kernel_size"]),
                static_dim=len(static_cols), arch=cfg["arch"])
            if args.mode == "pretrain":
                fit_frame = pooled_train
                seq_pre = tcn_data.SequencePreprocessor(
                    list(seq_config.seq_feature_columns)).fit(
                    bar_index.bar_mat_raw, dates=bar_index.dates,
                    train_end=split.train_end)
                static_pre = rm.FeaturePreprocessor(static_cols).fit(fit_frame)
                train_config = tm.TrainConfig(max_epochs=args.max_epochs,
                                              lr=float(cfg["lr"]),
                                              holdout_by_day=True)
                per_seed_probs = {}
                per_seed_meta = {}
                for sd in seeds:
                    state, history = tm.fit_sequence_model(
                        fit_frame, bar_index, seq_pre, static_pre, tmm.TOUCH_TARGET,
                        seq_config=seq_config, tcn_config=tcn_config,
                        train_config=train_config, seed=sd, device=args.device,
                        deterministic=deterministic)
                    model_path_s, _ = artifact_names(
                        args.model, args.mode, args.ft_symbol, args.out_suffix,
                        fold, sd, n_seeds=len(seeds))
                    tm.save_tcn(model_path_s, state, seq_pre=seq_pre,
                                static_pre=static_pre, seq_config=seq_config,
                                tcn_config=tcn_config, train_config=train_config,
                                history=history,
                                metadata={**meta, "seed": sd,
                                          "fit_rows": int(len(fit_frame)),
                                          "static_features": "STATIC_FEATURES_V1",
                                          "n_static": len(static_cols)})
                    per_seed_probs[sd] = np.asarray(tm.predict_sequence_model(
                        state, valid, bar_index, seq_pre, static_pre,
                        seq_config=seq_config, tcn_config=tcn_config,
                        device=args.device), dtype=float)
                    per_seed_meta[str(sd)] = {
                        "best_epoch": history.get("best_epoch"),
                        "holdout_auc": history.get("holdout_auc_best")}
                    if len(seeds) > 1:
                        print(f"    seed {sd}: best_epoch={history.get('best_epoch')} "
                              f"holdout_auc={_fmt_auc(history.get('holdout_auc_best'))}")
                prob = tm.ensemble_probs(list(per_seed_probs.values()))
                extra_cols = ({f"prob_seed{sd}": per_seed_probs[sd] for sd in seeds}
                              if len(seeds) > 1 else None)
                meta.update({"fit_rows": int(len(fit_frame)),
                             "per_seed": per_seed_meta,
                             "best_epoch": history.get("best_epoch"),
                             "holdout_auc": history.get("holdout_auc_best"),
                             "holdout_rows": history.get("holdout_rows")})
            elif args.mode == "finetune":
                pooled_path, _p = artifact_names(args.model, "pretrain", args.ft_symbol,
                                                 args.out_suffix, fold, args.seed)
                if not pooled_path.exists():
                    raise SystemExit(f"pooled artifact missing: {pooled_path} -- run "
                                     "--mode pretrain first.")
                art = tm.load_tcn(pooled_path)
                if art["metadata"].get("ft_symbol") != args.ft_symbol:
                    raise SystemExit("pooled artifact was trained with a different "
                                     f"--ft-symbol ({art['metadata'].get('ft_symbol')!r}) "
                                     "-- its tail exclusion does not match.")
                tail_pos = int(ft_es[tmm.TOUCH_TARGET].sum()) if len(ft_es) else 0
                if tail_pos < tmm.FT_MIN_POSITIVES:
                    print(f"  fold {fold}: FT SKIPPED -- tail positives "
                          f"{tail_pos} < {tmm.FT_MIN_POSITIVES} (pre-registered rule)")
                    continue
                fit_frame = pooled_train[pooled_train["symbol"] == args.ft_symbol] \
                    .reset_index(drop=True)
                seq_pre, static_pre = art["seq_pre"], art["static_pre"]
                tcn_config = art["tcn_config"]
                train_config = tm.TrainConfig(max_epochs=FT_MAX_EPOCHS,
                                              lr=float(cfg["lr"]) * FT_LR_SCALE,
                                              holdout_by_day=True)
                state, history = tm.fit_sequence_model(
                    fit_frame, bar_index, seq_pre, static_pre, tmm.TOUCH_TARGET,
                    seq_config=seq_config, tcn_config=tcn_config,
                    train_config=train_config, seed=args.seed, device=args.device,
                    deterministic=deterministic,
                    init_state=art["state_dict"], eval_frame=ft_es)
            else:                                       # scratch control
                fit_frame = pooled_train[pooled_train["symbol"] == args.ft_symbol] \
                    .reset_index(drop=True)
                seq_pre = tcn_data.SequencePreprocessor(
                    list(seq_config.seq_feature_columns)).fit(
                    bar_index.bar_mat_raw, dates=bar_index.dates,
                    train_end=split.train_end)
                static_pre = rm.FeaturePreprocessor(static_cols).fit(fit_frame)
                train_config = tm.TrainConfig(max_epochs=args.max_epochs,
                                              lr=float(cfg["lr"]),
                                              holdout_by_day=True)
                state, history = tm.fit_sequence_model(
                    fit_frame, bar_index, seq_pre, static_pre, tmm.TOUCH_TARGET,
                    seq_config=seq_config, tcn_config=tcn_config,
                    train_config=train_config, seed=args.seed, device=args.device,
                    deterministic=deterministic)
            if args.mode != "pretrain":     # pretrain saved/predicted per seed above
                tm.save_tcn(model_path, state, seq_pre=seq_pre, static_pre=static_pre,
                            seq_config=seq_config, tcn_config=tcn_config,
                            train_config=train_config, history=history,
                            metadata={**meta, "fit_rows": int(len(fit_frame)),
                                      "static_features": "STATIC_FEATURES_V1",
                                      "n_static": len(static_cols)})
                prob = tm.predict_sequence_model(state, valid, bar_index, seq_pre,
                                                 static_pre, seq_config=seq_config,
                                                 tcn_config=tcn_config,
                                                 device=args.device)
                meta.update({"fit_rows": int(len(fit_frame)),
                             "best_epoch": history.get("best_epoch"),
                             "holdout_auc": history.get("holdout_auc_best"),
                             "holdout_rows": history.get("holdout_rows")})

        y = valid[tmm.TOUCH_TARGET].astype(int).to_numpy()
        auc = rm.roc_auc_score_np(y, np.asarray(prob, dtype=float))
        auc_b1 = rm.roc_auc_score_np(y, tmm.apply_b1(b1, valid))
        meta.update({"validation_auc": float(auc), "validation_auc_b1": float(auc_b1)})
        write_predictions(pred_path, valid, np.asarray(prob, dtype=float),
                          tmm.apply_b0(b0, valid), tmm.apply_b1(b1, valid), meta,
                          extra=extra_cols)
        print(f"  fold {fold}: {time.monotonic() - t0:,.0f}s val_auc={auc:.4f} "
              f"(B1={auc_b1:.4f}) -> {pred_path.name}")

    print(f"\nnext: ..\\Python311-64\\python.exe scripts\\evaluate_touch_gate.py"
          + (f" --ft-symbol {args.ft_symbol}" if args.ft_symbol else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
