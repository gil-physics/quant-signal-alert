"""Step 16/17: train the TCN risk-model challenger (v4_tcn / v4_tcn_60m).

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Adopt nothing live:
v1.1 stays the served model until scripts/evaluate_tcn_gate.py returns GO (and the
Phase-2 serving work lands).

Step 16 (``--tcn-version v4_tcn``, default): challenger to the LightGBM v1.1 baseline.
Same rows / ``target_mae_risk`` / single date split as the champion; input = a causal
60-bar features_1m window + the v1 static features; the pre-registered 5-seed protocol
submits the MEAN of the per-seed probabilities (no seed selection).

Step 18 (``--tcn-version v5_cnn_60m``): performance-seeking plain-1D-CNN challenger
on the same 60m lineage/folds (single pre-registered seed 42; channels 64; mask-safe
triple pooling; whole-trading-day internal holdout; reuses the v4_tcn_60m frame cache).

Step 17 (``--tcn-version v4_tcn_60m``): challenger to the v1_60m champion on the
fixed-horizon ``target_mae_risk_60m`` label. Pre-registered protocol (USER-decided):
expanding walk-forward folds (risk_model.WALK_FORWARD_FOLDS_60M -- only the periods
differ between folds) x ONE seed (42) per fold; training and every reported metric use
the main population (decision time-of-day <= risk_model.HORIZON_60M_MAIN_CUTOFF, i.e.
always exactly 60 minutes of forward window), while later rows are still scored into
the parquet as the late-session DIAGNOSTIC. One invocation trains every fold; --fold k
re-runs a single fold. The time-of-day channel is KEPT (v1_60m gets the clock too).

Run (64-bit python + torch CPU wheel required):
  ..\\Python311-64\\python.exe scripts\\train_tcn_risk_model.py
  ..\\Python311-64\\python.exe scripts\\train_tcn_risk_model.py --tcn-version v4_tcn_60m
  ..\\Python311-64\\python.exe scripts\\train_tcn_risk_model.py --tcn-version v4_tcn_60m --fold 3
  smoke:    ... --tcn-version v4_tcn_60m --fold 4 --single-seed 42 --max-epochs 2 --out-suffix _v4_tcn_60m_smoke
  ablation: ... --tcn-version v4_tcn_60m --single-seed 42 --no-tod --out-suffix _v4_tcn_60m_ablC

The smoke invocation is NOT a reduced pipeline: it scores the full validation frame
and writes every artifact (model .pt/meta, predictions parquet + provenance sidecar,
summary/decile/monthly CSVs, sqlite mirrors) with the real schema, so the gate script
can be exercised end-to-end before the expensive real run.
"""

from __future__ import annotations

import argparse
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

from config import DB_PATH, PROCESSED_DIR, ensure_directories
import risk_model as rm
import tcn_data
import tcn_model as tm
from settings import load_active_symbols, load_data_config
from utils import now_kst_text

MODELS_DIR = ROOT_DIR / "models"
# v4_tcn's single-split defaults (fold versions take their dates from the fold registry
# and refuse explicit date args).
_DATE_DEFAULTS = ("2025-06-02", "2026-02-28", "2026-03-01", "2026-06-26")


def _canonical_suffix(version, fold=None):
    """The artifact suffix RESERVED for the pre-registered run of a version[/fold]."""
    return f"_{version}" if fold is None else f"_{version}_fold{int(fold)}"


def _tcn_reserved_suffixes():
    """Every canonical suffix across ALL TCN versions (incl. per-fold variants), so a
    diagnostic run of one version can never clobber another version's submission."""
    out = set()
    for v, vcfg in tm.TCN_VERSIONS.items():
        out.add(_canonical_suffix(v))
        if vcfg.get("folds"):
            out |= {_canonical_suffix(v, k)
                    for k in range(1, len(rm.WALK_FORWARD_FOLDS_60M) + 1)}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 16/17: train the TCN risk-model challenger (v4_tcn / v4_tcn_60m).")
    parser.add_argument("--tcn-version", default="v4_tcn", choices=list(tm.TCN_VERSIONS),
                        help="v4_tcn = Step 16 rest-of-day label, single split, 5-seed mean "
                             "ensemble; v4_tcn_60m = Step 17 fixed-horizon label, expanding "
                             "walk-forward folds, single pre-registered seed per fold.")
    parser.add_argument("--target", default=None,
                        choices=sorted({c["target"] for c in tm.TCN_VERSIONS.values()}),
                        help="default: the version's registry target (explicit mismatch refused).")
    parser.add_argument("--fold", type=int, default=None,
                        choices=range(1, len(rm.WALK_FORWARD_FOLDS_60M) + 1),
                        help="fold versions only: train ONE walk-forward fold (default: all).")
    parser.add_argument("--train-start", default=_DATE_DEFAULTS[0])
    parser.add_argument("--train-end", default=_DATE_DEFAULTS[1])
    parser.add_argument("--validation-start", default=_DATE_DEFAULTS[2])
    parser.add_argument("--validation-end", default=_DATE_DEFAULTS[3],
                        help="matches the shared gate frame (evaluate_v3_premarket_gate.py). "
                             "Fold versions ignore these: dates come from WALK_FORWARD_FOLDS_60M.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seed list; default = the version's pre-registered "
                             "registry seeds (v4_tcn: mean-of-5; v4_tcn_60m: single seed 42).")
    parser.add_argument("--single-seed", type=int, default=None,
                        help="train ONE seed (smoke / ablation runs). Overrides --seeds.")
    parser.add_argument("--seq-len", type=int, default=tm.SEQ_LEN)
    parser.add_argument("--channels", type=int, default=None,
                        help="conv width. Default: the version's registry pin "
                             "(tcn family 32; v5_cnn_60m 64).")
    parser.add_argument("--levels", type=int, default=None,
                        help="conv depth. Default: the version's registry pin; tcn family "
                             "falls back to the smallest level count whose receptive field "
                             "covers --seq-len (so --seq-len 120 auto-deepens).")
    parser.add_argument("--kernel-size", type=int, default=None,
                        help="conv kernel (single-kernel archs: tcn/cnn1d). Default: the "
                             "version's registry pin (tcn family 3; v5_cnn_60m 5). Refused "
                             "for the mscnn family (use --kernel-sizes).")
    parser.add_argument("--kernel-sizes", default=None,
                        help="mscnn family only: comma-separated parallel-branch kernels, "
                             "all odd. Default: the version's registry pin "
                             "(v6_mscnn_60m: 3,7,15). Refused for single-kernel archs.")
    parser.add_argument("--branch-channels", type=int, default=None,
                        help="mscnn family only: per-branch conv width before the 1x1 fuse. "
                             "Default: the version's registry pin (v6_mscnn_60m: 32).")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu",
                        help="torch device: cpu (default) or cuda / cuda:0 / cuda:1 for "
                             "GPU-server runs. Seeded per device but NOT bit-identical across "
                             "devices; recorded in every artifact's meta + provenance sidecar. "
                             "Not a protocol knob (does not affect submission status).")
    parser.add_argument("--deterministic", default="auto", choices=["auto", "on", "off"],
                        help="deterministic torch algorithms. auto (default) = on for cpu "
                             "(bit-reproducible, the pre-registered convention) and OFF for "
                             "cuda: cuDNN's deterministic dilated-conv backward kernels are "
                             "orders of magnitude slower (hours/seed instead of minutes). "
                             "cuda runs stay fully seeded either way; recorded in meta.")
    parser.add_argument("--frame-cache", default=None, metavar="DIR",
                        help="directory holding tcn_frame_<version>.parquet + "
                             "tcn_features_1m.parquet (written by --export-frame-cache). "
                             "Loaded INSTEAD of the sqlite: the GPU server needs no DB copy "
                             "and repeats no label math.")
    parser.add_argument("--export-frame-cache", default=None, metavar="DIR",
                        help="build the modelling frame + features_1m slice from --db, write "
                             "them to DIR as parquet, then EXIT without training (torch-free; "
                             "run locally, ship the two files, train remotely with "
                             "--frame-cache DIR).")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="torch intra-op threads (cpu device). Default: cpu_count-1. Pin it "
                             "to make a cpu submission bit-reproducible (CPU reduction order is "
                             "thread-count dependent); recorded in each artifact's meta.")
    parser.add_argument("--no-static", action="store_true",
                        help="ablation A: drop the static (alert + cross-symbol) head; sequence only.")
    parser.add_argument("--no-tod", action="store_true",
                        help="ablation C: drop the time-of-day channel (isolates the clock's "
                             "contribution; under the 60m label both lineage models hold the "
                             "clock, so this attributes the RESIDUAL tod value).")
    parser.add_argument("--out-suffix", default=None,
                        help="artifact suffix base (models/risk_model<sfx>_seed<k>.pt, "
                             "data/processed/risk_model_predictions<sfx>.parquet, ...); fold "
                             "versions append _fold<k> per fold. Default: the version's "
                             "canonical suffix, which is RESERVED for the pre-registered "
                             "run -- any diagnostic/smoke/ablation run must pass a distinct "
                             "base (e.g. _v4_tcn_60m_smoke) so it can never clobber it.")
    args = parser.parse_args()
    if args.frame_cache and args.export_frame_cache:
        parser.error("--frame-cache (consume) and --export-frame-cache (produce) are exclusive.")
    cfg = tm.tcn_version_config(args.tcn_version)
    if args.target is None:
        args.target = cfg["target"]
    elif args.target != cfg["target"]:
        parser.error(f"--target {args.target!r} conflicts with --tcn-version "
                     f"{args.tcn_version!r} (registry target {cfg['target']!r}); the "
                     "pairing is registry-driven to keep the lineages sealed.")
    if args.fold is not None and not cfg.get("folds"):
        parser.error(f"--fold is only valid for fold-based versions; {args.tcn_version!r} "
                     "uses a single split.")
    if cfg.get("folds"):
        got = (args.train_start, args.train_end, args.validation_start, args.validation_end)
        if got != _DATE_DEFAULTS:
            parser.error(f"{args.tcn_version!r} takes its dates from "
                         "risk_model.WALK_FORWARD_FOLDS_60M; do not pass explicit date args.")
    if cfg.get("split") is not None:
        # Step 20 production versions: the split is registry-pinned (a mistyped CLI
        # window could silently train into the sanity-check days).
        got = (args.train_start, args.train_end, args.validation_start, args.validation_end)
        if got != _DATE_DEFAULTS:
            parser.error(f"{args.tcn_version!r} dates are pinned by its registry 'split' "
                         "(risk_model.PRODUCTION_SPLIT_60M); do not pass explicit date args.")
        s = cfg["split"]
        args.train_start, args.train_end = s.train_start, s.train_end
        args.validation_start, args.validation_end = s.validation_start, s.validation_end
    if args.out_suffix is not None:
        if not args.out_suffix.startswith("_") or args.out_suffix.strip("_") == "":
            parser.error(f"--out-suffix must be non-empty and start with '_' (got "
                         f"{args.out_suffix!r}); an empty suffix would clobber the frozen "
                         "v1 artifact names.")
        # never let a TCN run overwrite a frozen LightGBM artifact family (v1/v2.x/v3.x/
        # short/v1_60m folds)
        reserved = {"_" + v.replace(".", "_") for v in rm.MODEL_VERSIONS} | {"", "_short"}
        reserved |= {"_" + v.replace(".", "_") + f"_fold{k}" for v in rm.MODEL_VERSIONS
                     for k in range(1, len(rm.WALK_FORWARD_FOLDS_60M) + 1)}
        if args.out_suffix in reserved:
            parser.error(f"--out-suffix {args.out_suffix!r} collides with a LightGBM artifact "
                         "family; choose a TCN-scoped suffix.")
    return args


def _table_exists(conn, name):
    return bool(conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()[0])


def _load_inputs(conn, model_source, symbols, seq_columns):
    """The five champion inputs (same SQL as train_risk_model.py) + features_1m bars."""
    if not _table_exists(conn, "directional_alerts"):
        raise SystemExit("directional_alerts table not found - run scripts/online_alert_backtest.py first.")

    alerts = pd.read_sql_query("SELECT * FROM directional_alerts WHERE source = ?", conn, params=(model_source,))
    features_merged = pd.read_sql_query("SELECT * FROM features_merged WHERE source = ?", conn, params=(model_source,))
    features_relative = pd.read_sql_query("SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,))
    labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
    bars = pd.read_sql_query("SELECT symbol, bar_start, low, high, close FROM bars_1m WHERE source = ?",
                             conn, params=(model_source,))
    seq_cols_sql = ", ".join(f'"{c}"' for c in seq_columns)
    features_1m = pd.read_sql_query(
        f"SELECT symbol, decision_time, {seq_cols_sql} FROM features_1m WHERE source = ?",
        conn, params=(model_source,))

    for frame in (alerts, features_merged, features_relative, labels, bars, features_1m):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return alerts, features_merged, features_relative, labels, bars, features_1m


def _summary_row(label, seed, args, tcn_config, train, valid, metrics, buckets, history,
                 fold=None, brier=None):
    top10 = buckets[buckets["risk_bucket"] == "top_10%"].iloc[0].to_dict() if len(buckets) else {}
    bottom20 = buckets[buckets["risk_bucket"] == "bottom_20%"].iloc[0].to_dict() if len(buckets) else {}
    row = {
        "created_at": now_kst_text(),
        "model_type": tm.MODEL_TYPE,
        "submission": label,
        "seed": seed,
        "target": args.target,
        "seq_len": args.seq_len,
        "seq_channels": tcn_config.in_channels,
        "static_dim": tcn_config.static_dim,
        "channels": tcn_config.channels,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_positive_rate": float(train[args.target].mean()) if len(train) else 0.0,
        "validation_positive_rate": float(valid[args.target].mean()) if len(valid) else 0.0,
        "auc": metrics["auc"],
        "pr_auc": metrics["pr_auc"],
        "logloss": metrics["logloss"],
        "precision_at_top10": metrics["precision_at_top10"],
        "recall_at_top10": metrics["recall_at_top10"],
        "top10_mae_risk_rate": top10.get("mae_risk_rate", 0.0),
        "bottom20_mae_risk_rate": bottom20.get("mae_risk_rate", 0.0),
        "holdout_auc_best": (history or {}).get("holdout_auc_best"),
        "best_epoch": (history or {}).get("best_epoch"),
        "epochs_ran": (history or {}).get("epochs_ran"),
        # Step 18 overfit watch: train-vs-internal-holdout divergence + external
        # calibration, one row per seed x fold
        "final_train_loss": (history or {}).get("final_train_loss"),
        "holdout_loss_best": (history or {}).get("holdout_loss_best"),
        "holdout_pr_auc_best": (history or {}).get("holdout_pr_auc_best"),
        "brier": brier,
    }
    if fold is not None:
        row["fold"] = fold
    return row


def _cache_paths(cache_dir, version):
    """Cache filenames. A version may alias another version's frame cache
    (registry frame_cache_alias) when their target/horizon are identical -- e.g.
    v5_cnn_60m reuses the v4_tcn_60m files already shipped to the GPU server."""
    cfg = tm.tcn_version_config(version)
    alias = cfg.get("frame_cache_alias", version)
    if alias != version:
        acfg = tm.tcn_version_config(alias)
        if (acfg["target"], acfg.get("horizon_minutes")) != \
                (cfg["target"], cfg.get("horizon_minutes")):
            raise SystemExit(f"frame_cache_alias {alias!r} is label-incompatible with "
                             f"{version!r} (target/horizon differ).")
    d = Path(cache_dir)
    return d / f"tcn_frame_{alias}.parquet", d / "tcn_features_1m.parquet"


def export_frame_cache_main(args, horizon_minutes) -> int:
    """Build the modelling frame (labels included) + the features_1m slice ONCE,
    locally, and write both as parquet -- the GPU server then trains from these two
    files (--frame-cache) with no sqlite copy and no label math. Torch-free."""
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    seq_cols = list(tcn_data.SequenceConfig().seq_feature_columns)
    print(f"[{now_kst_text()}] export frame cache version={args.tcn_version} "
          f"source={model_source} symbols={len(symbols)} horizon={horizon_minutes}")
    conn = sqlite3.connect(args.db)
    try:
        alerts, fm, rel, labels, bars, features_1m = _load_inputs(
            conn, model_source, symbols, seq_cols)
    finally:
        conn.close()
    data, _cols = rm.build_model_frame(alerts, fm, rel, labels, bars,
                                       horizon_minutes=horizon_minutes)
    frame_path, f1m_path = _cache_paths(args.export_frame_cache, args.tcn_version)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(frame_path, index=False)
    features_1m.to_parquet(f1m_path, index=False)
    for p, n in ((frame_path, len(data)), (f1m_path, len(features_1m))):
        print(f"  {p} rows={n:,} ({p.stat().st_size / 1024 / 1024:,.0f} MB)")
    print(f"\nserver: python scripts/train_tcn_risk_model.py --tcn-version {args.tcn_version} "
          "--device cuda:0 --frame-cache <DIR>")
    return 0


def load_frame_cache(args, seq_config, symbols):
    """Read the two cache parquets and re-apply the symbol filter (guards a config
    mismatch between the exporting and training machines)."""
    frame_path, f1m_path = _cache_paths(args.frame_cache, args.tcn_version)
    for p in (frame_path, f1m_path):
        if not p.exists():
            raise SystemExit(f"frame cache missing: {p} - run --export-frame-cache "
                             "locally and ship both files.")
    data = pd.read_parquet(frame_path)
    features_1m = pd.read_parquet(f1m_path)
    missing = [c for c in seq_config.seq_feature_columns if c not in features_1m.columns]
    if missing or args.target not in data.columns:
        raise SystemExit(f"frame cache is stale (missing {missing or [args.target]}) - "
                         "re-export it.")
    data = data[data["symbol"].isin(symbols)].reset_index(drop=True)
    features_1m = features_1m[features_1m["symbol"].isin(symbols)].reset_index(drop=True)
    return data, features_1m


def _default_knob_args(args) -> bool:
    """True only when EVERY knob that affects the trained weights is at its
    pre-registered default -- so a hyperparameter-shopped rerun can never be
    stamped as the submission (the reviewer's p-hacking hole). Architecture knobs
    (channels/levels/kernel) count as default when UNSET (None): each version's
    pre-registered values come from its registry pins."""
    return (args.seq_len == tm.SEQ_LEN and not args.no_static and not args.no_tod
            and args.max_epochs == 40 and args.channels is None and args.levels is None
            and args.kernel_size is None and args.kernel_sizes is None
            and args.branch_channels is None and abs(args.lr - 1e-3) < 1e-12
            and args.batch_size == 1024 and args.db == str(DB_PATH))


def _is_submission_run(args, cfg, seeds, suffix, fold) -> bool:
    if tuple(seeds) != tuple(cfg["seeds"]) or not _default_knob_args(args):
        return False
    if suffix != _canonical_suffix(args.tcn_version, fold):
        return False
    if cfg.get("folds"):
        return True   # fold dates are registry-driven; parse_args refuses explicit dates
    split = cfg.get("split")
    if split is not None:
        # Step 20 production versions: parse_args pinned the dates from the registry
        # split (and refused explicit ones), so they must match it exactly here.
        return (args.train_start, args.train_end,
                args.validation_start, args.validation_end) == (
                    split.train_start, split.train_end,
                    split.validation_start, split.validation_end)
    return (args.train_start, args.train_end,
            args.validation_start, args.validation_end) == _DATE_DEFAULTS


def run_fold(args, cfg, fold, suffix, seeds, is_submission, *, data, bar_index, seq_config,
             static_cols, tcn_config, train_config, torch, num_threads, model_source,
             device=None, deterministic=None):
    """Train/score/persist one (version, fold) run. ``fold=None`` = single-split v4_tcn."""
    horizon_minutes = cfg.get("horizon_minutes")
    split = rm.SplitConfig(train_start=args.train_start, train_end=args.train_end,
                           validation_start=args.validation_start,
                           validation_end=args.validation_end)
    train, valid = rm.split_by_date(data, split=split)
    train = train.dropna(subset=[args.target]).reset_index(drop=True)
    valid = valid.dropna(subset=[args.target]).reset_index(drop=True)
    if horizon_minutes is not None:
        # Main population (time <= cutoff) trains and drives every metric; later rows
        # (truncated-horizon label) stay in `valid` so the parquet scores them 1:1 for
        # the gate's late-session diagnostic.
        train = train[train["time"] <= rm.HORIZON_60M_MAIN_CUTOFF].reset_index(drop=True)
    if train.empty or valid.empty:
        raise SystemExit(f"empty train/validation split: train={len(train)} validation={len(valid)}")
    main_mask = (valid["time"] <= rm.HORIZON_60M_MAIN_CUTOFF if horizon_minutes is not None
                 else pd.Series(True, index=valid.index))
    valid_main = valid[main_mask].reset_index(drop=True)
    mae_col = "future_MAE_60m" if horizon_minutes is not None else "future_MAE"
    fold_tag = "" if fold is None else f" fold={fold}"
    print(f"\n[{now_kst_text()}]{fold_tag} split {args.train_start}..{args.train_end} -> "
          f"{args.validation_start}..{args.validation_end} suffix={suffix} "
          f"submission={'yes' if is_submission else 'no (diagnostic run)'}")
    print(f"  train={len(train)} val={len(valid)} val_main={len(valid_main)} "
          f"positive(train)={train[args.target].mean():.2%} "
          f"positive(val_main)={valid_main[args.target].mean():.2%}")

    seq_pre = tcn_data.SequencePreprocessor(list(seq_config.seq_feature_columns)).fit(
        bar_index.bar_mat_raw, dates=bar_index.dates, train_end=args.train_end)
    static_pre = rm.FeaturePreprocessor(static_cols).fit(train)
    n_train_bars = int((bar_index.dates <= args.train_end).sum())
    print(f"  bar_index: bars={len(bar_index.dates)} seq_pre_fit_bars={n_train_bars} "
          f"(dates <= {args.train_end})")

    seed_train_probs, seed_valid_probs, histories = {}, {}, {}
    for seed in seeds:
        t0 = time.monotonic()
        state, history = tm.fit_tcn(train, bar_index, seq_pre, static_pre, args.target,
                                    seq_config=seq_config, tcn_config=tcn_config,
                                    train_config=train_config, seed=seed, device=device,
                                    deterministic=deterministic)
        model_path = MODELS_DIR / f"risk_model{suffix}_seed{seed}.pt"
        tm.save_tcn(model_path, state, seq_pre=seq_pre, static_pre=static_pre,
                    seq_config=seq_config, tcn_config=tcn_config, train_config=train_config,
                    history=history, metadata={
                        "target": args.target,
                        "model_source": model_source,
                        "model_version": args.tcn_version,
                        "fold": fold,
                        "horizon_minutes": horizon_minutes,
                        "main_cutoff": (rm.HORIZON_60M_MAIN_CUTOFF
                                        if horizon_minutes is not None else None),
                        "out_suffix": suffix,
                        "submission_run": bool(is_submission),
                        "requires_real_relative_features": True,
                        "seq_feature_columns": list(seq_config.seq_feature_columns),
                        "static_feature_columns": static_cols,
                        "levels": tcn_config.levels,
                        "kernel_size": tcn_config.kernel_size,
                        "kernel_sizes": (list(tcn_config.kernel_sizes)
                                         if tcn_config.kernel_sizes else None),
                        "branch_channels": tcn_config.branch_channels,
                        "num_threads": int(num_threads),
                        "device": str(device) if device is not None else "cpu",
                        "deterministic": history.get("deterministic"),
                        "train_start": args.train_start,
                        "train_end": args.train_end,
                        "validation_start": args.validation_start,
                        "validation_end": args.validation_end,
                        "train_rows": len(train),
                        "validation_rows": len(valid),
                    })
        seed_train_probs[seed] = tm.predict_tcn(state, train, bar_index, seq_pre, static_pre,
                                                seq_config=seq_config, tcn_config=tcn_config,
                                                device=device)
        seed_valid_probs[seed] = tm.predict_tcn(state, valid, bar_index, seq_pre, static_pre,
                                                seq_config=seq_config, tcn_config=tcn_config,
                                                device=device)
        histories[seed] = history
        val_auc = rm.roc_auc_score_np(
            valid_main[args.target].astype(int).to_numpy(),
            seed_valid_probs[seed][main_mask.to_numpy()])
        print(f"  seed {seed}: {time.monotonic() - t0:,.0f}s epochs={history['epochs_ran']} "
              f"best_epoch={history['best_epoch']} holdout_auc={history['holdout_auc_best']} "
              f"val_main_auc={val_auc:.4f} -> {model_path.name}")

    train = train.copy()
    valid = valid.copy()
    train["risk_prob"] = tm.ensemble_probs([seed_train_probs[s] for s in seeds])
    valid["risk_prob"] = tm.ensemble_probs([seed_valid_probs[s] for s in seeds])
    for seed in seeds:
        train[f"risk_prob_seed{seed}"] = seed_train_probs[seed]
        valid[f"risk_prob_seed{seed}"] = seed_valid_probs[seed]
    train["split"] = "train"
    valid["split"] = "validation"
    valid_main = valid[main_mask.to_numpy()].reset_index(drop=True)

    y_main = valid_main[args.target].astype(float).to_numpy()

    def _brier(prob_col):
        return float(((valid_main[prob_col].to_numpy(dtype=float) - y_main) ** 2).mean())

    summary_rows = []
    for seed in seeds:
        m = rm.classification_summary(valid_main, target_col=args.target,
                                      prob_col=f"risk_prob_seed{seed}")
        b = rm.bucket_report(valid_main, target_col=args.target,
                             prob_col=f"risk_prob_seed{seed}", mae_col=mae_col)
        summary_rows.append(_summary_row("seed", seed, args, tcn_config, train, valid_main,
                                         m, b, histories[seed], fold=fold,
                                         brier=_brier(f"risk_prob_seed{seed}")))
    metrics = rm.classification_summary(valid_main, target_col=args.target)
    deciles = rm.bucket_report(valid_main, target_col=args.target, mae_col=mae_col)
    summary_rows.append(_summary_row("ensemble", "mean_of_seeds", args, tcn_config,
                                     train, valid_main, metrics, deciles, None, fold=fold,
                                     brier=_brier("risk_prob")))
    summary = pd.DataFrame(summary_rows)

    predictions_cols = [
        "split", "symbol", "decision_time", "date", "time", "source",
        "alert_type", "alert_grade", "risk_prob",
    ] + [f"risk_prob_seed{s}" for s in seeds] + [
        "target_mae_risk", "target_up_fail", "target_down2_60m", "target_up_risk",
        "future_MAE", "future_return_60m", "future_MFE",
        # Step 17 60m lineage (existence-filtered: absent on v4_tcn)
        "target_mae_risk_60m", "target_up_risk_60m", "future_MAE_60m", "future_MFE_60m",
    ]
    predictions = pd.concat([train, valid], ignore_index=True)
    predictions = predictions[[c for c in predictions_cols if c in predictions.columns]]

    vsfx = suffix
    summary_path = PROCESSED_DIR / f"risk_model_summary{vsfx}.csv"
    decile_path = PROCESSED_DIR / f"risk_prob_decile_report{vsfx}.csv"
    predictions_path = PROCESSED_DIR / f"risk_model_predictions{vsfx}.parquet"
    monthly_path = PROCESSED_DIR / f"risk_prob_monthly_auc{vsfx}.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    deciles.to_csv(decile_path, index=False, encoding="utf-8-sig")
    predictions.to_parquet(predictions_path, index=False)

    # Provenance sidecar keyed to THIS parquet (the gate reads it from
    # <predictions_path>.meta.json, not from the default model artifacts) so a
    # tuned/diagnostic run can never borrow the submission's provenance. The frame
    # fingerprint lets the gate detect a features_merged/relative rebuild between
    # training and gating (keys+labels alone can't); it covers the FULL validation
    # frame (main + late rows) -- the gate fingerprints before its cutoff filter.
    import json

    frame_fp = tm.frame_fingerprint(valid, list(rm.RELATIVE_FEATURE_COLUMNS)
                                    + list(rm.ALERT_FEATURE_COLUMNS))
    predictions_meta = {
        "model_version": args.tcn_version,
        "fold": fold,
        "horizon_minutes": horizon_minutes,
        "main_cutoff": rm.HORIZON_60M_MAIN_CUTOFF if horizon_minutes is not None else None,
        "out_suffix": suffix,
        "submission_run": bool(is_submission),
        "seeds": [int(s) for s in seeds],
        "submission_seeds": [int(s) for s in cfg["seeds"]],
        "seq_len": seq_config.seq_len,
        "n_seq_channels": seq_config.n_channels,
        "levels": tcn_config.levels,
        "kernel_size": tcn_config.kernel_size,
        "kernel_sizes": (list(tcn_config.kernel_sizes)
                         if tcn_config.kernel_sizes else None),
        "branch_channels": tcn_config.branch_channels,
        "channels": tcn_config.channels,
        "static_dim": tcn_config.static_dim,
        "arch": tcn_config.arch,
        "holdout_by_day": bool(cfg.get("holdout_by_day", False)),
        "num_threads": int(num_threads),
        "device": str(device) if device is not None else "cpu",
        "deterministic": args.deterministic,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "train_start": args.train_start, "train_end": args.train_end,
        "validation_start": args.validation_start, "validation_end": args.validation_end,
        "validation_rows": int(len(valid)),
        "validation_frame_fingerprint": frame_fp,
        "target": args.target,
    }
    Path(str(predictions_path) + ".meta.json").write_text(
        json.dumps(predictions_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    conn = sqlite3.connect(args.db)
    try:
        predictions.to_sql(f"risk_model_predictions{vsfx}", conn, if_exists="replace", index=False)
        summary.to_sql(f"risk_model_summary{vsfx}", conn, if_exists="replace", index=False)
        deciles.to_sql(f"risk_prob_decile_report{vsfx}", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print("\nvalidation metrics (per seed + ensemble; main population):")
    print(summary[["submission", "seed", "auc", "pr_auc", "logloss", "brier",
                   "top10_mae_risk_rate", "bottom20_mae_risk_rate", "holdout_auc_best",
                   "holdout_pr_auc_best", "holdout_loss_best", "final_train_loss",
                   "best_epoch", "epochs_ran"]].to_string(index=False))
    print("\nrisk_prob buckets (ensemble):")
    print(deciles.to_string(index=False))
    monthly = rm.monthly_auc(valid_main, target_col=args.target)
    if len(monthly):
        monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        print("\nvalidation AUC by month (ensemble; regime-decay check):")
        print(monthly.to_string(index=False))
    if horizon_minutes is not None:
        late = valid[~main_mask.to_numpy()]
        if len(late):
            late_auc = rm.roc_auc_score_np(late[args.target].astype(int).to_numpy(),
                                           late["risk_prob"].to_numpy(dtype=float))
            print(f"\nlate-session diagnostic (time > {rm.HORIZON_60M_MAIN_CUTOFF}; "
                  "truncated-horizon label, excluded from every metric above):")
            print(f"  n={len(late)} positive_rate={late[args.target].mean():.2%} "
                  f"auc={late_auc:.4f}")
    print(f"\npredictions -> db risk_model_predictions{vsfx} + {predictions_path}")
    print(f"summary -> {summary_path}")
    print(f"deciles -> {decile_path}")
    print(f"models -> {MODELS_DIR / ('risk_model' + vsfx + '_seed<k>.pt')}")


def main() -> int:
    args = parse_args()
    cfg = tm.tcn_version_config(args.tcn_version)
    horizon_minutes = cfg.get("horizon_minutes")
    if args.export_frame_cache:
        return export_frame_cache_main(args, horizon_minutes)   # torch-free, no training

    torch = tm._require_torch()
    import os

    device = tm.resolve_device(args.device)   # fail fast before any data load
    deterministic = None if args.deterministic == "auto" else (args.deterministic == "on")
    det_resolved = (device.type == "cpu") if deterministic is None else deterministic
    num_threads = args.num_threads if args.num_threads else max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(int(num_threads))
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [args.single_seed] if args.single_seed is not None else (
        [int(s) for s in str(args.seeds).split(",") if s.strip()]
        if args.seeds else list(cfg["seeds"]))

    model_source = load_data_config().model_source
    symbols = load_active_symbols()

    seq_config = tcn_data.SequenceConfig(seq_len=args.seq_len, add_tod_channel=not args.no_tod)
    static_cols = [] if args.no_static else list(cfg["static_features"])
    # architecture knobs resolve per version: explicit flag > registry pin > family default
    arch = cfg.get("arch", "tcn")
    channels = args.channels if args.channels is not None else int(cfg.get("channels", 32))
    if arch == "mscnn":
        # the kernel knob families are mutually exclusive so a mis-aimed flag can never
        # be silently ignored (it would also void submission status via _default_knob_args)
        if args.kernel_size is not None:
            raise SystemExit("--kernel-size is a single-kernel (tcn/cnn1d) knob; the mscnn "
                             "family takes --kernel-sizes (comma-separated odd ints).")
        kernel_size = None
        kernel_sizes = (tuple(int(k) for k in str(args.kernel_sizes).split(",") if k.strip())
                        if args.kernel_sizes is not None else tuple(cfg["kernel_sizes"]))
        if not kernel_sizes or any(k % 2 != 1 for k in kernel_sizes):
            raise SystemExit(f"--kernel-sizes needs a non-empty odd-int list "
                             f"(got {kernel_sizes}); even kernels break same-length padding.")
        branch_channels = (args.branch_channels if args.branch_channels is not None
                           else int(cfg["branch_channels"]))
    else:
        if args.kernel_sizes is not None or args.branch_channels is not None:
            raise SystemExit("--kernel-sizes/--branch-channels are mscnn-family knobs; "
                             f"{args.tcn_version} (arch={arch}) takes --kernel-size.")
        kernel_size = (args.kernel_size if args.kernel_size is not None
                       else int(cfg.get("kernel_size", 3)))
        kernel_sizes = None
        branch_channels = None
    levels = args.levels if args.levels is not None else cfg.get("levels")
    if levels is None:
        # tcn family: the smallest stack whose receptive field covers seq_len, so a
        # longer --seq-len is actually seen by the output (ablB fix) instead of ignored.
        levels = tm.required_levels(seq_config.seq_len, kernel_size)
    tcn_config = tm.TCNConfig(in_channels=seq_config.n_channels, channels=channels,
                              levels=int(levels), kernel_size=kernel_size,
                              static_dim=len(static_cols), arch=arch,
                              kernel_sizes=kernel_sizes, branch_channels=branch_channels)
    tm._check_receptive_field(tcn_config, seq_config)   # tcn arch only: RF must cover seq_len
    train_config = tm.TrainConfig(batch_size=args.batch_size, max_epochs=args.max_epochs,
                                  lr=args.lr,
                                  holdout_by_day=bool(cfg.get("holdout_by_day", False)))

    # plan every (fold, suffix, is_submission) up-front so the reserved-suffix guard
    # fires BEFORE any expensive work. The canonical suffixes are RESERVED for the
    # exact pre-registered run of their version[/fold]: a diagnostic run must not be
    # able to overwrite a submission's parquet/weights (critical review finding).
    folds = ([args.fold] if args.fold is not None
             else list(range(1, len(rm.WALK_FORWARD_FOLDS_60M) + 1))) \
        if cfg.get("folds") else [None]
    reserved_tcn = _tcn_reserved_suffixes()
    plan = []
    for fold in folds:
        base = args.out_suffix if args.out_suffix is not None else f"_{args.tcn_version}"
        suffix = base if fold is None else f"{base}_fold{fold}"
        is_submission = _is_submission_run(args, cfg, seeds, suffix, fold)
        if suffix in reserved_tcn and not is_submission:
            raise SystemExit(
                f"the suffix {suffix!r} is reserved for the pre-registered run of its "
                f"version (registry seeds={tuple(cfg['seeds'])}, all knobs at default). "
                f"This run (version={args.tcn_version}, seeds={tuple(seeds)}, "
                f"seq_len={args.seq_len}, static={not args.no_static}, tod={not args.no_tod}, "
                f"epochs={args.max_epochs}, channels={channels}, lr={args.lr}) differs "
                "-> pass a distinct --out-suffix, e.g. _v4_tcn_60m_smoke.")
        plan.append((fold, suffix, is_submission))

    rf = (tm.receptive_field(tcn_config.levels, tcn_config.kernel_size)
          if arch == "tcn" else "n/a(global-pool)")
    kernel_desc = (kernel_size if arch != "mscnn" else
                   "(" + ",".join(str(k) for k in kernel_sizes) + f")x{branch_channels}br")
    print(f"[{now_kst_text()}] train {args.tcn_version} challenger target={args.target} "
          f"source={model_source} symbols={len(symbols)} seeds={seeds} "
          f"folds={[f for f, _, _ in plan]} arch={arch} seq_len={args.seq_len} "
          f"channels={channels} levels={tcn_config.levels} kernel={kernel_desc} rf={rf} "
          f"static={len(static_cols)} tod={not args.no_tod} "
          f"holdout_by_day={train_config.holdout_by_day} "
          f"device={device} deterministic={det_resolved} torch={torch.__version__} "
          f"threads={torch.get_num_threads()}")

    if args.frame_cache:
        # pre-built locally by --export-frame-cache: no sqlite, no label math here
        data, features_1m = load_frame_cache(args, seq_config, symbols)
        print(f"  frame cache <- {args.frame_cache}: model_frame={len(data):,} "
              f"features_1m_bars={len(features_1m):,}")
    else:
        conn = sqlite3.connect(args.db)
        try:
            alerts, fm, rel, labels, bars, features_1m = _load_inputs(
                conn, model_source, symbols, seq_config.seq_feature_columns)
        finally:
            conn.close()
        print(f"  inputs: alerts={len(alerts)} features_merged={len(fm)} features_relative={len(rel)} "
              f"labels={len(labels)} features_1m_bars={len(features_1m)}")

        # identical modelling frame to the champion (v1 defaults; horizon_minutes adds the
        # 60m label columns for the 60m lineage) -> identical rows/labels; built ONCE and
        # sliced per fold (labels do not depend on the split).
        data, _cols = rm.build_model_frame(alerts, fm, rel, labels, bars,
                                           horizon_minutes=horizon_minutes)
        print(f"  model_frame={len(data)}")
    if static_cols:
        rm.available_feature_columns(data, static_cols)   # leakage-token guard
    bar_index = tcn_data.build_bar_index(features_1m, symbols=symbols, seq_config=seq_config)

    for fold, suffix, is_submission in plan:
        if fold is not None:
            fsplit = rm.walk_forward_fold(fold)
            args.train_start, args.train_end = fsplit.train_start, fsplit.train_end
            args.validation_start = fsplit.validation_start
            args.validation_end = fsplit.validation_end
        run_fold(args, cfg, fold, suffix, seeds, is_submission,
                 data=data, bar_index=bar_index, seq_config=seq_config,
                 static_cols=static_cols, tcn_config=tcn_config, train_config=train_config,
                 torch=torch, num_threads=num_threads, model_source=model_source,
                 device=device, deterministic=deterministic)

    print("\n(no feature_importance artifact: per-feature importance is not meaningful "
          "for the TCN; see the ablation rows in the gate report instead)")
    print(f"\nnext: ..\\Python311-64\\python.exe scripts\\evaluate_tcn_gate.py"
          + ("" if args.tcn_version == "v4_tcn" else f" --tcn-version {args.tcn_version}")
          + ("" if args.out_suffix is None else
             " --ablation-predictions <name>=<predictions parquet name>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
