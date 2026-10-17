"""Step 16: TCN risk-model challenger (v4_tcn) -- training/predict/artifact orchestration.

Research-only challenger to the LightGBM v1.1 long risk model: same rows, same
``target_mae_risk`` label, same date split; input = a causal 60-bar features_1m
window + the v1 static (alert + cross-symbol) features. Adoption is decided by the
same GO/NO-GO gate as v2.x/v3.0 (scripts/evaluate_tcn_gate.py); until a GO verdict
nothing here is wired to live serving.

House rules honored:
  * risk_model.py stays byte-identical -- v4_tcn lives in its OWN registry here
    (the MODEL_VERSIONS schema is tabular/LightGBM-shaped and load_model's artifact
    dispatch would misfire on a torch file).
  * torch is imported ONLY inside functions (same lazy pattern as lightgbm/sklearn),
    so this module imports fine on the 32-bit / dependency-free runtimes.
  * Pre-registered multi-seed protocol: SUBMISSION_SEEDS is fixed ahead of any
    validation scoring and the submitted score is the arithmetic MEAN of the
    per-seed probabilities -- no seed selection of any kind.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import risk_model as rm
from tcn_data import (
    SEQ_FEATURE_COLUMNS_V4,
    STATIC_FEATURE_COLUMNS_V4,
    SequenceConfig,
    SequencePreprocessor,
    attach_anchor_positions,
    gather_windows,
    iter_batch_indices,
)

MODEL_TYPE = "tcn_torch"
TCN_VERSION = "v4_tcn"
SEQ_LEN = 60

# Pre-registered BEFORE any validation scoring: the submission score is the mean of
# these seeds' probabilities. Changing this tuple after seeing validation results
# would be seed-picking bias; tests pin it.
SUBMISSION_SEEDS = (42, 43, 44, 45, 46)

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

TCN_VERSIONS = {
    "v4_tcn": {
        "seq_features": list(SEQ_FEATURE_COLUMNS_V4),
        "static_features": list(STATIC_FEATURE_COLUMNS_V4),
        "seq_len": SEQ_LEN,
        "seeds": SUBMISSION_SEEDS,
        "filename_pattern": "risk_model_v4_tcn_seed{seed}.pt",
        "requires_cross_symbol": True,   # challenger targets the v1.1 serving config
        "target": "target_mae_risk",
    },
    # v4_tcn_60m (Step 17): the fixed-horizon challenger vs the v1_60m LightGBM
    # champion. Same architecture/window/static head as v4_tcn -- the time-of-day
    # channel is KEPT on purpose: v1_60m also receives the clock (minutes_since_open),
    # so the gate measures architecture value on top of clock access. Pre-registered
    # protocol (USER-decided 2026-07-12): ONE seed (42) per expanding walk-forward fold
    # (rm.WALK_FORWARD_FOLDS_60M); a 5-seed rerun only if later justified. "folds": True
    # makes the trainer loop the folds and the gate stitch the four OOS months.
    "v4_tcn_60m": {
        "seq_features": list(SEQ_FEATURE_COLUMNS_V4),
        "static_features": list(STATIC_FEATURE_COLUMNS_V4),
        "seq_len": SEQ_LEN,
        "seeds": (42,),
        "filename_pattern": "risk_model_v4_tcn_60m_fold{fold}_seed{seed}.pt",
        "requires_cross_symbol": True,
        "target": "target_mae_risk_60m",
        "horizon_minutes": 60,
        "folds": True,
    },
    # v5_cnn_60m (Step 18): performance-seeking 1D-CNN challenger on the same 60m
    # lineage -- plain conv stack (channels 64, 4 layers, k=5; NOT parameter-matched
    # to the TCN) + mask-safe triple pooling (see tcn_net.CNN1dRiskNet). Same folds,
    # same features/tod, pre-registered SINGLE seed 42; 5-seed only if the escalation
    # criteria in the README pass. holdout_by_day: early stopping holds out whole
    # trading days. frame_cache_alias: identical target/horizon -> the server reuses
    # the v4_tcn_60m frame cache verbatim. Serving (if ever adopted) runs on the
    # local CPU -- ~80k params, single-row CPU inference is ms-scale.
    "v5_cnn_60m": {
        "seq_features": list(SEQ_FEATURE_COLUMNS_V4),
        "static_features": list(STATIC_FEATURE_COLUMNS_V4),
        "seq_len": SEQ_LEN,
        "seeds": (42,),
        "filename_pattern": "risk_model_v5_cnn_60m_fold{fold}_seed{seed}.pt",
        "requires_cross_symbol": True,
        "target": "target_mae_risk_60m",
        "horizon_minutes": 60,
        "folds": True,
        "arch": "cnn1d",
        "channels": 64,
        "levels": 4,
        "kernel_size": 5,
        "holdout_by_day": True,
        "frame_cache_alias": "v4_tcn_60m",
    },
    # v6_mscnn_60m (Step 19): multi-scale CNN tie-breaker. v5 tied the TCN (prob corr
    # ~0.97, 50/50 blend added nothing), so the next performance search changes the
    # ONE untested ingredient: parallel kernel scales per stage (3/7/15-bar branches
    # x 32ch -> 1x1 fuse to 64, residual stages after the stem; ~204k params, still
    # ms-scale single-row CPU inference). Everything else -- folds, label, features,
    # tod, day-holdout, pooling/head, single pre-registered seed 42, frame cache --
    # is v5 verbatim, so the gate attributes any delta to the multi-scale trunk.
    # NOTE: no "kernel_size" (singular) key -- the trainer refuses --kernel-size for
    # this family and records kernel_sizes/branch_channels instead.
    "v6_mscnn_60m": {
        "seq_features": list(SEQ_FEATURE_COLUMNS_V4),
        "static_features": list(STATIC_FEATURE_COLUMNS_V4),
        "seq_len": SEQ_LEN,
        "seeds": (42,),
        "filename_pattern": "risk_model_v6_mscnn_60m_fold{fold}_seed{seed}.pt",
        "requires_cross_symbol": True,
        "target": "target_mae_risk_60m",
        "horizon_minutes": 60,
        "folds": True,
        "arch": "mscnn",
        "channels": 64,
        "levels": 4,
        "kernel_sizes": (3, 7, 15),
        "branch_channels": 32,
        "holdout_by_day": True,
        "frame_cache_alias": "v4_tcn_60m",
    },
    # Step 20 (USER-decided 2026-07-16): PRODUCTION candidates -- the two adoption
    # candidates from Steps 17/18, retrained on rm.PRODUCTION_SPLIT_60M (through June,
    # first-July-days sanity window; "split" pins the dates, the CLI may not move them).
    # Both use the pre-registered 5-seed mean (the production-robustness rule -- a
    # single seed's regime luck must not pick the deployable weights). Selection
    # between them is scripts/select_prod_60m.py's pre-registered rule.
    "v4_tcn_60m_prod": {
        "seq_features": list(SEQ_FEATURE_COLUMNS_V4),
        "static_features": list(STATIC_FEATURE_COLUMNS_V4),
        "seq_len": SEQ_LEN,
        "seeds": SUBMISSION_SEEDS,
        "filename_pattern": "risk_model_v4_tcn_60m_prod_seed{seed}.pt",
        "requires_cross_symbol": True,
        "target": "target_mae_risk_60m",
        "horizon_minutes": 60,
        "split": rm.PRODUCTION_SPLIT_60M,
        "frame_cache_alias": "v4_tcn_60m",
    },
    "v5_cnn_60m_prod": {
        "seq_features": list(SEQ_FEATURE_COLUMNS_V4),
        "static_features": list(STATIC_FEATURE_COLUMNS_V4),
        "seq_len": SEQ_LEN,
        "seeds": SUBMISSION_SEEDS,
        "filename_pattern": "risk_model_v5_cnn_60m_prod_seed{seed}.pt",
        "requires_cross_symbol": True,
        "target": "target_mae_risk_60m",
        "horizon_minutes": 60,
        "split": rm.PRODUCTION_SPLIT_60M,
        "arch": "cnn1d",
        "channels": 64,
        "levels": 4,
        "kernel_size": 5,
        "holdout_by_day": True,
        "frame_cache_alias": "v4_tcn_60m",
    },
}


def tcn_version_config(version):
    if version not in TCN_VERSIONS:
        raise ValueError(f"unknown TCN model version {version!r}; choices={list(TCN_VERSIONS)}")
    return TCN_VERSIONS[version]


def resolve_tcn_model_path(version, seed, models_dir=None, *, fold=None):
    """Canonical .pt artifact for a (version, seed[, fold]); sidecar = <path>.meta.json.
    Fold-based versions (pattern contains ``{fold}``) require ``fold=``."""
    pattern = tcn_version_config(version)["filename_pattern"]
    if "{fold}" in pattern and fold is None:
        raise ValueError(f"TCN version {version!r} is fold-based; pass fold=1..N")
    return Path(models_dir or _MODELS_DIR) / pattern.format(
        seed=int(seed), fold=(fold if fold is None else int(fold)))


def ensemble_probs(per_seed_probs) -> np.ndarray:
    """The pre-registered submission rule: arithmetic mean of per-seed probabilities."""
    stacked = np.vstack([np.asarray(p, dtype=np.float64) for p in per_seed_probs])
    return stacked.mean(axis=0)


def assert_ensemble_consistency(frame, seed_cols, *, prob_col="risk_prob", atol=1e-9):
    """Enforce (not just document) the pre-registered rule at gate time: the gated
    ``prob_col`` must equal the arithmetic mean of the per-seed columns. Raises
    ValueError otherwise -- a parquet whose risk_prob is a best-seed pick or any
    post-hoc blend must never reach decide_verdict."""
    if not seed_cols:
        raise ValueError("no per-seed probability columns found; cannot verify the "
                         "mean-of-seeds submission rule")
    expected = frame[list(seed_cols)].to_numpy(dtype=np.float64).mean(axis=1)
    actual = frame[prob_col].to_numpy(dtype=np.float64)
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        worst = float(np.max(np.abs(actual - expected)))
        raise ValueError(
            f"{prob_col} is NOT the mean of {list(seed_cols)} (max abs diff {worst:.2e}); "
            "the pre-registered mean-of-seeds rule is violated")


def receptive_field(levels: int, kernel_size: int) -> int:
    """Receptive field of the Bai-style stack (mirrors TCNRiskNet.receptive_field,
    duplicated here so the torch-free path can validate configs)."""
    return 1 + 2 * (int(kernel_size) - 1) * (2 ** int(levels) - 1)


def required_levels(seq_len: int, kernel_size: int, *, min_levels: int = 4) -> int:
    """Smallest level count whose receptive field covers ``seq_len`` (>= min_levels).
    Without this, a longer --seq-len silently feeds bars the last-timestep output
    can never see (e.g. seq_len=120 with 4 levels: RF=61)."""
    levels = int(min_levels)
    while receptive_field(levels, kernel_size) < int(seq_len):
        levels += 1
    return levels


def _check_receptive_field(tcn_config, seq_config):
    if getattr(tcn_config, "arch", "tcn") != "tcn":
        return   # global-pooling archs (cnn1d/mscnn) see the full window by construction
    rf = receptive_field(tcn_config.levels, tcn_config.kernel_size)
    if rf < seq_config.seq_len:
        raise ValueError(
            f"receptive field {rf} (levels={tcn_config.levels}, k={tcn_config.kernel_size}) "
            f"< seq_len {seq_config.seq_len}: the oldest bars could never influence the "
            f"output; use levels >= {required_levels(seq_config.seq_len, tcn_config.kernel_size)}")


def frame_fingerprint(frame, columns) -> str:
    """Order-sensitive sha256 over the frame's rounded feature values (+ shape).

    Written into the predictions sidecar at train time and recomputed by the gate:
    if features_merged/features_relative were rebuilt between training and gating,
    the champion would be rescored on a different feature snapshot than the one the
    challenger's probabilities embed -- keys/labels can't detect that, values can."""
    import hashlib

    cols = [c for c in columns if c in frame.columns]
    x = frame[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(np.round(x, 6), nan=1e12, posinf=2e12, neginf=-2e12)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(",".join(cols).encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class TCNConfig:
    in_channels: int = SequenceConfig().n_channels   # 13 features + tod + mask = 15
    channels: int = 32
    levels: int = 4
    kernel_size: int = 3
    dropout: float = 0.10
    static_dim: int = len(STATIC_FEATURE_COLUMNS_V4)  # 13; 0 = ablation A (no static head)
    head_hidden: int = 64
    # Step 18: network family -- "tcn" (dilated causal, default: every pre-Step-18
    # artifact loads with it), "cnn1d" (plain conv + mask-safe pooling) or "mscnn"
    # (Step 19: parallel-kernel multi-scale stages).
    arch: str = "tcn"
    # Step 19: mscnn-family knobs. None on every other arch AND on every pre-Step-19
    # artifact (whose meta lacks the keys -> these defaults). JSON round-trips the
    # tuple as a list; consumers normalize with tuple(...).
    kernel_sizes: tuple = None
    branch_channels: int = None


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 1024
    max_epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 6
    holdout_frac: float = 0.15       # rm._time_ordered_holdout convention (train tail)
    lr_factor: float = 0.5
    lr_patience: int = 2
    min_lr: float = 1e-5
    predict_batch_size: int = 4096
    # Step 18: hold out whole TRADING DAYS from the train tail instead of raw rows
    # (early stopping never sees the morning of a day whose afternoon it trains on).
    # Default False = the exact pre-Step-18 behavior, so v4 protocols are unchanged.
    holdout_by_day: bool = False


def _day_boundary_holdout(frame, fraction):
    """Chronological tail holdout at TRADING-DAY granularity (Step 18).

    Whole calendar days are held out -- tail days accumulate until the holdout
    reaches ``fraction`` of the rows -- so early stopping never scores the morning
    of a day whose afternoon it trained on (the row-level split can cut mid-day).
    Mirrors rm._time_ordered_holdout's contract: (fit_pos, eval_pos) positional
    indices, or (None, None) when the split would be degenerate.
    """
    if not fraction or len(frame) < 400:
        return None, None
    dates = frame["date"].astype(str).to_numpy()
    target = float(fraction) * len(frame)
    cum = 0
    cut_dates = set()
    for d in sorted(set(dates), reverse=True):
        cut_dates.add(d)
        cum += int((dates == d).sum())
        if cum >= target:
            break
    eval_mask = np.isin(dates, sorted(cut_dates))
    fit_pos = np.nonzero(~eval_mask)[0]
    eval_pos = np.nonzero(eval_mask)[0]
    if len(fit_pos) == 0 or len(eval_pos) == 0:
        return None, None
    return fit_pos, eval_pos


def _weighted_bce_from_probs(y, probs, pos_weight):
    """The BCEWithLogitsLoss(pos_weight) value recomputed from probabilities --
    the holdout-loss series in the history uses the SAME loss the optimizer sees."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(-(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the v4_tcn challenger but is not installed in this "
            "(64-bit) runtime. Install the CPU wheel:\n"
            "  ..\\Python311-64\\python.exe -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from exc


def resolve_device(device=None):
    """'cpu' (default) / 'cuda' / 'cuda:0' / 'cuda:1' -> validated torch.device.

    Fails fast with a clear message when CUDA is requested but absent, instead of
    torch's lazy error surfacing mid-epoch. Training on GPU is numerically close but
    NOT bit-identical to the CPU runs (different reduction orders), so the device is
    recorded in every artifact's metadata/provenance."""
    torch = _require_torch()
    dev = torch.device(str(device) if device else "cpu")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False -- "
                "install a CUDA torch wheel on the GPU server (pip install torch) "
                "or pass --device cpu.")
        n = torch.cuda.device_count()
        if dev.index is not None and dev.index >= n:
            raise RuntimeError(f"cuda:{dev.index} requested but only {n} CUDA "
                               f"device(s) are visible.")
        # torch.use_deterministic_algorithms(True) requires this for CUDA matmuls;
        # must be set before the first cuBLAS call.
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return dev


def _build_net(tcn_config: TCNConfig):
    _require_torch()
    arch = getattr(tcn_config, "arch", "tcn")
    kwargs = dict(
        in_channels=tcn_config.in_channels,
        channels=tcn_config.channels,
        levels=tcn_config.levels,
        kernel_size=tcn_config.kernel_size,
        dropout=tcn_config.dropout,
        static_dim=tcn_config.static_dim,
        head_hidden=tcn_config.head_hidden,
    )
    if arch == "tcn":
        from tcn_net import TCNRiskNet
        return TCNRiskNet(**kwargs)
    if arch == "cnn1d":
        from tcn_net import CNN1dRiskNet
        return CNN1dRiskNet(**kwargs)
    if arch == "mscnn":
        from tcn_net import MultiScaleCNNRiskNet
        kwargs.pop("kernel_size")   # single-kernel knob; this family uses the ladder
        ks = getattr(tcn_config, "kernel_sizes", None)
        bc = getattr(tcn_config, "branch_channels", None)
        return MultiScaleCNNRiskNet(kernel_sizes=tuple(ks) if ks else (3, 7, 15),
                                    branch_channels=int(bc) if bc else 32, **kwargs)
    raise ValueError(f"unknown arch {arch!r}; choices=('tcn', 'cnn1d', 'mscnn')")


def _forward_probs(model, torch, indices, *, bar_mat_std, bar_tod, j, day_start,
                   static_x, seq_config, batch_size, device=None):
    """Sigmoid probabilities for the given row indices, chunked, order-preserving."""
    dev = device if device is not None else torch.device("cpu")
    out = np.empty(len(indices), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            sel = indices[start:start + int(batch_size)]
            seq = gather_windows(bar_mat_std, bar_tod, j[sel], day_start[sel],
                                 seq_config=seq_config)
            x_seq = torch.from_numpy(seq).permute(0, 2, 1).to(dev)
            x_static = (torch.from_numpy(static_x[sel]).to(dev)
                        if static_x is not None else None)
            logits = model(x_seq, x_static)
            out[start:start + len(sel)] = torch.sigmoid(logits).cpu().numpy()
    return out


def fit_sequence_model(train_frame, bar_index, seq_pre, static_pre, target_col, *,
                       seq_config: SequenceConfig = SequenceConfig(),
                       tcn_config: TCNConfig = TCNConfig(),
                       train_config: TrainConfig = TrainConfig(),
                       seed: int = 42, device=None, deterministic=None,
                       init_state=None, eval_frame=None, lr_schedule=None):
    """Train one seed on the TRAIN frame only and return (state_dict, history).

    Canonical name (the trainer dispatches TCN / CNN1d / multi-scale CNN through
    ``tcn_config.arch``, so the function is architecture-generic; the historical
    ``fit_tcn`` alias below is kept for the frozen risk-model lineages).

    Fine-tuning support (touch study; both default None so every risk-model lineage
    is untouched): ``init_state`` warm-starts the net from an existing state_dict;
    ``eval_frame`` replaces the INTERNAL train-tail holdout with an explicit
    early-stopping frame -- the caller guarantees it was excluded from every prior
    training (the pre-registered FT-tail block), so passing it here is the only
    place those rows are ever seen.

    The signature deliberately does not accept the validation frame: early stopping
    uses rm._time_ordered_holdout (the chronological tail of train), so validation
    metrics computed by the caller stay leakage-free. The best-holdout-AUC weights
    are returned (final weights when the frame is too small for a holdout), always
    moved to CPU so artifacts stay portable across devices.

    Fixed-epoch full-data refit (touch prod stage-2; default None so every existing
    lineage is untouched): pass ``train_config.holdout_frac=0`` (no holdout -> no
    early stopping, every epoch's weights are kept, the run ends at max_epochs) and
    ``lr_schedule`` = the per-epoch lr values recorded by the stage-1 run's history
    -- the refit replays that exact lr trajectory (the plateau scheduler cannot run
    without a holdout) instead of holding the initial lr flat.

    ``device``: None/'cpu' (default) or 'cuda'/'cuda:0'/'cuda:1' (GPU-server runs).
    ``deterministic``: None = device default -- True on cpu (bit-reproducible, the
    pre-registered convention), False on cuda: deterministic mode forces cuDNN's
    deterministic dilated-conv backward kernels, which are ORDERS OF MAGNITUDE
    slower (observed ~hours/seed vs minutes). Non-deterministic cuda runs are still
    fully SEEDED -- only kernel-level reduction order varies run-to-run -- and the
    flag is recorded in the history/metadata.
    """
    torch = _require_torch()
    dev = resolve_device(device)
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    det = (dev.type == "cpu") if deterministic is None else bool(deterministic)
    torch.use_deterministic_algorithms(det)
    if dev.type == "cuda" and not det:
        # fixed shapes (batch_size constant except the tail) -> autotuned kernels win
        torch.backends.cudnn.benchmark = True

    j, day_start = attach_anchor_positions(train_frame, bar_index)
    bar_mat_std = seq_pre.transform_matrix(bar_index.bar_mat_raw)
    static_x = None
    if tcn_config.static_dim:
        static_x = static_pre.transform(train_frame).astype(np.float32)
        if static_x.shape[1] != tcn_config.static_dim:
            raise ValueError(f"static matrix has {static_x.shape[1]} columns, "
                             f"config expects {tcn_config.static_dim}")
    y = train_frame[target_col].astype(int).to_numpy()

    eval_ext = None
    if eval_frame is not None:
        # explicit external ES frame (FT path): fit on ALL train rows, evaluate on
        # the caller-reserved frame (its own anchors/static transform).
        fit_pos = np.arange(len(train_frame))
        eval_pos = None
        if len(eval_frame):
            j_e, day_start_e = attach_anchor_positions(eval_frame, bar_index)
            static_e = (static_pre.transform(eval_frame).astype(np.float32)
                        if tcn_config.static_dim else None)
            eval_ext = (j_e, day_start_e, static_e,
                        eval_frame[target_col].astype(int).to_numpy())
    elif train_config.holdout_by_day:
        fit_pos, eval_pos = _day_boundary_holdout(train_frame, train_config.holdout_frac)
    else:
        fit_pos, eval_pos = rm._time_ordered_holdout(train_frame, train_config.holdout_frac)
    if fit_pos is None:
        fit_pos = np.arange(len(train_frame))
        eval_pos = None

    y_fit = y[fit_pos]
    pos = int(y_fit.sum())
    neg = int(len(y_fit) - pos)
    pos_weight_val = (neg / pos) if pos > 0 and neg > 0 else 1.0

    model = _build_net(tcn_config).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)      # warm start (FT path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr,
                                  weight_decay=train_config.weight_decay)
    scheduler = None
    if lr_schedule is None and (eval_pos is not None or eval_ext is not None):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=train_config.lr_factor,
            patience=train_config.lr_patience, min_lr=train_config.min_lr)
    if lr_schedule is not None:
        lr_schedule = [float(v) for v in lr_schedule]
        if not lr_schedule:
            raise ValueError("lr_schedule must hold at least one per-epoch lr value")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        pos_weight_val, dtype=torch.float32, device=dev))
    y_t = torch.from_numpy(y.astype(np.float32))

    rng = np.random.default_rng(int(seed))
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_log = []

    for epoch in range(1, int(train_config.max_epochs) + 1):
        if lr_schedule is not None:     # replay the stage-1 lr trajectory (prod refit)
            replay_lr = lr_schedule[min(epoch - 1, len(lr_schedule) - 1)]
            for group in optimizer.param_groups:
                group["lr"] = replay_lr
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for batch in iter_batch_indices(len(fit_pos), train_config.batch_size, rng=rng):
            sel = fit_pos[batch]
            seq = gather_windows(bar_mat_std, bar_tod=bar_index.bar_tod,
                                 j=j[sel], day_start=day_start[sel], seq_config=seq_config)
            x_seq = torch.from_numpy(seq).permute(0, 2, 1).to(dev)
            x_static = (torch.from_numpy(static_x[sel]).to(dev)
                        if static_x is not None else None)
            optimizer.zero_grad()
            logits = model(x_seq, x_static)
            loss = loss_fn(logits, y_t[sel].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(sel)
            n_seen += len(sel)
        train_loss = loss_sum / max(n_seen, 1)

        holdout_auc = float("nan")
        holdout_loss = float("nan")
        holdout_pr_auc = float("nan")
        y_eval = None
        if eval_ext is not None:
            j_e, day_start_e, static_e, y_eval = eval_ext
            probs = _forward_probs(model, torch, np.arange(len(y_eval)),
                                   bar_mat_std=bar_mat_std, bar_tod=bar_index.bar_tod,
                                   j=j_e, day_start=day_start_e, static_x=static_e,
                                   seq_config=seq_config,
                                   batch_size=train_config.predict_batch_size,
                                   device=dev)
        elif eval_pos is not None:
            y_eval = y[eval_pos]
            probs = _forward_probs(model, torch, eval_pos,
                                   bar_mat_std=bar_mat_std, bar_tod=bar_index.bar_tod,
                                   j=j, day_start=day_start, static_x=static_x,
                                   seq_config=seq_config,
                                   batch_size=train_config.predict_batch_size,
                                   device=dev)
        if y_eval is not None:
            holdout_auc = rm.roc_auc_score_np(y_eval, probs)
            # overfit watch (Step 18): the optimizer's own loss + PR-AUC on the holdout
            holdout_loss = _weighted_bce_from_probs(y_eval, probs, pos_weight_val)
            holdout_pr_auc = rm.average_precision_np(y_eval, probs)
            if scheduler is not None and holdout_auc == holdout_auc:
                scheduler.step(holdout_auc)

        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "holdout_auc": round(holdout_auc, 6) if holdout_auc == holdout_auc else None,
            "holdout_loss": round(holdout_loss, 6) if holdout_loss == holdout_loss else None,
            "holdout_pr_auc": (round(holdout_pr_auc, 6)
                               if holdout_pr_auc == holdout_pr_auc else None),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if eval_pos is None and eval_ext is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            continue
        if holdout_auc == holdout_auc and holdout_auc > best_auc:
            best_auc = holdout_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_config.es_patience):
                break

    if best_state is None:   # e.g. single-class holdout: keep the final weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch_log[-1]["epoch"] if epoch_log else 0

    best_entry = next((e for e in epoch_log if e["epoch"] == best_epoch), {})
    history = {
        "seed": int(seed),
        "epochs": epoch_log,
        "epochs_ran": len(epoch_log),
        "best_epoch": int(best_epoch),
        "holdout_auc_best": float(best_auc) if best_auc == best_auc and best_auc != -np.inf else None,
        # overfit watch (Step 18): the best-epoch snapshot + final train loss let a
        # reviewer read train-vs-holdout divergence without replaying the run
        "holdout_loss_best": best_entry.get("holdout_loss"),
        "holdout_pr_auc_best": best_entry.get("holdout_pr_auc"),
        "final_train_loss": epoch_log[-1]["train_loss"] if epoch_log else None,
        "holdout_by_day": ("external" if eval_ext is not None
                           else bool(train_config.holdout_by_day)),
        "lr_replayed": lr_schedule is not None,
        "holdout_rows": (int(len(eval_ext[3])) if eval_ext is not None
                         else int(len(eval_pos)) if eval_pos is not None else 0),
        "fit_rows": int(len(fit_pos)),
        "pos_weight": float(pos_weight_val),
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "device": str(dev),
        "deterministic": bool(det),
    }
    return best_state, history


# Historical alias: the Step 16-20 risk-model lineage scripts/tests call fit_tcn;
# the implementation is architecture-generic (arch dispatch in _build_net), so the
# canonical name is fit_sequence_model. Both refer to the SAME function object.
fit_tcn = fit_sequence_model


def predict_sequence_model(state_dict, frame, bar_index, seq_pre, static_pre, *,
                           seq_config: SequenceConfig = SequenceConfig(),
                           tcn_config: TCNConfig = TCNConfig(),
                           batch_size: int = 4096, device=None) -> np.ndarray:
    """Score a frame 1:1 (row-order preserving; padded rows still get a score).
    Canonical architecture-generic name; ``predict_tcn`` below is the historical
    alias kept for the frozen risk-model lineages."""
    torch = _require_torch()
    dev = resolve_device(device)
    j, day_start = attach_anchor_positions(frame, bar_index)
    bar_mat_std = seq_pre.transform_matrix(bar_index.bar_mat_raw)
    static_x = None
    if tcn_config.static_dim:
        static_x = static_pre.transform(frame).astype(np.float32)
    model = _build_net(tcn_config)
    model.load_state_dict(state_dict)
    model.to(dev)
    return _forward_probs(model, torch, np.arange(len(frame)),
                          bar_mat_std=bar_mat_std, bar_tod=bar_index.bar_tod,
                          j=j, day_start=day_start, static_x=static_x,
                          seq_config=seq_config, batch_size=batch_size, device=dev)


predict_tcn = predict_sequence_model   # historical alias (see fit_tcn note above)


def _seq_config_to_dict(seq_config: SequenceConfig):
    d = asdict(seq_config)
    d["seq_feature_columns"] = list(d["seq_feature_columns"])
    return d


def _seq_config_from_dict(payload):
    return SequenceConfig(
        seq_len=int(payload.get("seq_len", SEQ_LEN)),
        seq_feature_columns=tuple(payload.get("seq_feature_columns", SEQ_FEATURE_COLUMNS_V4)),
        add_tod_channel=bool(payload.get("add_tod_channel", True)),
        add_mask_channel=bool(payload.get("add_mask_channel", True)),
    )


def save_tcn(path, state_dict, *, seq_pre, static_pre, seq_config, tcn_config,
             train_config, history=None, metadata=None):
    """Write <path>.pt (state_dict) + <path>.pt.meta.json (house sidecar shape)."""
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(path))
    meta = {
        "model_type": MODEL_TYPE,
        "preprocessor": static_pre.to_dict(),
        "sequence_preprocessor": seq_pre.to_dict(),
        "seq_config": _seq_config_to_dict(seq_config),
        "tcn_config": asdict(tcn_config),
        "train_config": asdict(train_config),
        "history": history or {},
        "metadata": metadata or {},
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_tcn(path):
    """Reload one seed artifact -> dict with state_dict/preprocessors/configs/meta."""
    torch = _require_torch()
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"no TCN artifact at {path} (+ .meta.json)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model_type") != MODEL_TYPE:
        raise ValueError(f"{meta_path} is not a {MODEL_TYPE} artifact "
                         f"(model_type={meta.get('model_type')!r})")
    return {
        "state_dict": torch.load(str(path), map_location="cpu", weights_only=True),
        "static_pre": rm.FeaturePreprocessor.from_dict(meta.get("preprocessor", {})),
        "seq_pre": SequencePreprocessor.from_dict(meta.get("sequence_preprocessor", {})),
        "seq_config": _seq_config_from_dict(meta.get("seq_config", {})),
        "tcn_config": TCNConfig(**meta.get("tcn_config", {})),
        "train_config": TrainConfig(**meta.get("train_config", {})),
        "history": meta.get("history", {}),
        "metadata": meta.get("metadata", {}),
    }


def load_tcn_ensemble(version=TCN_VERSION, models_dir=None, *, fold=None):
    """Load every pre-registered seed artifact of a version (Phase-2 serving path).
    Fold-based versions load ONE fold's seed set (pass fold=)."""
    cfg = tcn_version_config(version)
    return [load_tcn(resolve_tcn_model_path(version, seed, models_dir, fold=fold))
            for seed in cfg["seeds"]]
