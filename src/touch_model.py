"""Touch-prediction study: MA5x20 cross -> +0.5% close-touch within 10 minutes.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Building blocks for
scripts/train_touch_model.py and scripts/evaluate_touch_gate.py; nothing here touches
serving or the risk-model lineages.

Pre-registered design (USER-decided 2026-07-17; plan v3):
  * Event = carry-variant close-MA MA5x20 crossover CONFIRMED in the regular session
    (premarket-confirmed crosses are excluded -- features_1m exists only for regular
    bars -- and counted). Label = P2r: entry open[t+1], timestamp window
    [t+1, t+1+10min), threshold +0.5% directional close-touch (TOUCH_THRESHOLD;
    chosen after a positives survey: 16,787 vs 9,130 at 0.7% -- the 0.7% touch is
    kept as an auxiliary diagnostic label). Incomplete windows are dropped.
  * Features join at decision_time = the confirmation bar's bar_end (never later);
    the sequence window ends at bar t; the label window starts at t+1 -> no overlap.
  * FIXED feature lists (no existence filtering): RELATIVE_FEATURES_V1 (exactly 8),
    STATIC_FEATURES_V1 (5 base + 8 relative + one symbol one-hot per active symbol),
    B2_FEATURES_V1 (static + the enumerated merged list). Joins assert coverage.
  * Baselines: B0 = train pooled touch rate per direction; B1 = train lookup table
    by (band x direction x early_flag); B2 champion = LightGBM on B2_FEATURES_V1.
  * TOUCH_FOLDS: five expanding walk-forward folds (val 2026-03..07; the July fold is
    a PARTIAL month and is confirmation-only, never used to select the architecture).
  * 000660 fine-tuning safety (user-picked option): the FT-tail block = the last
    FT_TAIL_DAYS trading days of each fold's train range, restricted to the FT
    symbol; those rows are EXCLUDED from every pooled training (champion included)
    and used exclusively as the FT early-stopping holdout.

Step 22 additions (USER-decided 2026-07-20, post-gate: winner=CNN GO, FT not
adopted): the production cut (TOUCH_PRODUCTION_SPLIT + touch_cnn_prod, 5-seed
two-stage refit), the frozen OOF grade map (build_grade_map/apply_grade -- built
ONLY from the 5-seed walk-forward OOF ensemble predictions, never from the prod
model's own training predictions), and build_touch_setups (the label-free
detection half of build_touch_events, shared with the live runner's parity check).
The +0.7% auxiliary label stays offline-only: the grade map and every new serving
surface use the +0.5% label exclusively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import risk_model as rm
from ma_cross_study import (
    PREMARKET_START,
    add_mas_and_atr,
    build_outcome_stream,
    classify_bands,
    detect_crossovers,
    directional_outcomes,
    mark_cluster_first,
    per_bar_forward_stats,
    prepare_variant,
)

# ---- pre-registered label constants -------------------------------------------------
TOUCH_THRESHOLD = 0.005          # primary label (+0.5% directional close-touch)
TOUCH_THRESHOLD_AUX = 0.007      # auxiliary diagnostic label
TOUCH_WINDOW_MIN = 10
TOUCH_TARGET = "target_touch_05"
TOUCH_TARGET_AUX = "target_touch_07"
# The feature stack (features_1m/merged/relative) only starts after a ~60-bar
# intraday warmup (first rows ~10:01), so the MODELLING population additionally
# requires feature coverage (filter_feature_covered) -- a TIME-based, label-blind
# exclusion. The carry variant's 09:00-09:20 recovered signals therefore stay in the
# raw statistics but are OUT OF SCOPE for the models until a shorter-warmup feature
# build exists. early_flag marks the first covered half-hour instead.
EARLY_SESSION_END = "10:30:00"
FT_TAIL_DAYS = 15                # per-fold FT early-stop block (trading days)
FT_MIN_POSITIVES = 20            # below this the fold's FT is skipped (reported)

# ---- pre-registered folds (fold 5 = partial July month, confirmation-only) ----------
TOUCH_FOLDS = (
    rm.SplitConfig(train_start="2025-06-02", train_end="2026-02-28",
                   validation_start="2026-03-01", validation_end="2026-03-31"),
    rm.SplitConfig(train_start="2025-06-02", train_end="2026-03-31",
                   validation_start="2026-04-01", validation_end="2026-04-30"),
    rm.SplitConfig(train_start="2025-06-02", train_end="2026-04-30",
                   validation_start="2026-05-01", validation_end="2026-05-31"),
    rm.SplitConfig(train_start="2025-06-02", train_end="2026-05-31",
                   validation_start="2026-06-01", validation_end="2026-06-30"),
    rm.SplitConfig(train_start="2025-06-02", train_end="2026-06-30",
                   validation_start="2026-07-01", validation_end="2026-07-31"),
)
SELECTION_FOLDS = (1, 2, 3, 4)        # architecture winner is picked here (2026-03..06)
CONFIRMATION_FOLD = 5                 # fully-future month; confirmation only
RECENT_MONTHS = ("2026-05", "2026-06")

# ---- production cut (Step 22) --------------------------------------------------------
# Seeds shared by the 5-seed OOF rerun (grade-map source) and the prod ensemble, so
# the frozen grade thresholds live in the same score space the serving ensemble emits.
TOUCH_SUBMISSION_SEEDS = (42, 43, 44, 45, 46)
# train_end=None = "the latest label-complete trading day at run time" (the touch
# window closes intraday, so every fully-collected day is label-complete); the trainer
# resolves it from the event population and records the resolved date in every
# sidecar. NO forward validation window (user decision: train through the latest
# date); the per-seed sanity metric is stage-1's day-boundary holdout AUC.
TOUCH_PRODUCTION_SPLIT = rm.SplitConfig(train_start="2025-06-02", train_end=None,
                                        validation_start=None, validation_end=None)


def touch_fold(fold):
    if not 1 <= int(fold) <= len(TOUCH_FOLDS):
        raise ValueError(f"fold must be 1..{len(TOUCH_FOLDS)}, got {fold!r}")
    return TOUCH_FOLDS[int(fold) - 1]


# ---- FIXED feature lists (user modification #2: explicit, no existence filter) ------
RELATIVE_FEATURES_V1 = (
    "active_symbol_count",
    "rank_ret_5m_1m",
    "rank_ret_20m_1m",
    "rank_volume_z_20m_1m",
    "universe_mean_ret_20m",
    "relative_ret_20m_to_universe",
    "relative_volume_z_to_universe",
    "universe_breadth",
)
STATIC_BASE_FEATURES_V1 = (
    "direction", "dist_pct", "dist_atr", "minutes_since_open", "early_flag",
)
# explicit frozen copy of rm.MERGED_FEATURE_COLUMNS + rm.OPTIONAL_DERIVED_COLUMNS;
# a test asserts equality so upstream drift is caught instead of silently absorbed.
MERGED_FEATURES_V1 = (
    "ret_1m_1m", "ret_3m_1m", "ret_5m_1m", "ret_10m_1m", "ret_20m_1m", "ret_60m_1m",
    "range_1m", "body_1m", "upper_shadow_1m", "lower_shadow_1m",
    "volume_z_20m_1m", "volume_z_60m_1m", "vwap_distance_1m", "return_from_open_1m",
    "distance_from_day_high_1m", "distance_from_day_low_1m",
    "ma5_daily_n", "ma5_intraday_n", "dist_ma5_daily", "dist_ma5_intraday",
    "ret_1bar_3m", "ret_2bar_3m", "ret_3bar_3m", "range_1bar_3m", "body_1bar_3m",
    "volume_z_10bar_3m", "volume_z_20bar_3m",
    "ret_1bar_5m", "ret_2bar_5m", "ret_3bar_5m", "range_1bar_5m", "body_1bar_5m",
    "volume_z_10bar_5m", "volume_z_20bar_5m",
    "above_ma5_intraday", "above_ma5_daily",
)
SYMBOL_ONEHOT_PREFIX = "sym_"
# these two MERGED_FEATURES_V1 entries are NOT features_merged table columns -- they
# are derived from dist_ma5_* after the join (same convention as risk_model.py's
# merge_alert_features / realtime_features._add_above_ma5).
DERIVED_MERGED_FLAGS = ("above_ma5_intraday", "above_ma5_daily")


def symbol_onehot_columns(symbols):
    """Deterministic one-hot column names from the ACTIVE symbol list (codes are
    configuration, never literals in code)."""
    return [f"{SYMBOL_ONEHOT_PREFIX}{s}" for s in sorted(symbols)]


def static_features_v1(symbols):
    """5 base + 8 relative + one-hot -- shared by the challengers' static head AND
    the champion (symbol identity is symmetric on purpose)."""
    return list(STATIC_BASE_FEATURES_V1) + list(RELATIVE_FEATURES_V1) \
        + symbol_onehot_columns(symbols)


def b2_features_v1(symbols):
    return static_features_v1(symbols) + list(MERGED_FEATURES_V1)


# ---- pre-registered model registry (input contract fully explicit) ------------------
# Pipeline-level contract facts shared by both sequence challengers (informative,
# asserted complete by tests; the per-version dicts pin every trainable knob).
TOUCH_INPUT_CONTRACT = {
    "seq_len": 60,
    "seq_channels": "SEQ_FEATURE_COLUMNS_V4 (13) + time-of-day + pad-mask = 15",
    "normalization_seq": "train_end-fitted per-channel z-score (SequencePreprocessor)",
    "normalization_static": "train-fitted median-impute + z-score (FeaturePreprocessor)",
    "padding_value": "0 after standardization",
    "padding_mask": "channel 15, valid=1",
    "missing_seq": "absent bar -> pad + mask 0",
    "missing_static": "NaN -> train median",
    "class_weight": "BCE pos_weight = neg/pos of the fit split",
    "label": f"{TOUCH_TARGET} (+{TOUCH_THRESHOLD:.3%} close-touch, {TOUCH_WINDOW_MIN}m, P2r)",
}
_SEQ_COMMON = {
    "kind": "sequence",
    "target": TOUCH_TARGET,
    "seq_len": 60,
    "seeds": (42,),
    "folds": True,
    "batch_size": 1024,
    "max_epochs": 40,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "es_patience": 6,
    "holdout_frac": 0.15,
    "holdout_by_day": True,
    "static_features": "STATIC_FEATURES_V1",
}
TOUCH_VERSIONS = {
    # B2 champion: LightGBM on the fixed tabular list (dist/regime/tod/symbol).
    "touch_lgbm": {
        "kind": "lgbm",
        "target": TOUCH_TARGET,
        "features": "B2_FEATURES_V1",
        "seeds": (42,),
        "folds": True,
        "filename_pattern": "touch_model_lgbm_fold{fold}.txt",
    },
    # CNN challenger: the Step 18/20 v5 architecture verbatim.
    "touch_cnn": {
        **_SEQ_COMMON,
        "arch": "cnn1d",
        "channels": 64,
        "levels": 4,
        "kernel_size": 5,
        "pooling": "masked triple (max + mean + last-valid)",
        "filename_pattern": "touch_model_cnn_fold{fold}_seed{seed}.pt",
    },
    # TCN challenger: the Step 16/17 v4 architecture verbatim (dilation 2^l, RF 61).
    "touch_tcn": {
        **_SEQ_COMMON,
        "arch": "tcn",
        "channels": 32,
        "levels": 4,
        "kernel_size": 3,
        "receptive_field": 61,
        "filename_pattern": "touch_model_tcn_fold{fold}_seed{seed}.pt",
    },
    # Production cut (Step 22): the gate winner's knobs FROZEN verbatim (a test pins
    # every shared knob against touch_cnn), 5-seed ensemble, single registry-pinned
    # split instead of folds, and the user-decided two-stage refit: stage-1 trains
    # with the day-boundary-holdout early stopping ONLY to pick best_epoch, stage-2
    # retrains from scratch on train+holdout for exactly that many epochs (replaying
    # stage-1's lr schedule) and only the stage-2 weights are saved.
    "touch_cnn_prod": {
        **_SEQ_COMMON,
        "arch": "cnn1d",
        "channels": 64,
        "levels": 4,
        "kernel_size": 5,
        "pooling": "masked triple (max + mean + last-valid)",
        "seeds": TOUCH_SUBMISSION_SEEDS,
        "folds": False,
        "split": TOUCH_PRODUCTION_SPLIT,
        "two_stage_refit": True,
        "filename_pattern": "touch_model_cnn_prod_seed{seed}.pt",
    },
}


def touch_version_config(version):
    if version not in TOUCH_VERSIONS:
        raise ValueError(f"unknown touch-model version {version!r}; "
                         f"choices={list(TOUCH_VERSIONS)}")
    return TOUCH_VERSIONS[version]


# ---- event/label construction --------------------------------------------------------
def build_touch_setups(bars_raw):
    """The label-free half of build_touch_events: carry-variant cross detection +
    setup features, WITHOUT the P2r outcome/labels (which need future bars).

    Shared by the offline population builder and the live runner / parity check, so
    the detection semantics can never fork. Returns (setups, stats): one row per
    regular-session-confirmed cross with symbol/date/time/bar_start/decision_time/
    month, direction, ma_fast/ma_slow/atr/open/close, dist_pct/dist_atr/band,
    early_flag/minutes_since_open/cluster_first; stats counts the eligible and the
    premarket-excluded crosses."""
    vf, gcols = prepare_variant(bars_raw, "carry")
    mf = add_mas_and_atr(vf, gcols, price_col="close")
    cf = detect_crossovers(mf, gcols, signal_start=PREMARKET_START)
    crosses = cf[(cf["cross"] != 0) & cf["eligible"]]
    n_premarket = int((crosses["session"] == "premarket").sum())
    ev = crosses[crosses["session"] == "regular"][
        ["symbol", "date", "time", "bar_start", "cross", "ma_fast", "ma_slow", "atr",
         "open", "close"]].reset_index(drop=True)
    ev = ev.rename(columns={"cross": "direction"})
    open_t = ev["open"].astype(float)
    ev["dist_pct"] = (open_t - ev["ma_fast"]).abs() / ev["ma_fast"]
    ev["dist_atr"] = (open_t - ev["ma_fast"]).abs() / ev["atr"]
    ev["band"] = classify_bands(ev["dist_pct"])
    ev["early_flag"] = (ev["time"] <= EARLY_SESSION_END).astype(float)
    ev["month"] = ev["date"].str.slice(0, 7)
    # decision_time = the confirmation bar's bar_end (features_1m anchor key)
    ev["decision_time"] = (pd.to_datetime(ev["bar_start"])
                           + pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M:%S")
    s = ev["decision_time"]
    ev["minutes_since_open"] = (s.str.slice(11, 13).astype(int) * 60
                                + s.str.slice(14, 16).astype(int) - 540)
    ev = mark_cluster_first(ev)
    stats = {"n_crosses_eligible": int(len(crosses)),
             "n_premarket_excluded": n_premarket}
    return ev, stats


def build_touch_events(bars_raw, *, window=TOUCH_WINDOW_MIN,
                       thresholds=(TOUCH_THRESHOLD, TOUCH_THRESHOLD_AUX)):
    """The modelling population from raw al_raw bars.

    Returns (events, stats): one row per carry-variant close-MA cross confirmed in
    the regular session with a COMPLETE P2r outcome window; columns = identifiers
    (symbol/date/time/bar_start/decision_time/month/session), setup features
    (direction/dist_pct/dist_atr/band/early_flag/minutes_since_open/cluster_first),
    the labels (target_touch_05 / target_touch_07) and diagnostics (w10_mae/
    w10_endpoint_ret/entry). stats counts the excluded populations."""
    stream = build_outcome_stream(bars_raw)
    per_bar = per_bar_forward_stats(stream, (int(window),))
    ev, setup_stats = build_touch_setups(bars_raw)

    # P2r outcome anchor = the NEXT bar in the outcome stream (entry open[t+1])
    ev = ev.merge(per_bar[["symbol", "bar_start", "pos"]],
                  on=["symbol", "bar_start"], how="left", validate="m:1")
    prim = [c for c in per_bar.columns
            if c.startswith(("fwd_", "endpoint_", "complete_"))]
    nxt = per_bar[["symbol", "date", "pos", "open"] + prim].copy()
    nxt["pos"] = nxt["pos"] - 1
    ev = ev.merge(nxt, on=["symbol", "date", "pos"], how="left",
                  validate="m:1", suffixes=("", "_entrybar"))
    ev["entry"] = ev["open_entrybar"].astype(float)
    ev[f"complete_w{window}"] = ev[f"complete_w{window}"].fillna(False).astype(bool)
    n_incomplete = int((~ev[f"complete_w{window}"]).sum())
    ev = ev[ev[f"complete_w{window}"] & ev["entry"].notna()].reset_index(drop=True)

    anchor = ev[["entry"] + prim].rename(columns={"entry": "open"})
    d = ev["direction"].to_numpy(dtype=float)
    for thr, name in ((thresholds[0], TOUCH_TARGET), (thresholds[1], TOUCH_TARGET_AUX)):
        out = directional_outcomes(anchor, d, (int(window),), threshold=thr)
        ev[name] = out[f"w{window}_close_touch"].astype(int)
        if name == TOUCH_TARGET:
            ev["w10_mae"] = out[f"w{window}_mae"]
            ev["w10_endpoint_ret"] = out[f"w{window}_endpoint_ret"]
    keep = ["symbol", "date", "time", "bar_start", "decision_time", "month",
            "direction", "dist_pct", "dist_atr", "band", "early_flag",
            "minutes_since_open", "cluster_first", "entry",
            TOUCH_TARGET, TOUCH_TARGET_AUX, "w10_mae", "w10_endpoint_ret"]
    stats = {**setup_stats,
             "n_incomplete_dropped": n_incomplete,
             "n_events": int(len(ev))}
    return ev[keep], stats


def filter_feature_covered(events, features_merged):
    """Drop events whose confirmation decision_time has no feature-stack row (the
    ~60-bar intraday warmup: features_1m starts ~10:01). Time-driven and label-blind
    -- purely a coverage constraint of the existing feature build. Returns
    (covered_events, n_dropped)."""
    keys = pd.MultiIndex.from_frame(features_merged[["symbol", "decision_time"]])
    ev_keys = pd.MultiIndex.from_frame(events[["symbol", "decision_time"]])
    covered = ev_keys.isin(keys)
    return events[covered].reset_index(drop=True), int((~covered).sum())


def join_event_features(events, features_merged, features_relative, symbols):
    """Fixed-list feature join at decision_time (match-rate contract >= 99.9%,
    mirroring the sequence pipeline's anchor contract) + symbol one-hots."""
    out = events.copy()
    merged_table_cols = tuple(c for c in MERGED_FEATURES_V1
                              if c not in DERIVED_MERGED_FLAGS)
    for frame, cols, tag in ((features_merged, merged_table_cols, "features_merged"),
                             (features_relative, RELATIVE_FEATURES_V1,
                              "features_relative")):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{tag} is missing fixed-list columns: {missing}")
        before = len(out)
        out = out.merge(frame[["symbol", "decision_time"] + list(cols)],
                        on=["symbol", "decision_time"], how="left",
                        validate="m:1")
        match = out[list(cols)[0]].notna().mean() if before else 1.0
        if match < 0.999:
            raise ValueError(f"{tag} join match rate {match:.4%} < 99.9% -- "
                             "feature snapshot does not cover the event population")
    # derived flags (risk_model.py convention: NaN dist -> 0.0)
    out["above_ma5_intraday"] = (out["dist_ma5_intraday"] > 0).astype(float)
    out["above_ma5_daily"] = (out["dist_ma5_daily"] > 0).astype(float)
    for col in symbol_onehot_columns(symbols):
        out[col] = (out["symbol"] == col[len(SYMBOL_ONEHOT_PREFIX):]).astype(float)
    return out


# ---- baselines -----------------------------------------------------------------------
def fit_b0(train, target=TOUCH_TARGET):
    """Constant per-direction touch rate (the user-defined all-crosses baseline)."""
    rates = train.groupby("direction")[target].mean().to_dict()
    overall = float(train[target].mean()) if len(train) else 0.0
    return {"rates": {int(k): float(v) for k, v in rates.items()},
            "overall": overall}


def apply_b0(model, frame):
    return frame["direction"].map(model["rates"]).fillna(model["overall"]) \
        .to_numpy(dtype=float)


def fit_b1(train, target=TOUCH_TARGET):
    """Lookup touch rate by (band x direction x early_flag), B0 fallback."""
    key = ["band", "direction", "early_flag"]
    table = train.groupby(key, observed=True)[target].mean()
    return {"table": {tuple(k): float(v) for k, v in table.items()},
            "b0": fit_b0(train, target)}


def apply_b1(model, frame):
    keys = list(zip(frame["band"], frame["direction"], frame["early_flag"]))
    fallback = apply_b0(model["b0"], frame)
    table = model["table"]
    return np.array([table.get(k, f) for k, f in zip(keys, fallback)], dtype=float)


# ---- fine-tune safety (user modification #3) ------------------------------------------
def ft_tail_dates(events, split, *, n_days=FT_TAIL_DAYS):
    """The last ``n_days`` TRADING days of the fold's train range (market days from
    the event population). These dates x the FT symbol are excluded from every
    pooled training and reserved for the FT early-stopping holdout."""
    days = np.sort(events.loc[(events["date"] >= split.train_start)
                              & (events["date"] <= split.train_end), "date"].unique())
    return set(days[-int(n_days):]) if len(days) else set()


def split_ft_rows(train, ft_symbol, tail_dates):
    """(pooled_train_without_tail, ft_es_holdout) -- the tail rows of the FT symbol
    leave the pooled train entirely (champion and challengers alike)."""
    if not ft_symbol or not tail_dates:
        return train, train.iloc[0:0]
    is_tail = (train["symbol"] == ft_symbol) & train["date"].isin(tail_dates)
    return (train[~is_tail].reset_index(drop=True),
            train[is_tail].reset_index(drop=True))


# ---- frozen OOF grade map (Step 22) ---------------------------------------------------
# User directive: the serving display is "raw prod score -> grade -> that grade's
# HISTORICAL OOF touch rate", never a calibrated probability. The map is built ONLY
# from the pooled 5-seed walk-forward OOF ensemble predictions (same seeds as the
# prod ensemble, so thresholds live in the serving score space) and then frozen.
GRADE_BOUNDARIES = (("A", 0.90), ("B", 0.70), ("C", 0.20), ("D", 0.0))
GRADE_MAP_NOTE = (
    "scores are class-weighted and UNCALIBRATED (ranks only); a grade maps a raw "
    "score to its pooled walk-forward OOF percentile bucket and reports that "
    "bucket's HISTORICAL +0.5% touch rate -- never quote the score as a probability")


def build_grade_map(oof, *, boundaries=GRADE_BOUNDARIES, score_col="prob",
                    target_col=TOUCH_TARGET):
    """Grade-map artifact from the pooled OOF frame (needs score_col, target_col and
    ``month``). Thresholds are raw-score percentiles of the pooled OOF distribution;
    each grade records its OOF touch rate + N (+ per-month breakdown, +0.5% ONLY --
    the 0.7% auxiliary label is deliberately absent from every serving surface).
    Hard-fails unless the per-grade touch rates are monotone non-increasing from the
    best grade down (pre-registered check) and every grade is non-empty."""
    if oof is None or not len(oof):
        raise ValueError("empty OOF frame")
    names = [g for g, _ in boundaries]
    pcts = [float(p) for _, p in boundaries]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate grade names in {names}")
    if pcts != sorted(pcts, reverse=True):
        raise ValueError("boundaries must be ordered best-first (descending percentile)")
    scores = oof[score_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite OOF scores -- refusing to build a grade map")
    y = oof[target_col].astype(int).to_numpy()
    thresholds = {g: float(np.quantile(scores, p)) for g, p in boundaries}
    partial = {"boundaries": [[g, p] for g, p in boundaries], "thresholds": thresholds}
    grades = apply_grade(scores, partial)
    month = oof["month"].astype(str).to_numpy()
    grade_stats = {}
    for g in names:
        m = grades == g
        n = int(m.sum())
        if n == 0:
            raise ValueError(f"grade {g!r} is empty -- degenerate OOF score "
                             "distribution (duplicate percentile thresholds?)")
        monthly = {}
        for mo in sorted(set(month[m])):
            mm = m & (month == mo)
            monthly[mo] = {"n": int(mm.sum()), "touch_rate": float(y[mm].mean())}
        grade_stats[g] = {"n": n, "touch_rate": float(y[m].mean()), "monthly": monthly}
    rates = [grade_stats[g]["touch_rate"] for g in names]
    if any(hi < lo for hi, lo in zip(rates, rates[1:])):
        raise ValueError(
            "grade touch rates are not monotone non-increasing: "
            f"{ {g: round(grade_stats[g]['touch_rate'], 4) for g in names} } -- "
            "the OOF ranking does not support this grade scheme")
    return {
        "boundaries": [[g, p] for g, p in boundaries],
        "thresholds": thresholds,
        "grades": grade_stats,
        "n_rows": int(len(oof)),
        "base_rate": float(y.mean()),
        "score_col": score_col,
        "target": target_col,
        "touch_threshold": TOUCH_THRESHOLD,
        "window_min": TOUCH_WINDOW_MIN,
        "note": GRADE_MAP_NOTE,
    }


def apply_grade(scores, grade_map):
    """Vectorized raw score -> grade letter (object array).

    A score takes the BEST grade whose threshold it reaches; the lowest grade is
    the catch-all, so live prod scores below the OOF minimum still grade instead of
    erroring. Non-finite scores return None (a scoring failure is the caller's to
    surface -- it must never be silently graded)."""
    arr = np.atleast_1d(np.asarray(scores, dtype=float))
    boundaries = [tuple(b) for b in grade_map["boundaries"]]
    thresholds = grade_map["thresholds"]
    names = [g for g, _ in boundaries]
    finite = np.isfinite(arr)
    out = np.full(len(arr), None, dtype=object)
    out[finite] = names[-1]
    assigned = ~finite
    for g in names[:-1]:
        hit = ~assigned & (arr >= float(thresholds[g]))
        out[hit] = g
        assigned |= hit
    return out
