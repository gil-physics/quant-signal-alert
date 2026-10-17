"""Step 14: Model 1 risk model for directional alerts.

The model predicts whether an alert is risky, starting with
``target_mae_risk``. Feature construction is deliberately causal: label/future
columns are created only after the feature matrix is fixed and are never part of
``FEATURE_COLUMNS``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_START = "2025-06-02"
TRAIN_END = "2026-02-28"
VALIDATION_START = "2026-03-01"
VALIDATION_END = "2026-06-19"
MAE_RISK_THRESHOLD = -0.015
# Step 15 short-entry mirror: adverse move for a SHORT is the rest-of-day HIGH running
# +1.5% above the decision-bar close (exact sign mirror of MAE_RISK_THRESHOLD).
UP_RISK_THRESHOLD = 0.015

# Step 17: fixed-horizon (60-minute) label lineage. Decisions with a time-of-day at or
# before this cutoff always observe EXACTLY horizon_minutes of forward window (14:30 +
# 60m = the 15:30 close); later decisions have a truncated window and form a separate
# late-session DIAGNOSTIC population, excluded from the main gate.
HORIZON_60M_MAIN_CUTOFF = "14:30:00"
# Step 17 event thresholds (USER-decided 2026-07-12 after scripts/
# survey_60m_label_thresholds.py): keep the legacy +-1.5% magnitudes. Survey basis:
# 60m future_MAE <= -1.5% covers 7.2% of train / 18.6% of validation main-population
# rows (17,963 / 24,814 events) -- sparser than the rest-of-day label but ample; MFE
# is near-symmetric. Pinned separately from the legacy constants so the two lineages
# can never drift into sharing a knob by accident.
MAE_RISK_THRESHOLD_60M = -0.015
UP_RISK_THRESHOLD_60M = 0.015

KEY_COLUMNS = ["symbol", "decision_time"]

ALERT_FEATURE_COLUMNS = [
    "up_score",
    "down_score",
    "alert_score",
    "direction_score",
    "risk_score",
]

RELATIVE_FEATURE_COLUMNS = [
    "active_symbol_count",
    "rank_ret_5m_1m",
    "rank_ret_20m_1m",
    "rank_volume_z_20m_1m",
    "universe_mean_ret_20m",
    "relative_ret_20m_to_universe",
    "relative_volume_z_to_universe",
    "universe_breadth",
]

MERGED_FEATURE_COLUMNS = [
    "ret_1m_1m",
    "ret_3m_1m",
    "ret_5m_1m",
    "ret_10m_1m",
    "ret_20m_1m",
    "ret_60m_1m",
    "range_1m",
    "body_1m",
    "upper_shadow_1m",
    "lower_shadow_1m",
    "volume_z_20m_1m",
    "volume_z_60m_1m",
    "vwap_distance_1m",
    "return_from_open_1m",
    "distance_from_day_high_1m",
    "distance_from_day_low_1m",
    "ma5_daily_n",
    "ma5_intraday_n",
    "dist_ma5_daily",
    "dist_ma5_intraday",
    "ret_1bar_3m",
    "ret_2bar_3m",
    "ret_3bar_3m",
    "range_1bar_3m",
    "body_1bar_3m",
    "volume_z_10bar_3m",
    "volume_z_20bar_3m",
    "ret_1bar_5m",
    "ret_2bar_5m",
    "ret_3bar_5m",
    "range_1bar_5m",
    "body_1bar_5m",
    "volume_z_10bar_5m",
    "volume_z_20bar_5m",
]

OPTIONAL_DERIVED_COLUMNS = [
    "above_ma5_intraday",
    "above_ma5_daily",
]

FEATURE_COLUMNS = (
    ALERT_FEATURE_COLUMNS
    + RELATIVE_FEATURE_COLUMNS
    + MERGED_FEATURE_COLUMNS
    + OPTIONAL_DERIVED_COLUMNS
)

# v2.0: 8 v1 cross-symbol features + 5 richer universe features (relative_features.py).
RELATIVE_FEATURE_COLUMNS_V2 = RELATIVE_FEATURE_COLUMNS + [
    "universe_disp_ret_20m",
    "universe_breadth_5m",
    "universe_breadth_60m",
    "ret_20m_gap_to_best",
    "universe_disp_volume_z",
]
FEATURE_COLUMNS_V2 = (
    ALERT_FEATURE_COLUMNS
    + RELATIVE_FEATURE_COLUMNS_V2
    + MERGED_FEATURE_COLUMNS
    + OPTIONAL_DERIVED_COLUMNS
)

# v2.1x: the 8 v1 cross-symbol features + 2 causal leader lead-lag features for ONE
# leader (relative_features.leader_columns(label)). Each variant is a SEPARATE
# single-leader hypothesis built on the v1 baseline, NOT on v2.0's (rejected) universe
# dispersion/breadth set. v2.1a uses leader "a" (005930), v2.1b uses leader "b" (402340);
# the label->code map lives in config.relative.leaders.
LEADER_A_COLUMNS = ["leader_a_ret_5m_1m", "leader_a_ret_20m_1m"]
LEADER_B_COLUMNS = ["leader_b_ret_5m_1m", "leader_b_ret_20m_1m"]
RELATIVE_FEATURE_COLUMNS_V2_1A = RELATIVE_FEATURE_COLUMNS + LEADER_A_COLUMNS
RELATIVE_FEATURE_COLUMNS_V2_1B = RELATIVE_FEATURE_COLUMNS + LEADER_B_COLUMNS
FEATURE_COLUMNS_V2_1A = (
    ALERT_FEATURE_COLUMNS
    + RELATIVE_FEATURE_COLUMNS_V2_1A
    + MERGED_FEATURE_COLUMNS
    + OPTIONAL_DERIVED_COLUMNS
)
FEATURE_COLUMNS_V2_1B = (
    ALERT_FEATURE_COLUMNS
    + RELATIVE_FEATURE_COLUMNS_V2_1B
    + MERGED_FEATURE_COLUMNS
    + OPTIONAL_DERIVED_COLUMNS
)

# v3.0x premarket-context features (built by src/premarket_features.py, one row per
# trade_date, joined onto each same-day regular-session row by date). v3.0 keeps the
# v1.1 baseline -- the v1 8 cross-symbol features served on the REAL cross-section -- and
# ATTACHES premarket (08:00-09:00) context. The premarket window closes before the open,
# so the attached values are causal for every regular-session decision that day.
#   v3.0a = target (000660) premarket only.
#   v3.0b = target + universe-wide premarket breadth/relative/rank (superset of v3.0a).
# The column names are 000660-specific for this experiment (the builder generates them
# from the configured target symbol; the literals here are guard-safe -- the digits are
# never quote-adjacent). v3.0c/v3.0d (005930 / 402340 anchors) are intentionally NOT
# registered yet.
PREMARKET_TARGET_COLUMNS = [
    "pm_000660_ret",
    "pm_000660_high_ret",
    "pm_000660_low_ret",
    "pm_000660_range",
    "pm_000660_volume_z",
    "pm_000660_close_to_vwap",
    "pm_000660_last_5m_ret",
    "pm_000660_has_data",
    "pm_000660_bar_count",
    "pm_000660_outlier_flag",
]
PREMARKET_UNIVERSE_COLUMNS = [
    "pm_universe_breadth_pos",
    "pm_universe_breadth_gt_0_5",
    "pm_universe_mean_ret",
    "pm_universe_median_ret",
    "pm_universe_std_ret",
    "pm_universe_dispersion_ret",
    "pm_universe_mean_range",
    "pm_universe_total_volume_z",
    "pm_000660_relative_ret_to_universe",
    "pm_000660_rank_ret",
    "pm_000660_rank_volume_z",
    "pm_universe_active_symbol_count",
    "pm_universe_missing_symbol_count",
]
PREMARKET_COLUMNS_V3_0A = list(PREMARKET_TARGET_COLUMNS)
PREMARKET_COLUMNS_V3_0B = PREMARKET_TARGET_COLUMNS + PREMARKET_UNIVERSE_COLUMNS

FEATURE_COLUMNS_V3_0A = FEATURE_COLUMNS + PREMARKET_COLUMNS_V3_0A
FEATURE_COLUMNS_V3_0B = FEATURE_COLUMNS + PREMARKET_COLUMNS_V3_0B

# Step 17: the fixed-horizon (60m) champion feature set = the v1 49 features + the
# causal minutes_since_open clock (emitted by build_model_frame(horizon_minutes=60);
# decision_time is the confirmed bar's bar_end, so the clock is known at decision
# time). BOTH lineage models receive the clock -- v1_60m as this tabular feature, the
# TCN as its time-of-day channel -- so the 60m gate measures architecture value on
# top of clock access, not clock access itself (the Step 16 ablation confound).
FEATURE_COLUMNS_60M = FEATURE_COLUMNS + ["minutes_since_open"]

# Risk-model version registry. "v1" is the frozen baseline (do not change). Serving a
# v2.x/v3.x model REQUIRES real cross-symbol features (--cross-symbol); v3.x additionally
# REQUIRES the premarket feature table to be joined at serving time (premarket_features.py).
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_VERSIONS = {
    "v1":    {"relative": RELATIVE_FEATURE_COLUMNS,       "features": FEATURE_COLUMNS,        "filename": "risk_model_lgbm.txt",       "requires_cross_symbol": False, "premarket": [],                       "requires_premarket": False},
    "v2.0":  {"relative": RELATIVE_FEATURE_COLUMNS_V2,    "features": FEATURE_COLUMNS_V2,     "filename": "risk_model_v2_0_lgbm.txt",  "requires_cross_symbol": True,  "premarket": [],                       "requires_premarket": False},
    "v2.1a": {"relative": RELATIVE_FEATURE_COLUMNS_V2_1A, "features": FEATURE_COLUMNS_V2_1A,  "filename": "risk_model_v2_1a_lgbm.txt", "requires_cross_symbol": True,  "premarket": [],                       "requires_premarket": False},
    "v2.1b": {"relative": RELATIVE_FEATURE_COLUMNS_V2_1B, "features": FEATURE_COLUMNS_V2_1B,  "filename": "risk_model_v2_1b_lgbm.txt", "requires_cross_symbol": True,  "premarket": [],                       "requires_premarket": False},
    "v3_0a_premarket_000660":          {"relative": RELATIVE_FEATURE_COLUMNS, "features": FEATURE_COLUMNS_V3_0A, "filename": "risk_model_v3_0a_premarket_000660_lgbm.txt",          "requires_cross_symbol": True, "premarket": PREMARKET_COLUMNS_V3_0A, "requires_premarket": True},
    "v3_0b_premarket_universe":        {"relative": RELATIVE_FEATURE_COLUMNS, "features": FEATURE_COLUMNS_V3_0B, "filename": "risk_model_v3_0b_premarket_universe_lgbm.txt",        "requires_cross_symbol": True, "premarket": PREMARKET_COLUMNS_V3_0B, "requires_premarket": True},
    # v1_short (Step 15): the up-risk (short-adverse) mirror of v1 -- same 49 features,
    # trained on target_up_risk with a SEPARATE filename so it can never overwrite the
    # v1 baseline risk_model_lgbm.txt (train_risk_model.py enforces the pairing).
    "v1_short": {"relative": RELATIVE_FEATURE_COLUMNS,   "features": FEATURE_COLUMNS,        "filename": "short_risk_model_lgbm.txt", "requires_cross_symbol": False, "premarket": [],                       "requires_premarket": False},
    # v1_60m (Step 17): the fixed-horizon champion -- v1 features + minutes_since_open,
    # trained on target_mae_risk_60m over the expanding walk-forward folds
    # (train_risk_model.py --fold k derives risk_model_v1_60m_fold<k>_lgbm.txt from this
    # filename). Research-only until its own gate GO; v1_short_60m is deliberately NOT
    # registered (one lineage at a time, the v3.0c/d precedent).
    "v1_60m": {"relative": RELATIVE_FEATURE_COLUMNS,     "features": FEATURE_COLUMNS_60M,    "filename": "risk_model_v1_60m_lgbm.txt", "requires_cross_symbol": False, "premarket": [],                      "requires_premarket": False},
    # Step 20: the PRODUCTION champion baseline -- v1_60m verbatim, trained on
    # PRODUCTION_SPLIT_60M (through June, July sanity window) under its own artifact
    # name so the frozen fold artifacts are never clobbered.
    "v1_60m_prod": {"relative": RELATIVE_FEATURE_COLUMNS, "features": FEATURE_COLUMNS_60M,   "filename": "risk_model_v1_60m_prod_lgbm.txt", "requires_cross_symbol": False, "premarket": [],                 "requires_premarket": False},
}


def model_version_config(version):
    if version not in MODEL_VERSIONS:
        raise ValueError(f"unknown risk-model version {version!r}; choices={list(MODEL_VERSIONS)}")
    return MODEL_VERSIONS[version]


def resolve_model_path(version, models_dir=None):
    """Canonical LightGBM model file for a version (override with --model-path)."""
    return Path(models_dir or _MODELS_DIR) / model_version_config(version)["filename"]

TARGET_COLUMNS = ["target_mae_risk", "target_up_fail", "target_down2_60m", "target_up_risk",
                  # Step 17 fixed-horizon lineage (emitted only when build_model_frame is
                  # called with horizon_minutes=60; keep-list is existence-filtered).
                  # NaN semantics differ from the legacy pair: no bars in the window ->
                  # target NaN (train drops it), NOT 0.
                  "target_mae_risk_60m", "target_up_risk_60m"]
FUTURE_EVAL_COLUMNS = ["future_MAE", "future_return_60m", "future_MFE",
                       "future_MAE_60m", "future_MFE_60m"]
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "future",
    "label_",
    "up_fail",
    "down2",
    "up2_",
    "up3_",
    "up5_",
    "down3_",
    "down5_",
    "event",
)


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    validation_start: str = VALIDATION_START
    validation_end: str = VALIDATION_END


# Step 17 (USER-decided 2026-07-12): expanding-window walk-forward folds for the 60m
# lineage, to survive the 2026 high-volatility regime months. ONLY the periods differ
# between folds -- features / targets / hyperparameters / seed are identical -- so each
# validation month is scored by a model that never saw it, and the gate stitches the
# four genuinely out-of-sample months (2026-03..06) into one walk-forward frame.
WALK_FORWARD_FOLDS_60M = (
    SplitConfig(train_start="2025-06-02", train_end="2026-02-28",
                validation_start="2026-03-01", validation_end="2026-03-31"),
    SplitConfig(train_start="2025-06-02", train_end="2026-03-31",
                validation_start="2026-04-01", validation_end="2026-04-30"),
    SplitConfig(train_start="2025-06-02", train_end="2026-04-30",
                validation_start="2026-05-01", validation_end="2026-05-31"),
    SplitConfig(train_start="2025-06-02", train_end="2026-05-31",
                validation_start="2026-06-01", validation_end="2026-06-30"),
)


def walk_forward_fold(fold):
    """1-based accessor into WALK_FORWARD_FOLDS_60M (fold 1..4)."""
    if not 1 <= int(fold) <= len(WALK_FORWARD_FOLDS_60M):
        raise ValueError(f"fold must be 1..{len(WALK_FORWARD_FOLDS_60M)}, got {fold!r}")
    return WALK_FORWARD_FOLDS_60M[int(fold) - 1]


# Step 20 (USER-decided 2026-07-16): the PRODUCTION split for the 60m lineage -- train
# through June (so the models finally see the June drawdown regime) and sanity-check on
# the first July trading days. NOT a fifth walk-forward fold: the frozen 4-fold tuple
# above is a sealed research protocol, while this split exists to train the deployable
# candidates (v1_60m_prod / v4_tcn_60m_prod / v5_cnn_60m_prod). The July window is tiny
# (~8 trading days) -- selection between the TCN and CNN candidates follows the
# pre-registered rule in scripts/select_prod_60m.py, never eyeballing.
PRODUCTION_SPLIT_60M = SplitConfig(train_start="2025-06-02", train_end="2026-06-30",
                                   validation_start="2026-07-01",
                                   validation_end="2026-07-10")


def available_feature_columns(frame, feature_columns=None):
    feature_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    cols = [c for c in feature_columns if c in frame.columns]
    leaked = [
        c
        for c in cols
        if any(token in c.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaked:
        raise ValueError(f"future/target-like columns cannot be features: {leaked}")
    return cols


def merge_alert_features(alerts, features_merged, features_relative, *, relative_columns=None):
    """Join current-time features onto Step 13_6 directional alerts."""
    if alerts is None or len(alerts) == 0:
        return pd.DataFrame()
    relative_columns = RELATIVE_FEATURE_COLUMNS if relative_columns is None else relative_columns

    df = alerts.copy()
    if "date" not in df.columns:
        df["date"] = df["decision_time"].str.slice(0, 10)
    if "time" not in df.columns:
        df["time"] = df["decision_time"].str.slice(11, 19)

    rel_cols = KEY_COLUMNS + [c for c in relative_columns if c in features_relative.columns]
    if features_relative is not None and len(features_relative) and len(rel_cols) > len(KEY_COLUMNS):
        rel = features_relative[rel_cols].drop_duplicates(KEY_COLUMNS)
        dup = [c for c in rel.columns if c in df.columns and c not in KEY_COLUMNS]
        if dup:
            df = df.drop(columns=dup)
        df = df.merge(rel, on=KEY_COLUMNS, how="left", validate="many_to_one")

    merged_cols = KEY_COLUMNS + [c for c in MERGED_FEATURE_COLUMNS if c in features_merged.columns]
    if features_merged is not None and len(features_merged) and len(merged_cols) > len(KEY_COLUMNS):
        fm = features_merged[merged_cols].drop_duplicates(KEY_COLUMNS)
        dup = [c for c in fm.columns if c in df.columns and c not in KEY_COLUMNS]
        if dup:
            df = df.drop(columns=dup)
        df = df.merge(fm, on=KEY_COLUMNS, how="left", validate="many_to_one")

    if "dist_ma5_intraday" in df.columns:
        df["above_ma5_intraday"] = (df["dist_ma5_intraday"] > 0).astype(float)
    if "dist_ma5_daily" in df.columns:
        df["above_ma5_daily"] = (df["dist_ma5_daily"] > 0).astype(float)
    return df


def merge_premarket_features(frame, premarket, premarket_columns):
    """Join the per-day premarket features onto regular-session rows by calendar date.

    No-op when ``premarket``/``premarket_columns`` are empty, so v1/v2 model frames are
    byte-identical. The join is many-to-one on the trade date: the premarket table holds
    one row per day, broadcast onto every same-day decision row. The premarket window
    (08:00-09:00) closes before the regular open, so this is strictly causal for the
    09:00+ decisions it attaches to.
    """
    if premarket is None or len(premarket) == 0 or not premarket_columns:
        return frame
    df = frame.copy()
    if "date" not in df.columns:
        df["date"] = df["decision_time"].str.slice(0, 10)
    pm = premarket.copy()
    key = "trade_date" if "trade_date" in pm.columns else "date"
    cols = [c for c in premarket_columns if c in pm.columns]
    if not cols:
        return df
    pm = pm[[key] + cols].drop_duplicates(key).rename(columns={key: "date"})
    dup = [c for c in cols if c in df.columns and c != "date"]
    if dup:
        df = df.drop(columns=dup)
    return df.merge(pm, on="date", how="left", validate="many_to_one")


def add_label_targets(frame, labels):
    df = frame.copy()
    lab_cols = KEY_COLUMNS + [c for c in ("down2_60m", "up5_event") if c in labels.columns]
    if labels is not None and len(labels) and len(lab_cols) > len(KEY_COLUMNS):
        lab = labels[lab_cols].drop_duplicates(KEY_COLUMNS)
        df = df.merge(lab, on=KEY_COLUMNS, how="left", validate="many_to_one")
    df["target_up_fail"] = (df.get("up5_event", "") == "UP_FAIL").astype(int)
    df["target_down2_60m"] = df.get("down2_60m", 0).fillna(0).astype(int)
    return df


def _prepare_bars_by_day(bars):
    b = bars.copy()
    b["_dt"] = pd.to_datetime(b["bar_start"])
    b["_date"] = b["bar_start"].str.slice(0, 10)
    has_high = "high" in b.columns   # optional: only needed for the up-risk (short) label
    out = {}
    for (symbol, date), group in b.groupby(["symbol", "_date"], sort=False):
        g = group.sort_values("_dt")
        arr = {
            "dt": g["_dt"].to_numpy(),
            "low": g["low"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
        }
        if has_high:
            arr["high"] = g["high"].to_numpy(dtype=float)
        out[(symbol, date)] = arr
    return out


def add_future_risk_targets(frame, bars, *, mae_threshold=MAE_RISK_THRESHOLD,
                            up_threshold=UP_RISK_THRESHOLD,
                            horizon_minutes=None,
                            horizon_mae_threshold=None,
                            horizon_up_threshold=None):
    """Add future_MAE/MFE/return_60m and the MAE-risk / up-risk targets from bars.

    future_MFE (rest-of-day suffix-max of ``high`` relative to the decision-bar close,
    same forward-window convention as future_MAE) is only computable when ``bars`` has
    a ``high`` column; otherwise it stays NaN and target_up_risk is 0 -- the columns are
    ALWAYS emitted so the output schema is stable, and the MAE path is unaffected.

    Step 17 (additive; ``horizon_minutes=None`` keeps the output byte-identical):
    when ``horizon_minutes`` is set, ADDITIONALLY emit fixed-horizon columns
    ``future_MAE_<H>m`` / ``future_MFE_<H>m`` = min(low) / max(high) over the bars with
    bar_start in [decision_time, decision_time + H minutes) -- same half-open end
    convention as future_return_60m, so late decisions truncate at the session close --
    plus ``target_mae_risk_<H>m`` / ``target_up_risk_<H>m``. The horizon targets use
    NaN PROPAGATION (no bars in the window -> target NaN; training drops it), deliberately
    unlike the legacy pair's NaN->0; horizon thresholds default to the survey-pinned
    MAE_RISK_THRESHOLD_60M / UP_RISK_THRESHOLD_60M (``horizon_*_threshold`` overrides),
    never to the legacy kwargs -- the lineages share no knob.
    """
    h_sfx = f"_{int(horizon_minutes)}m" if horizon_minutes is not None else None
    h_mae_thr = MAE_RISK_THRESHOLD_60M if horizon_mae_threshold is None else horizon_mae_threshold
    h_up_thr = UP_RISK_THRESHOLD_60M if horizon_up_threshold is None else horizon_up_threshold

    df = frame.copy()
    if df.empty:
        df["future_MAE"] = []
        df["future_return_60m"] = []
        df["target_mae_risk"] = []
        df["future_MFE"] = []
        df["target_up_risk"] = []
        if h_sfx is not None:
            for col in (f"future_MAE{h_sfx}", f"future_MFE{h_sfx}",
                        f"target_mae_risk{h_sfx}", f"target_up_risk{h_sfx}"):
                df[col] = []
        return df

    df["future_MAE"] = np.nan
    df["future_return_60m"] = np.nan
    df["future_MFE"] = np.nan
    if h_sfx is not None:
        df[f"future_MAE{h_sfx}"] = np.nan
        df[f"future_MFE{h_sfx}"] = np.nan
    bars_by_day = _prepare_bars_by_day(bars)

    for (symbol, date), idx in df.groupby(["symbol", "date"], sort=False).groups.items():
        arr = bars_by_day.get((symbol, date))
        if arr is None:
            continue
        positions = np.asarray(list(idx), dtype=int)
        decisions = pd.to_datetime(df.loc[positions, "decision_time"]).to_numpy()
        close_t = df.loc[positions, "close"].to_numpy(dtype=float)
        dt = arr["dt"]
        lows = arr["low"]
        closes = arr["close"]
        if len(dt) == 0:
            continue

        # decision_time == the decision (feature) bar's bar_END, and close is that
        # bar's close (see feature_lib.build_minute_features: decision_time = bar_end).
        # The forward window is the bars with bar_start >= decision_time, i.e. the bar
        # that STARTS at decision_time onward -- the first genuinely-future bar.
        # side="left" starts exactly there while still excluding the feature bar (whose
        # bar_start < decision_time). Matches labeling.py's convention; side="right"
        # would skip the first forward minute (an off-by-one).
        start = np.searchsorted(dt, decisions, side="left")
        valid = start < len(dt)
        if not valid.any():
            continue

        suffix_low = np.minimum.accumulate(lows[::-1])[::-1]
        mae = np.full(len(positions), np.nan, dtype=float)
        mae[valid] = suffix_low[start[valid]] / close_t[valid] - 1
        df.loc[positions, "future_MAE"] = mae

        # Up-risk mirror (Step 15): rest-of-day suffix-MAX of high over the same
        # forward window (first forward bar included, decision bar excluded).
        highs = arr.get("high")
        if highs is not None:
            suffix_high = np.maximum.accumulate(highs[::-1])[::-1]
            mfe = np.full(len(positions), np.nan, dtype=float)
            mfe[valid] = suffix_high[start[valid]] / close_t[valid] - 1
            df.loc[positions, "future_MFE"] = mfe

        end = np.searchsorted(dt, decisions + np.timedelta64(60, "m"), side="left") - 1
        valid60 = valid & (end >= start)
        ret60 = np.full(len(positions), np.nan, dtype=float)
        ret60[valid60] = closes[end[valid60]] / close_t[valid60] - 1
        df.loc[positions, "future_return_60m"] = ret60

        if h_sfx is not None:
            # Fixed-horizon window [start, end_h]: same half-open end convention as
            # future_return_60m (the bar starting exactly at decision+H is excluded;
            # late decisions truncate at the close because no later bars exist).
            end_h = np.searchsorted(
                dt, decisions + np.timedelta64(int(horizon_minutes), "m"), side="left") - 1
            valid_h = valid & (end_h >= start)
            mae_h = np.full(len(positions), np.nan, dtype=float)
            mfe_h = np.full(len(positions), np.nan, dtype=float)
            if valid_h.any():
                # windows are time-based (gap-robust): with 1-minute bars a H-minute
                # window holds at most H bars, so a (n, span) offset gather covers it.
                span = int((end_h[valid_h] - start[valid_h]).max()) + 1
                idx_h = start[:, None] + np.arange(span)[None, :]
                in_win = valid_h[:, None] & (idx_h <= end_h[:, None])
                safe = np.where(in_win, idx_h, 0)
                win_low = np.where(in_win, lows[safe], np.inf)
                mae_h[valid_h] = win_low.min(axis=1)[valid_h] / close_t[valid_h] - 1
                if highs is not None:
                    win_high = np.where(in_win, highs[safe], -np.inf)
                    mfe_h[valid_h] = win_high.max(axis=1)[valid_h] / close_t[valid_h] - 1
            df.loc[positions, f"future_MAE{h_sfx}"] = mae_h
            df.loc[positions, f"future_MFE{h_sfx}"] = mfe_h

    df["target_mae_risk"] = (df["future_MAE"] <= mae_threshold).astype(int)
    # NaN future_MFE (bars without high / no forward bars) compares False -> 0,
    # exactly the future_MAE NaN convention.
    df["target_up_risk"] = (df["future_MFE"] >= up_threshold).astype(int)
    if h_sfx is not None:
        # NaN PROPAGATION (unlike the legacy pair): no bars in the horizon window ->
        # the label is unknowable -> target NaN (float column); training drops it.
        mae_h_all = df[f"future_MAE{h_sfx}"]
        mfe_h_all = df[f"future_MFE{h_sfx}"]
        df[f"target_mae_risk{h_sfx}"] = np.where(
            mae_h_all.isna(), np.nan, (mae_h_all <= h_mae_thr).astype(float))
        df[f"target_up_risk{h_sfx}"] = np.where(
            mfe_h_all.isna(), np.nan, (mfe_h_all >= h_up_thr).astype(float))
    return df


def build_model_frame(alerts, features_merged, features_relative, labels, bars, *,
                      relative_columns=None, feature_columns=None,
                      premarket=None, premarket_columns=None,
                      horizon_minutes=None):
    """Build one modelling table from Step 14 inputs. ``relative_columns`` /
    ``feature_columns`` default to the v1 sets; pass the v2 sets for the v2.0 model.
    Pass ``premarket`` (the per-day premarket table) + ``premarket_columns`` for the
    v3.0 models -- those columns are joined by date and added to the feature matrix.
    Step 17: ``horizon_minutes=60`` additionally emits the fixed-horizon label columns
    (see add_future_risk_targets) plus the causal ``minutes_since_open`` clock feature
    (decision_time is the confirmed bar's bar_end, so it is known at decision time);
    the default ``None`` keeps the output byte-identical."""
    df = merge_alert_features(alerts, features_merged, features_relative,
                              relative_columns=relative_columns)
    df = merge_premarket_features(df, premarket, premarket_columns)
    df = add_label_targets(df, labels)
    df = add_future_risk_targets(df, bars, horizon_minutes=horizon_minutes)
    if horizon_minutes is not None:
        s = df["decision_time"].astype(str)
        df["minutes_since_open"] = (s.str.slice(11, 13).astype(int) * 60
                                    + s.str.slice(14, 16).astype(int) - 540)
    feature_cols = available_feature_columns(df, feature_columns)
    keep = [
        "symbol",
        "decision_time",
        "date",
        "time",
        "source",
        "alert_type",
        "alert_grade",
        "close",
    ] + feature_cols + TARGET_COLUMNS + FUTURE_EVAL_COLUMNS
    if horizon_minutes is not None and "minutes_since_open" not in keep:
        keep.append("minutes_since_open")
    keep = [c for c in keep if c in df.columns]
    return df[keep].replace([np.inf, -np.inf], np.nan).reset_index(drop=True), feature_cols


def split_by_date(frame, *, split=SplitConfig()):
    train = frame[(frame["date"] >= split.train_start) & (frame["date"] <= split.train_end)].copy()
    valid = frame[(frame["date"] >= split.validation_start) & (frame["date"] <= split.validation_end)].copy()
    return train.reset_index(drop=True), valid.reset_index(drop=True)


class FeaturePreprocessor:
    def __init__(self, feature_columns):
        self.feature_columns = list(feature_columns)
        self.median_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, frame):
        x = frame[self.feature_columns].astype(float)
        median = x.median(axis=0).fillna(0.0)
        filled = x.fillna(median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0).replace(0, 1.0).fillna(1.0)
        self.median_ = median
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, frame):
        if self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("preprocessor is not fitted")
        x = frame[self.feature_columns].astype(float).fillna(self.median_)
        return ((x - self.mean_) / self.scale_).to_numpy(dtype=float)

    def to_dict(self):
        return {
            "feature_columns": self.feature_columns,
            "median": self.median_.to_dict() if self.median_ is not None else {},
            "mean": self.mean_.to_dict() if self.mean_ is not None else {},
            "scale": self.scale_.to_dict() if self.scale_ is not None else {},
        }

    @classmethod
    def from_dict(cls, payload):
        pre = cls(payload.get("feature_columns", []))
        cols = pre.feature_columns
        pre.median_ = pd.Series(payload.get("median", {})).reindex(cols).astype(float)
        pre.mean_ = pd.Series(payload.get("mean", {})).reindex(cols).astype(float)
        pre.scale_ = pd.Series(payload.get("scale", {})).reindex(cols).fillna(1.0).astype(float)
        return pre


class NumpyLogisticRiskModel:
    """Small fallback binary classifier when LightGBM/sklearn are unavailable."""

    model_type = "numpy_logistic_regression"

    def __init__(self, *, learning_rate=0.08, max_iter=120, l2=1e-3, class_weight="balanced"):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.coef_ = None
        self.intercept_ = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n, p = x.shape
        if n == 0:
            raise ValueError("empty training data")
        w = np.zeros(p, dtype=float)
        b = 0.0

        sample_weight = np.ones(n, dtype=float)
        if self.class_weight == "balanced":
            pos = y.sum()
            neg = n - pos
            if pos > 0 and neg > 0:
                sample_weight[y == 1] = n / (2.0 * pos)
                sample_weight[y == 0] = n / (2.0 * neg)
        denom = sample_weight.sum()

        for _ in range(self.max_iter):
            pred = self._sigmoid(x @ w + b)
            err = (pred - y) * sample_weight
            grad_w = (x.T @ err) / denom + self.l2 * w
            grad_b = float(err.sum() / denom)
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict_proba(self, x):
        if self.coef_ is None:
            raise ValueError("model is not fitted")
        p = self._sigmoid(np.asarray(x, dtype=float) @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p, p])

    def feature_importance(self, feature_columns):
        if self.coef_ is None:
            return pd.DataFrame({"feature": feature_columns, "importance": 0.0})
        out = pd.DataFrame({
            "feature": feature_columns,
            "importance": np.abs(self.coef_),
            "coefficient": self.coef_,
        })
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path, *, preprocessor, metadata=None):
        payload = {
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "l2": self.l2,
            "class_weight": self.class_weight,
            "coef": self.coef_.tolist() if self.coef_ is not None else [],
            "intercept": self.intercept_,
            "preprocessor": preprocessor.to_dict(),
            "metadata": metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


MODEL_KINDS = ("auto", "lightgbm", "sklearn_hgb", "sklearn_logistic", "numpy")


def _has_module(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None


def resolve_model_kind(kind):
    """Resolve ``auto`` to the best available backend (lightgbm > sklearn > numpy)."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind: {kind!r} (choose from {MODEL_KINDS})")
    if kind != "auto":
        return kind
    if _has_module("lightgbm"):
        return "lightgbm"
    if _has_module("sklearn"):
        return "sklearn_hgb"
    return "numpy"


def _balanced_sample_weight(y):
    """sklearn-style 'balanced' sample weights (same scheme as the numpy fallback)."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    w = np.ones(n, dtype=float)
    if pos > 0 and neg > 0:
        w[y == 1] = n / (2.0 * pos)
        w[y == 0] = n / (2.0 * neg)
    return w


def _time_ordered_holdout(frame, fraction):
    """Chronological (by decision_time) (fit_idx, eval_idx) positions for an
    internal early-stopping holdout carved from the TAIL of train.

    Returns ``(None, None)`` when the split is too small to be useful. The real
    validation set is never touched, so reported metrics stay leakage-free.
    """
    if not fraction or len(frame) < 400:
        return None, None
    order = np.argsort(frame["decision_time"].to_numpy(dtype=str), kind="stable")
    cut = int(len(order) * (1.0 - fraction))
    if cut <= 0 or cut >= len(order):
        return None, None
    return order[:cut], order[cut:]


def _named_frame(x, feature_columns):
    """Wrap a transformed feature matrix as a named DataFrame so the sklearn-API
    models record real feature names (predictions then align by name, not just
    position, and the spurious 'no valid feature names' warning disappears)."""
    return pd.DataFrame(np.asarray(x, dtype=float), columns=list(feature_columns))


def _fit_lightgbm(train, x_train, y_train, feature_columns, *, early_stopping_frac, random_state):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight="balanced",   # target_mae_risk is imbalanced
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    xdf = _named_frame(x_train, feature_columns)
    fit_pos, eval_pos = _time_ordered_holdout(train, early_stopping_frac)
    best_iter = None
    if fit_pos is not None and len(np.unique(y_train[eval_pos])) == 2:
        model.fit(
            xdf.iloc[fit_pos], y_train[fit_pos],
            eval_set=[(xdf.iloc[eval_pos], y_train[eval_pos])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        best_iter = int(model.best_iteration_) if model.best_iteration_ else None
    else:
        model.fit(xdf, y_train)
    model.model_type = "lightgbm"
    model.best_iteration_used = best_iter
    return model


def _fit_sklearn_hgb(x_train, y_train, feature_columns, *, random_state):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
    )
    model.fit(_named_frame(x_train, feature_columns), y_train,
              sample_weight=_balanced_sample_weight(y_train))
    model.model_type = "sklearn_hgb"
    return model


def _fit_sklearn_logistic(x_train, y_train, feature_columns):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    model.fit(_named_frame(x_train, feature_columns), y_train)
    model.model_type = "sklearn_logistic"
    return model


def fit_binary_model(train, valid, feature_columns, target_col, *, model_kind="auto",
                     early_stopping_frac=0.15, random_state=42, prefer_lightgbm=None):
    """Fit a binary risk classifier.

    ``model_kind`` selects the backend (``auto`` picks lightgbm > sklearn > numpy).
    LightGBM uses class balancing and early stops on a chronological holdout carved
    from the TAIL of ``train`` -- the ``valid`` frame is never seen during fitting,
    so the validation metrics the caller computes are leakage-free.
    """
    if prefer_lightgbm is False and model_kind == "auto":
        model_kind = "numpy"   # backward-compat with the old --no-lightgbm flag

    pre = FeaturePreprocessor(feature_columns).fit(train)
    x_train = pre.transform(train)
    y_train = train[target_col].astype(int).to_numpy()

    kind = resolve_model_kind(model_kind)
    if kind == "lightgbm":
        model = _fit_lightgbm(train, x_train, y_train, feature_columns,
                              early_stopping_frac=early_stopping_frac, random_state=random_state)
    elif kind == "sklearn_hgb":
        model = _fit_sklearn_hgb(x_train, y_train, feature_columns, random_state=random_state)
    elif kind == "sklearn_logistic":
        model = _fit_sklearn_logistic(x_train, y_train, feature_columns)
    else:
        model = NumpyLogisticRiskModel().fit(x_train, y_train)
        model.model_type = NumpyLogisticRiskModel.model_type
    return model, pre


def model_feature_importance(model, feature_columns):
    """Uniform feature-importance frame across the model backends."""
    if isinstance(model, NumpyLogisticRiskModel):
        return model.feature_importance(feature_columns)
    if hasattr(model, "feature_importances_"):           # lightgbm
        imp = np.asarray(model.feature_importances_, dtype=float)
        return (pd.DataFrame({"feature": feature_columns, "importance": imp})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    if hasattr(model, "coef_"):                          # logistic regression
        coef = np.ravel(model.coef_).astype(float)
        return (pd.DataFrame({"feature": feature_columns, "importance": np.abs(coef), "coefficient": coef})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    return pd.DataFrame({"feature": feature_columns, "importance": 0.0})   # e.g. HGB has no native importance


def predict_with_model(model, preprocessor, frame):
    x = preprocessor.transform(frame)
    model_type = getattr(model, "model_type", "")
    if model_type == "lightgbm" or model_type.startswith("sklearn"):
        x = _named_frame(x, preprocessor.feature_columns)   # align by name, silence warning
    return model.predict_proba(x)[:, 1]


def roc_auc_score_np(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision_np(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = int(y.sum())
    if pos == 0:
        return np.nan
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / pos)


def logloss_np(y, prob):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def top_fraction_mask(prob, fraction):
    prob = np.asarray(prob, dtype=float)
    n = len(prob)
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-prob)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def classification_summary(frame, *, target_col, prob_col="risk_prob"):
    y = frame[target_col].astype(int).to_numpy()
    p = frame[prob_col].to_numpy(dtype=float)
    top10 = top_fraction_mask(p, 0.10)
    positives = int(y.sum())
    top_pos = int(y[top10].sum())
    two_class = 0 < positives < len(y)
    # Prefer sklearn (handles tied scores exactly in AP); fall back to the
    # dependency-free numpy implementations (used by the 32-bit runtime + tests).
    try:
        from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

        auc = float(roc_auc_score(y, p)) if two_class else float("nan")
        pr_auc = float(average_precision_score(y, p)) if positives else float("nan")
        logloss = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        auc, pr_auc, logloss = roc_auc_score_np(y, p), average_precision_np(y, p), logloss_np(y, p)
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "logloss": logloss,
        "precision_at_top10": float(top_pos / top10.sum()) if top10.sum() else 0.0,
        "recall_at_top10": float(top_pos / positives) if positives else 0.0,
    }


def monthly_auc(frame, *, target_col, prob_col="risk_prob"):
    """Validation AUC per calendar month — surfaces regime decay (the model's main
    operational caveat). Returns a DataFrame [month, n, positive_rate, auc]."""
    if frame is None or len(frame) == 0 or "date" not in frame.columns:
        return pd.DataFrame(columns=["month", "n", "positive_rate", "auc"])
    rows = []
    work = frame.copy()
    work["month"] = work["date"].str.slice(0, 7)
    for month, g in work.groupby("month", sort=True):
        y = g[target_col].astype(int).to_numpy()
        rows.append({
            "month": month, "n": int(len(g)),
            "positive_rate": round(float(y.mean()), 4),
            "auc": round(roc_auc_score_np(y, g[prob_col].to_numpy(dtype=float)), 4),
        })
    return pd.DataFrame(rows)


def bucket_report(validation, *, target_col="target_mae_risk", prob_col="risk_prob",
                  mae_col="future_MAE"):
    if validation.empty:
        return pd.DataFrame()
    p = validation[prob_col].to_numpy(dtype=float)
    buckets = [
        ("top_10%", top_fraction_mask(p, 0.10)),
        ("top_20%", top_fraction_mask(p, 0.20)),
        ("middle_40%", (p >= np.quantile(p, 0.30)) & (p <= np.quantile(p, 0.70))),
        ("bottom_20%", p <= np.quantile(p, 0.20)),
    ]
    rows = []
    for name, mask in buckets:
        g = validation.loc[mask]
        n = int(len(g))
        rows.append({
            "risk_bucket": name,
            "n": n,
            "mae_risk_rate": float(g[target_col].mean()) if n else 0.0,
            "up_fail_rate": float(g["target_up_fail"].mean()) if n else 0.0,
            "down2_60m_rate": float(g["target_down2_60m"].mean()) if n else 0.0,
            "avg_MAE": float(g[mae_col].mean()) if n else 0.0,
            "avg_return_60m": float(g["future_return_60m"].mean()) if n else 0.0,
            "avg_risk_prob": float(g[prob_col].mean()) if n else 0.0,
        })
    return pd.DataFrame(rows)


def existing_risk_grade_report(validation, *, target_col="target_mae_risk",
                               mae_col="future_MAE"):
    risk = validation[validation["alert_grade"] == "RISK"]
    not_risk = validation[validation["alert_grade"] != "RISK"]

    def row(name, g):
        n = int(len(g))
        return {
            "group": name,
            "n": n,
            "mae_risk_rate": float(g[target_col].mean()) if n else 0.0,
            "up_fail_rate": float(g["target_up_fail"].mean()) if n else 0.0,
            "down2_60m_rate": float(g["target_down2_60m"].mean()) if n else 0.0,
            "avg_MAE": float(g[mae_col].mean()) if n else 0.0,
        }

    return pd.DataFrame([row("existing_RISK_grade", risk), row("non_RISK_grade", not_risk)])


def save_model(model, path, *, preprocessor, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = getattr(model, "model_type", type(model).__name__)
    meta = {"model_type": model_type, "preprocessor": preprocessor.to_dict(), "metadata": metadata or {}}
    if model_type == "lightgbm":
        model.booster_.save_model(str(path))
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    elif model_type.startswith("sklearn"):
        import pickle

        with open(path.with_suffix(".pkl"), "wb") as fh:
            pickle.dump({"model": model, "preprocessor": preprocessor, "metadata": metadata or {}}, fh)
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        model.save(path, preprocessor=preprocessor, metadata=metadata)


class LoadedBoosterModel:
    """Thin predict_proba wrapper around a saved LightGBM Booster (for serving)."""

    model_type = "lightgbm"

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, x):
        p = np.asarray(self.booster.predict(x), dtype=float)
        return np.column_stack([1.0 - p, p])


def load_model(path):
    """Reload a model + its preprocessor saved by :func:`save_model`.

    Returns ``(model, preprocessor, metadata)``. Works for the lightgbm booster
    (``.txt`` + ``.meta.json``), the sklearn pickle (``.pkl``), and the numpy
    logistic json. Raises FileNotFoundError if no artifact is found.
    """
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    pkl_path = path.with_suffix(".pkl")

    if meta_path.exists() and path.exists():       # lightgbm booster
        import lightgbm as lgb

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(path))
        pre = FeaturePreprocessor.from_dict(meta.get("preprocessor", {}))
        return LoadedBoosterModel(booster), pre, meta.get("metadata", {})

    if meta_path.exists() and pkl_path.exists():    # sklearn pickle
        import pickle

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with open(pkl_path, "rb") as fh:
            payload = pickle.load(fh)
        model = payload["model"]
        if not hasattr(model, "model_type"):
            model.model_type = meta.get("model_type", "sklearn")
        return model, payload["preprocessor"], payload.get("metadata", {})

    if path.exists():                               # numpy logistic json
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = NumpyLogisticRiskModel(
            learning_rate=payload.get("learning_rate", 0.08),
            max_iter=payload.get("max_iter", 120),
            l2=payload.get("l2", 1e-3),
            class_weight=payload.get("class_weight", "balanced"),
        )
        model.coef_ = np.asarray(payload.get("coef", []), dtype=float)
        model.intercept_ = float(payload.get("intercept", 0.0))
        model.model_type = NumpyLogisticRiskModel.model_type
        pre = FeaturePreprocessor.from_dict(payload.get("preprocessor", {}))
        return model, pre, payload.get("metadata", {})

    raise FileNotFoundError(f"no model artifact found at {path} (.meta.json/.pkl/json)")
