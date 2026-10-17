"""Serving-time feature + alert + risk scoring for the real-time trend-alert engine.

Reproduces the SAME feature pipeline the models were trained on so train/serve
features match exactly (Step 7 1m features, Step 5 resample -> Step 7 3m/5m
features, Step 8 leakage-safe as-of merge), then runs the Step 11 rule signal,
the Step 13_5 directional alert, and the Step 14 risk model.

Single-symbol mode: the cross-sectional Step 9 features (rank / relative /
universe breadth) need the whole active universe, which a one-symbol live feed
does not have, so they are filled with NEUTRAL values (no rank bonus/penalty,
zero relative, breadth 0.5). The risk model's strongest features are all
within-symbol (dist_ma5, vwap_distance, volume_z, range, distance_from_day_high),
so this degrades gracefully. Everything here is strictly causal: it only ever
reads bars up to each decision_time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from directional_alert import AlertThresholds, alert_fire_gate, compute_directional_alerts
from feature_lib import build_bar_features, build_minute_features
from feature_merge import merge_asof_features
from relative_features import INPUT_COLUMNS as REL_INPUT_COLUMNS
from relative_features import OUTPUT_COLUMNS as REL_OUTPUT_COLUMNS
from relative_features import RANK_INPUTS, build_relative_features
from resampler import resample_ohlcv
from risk_model import FEATURE_COLUMNS, available_feature_columns, predict_with_model
from rule_signal import INPUT_COLUMNS as RULE_INPUT_COLUMNS
from rule_signal import NUMERIC_INPUTS as RULE_NUMERIC_INPUTS
from rule_signal import build_rule_signals

# the 8 cross-sectional (Step 9) columns build_relative_features appends after the
# INPUT_COLUMNS -- the exact set _fill_neutral_relative neutralises for single-symbol.
REL_COLUMNS = REL_OUTPUT_COLUMNS[len(REL_INPUT_COLUMNS):]

# cross-sectional (universe) columns that a single-symbol feed cannot compute
NEUTRAL_RELATIVE = {
    "universe_mean_ret_20m": 0.0,
    "relative_ret_20m_to_universe": 0.0,
    "relative_volume_z_to_universe": 0.0,
    "universe_breadth": 0.5,
    # v2.0 richer universe features (neutral = no dispersion / balanced breadth / at-leader)
    "universe_disp_ret_20m": 0.0,
    "universe_disp_volume_z": 0.0,
    "universe_breadth_5m": 0.5,
    "universe_breadth_60m": 0.5,
    "ret_20m_gap_to_best": 0.0,
}
RULE_SCORE_COLUMNS = ["up_score", "down_score", "is_up_candidate", "is_down_candidate", "primary_side"]
RESULT_COLUMNS = [
    "symbol", "decision_time", "close",
    "alert_type", "alert_grade", "alert_score", "risk_score", "direction_score",
    "up_score", "down_score", "reason_1", "reason_2", "reason_3", "risk_prob",
]


def _fill_neutral_relative(df, active_count):
    """Add the universe Step-9 columns with neutral values (single-symbol serving)."""
    n = max(1, int(active_count))
    mid_rank = max(1, math.ceil(n / 2))   # neither top-third nor bottom-third bucket
    df["active_symbol_count"] = n
    df["rank_ret_5m_1m"] = mid_rank
    df["rank_ret_20m_1m"] = mid_rank
    df["rank_volume_z_20m_1m"] = mid_rank
    for col, val in NEUTRAL_RELATIVE.items():
        df[col] = val
    return df


def _build_merged(bars_window, source):
    """``features_merged`` (1m + as-of 3m/5m) from a 1m bars window, deduped on
    (symbol, decision_time). Shared by single-symbol and cross-symbol serving;
    multi-symbol safe (the builders group by (symbol, _date)). Returns an empty
    frame when there is nothing to build. The cross-sectional Step-9 columns are
    NOT added here -- the caller fills them (neutral or real)."""
    if bars_window is None or len(bars_window) == 0:
        return pd.DataFrame()

    # incoming live/replay bars may arrive as object dtype (dict rows, NULL volume);
    # the feature pipeline needs numeric OHLCV, and the string keys must share one
    # dtype or the downstream merge_asof (1m vs 3m/5m) raises on mixed object /
    # StringDtype symbol keys.
    bars_window = bars_window.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in bars_window.columns:
            bars_window[col] = pd.to_numeric(bars_window[col], errors="coerce")
    for col in ("symbol", "bar_start", "bar_end"):
        if col in bars_window.columns:
            bars_window[col] = bars_window[col].astype(str)

    f1 = build_minute_features(bars_window, source)
    if f1.empty:
        return pd.DataFrame()
    b3 = resample_ohlcv(bars_window, 3)
    b5 = resample_ohlcv(bars_window, 5)
    f3 = build_bar_features(b3, "3m", source)
    f5 = build_bar_features(b5, "5m", source)

    merged = merge_asof_features(f1, f3, f5)
    # guard against a duplicate decision_time (e.g. a closing-auction bar sharing a
    # bar_end, or an overlapping seed) so the downstream one-to-one merges hold.
    return merged.drop_duplicates(subset=["symbol", "decision_time"], keep="last").reset_index(drop=True)


def _add_above_ma5(df):
    """Step 14 OPTIONAL_DERIVED flags (see risk_model.merge_alert_features)."""
    if "dist_ma5_intraday" in df.columns:
        df["above_ma5_intraday"] = (df["dist_ma5_intraday"] > 0).astype(float)
    if "dist_ma5_daily" in df.columns:
        df["above_ma5_daily"] = (df["dist_ma5_daily"] > 0).astype(float)
    return df


def build_serving_features(bars_window, *, source, active_count):
    """Build the merged feature frame for a SINGLE symbol from a multi-day 1m window.

    ``bars_window`` must hold enough PRIOR trading days (>= ~6) for the 5-day MA
    features, plus the current day. Columns: symbol, bar_start, bar_end, open,
    high, low, close, volume. Returns one row per 1m decision_time with all
    ``features_merged`` columns + NEUTRAL ``features_relative`` columns + the two
    derived above_ma5 flags. Cross-sectional columns are neutral (see module doc);
    use build_serving_features_cross for REAL cross-symbol values.
    """
    merged = _build_merged(bars_window, source)
    if merged.empty:
        return pd.DataFrame()
    merged = _fill_neutral_relative(merged, active_count)
    return _add_above_ma5(merged).reset_index(drop=True)


def build_serving_features_cross(peer_bars, *, target, source, active_count, fallback=True):
    """Serving features for ``target`` with REAL cross-symbol Step-9 features.

    ``peer_bars`` is a MULTI-symbol 1m window (history + today for all active
    symbols; same columns as build_serving_features). Per decision_time the
    cross-section (rank/relative/breadth) is computed by
    relative_features.build_relative_features over the symbols whose rank inputs are
    complete, then merged onto the target's rows. Causality is preserved:
    build_relative_features only compares rows sharing a decision_time, and the
    caller passes only peer bars with bar_end <= each decision_time. Target rows
    with no aligned peers (active_symbol_count <= 1) fall back to neutral when
    ``fallback`` is True. Returns the SAME columns as build_serving_features for the
    target (so score_serving is unchanged).
    """
    merged_all = _build_merged(peer_bars, source)
    if merged_all.empty:
        return pd.DataFrame()
    tgt = merged_all[merged_all["symbol"] == target].copy()
    if tgt.empty:
        return pd.DataFrame()
    tgt = tgt.drop(columns=[c for c in REL_COLUMNS if c in tgt.columns])

    # real cross-section over symbols with COMPLETE rank inputs at each decision_time;
    # warming-up peers (NaN rank inputs) are dropped so build_relative_features (which
    # raises on NaN) does not fail and does not poison the ranks.
    nonan = merged_all.dropna(subset=RANK_INPUTS)
    if not nonan.empty:
        rel = build_relative_features(nonan)
        rel_t = rel.loc[rel["symbol"] == target, ["decision_time"] + REL_COLUMNS]
        tgt = tgt.merge(rel_t, on="decision_time", how="left", validate="one_to_one")
    else:
        for c in REL_COLUMNS:
            tgt[c] = np.nan

    # neutral fallback where the cross-section is missing or degenerate (singleton)
    if fallback:
        need = tgt[REL_COLUMNS[0]].isna() | (tgt["active_symbol_count"].fillna(1) <= 1)
        if need.any():
            tgt.loc[need, REL_COLUMNS] = _fill_neutral_relative(
                tgt.loc[need].copy(), active_count)[REL_COLUMNS]
    return _add_above_ma5(tgt).reset_index(drop=True)


def score_serving(serving_df, model, preprocessor, *, thresholds: AlertThresholds | None = None):
    """Attach rule scores, directional alert (type/grade/reasons), and risk_prob.

    Only rows whose rule inputs are fully populated (past the warm-up) are scored;
    earlier rows are dropped (they cannot be evaluated live either). Returns a frame
    with RESULT_COLUMNS (+ ``fires`` if ``thresholds`` given), one row per scorable
    decision_time, in time order.
    """
    empty = pd.DataFrame(columns=RESULT_COLUMNS)
    if serving_df is None or len(serving_df) == 0:
        return empty
    missing_inputs = [c for c in RULE_INPUT_COLUMNS if c not in serving_df.columns]
    if missing_inputs:
        raise ValueError(f"serving frame missing rule inputs: {missing_inputs}")

    valid = serving_df[RULE_NUMERIC_INPUTS].notna().all(axis=1)
    work = serving_df.loc[valid].copy()
    if work.empty:
        return empty

    signals = build_rule_signals(work[RULE_INPUT_COLUMNS])
    # `work` already carries every rule INPUT column from build_serving_features; only
    # the SCORE columns are new, so merge just those (dropping any stale copies first).
    work = work.drop(columns=[c for c in RULE_SCORE_COLUMNS if c in work.columns])
    work = work.merge(
        signals[["symbol", "decision_time"] + RULE_SCORE_COLUMNS],
        on=["symbol", "decision_time"], how="left", validate="one_to_one",
    )

    alerts = compute_directional_alerts(work)
    missing_feat = [c for c in preprocessor.feature_columns if c not in alerts.columns]
    if missing_feat:
        raise ValueError(f"serving frame missing model features: {missing_feat}")
    alerts["risk_prob"] = predict_with_model(model, preprocessor, alerts)
    if thresholds is not None:
        alerts["fires"] = alert_fire_gate(alerts, thresholds).to_numpy()

    keep = RESULT_COLUMNS + (["fires"] if thresholds is not None else [])
    return alerts[[c for c in keep if c in alerts.columns]].reset_index(drop=True)
