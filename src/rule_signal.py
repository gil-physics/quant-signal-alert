"""Step 11: rule-based signal scoring v0.

The input is Step 9 ``features_relative``. Labels and future prices are not used
here; this module only reads features that were already available at
``decision_time``.
"""

from __future__ import annotations

import math


INPUT_COLUMNS = [
    "symbol",
    "source",
    "decision_time",
    "close",
    "ret_5m_1m",
    "ret_20m_1m",
    "volume_z_20m_1m",
    "active_symbol_count",
    "rank_ret_5m_1m",
    "rank_ret_20m_1m",
    "rank_volume_z_20m_1m",
    "universe_mean_ret_20m",
    "relative_ret_20m_to_universe",
    "relative_volume_z_to_universe",
    "universe_breadth",
]

NUMERIC_INPUTS = [c for c in INPUT_COLUMNS if c not in {"symbol", "source", "decision_time"}]

SCORE_COLUMNS = [
    "up_score",
    "down_score",
    "is_up_candidate",
    "is_down_candidate",
    "primary_side",
    "reason_1",
    "reason_2",
    "reason_3",
    "up_reasons",
    "down_reasons",
]

OUTPUT_COLUMNS = INPUT_COLUMNS + SCORE_COLUMNS

DEFAULT_UP_THRESHOLD = 3.0
DEFAULT_DOWN_THRESHOLD = 3.0


def _top_bucket(rank, active_count):
    if active_count <= 1:
        return rank <= 1
    return rank <= max(1, math.ceil(active_count / 3))


def _bottom_bucket(rank, active_count):
    if active_count <= 1:
        return rank <= 1
    return rank > active_count - max(1, math.ceil(active_count / 3))


def _score_row(row):
    up_score = 0.0
    down_score = 0.0
    up_reasons = []
    down_reasons = []

    ret5 = float(row["ret_5m_1m"])
    ret20 = float(row["ret_20m_1m"])
    volz = float(row["volume_z_20m_1m"])
    rel_ret20 = float(row["relative_ret_20m_to_universe"])
    rel_vol = float(row["relative_volume_z_to_universe"])
    breadth = float(row["universe_breadth"])
    active_count = int(row["active_symbol_count"])
    rank5 = int(row["rank_ret_5m_1m"])
    rank20 = int(row["rank_ret_20m_1m"])
    rank_vol = int(row["rank_volume_z_20m_1m"])

    if ret20 >= 0.008:
        up_score += 2.0
        up_reasons.append("ret20>=0.8%")
    elif ret20 > 0.003:
        up_score += 1.0
        up_reasons.append("ret20>0.3%")

    if ret20 <= -0.008:
        down_score += 2.0
        down_reasons.append("ret20<=-0.8%")
    elif ret20 < -0.003:
        down_score += 1.0
        down_reasons.append("ret20<-0.3%")

    if ret5 > 0:
        up_score += 0.75
        up_reasons.append("ret5>0")
    elif ret5 < 0:
        down_score += 0.75
        down_reasons.append("ret5<0")

    if volz >= 2.0:
        up_score += 1.0
        down_score += 1.0
        up_reasons.append("volume_z>=2")
        down_reasons.append("volume_z>=2")
    elif volz >= 1.0:
        up_score += 0.5
        down_score += 0.5
        up_reasons.append("volume_z>=1")
        down_reasons.append("volume_z>=1")

    if rel_ret20 > 0:
        up_score += 1.0
        up_reasons.append("relative_ret20>0")
    elif rel_ret20 < 0:
        down_score += 1.0
        down_reasons.append("relative_ret20<0")

    if _top_bucket(rank20, active_count):
        up_score += 1.0
        up_reasons.append("rank_ret20_top")
    if _bottom_bucket(rank20, active_count):
        down_score += 1.0
        down_reasons.append("rank_ret20_bottom")

    if _top_bucket(rank5, active_count):
        up_score += 0.5
        up_reasons.append("rank_ret5_top")
    if _bottom_bucket(rank5, active_count):
        down_score += 0.5
        down_reasons.append("rank_ret5_bottom")

    if _top_bucket(rank_vol, active_count) or rel_vol > 0:
        up_score += 0.25
        down_score += 0.25
        up_reasons.append("relative_volume_active")
        down_reasons.append("relative_volume_active")

    if breadth > 0.5:
        up_score += 0.5
        up_reasons.append("universe_breadth>0.5")
    elif breadth < 0.5:
        down_score += 0.5
        down_reasons.append("universe_breadth<0.5")

    return round(up_score, 4), round(down_score, 4), up_reasons, down_reasons


def _primary_side(up_score, down_score, is_up, is_down):
    if is_up and not is_down:
        return "UP"
    if is_down and not is_up:
        return "DOWN"
    if is_up and is_down:
        if up_score > down_score:
            return "UP"
        if down_score > up_score:
            return "DOWN"
        return "MIXED"
    return "NONE"


def build_rule_signals(features, up_threshold=DEFAULT_UP_THRESHOLD, down_threshold=DEFAULT_DOWN_THRESHOLD):
    """Build rule signal rows from a features_relative-like DataFrame."""
    import pandas as pd

    if features is None or len(features) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    missing = [c for c in INPUT_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"features_relative missing required columns: {missing}")

    df = features[INPUT_COLUMNS].copy()
    nan_mask = df[NUMERIC_INPUTS].isna().any(axis=1)
    if nan_mask.any():
        sample = df.loc[nan_mask, ["symbol", "decision_time"]].head(5).to_dict("records")
        raise ValueError(f"rule signal inputs contain NaN in {int(nan_mask.sum())} row(s); e.g. {sample}")

    rows = []
    for record in df.to_dict("records"):
        up_score, down_score, up_reasons, down_reasons = _score_row(record)
        is_up = int(up_score >= up_threshold)
        is_down = int(down_score >= down_threshold)
        side = _primary_side(up_score, down_score, is_up, is_down)
        if side == "DOWN":
            main_reasons = down_reasons
        elif side in {"UP", "MIXED"}:
            main_reasons = up_reasons
        else:
            main_reasons = up_reasons if up_score >= down_score else down_reasons

        out = dict(record)
        out.update({
            "up_score": up_score,
            "down_score": down_score,
            "is_up_candidate": is_up,
            "is_down_candidate": is_down,
            "primary_side": side,
            "reason_1": main_reasons[0] if len(main_reasons) > 0 else "",
            "reason_2": main_reasons[1] if len(main_reasons) > 1 else "",
            "reason_3": main_reasons[2] if len(main_reasons) > 2 else "",
            "up_reasons": ";".join(up_reasons),
            "down_reasons": ";".join(down_reasons),
        })
        rows.append(out)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def summarize_rule_signals(signals):
    """Compact count summary for logging/printing."""
    total = int(len(signals))
    if total == 0:
        return {
            "rows": 0,
            "up_candidates": 0,
            "down_candidates": 0,
            "both_candidates": 0,
            "symbols": 0,
        }

    return {
        "rows": total,
        "symbols": int(signals["symbol"].nunique()),
        "up_candidates": int(signals["is_up_candidate"].sum()),
        "down_candidates": int(signals["is_down_candidate"].sum()),
        "both_candidates": int(((signals["is_up_candidate"] == 1) & (signals["is_down_candidate"] == 1)).sum()),
    }
