"""Step 13_5: rule_signals reframed as a DIRECTIONAL ALERT system.

The earlier steps treated ``rule_signals`` as automated entry signals. Step 13_5
drops that framing: a row is no longer "buy now" but an *alert candidate* carrying
a *type* and a *grade* that tell a human what is happening:

    UP_ALERT          상방 가능성        (bullish lean)
    UP_CONTINUATION   상방 유지 가능성    (strong trend likely to continue)
    RISK_ALERT        하방 위험          (downside risk dominant)
    WEAKENING_ALERT   추세 약화          (momentum fading / rejection)
    VOLATILITY_ALERT  변동성 확대        (big movement, no clear direction)

Grades A/B/C rank the *quality* of a (mostly bullish) alert; RISK is a separate
grade flagging that downside risk dominates regardless of the bull case.

Everything here is computed from features already known at ``decision_time``
(rule_signals scores + leakage-safe microstructure features). NO label or future
price ever enters alert generation -- labels are evaluation-only and live in the
backtest script, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
ALERT_TYPES = ["UP_ALERT", "UP_CONTINUATION", "RISK_ALERT", "WEAKENING_ALERT", "VOLATILITY_ALERT"]
ALERT_GRADES = ["A", "B", "C", "RISK"]
UP_TYPES = {"UP_ALERT", "UP_CONTINUATION"}
NO_ALERT = ""

ALERT_COLUMNS = [
    "alert_score",
    "risk_score",
    "direction_score",
    "alert_type",
    "alert_grade",
    "reason_1",
    "reason_2",
    "reason_3",
]

# --- score / type / grade thresholds (in feature units; returns are fractions) --- #
# microstructure risk triggers
UPPER_REJECTION = 0.003      # upper_shadow_1m >= 0.3% of price -> rejection wick
EXTENDED_VWAP = 0.010        # close >= 1.0% above intraday VWAP -> stretched
AT_DAY_HIGH = -0.001         # within 0.1% of the running day high
RUN_FROM_OPEN = 0.020        # +2% from the open (used with at-high -> exhaustion)
VOL_SPIKE = 2.5              # volume_z_20m >= 2.5 -> instability
VOL_ELEVATED = 2.0           # volume_z_20m >= 2.0 -> elevated (volatility flag)
WIDE_RANGE = 0.006           # bar high/low range >= 0.6%
TREND_UP_20M = 0.008         # ret_20m >= 0.8% -> strong up trend
TREND_DN_20M = -0.003        # ret_20m <= -0.3% -> down trend

# type cut-offs
RISK_TYPE_SCORE = 3.0        # risk_score >= 3 and risk dominates -> RISK_ALERT
UP_CONT_SCORE = 4.0          # alert_score for a continuation alert
UP_CONT_NEAR_HIGH = -0.005   # within 0.5% of day high
UP_ALERT_DIR = 1.5           # direction_score for a plain up alert
UP_ALERT_SCORE = 3.0
VOL_FLAT_DIR = 1.5           # |direction_score| below this -> "no clear direction"

# grade cut-offs
RISK_GRADE_SCORE = 3.0       # risk_score >= 3 -> grade RISK (overrides A/B/C)
A_SCORE, A_RISK, A_MARGIN = 4.5, 1.5, 3.0
B_SCORE, B_RISK, B_MARGIN = 3.5, 2.5, 1.5

# selective FIRING gates for the informational (non-up) alert types. These keep
# the emitted stream noteworthy -- the raw typed population is far too broad to
# surface. The up-types are gated by the swept AlertThresholds instead.
RISK_FIRE_SCORE = 4.0        # risk_score for a fired RISK_ALERT
RISK_FIRE_DIR = -3.0         # direction_score (down-leading) for a fired RISK_ALERT
WEAK_FIRE_UP = 3.0           # was meaningfully bullish before fading
VOL_FIRE_SPIKE = 2.5         # volume_z and ...
VOL_FIRE_RANGE = 0.006       # ... wide range, both, for a fired VOLATILITY_ALERT

# columns the scorer needs; microstructure ones are optional (default 0.0)
REQUIRED_COLUMNS = [
    "up_score", "down_score", "ret_5m_1m", "ret_20m_1m", "volume_z_20m_1m",
    "rank_ret_20m_1m", "active_symbol_count", "relative_ret_20m_to_universe",
    "universe_breadth",
]
OPTIONAL_COLUMNS = [
    "upper_shadow_1m", "vwap_distance_1m", "distance_from_day_high_1m",
    "return_from_open_1m", "range_1m", "dist_ma5_intraday",
]


@dataclass(frozen=True)
class AlertThresholds:
    """Knobs that gate ONLINE alert firing (Step 13_5 instruction 10)."""

    name: str = "baseline"
    up_threshold: float = 4.0
    score_margin: float = 3.0
    down_risk_limit: float | None = 2.0
    current_rank_limit: int = 5
    daily_alert_limit: int = 5
    cooldown_minutes: int = 30
    entry_delay_minutes: int = 3


# --------------------------------------------------------------------------- #
# Alert attribute computation (causal; no labels)
# --------------------------------------------------------------------------- #
def _col(frame, name, default=0.0):
    import pandas as pd

    if name in frame.columns:
        return frame[name].astype(float).to_numpy()
    return np.full(len(frame), default, dtype=float)


def compute_directional_alerts(frame):
    """Return ``frame`` with the Step 13_5 alert columns appended.

    ``frame`` must carry the rule_signals scores plus the relative features in
    ``REQUIRED_COLUMNS``; the microstructure columns in ``OPTIONAL_COLUMNS``
    sharpen the risk/volatility reads when present and default to 0 otherwise.
    """
    import pandas as pd

    if frame is None or len(frame) == 0:
        out = (frame.copy() if frame is not None else pd.DataFrame())
        for col in ALERT_COLUMNS:
            out[col] = pd.Series(dtype="float64" if col.endswith("score") else "object")
        return out

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"directional alert inputs missing columns: {missing}")

    up = _col(frame, "up_score")
    down = _col(frame, "down_score")
    ret5 = _col(frame, "ret_5m_1m")
    ret20 = _col(frame, "ret_20m_1m")
    volz = _col(frame, "volume_z_20m_1m")
    rank20 = _col(frame, "rank_ret_20m_1m")
    active = _col(frame, "active_symbol_count")
    rel20 = _col(frame, "relative_ret_20m_to_universe")
    breadth = _col(frame, "universe_breadth")

    upper = _col(frame, "upper_shadow_1m")
    vwapd = _col(frame, "vwap_distance_1m")
    dist_high = _col(frame, "distance_from_day_high_1m")
    from_open = _col(frame, "return_from_open_1m")
    rng = _col(frame, "range_1m")
    dist_ma5 = _col(frame, "dist_ma5_intraday")

    n = len(frame)
    direction = np.round(up - down, 4)

    above_ma5 = dist_ma5 > 0
    alert_score = np.round(up + np.where(above_ma5, 1.0, 0.0), 4)

    rejection = upper >= UPPER_REJECTION
    extended = vwapd >= EXTENDED_VWAP
    at_high = dist_high >= AT_DAY_HIGH
    exhaustion = at_high & (from_open >= RUN_FROM_OPEN)
    spike = volz >= VOL_SPIKE
    risk_score = np.round(
        down
        + np.where(rejection, 1.5, 0.0)
        + np.where(extended, 1.0, 0.0)
        + np.where(exhaustion, 1.0, 0.0)
        + np.where(spike, 0.5, 0.0),
        4,
    )

    # --- alert_type (priority: RISK > WEAKENING > UP_CONTINUATION > UP_ALERT > VOLATILITY) ---
    is_risk = (risk_score >= RISK_TYPE_SCORE) & (risk_score > alert_score)
    fading = ((ret20 > 0) & (ret5 < 0)) | ((up >= 3.0) & rejection)
    is_weak = fading & ~is_risk
    is_cont = (
        (alert_score >= UP_CONT_SCORE)
        & above_ma5
        & (dist_high >= UP_CONT_NEAR_HIGH)
        & (ret5 > 0)
    ) & ~is_risk & ~is_weak
    is_up = (direction >= UP_ALERT_DIR) & (up >= UP_ALERT_SCORE) & ~is_risk & ~is_weak & ~is_cont
    vol_flag = (volz >= VOL_ELEVATED) | (rng >= WIDE_RANGE)
    is_vol = vol_flag & (np.abs(direction) < VOL_FLAT_DIR) & ~is_risk & ~is_weak & ~is_cont & ~is_up

    alert_type = np.select(
        [is_risk, is_weak, is_cont, is_up, is_vol],
        ["RISK_ALERT", "WEAKENING_ALERT", "UP_CONTINUATION", "UP_ALERT", "VOLATILITY_ALERT"],
        default=NO_ALERT,
    ).astype(object)

    # --- alert_grade ---
    has_alert = alert_type != NO_ALERT
    is_a = (alert_score >= A_SCORE) & (risk_score <= A_RISK) & (direction >= A_MARGIN) & (rel20 > 0)
    is_b = (alert_score >= B_SCORE) & (risk_score <= B_RISK) & (direction >= B_MARGIN)
    grade_risk = (risk_score >= RISK_GRADE_SCORE) | (alert_type == "RISK_ALERT")
    alert_grade = np.select(
        [~has_alert, grade_risk, is_a, is_b],
        [NO_ALERT, "RISK", "A", "B"],
        default="C",
    ).astype(object)

    # --- reasons (priority-ordered fill, type-agnostic but risk-first) ---
    rank_top = rank20 <= np.maximum(1, np.ceil(np.where(active > 0, active, 1) / 3.0))
    reason_specs = [
        (ret20 <= TREND_DN_20M, "downtrend_20m"),
        (rejection, "upper_rejection"),
        (extended, "extended_above_vwap"),
        (exhaustion, "at_day_high_exhaustion"),
        (spike, "volume_spike"),
        ((ret20 > 0) & (ret5 < 0), "momentum_fading_5m"),
        (rng >= WIDE_RANGE, "wide_range"),
        (ret20 >= TREND_UP_20M, "uptrend_20m"),
        (above_ma5, "above_5ma"),
        (rank_top, "rank_top"),
        (rel20 > 0, "rel_strength"),
        (volz >= 1.0, "volume_support"),
        (breadth > 0.5, "breadth_up"),
    ]
    r1, r2, r3 = (np.full(n, "", dtype=object) for _ in range(3))
    filled = np.zeros(n, dtype=int)
    slots = (r1, r2, r3)
    for mask, label in reason_specs:
        elig = mask & has_alert & (filled < 3)
        for pos, arr in enumerate(slots):
            arr[elig & (filled == pos)] = label
        filled = filled + elig.astype(int)

    out = frame.copy()
    out["alert_score"] = alert_score
    out["risk_score"] = risk_score
    out["direction_score"] = direction
    out["alert_type"] = alert_type
    out["alert_grade"] = alert_grade
    out["reason_1"] = r1
    out["reason_2"] = r2
    out["reason_3"] = r3
    return out


# --------------------------------------------------------------------------- #
# Online firing gate + causal selection
# --------------------------------------------------------------------------- #
def up_alert_gate(alerts, thr: AlertThresholds):
    """Boolean Series: which rows fire as a (bullish) directional alert under ``thr``.

    Only UP_ALERT / UP_CONTINUATION types are gated by the swept knobs; the gate
    deliberately does *not* hard-exclude high risk_score, so a stretched/rejected
    but still-bullish row fires and is then graded RISK. ``down_risk_limit`` is the
    knob that, when tightened, removes those risky alerts from the stream.
    """
    import pandas as pd

    m = alerts["alert_type"].isin(UP_TYPES)
    m &= alerts["up_score"] >= thr.up_threshold
    m &= alerts["direction_score"] >= thr.score_margin
    m &= alerts["rank_ret_20m_1m"] <= thr.current_rank_limit
    if thr.down_risk_limit is not None:
        m &= alerts["down_score"] <= thr.down_risk_limit
    return m.fillna(False)


def alert_fire_gate(alerts, thr: AlertThresholds):
    """Boolean Series: which rows the system would EMIT as an alert under ``thr``.

    Up-direction alerts are gated by the swept knobs (:func:`up_alert_gate`); the
    informational types (RISK / WEAKENING / VOLATILITY) use fixed selective gates
    so the emitted catalogue stays noteworthy rather than covering every row whose
    type is non-empty.
    """
    import pandas as pd

    up = up_alert_gate(alerts, thr)
    risk = (
        (alerts["alert_type"] == "RISK_ALERT")
        & (alerts["risk_score"] >= RISK_FIRE_SCORE)
        & (alerts["direction_score"] <= RISK_FIRE_DIR)
    )
    weak = (alerts["alert_type"] == "WEAKENING_ALERT") & (alerts["up_score"] >= WEAK_FIRE_UP)
    rng = alerts["range_1m"] if "range_1m" in alerts.columns else 0.0
    vol = (
        (alerts["alert_type"] == "VOLATILITY_ALERT")
        & (alerts["volume_z_20m_1m"] >= VOL_FIRE_SPIKE)
        & (rng >= VOL_FIRE_RANGE)
    )
    return (up | risk | weak | vol).fillna(False)


def _with_day_dt(frame):
    import pandas as pd

    df = frame.copy()
    if "date" not in df.columns:
        df["date"] = df["decision_time"].str.slice(0, 10)
    if "_decision_dt" not in df.columns:
        df["_decision_dt"] = pd.to_datetime(df["decision_time"])
    return df


def select_online_alerts(gated, *, daily_alert_limit, cooldown_minutes):
    """Fire alerts in time order each day (causal), capping per day and applying a
    per-symbol cooldown. No future row influences an earlier firing decision.

    Returns the fired rows with an ``alert_seq`` (1-based within-day order).
    """
    import pandas as pd

    if gated is None or len(gated) == 0:
        cols = list(gated.columns) if gated is not None else []
        return pd.DataFrame(columns=cols + ["alert_seq"])

    df = _with_day_dt(gated)
    kept_idx = []
    seqs = {}
    for date, day in df.groupby("date", sort=True):
        day = day.sort_values(["_decision_dt", "symbol"])
        last_fire = {}
        count = 0
        for idx, t, sym in zip(day.index, day["_decision_dt"], day["symbol"]):
            if count >= daily_alert_limit:
                break
            prior = last_fire.get(sym)
            if prior is not None and (t - prior).total_seconds() < cooldown_minutes * 60:
                continue
            last_fire[sym] = t
            count += 1
            kept_idx.append(idx)
            seqs[idx] = count

    out = df.loc[kept_idx].copy()
    out["alert_seq"] = [seqs[i] for i in out.index]
    return out.reset_index(drop=True)


def select_topk_diagnostic(gated, *, top_k=2):
    """Daily top-k by alert_score -- RANKING DIAGNOSTIC ONLY (sees the whole day,
    so it is not causal). Kept to compare against the online stream, never to trade.
    """
    import pandas as pd

    if gated is None or len(gated) == 0:
        cols = list(gated.columns) if gated is not None else []
        return pd.DataFrame(columns=cols + ["daily_rank"])

    df = _with_day_dt(gated)
    sort_cols = ["alert_score", "direction_score", "_decision_dt"]
    ascending = [False, False, True]
    rows = []
    for _, day in df.groupby("date", sort=True):
        ordered = day.sort_values(sort_cols, ascending=ascending).head(top_k).copy()
        ordered["daily_rank"] = range(1, len(ordered) + 1)
        rows.append(ordered)
    out = pd.concat(rows, ignore_index=True) if rows else df.iloc[0:0].assign(daily_rank=[])
    return out.reset_index(drop=True)
