"""Step 15: conservative DOWNTREND trend-alert module -- pure rule/scoring logic.

Decision-support ONLY: this module classifies what is happening on the DOWN side
of a symbol and emits typed trend alerts for a human -- it is the symmetric
mirror of the long side's directional-alert taxonomy, not an entry-signal
system. It never touches an order API, model files, sqlite, or Kiwoom --
models are passed in already loaded, and all persistence lives in the runner
scripts.

Alert taxonomy (priority order; sign-flipped mirror of directional_alert's
RISK > WEAKENING > UP_CONTINUATION > UP_ALERT > VOLATILITY):

    SHORT_RISK_ALERT      상승 위험          (upside/bounce risk dominant)
    DOWN_WEAKENING_ALERT  하락 추세 약화      (down momentum fading / support bounce)
    DOWN_CONTINUATION     하방 유지 가능성    (strong down trend likely to continue)
    DOWN_ALERT            하방 가능성        (bearish lean)
    VOLATILITY_ALERT      변동성 확대        (big movement, no clear direction)

Sign-flipped mirror of the long-side stack (the long modules are NOT modified):

    directional_alert.compute_directional_alerts -> compute_short_entry_columns
    directional_alert.up_alert_gate              -> short_entry_gate (DOWN types)
    directional_alert.alert_fire_gate            -> short_alert_fire_gate
    realtime_trend_alert.TrendAlertEngine._maybe_fire -> ShortFireGate.maybe_fire
    realtime_features.score_serving              -> score_serving_short

Everything here is computed from features already known at ``decision_time``
(decision_time == the decision bar's bar_end); no label or future price enters.
The v1_short model is ANNOTATION-FIRST (mirror of the long system's risk model):
``short_adverse_prob`` -- P(adverse UP move against the down case) -- and its
``short_risk_band`` are displayed on every alert. The optional veto knob
``short_adverse_prob_max`` defaults to None (annotation only); when explicitly
set it vetoes DOWN types only, and a row with an UNKNOWN probability is then
NOT fired (conservative). Informational types (SHORT_RISK / WEAKENING /
VOLATILITY) are never vetoed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from directional_alert import compute_directional_alerts
from realtime_features import RULE_SCORE_COLUMNS
from risk_model import available_feature_columns, predict_with_model
from rule_signal import INPUT_COLUMNS as RULE_INPUT_COLUMNS
from rule_signal import NUMERIC_INPUTS as RULE_NUMERIC_INPUTS
from rule_signal import build_rule_signals

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
# priority order (first match wins), mirror of directional_alert.ALERT_TYPES
DOWN_ALERT_TYPES = [
    "SHORT_RISK_ALERT",
    "DOWN_WEAKENING_ALERT",
    "DOWN_CONTINUATION",
    "DOWN_ALERT",
    "VOLATILITY_ALERT",
]
DOWN_TYPES = {"DOWN_ALERT", "DOWN_CONTINUATION"}   # mirror of UP_TYPES (veto-able)
NO_ALERT = ""
SHORT_GRADES = ["A", "B", "C", "SHORT_RISK"]    # SHORT_RISK = upside/bounce risk dominates
                                                # (mirror of grade RISK)
SHORT_HIGH, SHORT_MED = 0.60, 0.45              # short_adverse_prob display bands (recalibrate to the
                                                # v1_short decile report after training)

SHORT_ENTRY_COLUMNS = [
    "direction_score",
    "short_score",
    "upside_risk_score",
    "alert_type",
    "short_grade",
    "reason_1",
    "reason_2",
    "reason_3",
]

# score_serving_short output schema (one row per scorable decision_time, time order)
SHORT_RESULT_COLUMNS = [
    "symbol", "decision_time", "close",
    "up_score", "down_score", "direction_score", "short_score", "upside_risk_score",
    "alert_type", "short_grade", "short_fires_rule", "short_adverse_prob",
    "rank_ret_20m_1m", "active_symbol_count", "relative_ret_20m_to_universe",
    "ret_5m_1m", "ret_20m_1m", "volume_z_20m_1m", "dist_ma5_intraday", "vwap_distance_1m",
    "reason_1", "reason_2", "reason_3",
]

# --- mirrored microstructure thresholds (sign-flips of directional_alert's) --- #
SUPPORT_WICK = 0.003         # lower_shadow_1m >= 0.3% -> support/bounce wick (mirror of UPPER_REJECTION)
STRETCHED_VWAP = -0.010      # close <= 1.0% below intraday VWAP -> over-extended down (mirror of EXTENDED_VWAP)
AT_DAY_LOW = 0.001           # within 0.1% of the running day low (mirror of AT_DAY_HIGH)
DROP_FROM_OPEN = -0.020      # -2% from the open (with at-low -> capitulation; mirror of RUN_FROM_OPEN)
VOL_SPIKE = 2.5              # volume_z_20m >= 2.5 -> instability (same value as the long side)
TREND_UP_20M = 0.003         # ret_20m >= +0.3% -> counter-trend drift against the down case
TREND_DN_20M = -0.008        # ret_20m <= -0.8% -> established down trend (mirror of TREND_UP_20M)

# --- type cut-offs (sign-flipped mirrors of directional_alert's type cut-offs) --- #
SHORT_RISK_TYPE_SCORE = 3.0  # upside_risk_score >= 3 and it dominates -> SHORT_RISK_ALERT
DOWN_CONT_SCORE = 4.0        # short_score for a continuation alert
DOWN_CONT_NEAR_LOW = 0.005   # within 0.5% of the day low (mirror of UP_CONT_NEAR_HIGH)
DOWN_ALERT_DIR = 1.5         # direction_score <= -1.5 for a plain down alert
DOWN_ALERT_SCORE = 3.0       # down_score >= 3.0 for a plain down alert
VOL_ELEVATED = 2.0           # volume_z_20m >= 2.0 -> elevated (volatility flag; same as long)
WIDE_RANGE = 0.006           # bar high/low range >= 0.6% (same as long)
VOL_FLAT_DIR = 1.5           # |direction_score| below this -> "no clear direction"

# grade cut-offs (mirror of directional_alert's A/B/RISK grade cut-offs)
SHORT_RISK_GRADE_SCORE = 3.0  # upside_risk_score >= 3 -> grade SHORT_RISK (overrides A/B/C)
A_SCORE, A_RISK, A_MARGIN = 4.5, 1.5, 3.0
B_SCORE, B_RISK, B_MARGIN = 3.5, 2.5, 1.5

# selective FIRING gates for the informational (non-DOWN) alert types (mirror of
# directional_alert's RISK/WEAK/VOL fire constants, sides flipped). The DOWN types
# are gated by the ShortThresholds knobs instead.
SHORT_RISK_FIRE_SCORE = 4.0  # upside_risk_score for a fired SHORT_RISK_ALERT
SHORT_RISK_FIRE_DIR = 3.0    # direction_score (up-leading) for a fired SHORT_RISK_ALERT
WEAK_FIRE_DOWN = 3.0         # was meaningfully bearish before fading
VOL_FIRE_SPIKE = 2.5         # volume_z and ...
VOL_FIRE_RANGE = 0.006       # ... wide range, both, for a fired VOLATILITY_ALERT

# columns the scorer needs; microstructure ones are optional (default 0.0)
REQUIRED_COLUMNS = [
    "up_score", "down_score", "ret_5m_1m", "ret_20m_1m", "volume_z_20m_1m",
    "rank_ret_20m_1m", "active_symbol_count", "relative_ret_20m_to_universe",
    "universe_breadth",
]
OPTIONAL_COLUMNS = [
    "lower_shadow_1m", "vwap_distance_1m", "distance_from_day_low_1m",
    "return_from_open_1m", "range_1m", "dist_ma5_intraday",
]


@dataclass(frozen=True)
class ShortThresholds:
    """Knobs that gate ONLINE firing of the DOWN alert types (mirror of
    AlertThresholds). The informational types use the fixed module-level fire
    constants instead.

    ``name`` is recorded on every fired alert/ledger row so a feed can always be
    traced back to the threshold combination that produced it.
    ``short_adverse_prob_max`` defaults to None: the model annotates instead of
    vetoing; when set, it vetoes DOWN types only.
    """

    name: str = "mirror_baseline"
    down_threshold: float = 4.0          # down_score >= (mirror of up_threshold=4.0)
    up_risk_limit: float | None = 1.0    # up_score <=  (mirror of down_risk_limit; None skips)
    score_margin: float = 3.0            # direction_score <= -score_margin
    bottom_rank_limit: int | None = 5    # rank_ret_20m_1m > active_symbol_count - limit (None skips;
                                         # only meaningful with REAL cross-symbol ranks)
    require_below_ma5: bool = False      # optional confirm: dist_ma5_intraday < 0
    require_rel_weakness: bool = False   # optional confirm: relative_ret_20m_to_universe < 0
    short_adverse_prob_max: float | None = None  # optional model veto on DOWN types:
                                                 # short_adverse_prob >= max -> no fire;
                                                 # None (default) annotates without vetoing
    daily_alert_limit: int | None = None  # None (default) = no daily cap; the per-type cooldown
                                           # is the throttle. Set an int to cap fired alerts/day.
    cooldown_minutes: int = 15           # per alert TYPE (mirror of the engine's de-spam gate);
                                         # keep in the 5-15 min range (the only throttle when
                                         # daily_alert_limit is off)
    min_time: str = "09:10:00"
    max_time: str = "15:00:00"


# --------------------------------------------------------------------------- #
# Down-alert attribute computation (causal; no labels)
# --------------------------------------------------------------------------- #
def _col(frame, name, default=0.0):
    if name in frame.columns:
        return frame[name].astype(float).to_numpy()
    return np.full(len(frame), default, dtype=float)


def compute_short_entry_columns(frame):
    """Return ``frame`` with the Step 15 down-alert columns appended.

    Sign-flipped mirror of :func:`directional_alert.compute_directional_alerts`:
    ``short_score`` confirms the down case (down_score + below-MA5 bonus),
    ``upside_risk_score`` accumulates bounce evidence AGAINST it (up_score +
    support wick + over-extension below VWAP + day-low capitulation + volume
    spike), and ``alert_type`` classifies the row into the 5-type taxonomy in
    DOWN_ALERT_TYPES priority order (empty string = no alert). ``frame`` must
    carry the rule_signals scores plus REQUIRED_COLUMNS; the microstructure
    columns in OPTIONAL_COLUMNS sharpen the reads when present and default to 0
    otherwise. Overwrites ``direction_score`` (same up-down value as the long
    side), ``alert_type`` (short taxonomy) and ``reason_1..3`` (short reasons)
    when a long-alert pass wrote them first.
    """
    if frame is None or len(frame) == 0:
        out = (frame.copy() if frame is not None else pd.DataFrame())
        for col in SHORT_ENTRY_COLUMNS:
            out[col] = pd.Series(dtype="float64" if col.endswith("score") else "object")
        return out

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"short alert inputs missing columns: {missing}")

    up = _col(frame, "up_score")
    down = _col(frame, "down_score")
    ret5 = _col(frame, "ret_5m_1m")
    ret20 = _col(frame, "ret_20m_1m")
    volz = _col(frame, "volume_z_20m_1m")
    rank20 = _col(frame, "rank_ret_20m_1m")
    active = _col(frame, "active_symbol_count")
    rel20 = _col(frame, "relative_ret_20m_to_universe")
    breadth = _col(frame, "universe_breadth")

    lower = _col(frame, "lower_shadow_1m")
    vwapd = _col(frame, "vwap_distance_1m")
    dist_low = _col(frame, "distance_from_day_low_1m")
    from_open = _col(frame, "return_from_open_1m")
    rng = _col(frame, "range_1m")
    dist_ma5 = _col(frame, "dist_ma5_intraday")

    n = len(frame)
    direction = np.round(up - down, 4)

    below_ma5 = dist_ma5 < 0
    short_score = np.round(down + np.where(below_ma5, 1.0, 0.0), 4)

    support_wick = lower >= SUPPORT_WICK
    stretched = vwapd <= STRETCHED_VWAP
    at_low = dist_low <= AT_DAY_LOW
    capitulation = at_low & (from_open <= DROP_FROM_OPEN)
    spike = volz >= VOL_SPIKE
    upside_risk = np.round(
        up
        + np.where(support_wick, 1.5, 0.0)
        + np.where(stretched, 1.0, 0.0)
        + np.where(capitulation, 1.0, 0.0)
        + np.where(spike, 0.5, 0.0),
        4,
    )

    # --- alert_type (priority: SHORT_RISK > DOWN_WEAKENING > DOWN_CONTINUATION
    # > DOWN_ALERT > VOLATILITY; sign-flipped mirror of the long np.select) ---
    is_short_risk = (upside_risk >= SHORT_RISK_TYPE_SCORE) & (upside_risk > short_score)
    fading = ((ret20 < 0) & (ret5 > 0)) | ((down >= WEAK_FIRE_DOWN) & support_wick)
    is_down_weak = fading & ~is_short_risk
    is_down_cont = (
        (short_score >= DOWN_CONT_SCORE)
        & below_ma5
        & (dist_low <= DOWN_CONT_NEAR_LOW)
        & (ret5 < 0)
    ) & ~is_short_risk & ~is_down_weak
    is_down = ((direction <= -DOWN_ALERT_DIR) & (down >= DOWN_ALERT_SCORE)
               & ~is_short_risk & ~is_down_weak & ~is_down_cont)
    vol_flag = (volz >= VOL_ELEVATED) | (rng >= WIDE_RANGE)
    is_vol = (vol_flag & (np.abs(direction) < VOL_FLAT_DIR)
              & ~is_short_risk & ~is_down_weak & ~is_down_cont & ~is_down)

    alert_type = np.select(
        [is_short_risk, is_down_weak, is_down_cont, is_down, is_vol],
        DOWN_ALERT_TYPES,
        default=NO_ALERT,
    ).astype(object)

    # --- short_grade (SHORT_RISK overrides A/B/C, mirror of grade RISK) ---
    is_risk_grade = upside_risk >= SHORT_RISK_GRADE_SCORE
    is_a = (short_score >= A_SCORE) & (upside_risk <= A_RISK) & (direction <= -A_MARGIN) & (rel20 < 0)
    is_b = (short_score >= B_SCORE) & (upside_risk <= B_RISK) & (direction <= -B_MARGIN)
    short_grade = np.select(
        [is_risk_grade, is_a, is_b],
        ["SHORT_RISK", "A", "B"],
        default="C",
    ).astype(object)

    # --- reasons (priority-ordered fill: bounce-risk warnings first, then confirms) ---
    active_n = np.where(active > 0, active, 1)
    rank_bottom = rank20 > active_n - np.maximum(1, np.ceil(active_n / 3.0))
    reason_specs = [
        (ret20 >= TREND_UP_20M, "uptrend_20m"),
        (support_wick, "support_bounce_wick"),
        (stretched, "stretched_below_vwap"),
        (capitulation, "at_day_low_capitulation"),
        (spike, "volume_spike"),
        (ret20 <= TREND_DN_20M, "downtrend_20m"),
        (ret5 < 0, "momentum_falling_5m"),
        (below_ma5, "below_5ma"),
        (rank_bottom, "rank_bottom"),
        (rel20 < 0, "rel_weakness"),
        (breadth < 0.5, "breadth_down"),
    ]
    r1, r2, r3 = (np.full(n, "", dtype=object) for _ in range(3))
    filled = np.zeros(n, dtype=int)
    slots = (r1, r2, r3)
    for mask, label in reason_specs:
        elig = mask & (filled < 3)
        for pos, arr in enumerate(slots):
            arr[elig & (filled == pos)] = label
        filled = filled + elig.astype(int)

    out = frame.copy()
    out["direction_score"] = direction
    out["short_score"] = short_score
    out["upside_risk_score"] = upside_risk
    out["alert_type"] = alert_type
    out["short_grade"] = short_grade
    out["reason_1"] = r1
    out["reason_2"] = r2
    out["reason_3"] = r3
    return out


# --------------------------------------------------------------------------- #
# Rule gates (model veto excluded -- ShortFireGate applies it online)
# --------------------------------------------------------------------------- #
def short_entry_gate(frame, thr: ShortThresholds):
    """Boolean Series: which rows pass the DOWN-type threshold gate under ``thr``.

    Mirror of :func:`directional_alert.up_alert_gate`; the model veto
    (``short_adverse_prob_max``) is deliberately NOT applied here -- it is an
    online check owned by :class:`ShortFireGate`, so the same rule flag can be
    re-gated offline against different veto settings. The gate does not
    hard-exclude SHORT_RISK-graded rows; ``up_risk_limit`` is the knob that,
    when tightened, removes bouncy candidates from the stream. The bottom-rank
    knob only separates rows when the serving frame carries REAL cross-symbol
    ranks: single-symbol serving fills the NEUTRAL mid rank, which never reaches
    the bottom bucket, so non-cross callers must pass ``bottom_rank_limit=None``.
    """
    m = frame["down_score"] >= thr.down_threshold
    if thr.up_risk_limit is not None:
        m &= frame["up_score"] <= thr.up_risk_limit
    m &= frame["direction_score"] <= -thr.score_margin
    if thr.bottom_rank_limit is not None:
        m &= frame["rank_ret_20m_1m"] > frame["active_symbol_count"] - thr.bottom_rank_limit
    if thr.require_below_ma5:
        m &= frame["dist_ma5_intraday"] < 0
    if thr.require_rel_weakness:
        m &= frame["relative_ret_20m_to_universe"] < 0
    return m.fillna(False)


def short_alert_fire_gate(frame, thr: ShortThresholds):
    """Boolean Series: which rows the system would EMIT as a down alert under ``thr``.

    Mirror of :func:`directional_alert.alert_fire_gate`: DOWN_ALERT /
    DOWN_CONTINUATION are gated by the ShortThresholds knobs
    (:func:`short_entry_gate`); the informational types (SHORT_RISK / WEAKENING
    / VOLATILITY) use fixed selective gates so the emitted catalogue stays
    noteworthy rather than covering every row whose type is non-empty.
    """
    down = frame["alert_type"].isin(DOWN_TYPES) & short_entry_gate(frame, thr)
    risk = (
        (frame["alert_type"] == "SHORT_RISK_ALERT")
        & (frame["upside_risk_score"] >= SHORT_RISK_FIRE_SCORE)
        & (frame["direction_score"] >= SHORT_RISK_FIRE_DIR)
    )
    weak = (frame["alert_type"] == "DOWN_WEAKENING_ALERT") & (frame["down_score"] >= WEAK_FIRE_DOWN)
    rng = frame["range_1m"] if "range_1m" in frame.columns else 0.0
    vol = (
        (frame["alert_type"] == "VOLATILITY_ALERT")
        & (frame["volume_z_20m_1m"] >= VOL_FIRE_SPIKE)
        & (rng >= VOL_FIRE_RANGE)
    )
    return (down | risk | weak | vol).fillna(False)


# --------------------------------------------------------------------------- #
# Display band + message
# --------------------------------------------------------------------------- #
def short_risk_band(p):
    """Display band for short_adverse_prob ("NA" when the model gave no score)."""
    if p != p:
        return "NA"
    return "HIGH" if p >= SHORT_HIGH else ("MED" if p >= SHORT_MED else "LOW")


def short_message(row):
    """Human-readable down-alert line (mirror of trend_message): direction from
    the alert type, bounce risk annotated from the model."""
    typ = row.get("alert_type", "")
    grade = row.get("short_grade", "")
    p = float(row.get("short_adverse_prob", float("nan")))
    band = short_risk_band(p)
    head = {
        "DOWN_ALERT": "하방 가능성", "DOWN_CONTINUATION": "하방 유지 가능성",
        "SHORT_RISK_ALERT": "상승 위험", "DOWN_WEAKENING_ALERT": "하락 추세 약화",
        "VOLATILITY_ALERT": "변동성 확대",
    }.get(typ, typ)
    note = ""
    if typ in DOWN_TYPES and band == "LOW":
        note = "  ✓ 저위험 하방"
    elif typ in DOWN_TYPES and band == "HIGH":
        note = "  ⚠ 모델 상승위험: 하방이지만 반등 주의"
    reasons = "/".join(r for r in (row.get("reason_1"), row.get("reason_2")) if r)
    return (f"{head} [{typ} {grade}] "
            f"short_adverse_prob={p:.2f}({band}){note}  reasons={reasons}")


# --------------------------------------------------------------------------- #
# Stateful ONLINE firing gate
# --------------------------------------------------------------------------- #
class ShortFireGate:
    """Stateful firing gate for the down alert types (mirror of the trend
    engine's ``_maybe_fire``). The realtime runner and the replay-feed builder
    must share THIS class so a replayed feed matches what a live session would
    have fired.

    ``cooldown_scope`` selects the de-spam rule (keyword-only; the default
    keeps every existing caller byte-compatible):

    * ``"per_type"`` (default) -- the cooldown is PER ALERT TYPE: the same type
      is suppressed within ``cooldown_minutes`` while a different type fires
      immediately, and each type keeps its own independent timer.
    * ``"single_slot"`` -- the long engine's historical rule (one slot holding
      the LAST fired type+time): the incoming type is suppressed only when it
      equals the last fired type AND is within the cooldown; a different type
      fires immediately and REPLACES the slot, so the earlier type may re-fire
      before its own ``cooldown_minutes`` elapse.

    ``daily_alert_limit`` counts fired alerts of ALL types and is OFF by
    default (None) -- the cooldown is the throttle; set an int to cap fired
    alerts per day.
    State is per calendar day and auto-resets when a row from a new date
    arrives; it is fully separate from the long engine's cooldown state (the
    two must never share state).
    """

    COOLDOWN_SCOPES = ("per_type", "single_slot")

    def __init__(self, symbol, thresholds: ShortThresholds | None = None, *,
                 cooldown_scope: str = "per_type"):
        if cooldown_scope not in self.COOLDOWN_SCOPES:
            raise ValueError(
                f"cooldown_scope must be one of {self.COOLDOWN_SCOPES}, got {cooldown_scope!r}")
        self.symbol = symbol
        self.thresholds = thresholds if thresholds is not None else ShortThresholds()
        self.cooldown_scope = cooldown_scope
        self._cooldown = pd.Timedelta(minutes=self.thresholds.cooldown_minutes)
        self._day = None
        self._last_fire_at = {}     # alert_type -> last fire Timestamp (per_type scope keeps
                                    # one entry per type; single_slot keeps at most ONE entry)
        self._fired_today = 0

    def reset_day(self, day=None):
        """Clear the per-day state (per-type cooldowns + daily count); called on
        date roll."""
        self._day = day
        self._last_fire_at = {}
        self._fired_today = 0

    def maybe_fire(self, row, *, short_adverse_prob):
        """Return the alert dict when ``row`` fires under the thresholds, else None.

        Check order: session window -> rule flag (``short_fires_rule``) -> typed
        row -> model veto -> per-type cooldown -> daily limit. The veto only
        exists when ``short_adverse_prob_max`` is explicitly set (default None =
        annotation only) and then applies ONLY to the DOWN types
        (DOWN_ALERT/DOWN_CONTINUATION); with a configured veto an UNKNOWN
        probability (NaN) does NOT fire a DOWN row (conservative). The
        informational types are never vetoed. The veto and the emitted
        probability use the ``short_adverse_prob`` ARGUMENT (the caller's
        authoritative score), not any same-named row column.
        """
        dt = str(row["decision_time"])
        if dt[:10] != self._day:
            self.reset_day(dt[:10])
        thr = self.thresholds
        tod = dt[11:19]
        if not (thr.min_time <= tod < thr.max_time):
            return None
        if not bool(row.get("short_fires_rule", False)):
            return None
        typ = str(row.get("alert_type", "") or "")
        if not typ:
            return None
        prob = float(short_adverse_prob) if short_adverse_prob is not None else float("nan")
        if thr.short_adverse_prob_max is not None and typ in DOWN_TYPES:
            # NaN fails the `<` comparison too, so an unscored DOWN row is vetoed
            if not (prob < thr.short_adverse_prob_max):
                return None
        t = pd.Timestamp(dt)
        # per_type: each type has its own timer. single_slot: the dict holds at most
        # the LAST fired type, so a different incoming type never finds an entry
        # (fires immediately) and the fire below evicts the previous slot.
        last = self._last_fire_at.get(typ)
        if last is not None and (t - last) < self._cooldown:
            return None
        if thr.daily_alert_limit is not None and self._fired_today >= thr.daily_alert_limit:
            return None
        if self.cooldown_scope == "single_slot":
            self._last_fire_at = {typ: t}
        else:
            self._last_fire_at[typ] = t
        self._fired_today += 1
        alert = {
            "time": dt,
            "symbol": self.symbol,
            "close": float(row["close"]),
            "alert_type": typ,
            "short_grade": row.get("short_grade", ""),
            "up_score": float(row.get("up_score", float("nan"))),
            "down_score": float(row.get("down_score", float("nan"))),
            "direction_score": float(row.get("direction_score", float("nan"))),
            "short_score": float(row.get("short_score", float("nan"))),
            "upside_risk_score": float(row.get("upside_risk_score", float("nan"))),
            "short_adverse_prob": prob,
            "short_risk_band": short_risk_band(prob),
            "reason_1": row.get("reason_1", ""),
            "reason_2": row.get("reason_2", ""),
            "reason_3": row.get("reason_3", ""),
            "thresholds_name": thr.name,
        }
        alert["message"] = short_message(alert)
        return alert


# --------------------------------------------------------------------------- #
# Serving-frame scoring (mirror of realtime_features.score_serving)
# --------------------------------------------------------------------------- #
def score_serving_short(serving_df, short_model, short_pre, *, thresholds):
    """Attach rule scores, down-alert columns, the fire-gate flag, and
    ``short_adverse_prob`` to a serving frame.

    ``short_model``/``short_pre`` are ALREADY-LOADED artifacts (this module never
    reads model files); ``short_model=None`` serves a rule-only frame whose
    ``short_adverse_prob`` is all-NaN. Only rows whose rule inputs are fully
    populated (past the warm-up) are scored; earlier rows are dropped (they
    cannot be evaluated live either). ``short_fires_rule`` (name kept for
    downstream compatibility) is :func:`short_alert_fire_gate` -- typed rows
    that pass their per-type fire condition. Returns SHORT_RESULT_COLUMNS, one
    row per scorable decision_time, in time order.
    """
    empty = pd.DataFrame(columns=SHORT_RESULT_COLUMNS)
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
    work = work.drop(columns=[c for c in RULE_SCORE_COLUMNS if c in work.columns])
    work = work.merge(
        signals[["symbol", "decision_time"] + RULE_SCORE_COLUMNS],
        on=["symbol", "decision_time"], how="left", validate="one_to_one",
    )

    # the long alert columns (alert_score/risk_score/...) are model features for the
    # v1-mirror short model, so compute them BEFORE the short overlay; the overlay
    # then overwrites direction_score (same value), alert_type (short taxonomy) and
    # reason_1..3 (short reasons).
    work = compute_directional_alerts(work)
    work = compute_short_entry_columns(work)
    work["short_fires_rule"] = short_alert_fire_gate(work, thresholds).to_numpy()

    if short_model is None:
        work["short_adverse_prob"] = np.nan
    else:
        missing_feat = [c for c in short_pre.feature_columns if c not in work.columns]
        if missing_feat:
            raise ValueError(f"serving frame missing short-model features: {missing_feat}")
        # leakage-token guard: refuses target/future-like names in the feature set
        available_feature_columns(work, short_pre.feature_columns)
        work["short_adverse_prob"] = predict_with_model(short_model, short_pre, work)

    return work[SHORT_RESULT_COLUMNS].reset_index(drop=True)
