"""Position exit-warning rules for an ALREADY-OPEN manual position (long-only v1).

This is a *decision-support* layer, NOT an order system: it only classifies whether
an open position should be HELD or whether a sell WARNING should be raised, and at
what strength. It never places or cancels orders.

Separation of concerns:
  * the realtime TREND engine (scripts/realtime_trend_alert.py) answers
    "what is the market/trend doing right now?"  -> a trend_snapshot
  * THIS module answers "given that trend snapshot and how my open position has
    behaved since entry, should I hold or get a sell warning?"

Pure + dependency-light (stdlib only) so it unit-tests without Kiwoom, pandas, or a
model. The monitor script feeds it a PositionState (price excursions since entry) and
a trend_snapshot dict (engine output); this module returns an ExitSignal.

Long-only for now; the side is carried on PositionState and the metric math branches
on it, so short-side excursions/rules can be added later without reshaping callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Mapping, Optional

# --------------------------------------------------------------------------- #
# Exit states (weakest -> strongest) and their ordering
# --------------------------------------------------------------------------- #
HOLD = "HOLD"
WEAK_SELL_WARNING = "WEAK_SELL_WARNING"
SELL_WARNING = "SELL_WARNING"
STRONG_SELL_WARNING = "STRONG_SELL_WARNING"

STATE_RANK = {HOLD: 0, WEAK_SELL_WARNING: 1, SELL_WARNING: 2, STRONG_SELL_WARNING: 3}

# Every reason in GLOBAL priority order (highest priority first). The position in
# this list is both the within-state tie-breaker and the order of reason_details.
# Each entry also pins which exit_state the reason belongs to.
REASON_PRIORITY = [
    (STRONG_SELL_WARNING, "hard_stop"),
    (STRONG_SELL_WARNING, "opposite_signal"),
    (STRONG_SELL_WARNING, "risk_high_trend_weakening"),
    (SELL_WARNING,        "profit_giveback_to_breakeven"),
    (SELL_WARNING,        "trailing_drawdown"),
    (SELL_WARNING,        "ma5_vwap_break"),
    (SELL_WARNING,        "time_stop_no_progress"),
    (WEAK_SELL_WARNING,   "ma5_break"),
    (WEAK_SELL_WARNING,   "vwap_break"),
    (WEAK_SELL_WARNING,   "risk_elevated"),
]
_REASON_RANK = {reason: i for i, (_, reason) in enumerate(REASON_PRIORITY)}  # lower = higher priority
_REASON_STATE = {reason: state for state, reason in REASON_PRIORITY}


# --------------------------------------------------------------------------- #
# Data holders
# --------------------------------------------------------------------------- #
@dataclass
class PositionState:
    """An open position's state and price excursions since entry.

    The excursion metrics (current_return / mfe / mae / drawdown_from_peak) are what
    the rules read; build via :meth:`from_prices` so they are derived consistently
    from the raw marks. For a LONG position:
        current_return     = current_price / entry_price - 1
        mfe                = highest_price_since_entry / entry_price - 1   (max favorable)
        mae                = lowest_price_since_entry  / entry_price - 1   (max adverse)
        drawdown_from_peak = current_price / highest_price_since_entry - 1 (<= 0)
    """

    symbol: str
    side: str            # "long" for now; "short" reserved
    entry_time: datetime
    entry_price: float
    current_time: datetime
    current_price: float
    highest_price_since_entry: float
    lowest_price_since_entry: float
    bars_since_entry: int
    current_return: float
    mfe: float
    mae: float
    drawdown_from_peak: float

    @classmethod
    def from_prices(cls, *, symbol, side, entry_time, entry_price, current_time,
                    current_price, highest_price_since_entry, lowest_price_since_entry,
                    bars_since_entry):
        """Build a PositionState, deriving the excursion metrics from raw marks."""
        entry_price = float(entry_price)
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        cur = float(current_price)
        hi = float(highest_price_since_entry)
        lo = float(lowest_price_since_entry)
        if side == "long":
            current_return = cur / entry_price - 1.0
            mfe = hi / entry_price - 1.0
            mae = lo / entry_price - 1.0
            drawdown_from_peak = (cur / hi - 1.0) if hi > 0 else 0.0
        else:
            # Long-only v1. Short metrics mirror the favorable direction; wire the
            # rules for short before enabling a non-long side end to end.
            raise NotImplementedError(f"side {side!r} not supported yet (long-only v1)")
        return cls(
            symbol=symbol, side=side, entry_time=entry_time, entry_price=entry_price,
            current_time=current_time, current_price=cur,
            highest_price_since_entry=hi, lowest_price_since_entry=lo,
            bars_since_entry=int(bars_since_entry), current_return=current_return,
            mfe=mfe, mae=mae, drawdown_from_peak=drawdown_from_peak,
        )


@dataclass
class ExitSignal:
    exit_state: str               # HOLD / WEAK_SELL_WARNING / SELL_WARNING / STRONG_SELL_WARNING
    exit_reason: str              # highest-priority reason at the winning state ("" for HOLD)
    reason_details: List[str] = field(default_factory=list)  # all matched reasons, priority-ordered


@dataclass
class ExitConfig:
    """Exit-rule thresholds (all overridable from the monitor CLI)."""

    hard_stop: float = -0.008

    breakeven_mfe_trigger: float = 0.007
    breakeven_floor: float = 0.001

    trailing_mfe_trigger: float = 0.012
    trailing_drawdown: float = -0.005

    time_stop_minutes: int = 30
    time_stop_min_mfe: float = 0.005
    time_stop_floor: float = 0.001

    risk_high_threshold: float = 0.80
    risk_med_threshold: float = 0.60


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def _below(close, level, above_flag) -> bool:
    """Is price below a structural line? Prefer an explicit close-vs-level compare;
    fall back to the boolean ``above_*`` flag; if neither is known, NOT below (we
    don't fire a break we can't confirm)."""
    if _is_number(close) and _is_number(level):
        return close < level
    if above_flag is not None:
        return above_flag is False
    return False


# weak trend states that, combined with high risk, justify a strong exit (rule 5)
_WEAK_TREND_STATES = {"RISK", "WEAKENING", "CHOPPY", "DOWN"}
# explicit opposite-direction alert types (none exist in the current rule set, but
# kept so a future DOWN/SHORT alert flows straight into the opposite-signal rule)
_OPPOSITE_ALERT_TYPES = {"DOWN_ALERT", "SHORT_ALERT"}


def evaluate_exit(position: PositionState, trend_snapshot: Optional[Mapping],
                  config: Optional[ExitConfig] = None) -> ExitSignal:
    """Classify an open LONG position into HOLD / *_SELL_WARNING.

    ``trend_snapshot`` is a dict-like from the trend engine; any of these keys is
    used when present: trend_state, alert_type, alert_grade, risk_prob, risk_bucket,
    close, ma5, vwap, above_ma5, above_vwap. Missing keys simply disable the rules
    that need them (no false warnings). When several rules fire, the STRONGEST state
    wins; the primary ``exit_reason`` is the highest-priority reason at that state and
    ``reason_details`` lists every matched reason in priority order.
    """
    if position.side != "long":
        raise NotImplementedError("evaluate_exit currently supports long positions only")
    config = config or ExitConfig()
    snap = trend_snapshot or {}

    ret = position.current_return
    mfe = position.mfe
    dd = position.drawdown_from_peak

    risk = snap.get("risk_prob")
    trend_state = snap.get("trend_state")
    alert_type = snap.get("alert_type")
    close = snap.get("close")
    ma5 = snap.get("ma5")
    vwap = snap.get("vwap")

    matched: List[str] = []

    # 1) hard stop
    if ret <= config.hard_stop:
        matched.append("hard_stop")

    # 6) opposite / down signal
    if alert_type in _OPPOSITE_ALERT_TYPES or trend_state == "DOWN":
        matched.append("opposite_signal")

    # 5) elevated risk (+ weakening trend -> strong)
    if _is_number(risk):
        if risk >= config.risk_high_threshold and trend_state in _WEAK_TREND_STATES:
            matched.append("risk_high_trend_weakening")
        elif risk >= config.risk_med_threshold:
            matched.append("risk_elevated")

    # 2) profit giveback / break-even protection
    if mfe >= config.breakeven_mfe_trigger and ret <= config.breakeven_floor:
        matched.append("profit_giveback_to_breakeven")

    # 3) trailing profit protection
    if mfe >= config.trailing_mfe_trigger and dd <= config.trailing_drawdown:
        matched.append("trailing_drawdown")

    # 4) trend invalidation (MA5 / VWAP structure)
    below_ma5 = _below(close, ma5, snap.get("above_ma5"))
    below_vwap = _below(close, vwap, snap.get("above_vwap"))
    if below_ma5 and below_vwap:
        matched.append("ma5_vwap_break")
    elif below_ma5:
        matched.append("ma5_break")
    elif below_vwap:
        matched.append("vwap_break")

    # 7) time stop (no progress)
    if (position.bars_since_entry >= config.time_stop_minutes
            and mfe < config.time_stop_min_mfe and ret <= config.time_stop_floor):
        matched.append("time_stop_no_progress")

    if not matched:
        return ExitSignal(exit_state=HOLD, exit_reason="", reason_details=[])

    ordered = sorted(set(matched), key=lambda r: _REASON_RANK[r])
    strongest_rank = max(STATE_RANK[_REASON_STATE[r]] for r in ordered)
    state = next(s for s, rank in STATE_RANK.items() if rank == strongest_rank)
    primary = next(r for r in ordered if STATE_RANK[_REASON_STATE[r]] == strongest_rank)
    return ExitSignal(exit_state=state, exit_reason=primary, reason_details=ordered)
