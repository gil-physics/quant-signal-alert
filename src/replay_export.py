"""Pure helpers for the replay-app export (scripts/export_replay_bundle.py).

The replay app (app/) re-plays a past trading day minute by minute and shows the
alerts the realtime runners would have printed. Nothing is recomputed in the
browser: the exporter runs the runners' own replay code and stores the results.
This module holds the side-effect-free pieces so they can be unit-tested:

  * Korean display labels for alert types and reason tokens,
  * the direction category that drives the app's colours (the runner's ▲ means
    "the LONG engine fired", which includes its downside RISK_ALERT, so the app
    colours by what the alert MEANS instead),
  * forward-outcome math with an explicit ``reveal`` bar, so the app can hold an
    outcome back until the replay clock has actually passed its window.

Decision-support/education only -- no orders, no Kiwoom, no SendOrder.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
LONG_TYPE_KO = {
    "UP_ALERT": "상방 가능성",
    "UP_CONTINUATION": "상방 유지 가능성",
    "RISK_ALERT": "하방 위험",
    "WEAKENING_ALERT": "추세 약화",
    "VOLATILITY_ALERT": "변동성 확대",
}
SHORT_TYPE_KO = {
    "DOWN_ALERT": "하방 가능성",
    "DOWN_CONTINUATION": "하방 유지 가능성",
    "SHORT_RISK_ALERT": "상승 위험",
    "DOWN_WEAKENING_ALERT": "하락 추세 약화",
    "VOLATILITY_ALERT": "변동성 확대",
}

# what the alert means for price direction (app colour/icon), per side
LONG_TYPE_DIR = {
    "UP_ALERT": "up", "UP_CONTINUATION": "up", "RISK_ALERT": "down",
    "WEAKENING_ALERT": "neutral", "VOLATILITY_ALERT": "neutral",
}
SHORT_TYPE_DIR = {
    "DOWN_ALERT": "down", "DOWN_CONTINUATION": "down", "SHORT_RISK_ALERT": "up",
    "DOWN_WEAKENING_ALERT": "neutral", "VOLATILITY_ALERT": "neutral",
}

# reason tokens (directional_alert.py / short_entry.py) -> short Korean phrases
REASON_KO = {
    "downtrend_20m": "20분 하락 추세",
    "uptrend_20m": "20분 상승 추세",
    "upper_rejection": "윗꼬리 저항",
    "extended_above_vwap": "VWAP 위로 과열",
    "at_day_high_exhaustion": "당일 고점 부근 힘 빠짐",
    "volume_spike": "거래량 급증",
    "momentum_fading_5m": "5분 모멘텀 둔화",
    "wide_range": "변동폭 확대",
    "above_5ma": "5일 평균가 위",
    "rank_top": "종목 간 상승률 상위",
    "rel_strength": "시장 대비 강함",
    "volume_support": "거래량 뒷받침",
    "breadth_up": "시장 전반 상승",
    "support_bounce_wick": "아래꼬리 반등",
    "stretched_below_vwap": "VWAP 아래로 과매도",
    "at_day_low_capitulation": "당일 저점 부근 투매",
    "momentum_falling_5m": "5분 모멘텀 하락",
    "below_5ma": "5일 평균가 아래",
    "rank_bottom": "종목 간 상승률 하위",
    "rel_weakness": "시장 대비 약함",
    "breadth_down": "시장 전반 하락",
}

BAND_KO = {"HIGH": "높음", "MED": "보통", "LOW": "낮음", "NA": "없음"}

FORWARD_MINUTES = 60          # the "이후 60분" outcome window (matches *_60m labels)
TOUCH_WINDOW_MINUTES = 10     # the touch label window (touch_model.TOUCH_WINDOW_MIN)


def reasons_ko(tokens: Iterable[Optional[str]]) -> List[str]:
    """Reason tokens -> Korean phrases (unknown tokens pass through unchanged)."""
    out = []
    for t in tokens:
        t = (t or "").strip()
        if t:
            out.append(REASON_KO.get(t, t))
    return out


def parse_message_reasons(message: str) -> List[str]:
    """The runners append ``  reasons=a/b`` to every alert message."""
    if not message or "reasons=" not in message:
        return []
    tail = message.rsplit("reasons=", 1)[1].strip()
    return [t for t in tail.split("/") if t]


def type_ko(side: str, alert_type: str) -> str:
    table = LONG_TYPE_KO if side == "long" else SHORT_TYPE_KO
    return table.get(alert_type, alert_type)


def type_dir(side: str, alert_type: str) -> str:
    table = LONG_TYPE_DIR if side == "long" else SHORT_TYPE_DIR
    return table.get(alert_type, "neutral")


def finite_or_none(x, digits: int = 4):
    """JSON-safe float (NaN/inf/None -> None), rounded to keep bundles small."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, digits)


def hhmm(ts: str) -> str:
    """'YYYY-MM-DD HH:MM:SS' -> 'HH:MM'."""
    return str(ts)[11:16]


def bar_label_for_decision(decision_time: str) -> str:
    """decision_time (= the feature bar's bar_end) -> that bar's bar_start HH:MM,
    the chart-candle label every app element is keyed on (bars are 1 minute)."""
    return (pd.Timestamp(decision_time) - pd.Timedelta(minutes=1)).strftime("%H:%M")


# --------------------------------------------------------------------------- #
# forward outcomes (evaluation only -- the app reveals them after their window)
# --------------------------------------------------------------------------- #
def forward_window(day_bars: pd.DataFrame, start_ts: str, minutes: Optional[int]):
    """Bars with bar_start in [start_ts, start_ts + minutes) -- the same half-open
    window as labeling.py / risk_model.add_future_risk_targets. ``minutes=None``
    means "to the end of the day". ``day_bars`` = one symbol, one day, sorted."""
    t0 = pd.Timestamp(start_ts)
    bs = pd.to_datetime(day_bars["bar_start"])
    mask = bs >= t0
    if minutes is not None:
        mask &= bs < t0 + pd.Timedelta(minutes=int(minutes))
    return day_bars[mask.to_numpy()]


def forward_outcome(day_bars: pd.DataFrame, decision_time: str, ref_price: float,
                    minutes: Optional[int]) -> Optional[Dict]:
    """Max-up / max-down excursion vs ``ref_price`` over the forward window that
    starts AT decision_time (the first bar after the decision). Returns None when
    the window is empty (e.g. a decision on the closing bar). ``reveal`` is the
    bar label of the window's last bar: the app shows the outcome only once the
    replay clock reaches it, so no future value leaks into the replay."""
    win = forward_window(day_bars, decision_time, minutes)
    if win.empty or not ref_price:
        return None
    ref = float(ref_price)
    hi = float(win["high"].astype(float).max()) / ref - 1.0
    lo = float(win["low"].astype(float).min()) / ref - 1.0
    full = True
    if minutes is not None:
        expected_last = pd.Timestamp(decision_time) + pd.Timedelta(minutes=int(minutes) - 1)
        full = pd.Timestamp(win["bar_start"].iloc[-1]) >= expected_last
    return {
        "hi": finite_or_none(hi, 5),
        "lo": finite_or_none(lo, 5),
        "reveal": hhmm(win["bar_start"].iloc[-1]),
        "full": bool(full),
    }


def touch_reveal_label(decision_time: str) -> str:
    """The touch window is [t+1, t+1+10m) with t+1 = decision_time; its last bar
    starts at decision_time + 9 minutes."""
    return (pd.Timestamp(decision_time)
            + pd.Timedelta(minutes=TOUCH_WINDOW_MINUTES - 1)).strftime("%H:%M")


# --------------------------------------------------------------------------- #
# bundle validation (also used by the tests)
# --------------------------------------------------------------------------- #
def _minutes(label: str) -> int:
    h, m = label.split(":")
    return int(h) * 60 + int(m)


def validate_bundle(bundle: Dict) -> List[str]:
    """Structural checks the app relies on. Returns a list of problems (empty =
    OK): every event sits on an exported bar, and no outcome is revealed at or
    before the bar its alert was decided on (the no-future-leak guarantee)."""
    problems = []
    labels = [b[0] for b in bundle.get("bars", [])]
    if labels != sorted(labels):
        problems.append("bars are not in time order")
    known = set(labels)
    for ev in bundle.get("events", []):
        bar = ev.get("bar")
        if bar not in known:
            problems.append(f"event {ev.get('id')} on unknown bar {bar}")
            continue
        for key, out in (ev.get("outcome") or {}).items():
            if out is None:
                continue
            reveal = out.get("reveal")
            if reveal is None:
                problems.append(f"event {ev.get('id')} outcome {key} lacks reveal")
            elif _minutes(reveal) <= _minutes(bar):
                problems.append(f"event {ev.get('id')} outcome {key} reveals at "
                                f"{reveal} <= its bar {bar}")
    return problems
