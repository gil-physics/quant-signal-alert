"""Pure logic for the morning-signal -> afternoon-extreme validation study.

Research/validation only (no trading, no orders). Kept dependency-light (pandas/numpy
+ stdlib) and free of DB/engine imports so it unit-tests without Kiwoom or the model.
The CLI layer (scripts/validate_morning_signal_afternoon_extreme.py) does the DB I/O
and file writing; everything here is a deterministic function of in-memory frames.

Hypothesis under test: a morning 🔔 signal (before a cutoff, default 11:00) predicts a
useful afternoon (default 13:00-15:30) extreme relative to the alert price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Direction parser (ready for future DOWN/SHORT alerts; today mostly finds UP)
# --------------------------------------------------------------------------- #
UP_ALERT_TYPES = {"UP_ALERT", "UP_CONTINUATION", "LONG_ALERT"}
DOWN_ALERT_TYPES = {"DOWN_ALERT", "SHORT_ALERT"}
# RISK_ALERT / WEAKENING_ALERT / VOLATILITY_ALERT are not directional long/short entry
# signals for this hypothesis -> classified OTHER and excluded from selection.


def classify_direction(alert_type, *, risk_as_down=False) -> str:
    """Map an alert_type to a tradeable direction: 'UP', 'DOWN', or 'OTHER'.

    ``risk_as_down=True`` treats RISK_ALERT (하방 위험) as a SHORT/DOWN signal so the
    short side can be validated even before real DOWN/SHORT alert types exist. Caveat:
    in this project RISK_ALERT has historically often preceded mean-reverting UP moves,
    so treating it as short is a hypothesis to be tested, not an assumption."""
    t = (alert_type or "").strip().upper()
    if t in UP_ALERT_TYPES:
        return "UP"
    if t in DOWN_ALERT_TYPES:
        return "DOWN"
    if risk_as_down and t == "RISK_ALERT":
        return "DOWN"
    return "OTHER"


# coarse trend_state label for context/output (no engine dependency)
def trend_state_from_alert(alert_type) -> str:
    t = (alert_type or "").strip().upper()
    if t in UP_ALERT_TYPES:
        return "UP"
    if t in DOWN_ALERT_TYPES:
        return "DOWN"
    if t == "RISK_ALERT":
        return "RISK"
    if t == "WEAKENING_ALERT":
        return "WEAKENING"
    if t == "VOLATILITY_ALERT":
        return "CHOPPY"
    return "OTHER"


# --------------------------------------------------------------------------- #
# Risk band (mirrors realtime_trend_alert.risk_band; kept local to stay light)
# --------------------------------------------------------------------------- #
RISK_HIGH, RISK_MED = 0.60, 0.45


def risk_band(p) -> str:
    if p is None or (isinstance(p, float) and p != p):
        return "NA"
    return "HIGH" if p >= RISK_HIGH else ("MED" if p >= RISK_MED else "LOW")


GRADE_RANK = {"A": 0, "B": 1, "C": 2}


def grade_rank(g) -> int:
    """Lower = stronger. Non-A/B/C grades (RISK, '') sort last."""
    return GRADE_RANK.get(str(g).strip().upper(), 9)


# --------------------------------------------------------------------------- #
# Signal selection (one row per day for first/last/strongest/risk modes; all rows
# for 'all'). Operates on a frame already filtered to one symbol.
# --------------------------------------------------------------------------- #
SELECTION_MODES = ("first", "last", "strongest", "lowest-risk-up", "highest-risk-down", "all")


def select_signals(alerts: pd.DataFrame, *, selection: str, cutoff_tod: str) -> pd.DataFrame:
    """Select morning signals (tod < cutoff_tod, direction UP/DOWN) per the mode.

    ``alerts`` must have columns: date, tod, signal_direction, alert_grade, alert_score,
    risk_prob (risk_prob may be NaN). Returns the selected rows (a subset/copy), one per
    date except for 'all'. Deterministic — ties always break to the LATEST signal.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(f"unknown selection {selection!r}; choices={SELECTION_MODES}")
    if alerts is None or len(alerts) == 0:
        return alerts.iloc[0:0].copy() if alerts is not None else pd.DataFrame()

    morning = alerts[alerts["tod"] < cutoff_tod]
    valid = morning[morning["signal_direction"].isin(("UP", "DOWN"))].copy()
    if valid.empty:
        return valid

    if selection == "all":
        return valid.sort_values(["date", "tod"]).reset_index(drop=True)

    if selection in ("lowest-risk-up", "highest-risk-down"):
        want = "UP" if selection == "lowest-risk-up" else "DOWN"
        valid = valid[valid["signal_direction"] == want].copy()
        if valid.empty:
            return valid
        ascending = selection == "lowest-risk-up"      # lowest risk for UP, highest for DOWN
        # tie-break: latest signal. Sort so the desired row is first within each date.
        valid = valid.sort_values(
            ["date", "risk_prob", "tod"],
            ascending=[True, ascending, False],
            na_position="last",
            kind="stable",
        )
        return valid.groupby("date", sort=False, as_index=False).head(1).reset_index(drop=True)

    # first/last/strongest pick ONE signal per day overall — the day's direction is
    # whatever that single signal is (first UP -> long day; first RISK/short -> short day).
    by = ["date"]
    if selection == "strongest":
        valid["_grank"] = valid["alert_grade"].map(grade_rank)
        valid = valid.sort_values(
            ["date", "_grank", "alert_score", "tod"],
            ascending=[True, True, False, False],   # best grade, then highest score, then latest
            na_position="last",
            kind="stable",
        )
        out = valid.groupby(by, sort=False, as_index=False).head(1)
        return out.drop(columns=["_grank"]).reset_index(drop=True)

    # first / last
    ascending = selection == "first"
    valid = valid.sort_values(["date", "tod"], ascending=[True, ascending], kind="stable")
    return valid.groupby(by, sort=False, as_index=False).head(1).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Price-window metrics for one (signal, day)
# --------------------------------------------------------------------------- #
def _ret(price, base):
    if base is None or base != base or base == 0 or price is None or price != price:
        return float("nan")
    return price / base - 1.0


def compute_windows(day_bars: pd.DataFrame, *, alert_price, signal_tod, direction,
                    afternoon_start, afternoon_end) -> dict:
    """Compute afternoon / pre-13 / full-day excursion metrics for one signal.

    ``day_bars``: that symbol+date's 1m bars, columns tod (HH:MM:SS), bar_end (str),
    high, low, close (floats), sorted by tod. ``alert_price``: the signal's reference
    price. ``signal_tod``: HH:MM:SS of the signal. Returns a dict of metrics + a
    ``notes`` list (e.g. missing_afternoon_bars). Hindsight extremes — measures
    predictive potential, not a tradable exit. Never raises on missing data.
    """
    notes: list[str] = []
    nan = float("nan")
    out = {
        "alert_price": alert_price,
        "price_13_00": nan, "signal_to_13_return": nan,
        "afternoon_high": nan, "afternoon_high_time": None, "afternoon_high_return": nan,
        "afternoon_low": nan, "afternoon_low_time": None, "afternoon_low_return": nan,
        "short_potential": nan,
        "afternoon_close": nan, "afternoon_close_time": None, "afternoon_close_return": nan,
        "pre_13_high": nan, "pre_13_low": nan, "pre_13_mfe": nan, "pre_13_mae": nan,
        "full_day_mfe_after_signal": nan, "full_day_mae_after_signal": nan,
        "bars_before_13": 0, "bars_in_afternoon": 0, "notes": notes,
    }
    if alert_price is None or alert_price != alert_price or alert_price <= 0:
        notes.append("missing_alert_price")
        return out
    if day_bars is None or len(day_bars) == 0:
        notes.append("missing_afternoon_bars")
        return out

    tod = day_bars["tod"]
    aft = day_bars[(tod >= afternoon_start) & (tod <= afternoon_end)]
    out["bars_in_afternoon"] = int(len(aft))
    pre13 = day_bars[(tod > signal_tod) & (tod < afternoon_start)]
    out["bars_before_13"] = int(len(pre13))
    after_sig = day_bars[(tod > signal_tod) & (tod <= afternoon_end)]
    pre_aft = day_bars[tod <= afternoon_start]

    # price at ~13:00 = last close at or before the afternoon start
    if not pre_aft.empty:
        out["price_13_00"] = float(pre_aft["close"].iloc[-1])
        out["signal_to_13_return"] = _ret(out["price_13_00"], alert_price)

    if aft.empty:
        notes.append("missing_afternoon_bars")
    else:
        hi_i = aft["high"].idxmax()
        lo_i = aft["low"].idxmin()
        last = aft.iloc[-1]
        out["afternoon_high"] = float(aft.loc[hi_i, "high"])
        out["afternoon_high_time"] = aft.loc[hi_i, "bar_end"]
        out["afternoon_low"] = float(aft.loc[lo_i, "low"])
        out["afternoon_low_time"] = aft.loc[lo_i, "bar_end"]
        out["afternoon_close"] = float(last["close"])
        out["afternoon_close_time"] = last["bar_end"]
        # raw returns (same formula for both directions, for transparent interpretation)
        out["afternoon_high_return"] = _ret(out["afternoon_high"], alert_price)
        out["afternoon_low_return"] = _ret(out["afternoon_low"], alert_price)
        out["afternoon_close_return"] = _ret(out["afternoon_close"], alert_price)
        if out["afternoon_low"] and out["afternoon_low"] == out["afternoon_low"]:
            out["short_potential"] = alert_price / out["afternoon_low"] - 1.0

    if pre13.empty:
        notes.append("no_pre_13_window")
    else:
        out["pre_13_high"] = float(pre13["high"].max())
        out["pre_13_low"] = float(pre13["low"].min())

    fd_high = float(after_sig["high"].max()) if not after_sig.empty else nan
    fd_low = float(after_sig["low"].min()) if not after_sig.empty else nan

    # direction-aware favorable (MFE) / adverse (MAE) excursions
    if direction == "DOWN":
        out["pre_13_mfe"] = (alert_price / out["pre_13_low"] - 1.0) if out["pre_13_low"] == out["pre_13_low"] else nan
        out["pre_13_mae"] = _ret(out["pre_13_high"], alert_price)
        out["full_day_mfe_after_signal"] = (alert_price / fd_low - 1.0) if fd_low == fd_low else nan
        out["full_day_mae_after_signal"] = _ret(fd_high, alert_price)
    else:  # UP (and OTHER, treated long-style)
        out["pre_13_mfe"] = _ret(out["pre_13_high"], alert_price)
        out["pre_13_mae"] = _ret(out["pre_13_low"], alert_price)
        out["full_day_mfe_after_signal"] = _ret(fd_high, alert_price)
        out["full_day_mae_after_signal"] = _ret(fd_low, alert_price)
    return out


# --------------------------------------------------------------------------- #
# Aggregation / hit rates / verdict
# --------------------------------------------------------------------------- #
HIT_THRESHOLDS = (0.005, 0.010, 0.015, 0.020)
_HIT_TAGS = ("0_5", "1_0", "1_5", "2_0")


def _col(df, name):
    """Numeric Series for a column, or an empty float Series if it is absent."""
    if name in getattr(df, "columns", []):
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(dtype=float)


def _avg(s):
    return float(s.mean()) if len(s) and s.notna().any() else float("nan")


def _median(s):
    return float(s.median()) if len(s) and s.notna().any() else float("nan")


def _hit_rate(s, thr):
    s = s.dropna()
    return float((s >= thr).mean()) if len(s) else float("nan")


DAILY_COLUMNS = [
    "trade_date", "direction", "signal_time", "alert_type", "alert_grade", "risk_prob",
    "risk_bucket", "alert_price", "signal_return",
    "baseline_time", "baseline_price", "baseline_return", "edge_vs_baseline",
    "afternoon_high", "afternoon_high_time", "afternoon_low", "afternoon_low_time",
    "afternoon_close", "afternoon_close_time",
    "afternoon_high_return", "afternoon_low_return", "short_potential", "afternoon_close_return",
    "signal_to_13_return", "pre_13_mfe", "pre_13_mae",
    "full_day_mfe_after_signal", "full_day_mae_after_signal", "notes",
]


def daily_comparison(trades: pd.DataFrame, baseline_last: pd.DataFrame) -> pd.DataFrame:
    """One row per validated DAY: the day's direction is set by its single selected
    signal (long if UP, short if DOWN), with the direction-appropriate favorable return
    (long -> afternoon_high_return, short -> short_potential) alongside the same-direction
    baseline (last_bar_before_11) return and the edge. Requires ``trades`` to be one row
    per day (first/last/strongest selection) -> each day is exactly ONE direction, never
    both. ``baseline_last``: the last_bar_before_11 baseline rows."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    base = (baseline_last.drop_duplicates("trade_date").set_index("trade_date")
            if baseline_last is not None and len(baseline_last) else None)
    rows = []
    for r in trades.to_dict("records"):
        is_long = r["signal_direction"] == "UP"
        sig_ret = r.get("afternoon_high_return") if is_long else r.get("short_potential")
        b = base.loc[r["trade_date"]] if (base is not None and r["trade_date"] in base.index) else None
        if b is not None:
            base_ret = b["afternoon_high_return"] if is_long else b["short_potential"]
            base_time, base_price = b["signal_time"], b["alert_price"]
        else:
            base_ret, base_time, base_price = float("nan"), None, float("nan")
        edge = (sig_ret - base_ret) if (sig_ret == sig_ret and base_ret == base_ret) else float("nan")
        rows.append({
            "trade_date": r["trade_date"], "direction": "long" if is_long else "short",
            "signal_time": r.get("signal_time"), "alert_type": r.get("alert_type"),
            "alert_grade": r.get("alert_grade"), "risk_prob": r.get("risk_prob"),
            "risk_bucket": r.get("risk_bucket"), "alert_price": r.get("alert_price"),
            "signal_return": sig_ret, "baseline_time": base_time, "baseline_price": base_price,
            "baseline_return": base_ret, "edge_vs_baseline": edge,
            "afternoon_high": r.get("afternoon_high"), "afternoon_high_time": r.get("afternoon_high_time"),
            "afternoon_low": r.get("afternoon_low"), "afternoon_low_time": r.get("afternoon_low_time"),
            "afternoon_close": r.get("afternoon_close"), "afternoon_close_time": r.get("afternoon_close_time"),
            "afternoon_high_return": r.get("afternoon_high_return"),
            "afternoon_low_return": r.get("afternoon_low_return"),
            "short_potential": r.get("short_potential"),
            "afternoon_close_return": r.get("afternoon_close_return"),
            "signal_to_13_return": r.get("signal_to_13_return"),
            "pre_13_mfe": r.get("pre_13_mfe"), "pre_13_mae": r.get("pre_13_mae"),
            "full_day_mfe_after_signal": r.get("full_day_mfe_after_signal"),
            "full_day_mae_after_signal": r.get("full_day_mae_after_signal"), "notes": r.get("notes"),
        })
    return pd.DataFrame(rows, columns=DAILY_COLUMNS).sort_values("trade_date").reset_index(drop=True)


def aggregate(df: pd.DataFrame) -> dict:
    """Summary metrics over a set of signal/baseline rows (hit_up on
    afternoon_high_return, hit_down on short_potential)."""
    high = _col(df, "afternoon_high_return")
    short = _col(df, "short_potential")
    row = {"n": int(len(df))}
    row["avg_afternoon_high_return"] = _avg(high)
    row["median_afternoon_high_return"] = _median(high)
    for thr, tag in zip(HIT_THRESHOLDS, _HIT_TAGS):
        row[f"hit_up_{tag}"] = _hit_rate(high, thr)
    row["avg_short_potential"] = _avg(short)
    row["median_short_potential"] = _median(short)
    for thr, tag in zip(HIT_THRESHOLDS, _HIT_TAGS):
        row[f"hit_down_{tag}"] = _hit_rate(short, thr)
    row["avg_afternoon_close_return"] = _avg(_col(df, "afternoon_close_return"))
    row["median_afternoon_close_return"] = _median(_col(df, "afternoon_close_return"))
    row["avg_signal_to_13_return"] = _avg(_col(df, "signal_to_13_return"))
    row["avg_pre_13_mfe"] = _avg(_col(df, "pre_13_mfe"))
    row["avg_pre_13_mae"] = _avg(_col(df, "pre_13_mae"))
    row["avg_full_day_mfe_after_signal"] = _avg(_col(df, "full_day_mfe_after_signal"))
    row["avg_full_day_mae_after_signal"] = _avg(_col(df, "full_day_mae_after_signal"))
    return row


# verdict thresholds (documented in the report)
MIN_EDGE = 0.001          # signal avg afternoon_high_return must beat baseline by this
MIN_DAYS = 20             # below this the result is "concentrated in a few days"
MAX_TOLERABLE_PRE13_MAE = -0.012   # avg pre-13 adverse move worse than this = not tolerable


def decide_verdict(stats: dict, baseline_stats: dict, *, direction: str = "UP") -> tuple:
    """Return (verdict, reasons[]) comparing signals vs the baseline.

    direction='UP'   : favorable = afternoon_high_return (long upside).
    direction='DOWN' : favorable = short_potential (short downside profit).
    PASS  : beats baseline on mean AND median favorable return, enough days, and the
            pre-13 adverse move is tolerable.
    PARTIAL: potential exists but one criterion fails.  FAIL: doesn't beat baseline mean.
    """
    if direction == "DOWN":
        fav_key, med_key, label = "avg_short_potential", "median_short_potential", "short_potential"
    else:
        fav_key, med_key, label = "avg_afternoon_high_return", "median_afternoon_high_return", "afternoon_high_return"
    n = stats.get("n", 0)
    avg = stats.get(fav_key, float("nan"))
    med = stats.get(med_key, float("nan"))
    mae = stats.get("avg_pre_13_mae", float("nan"))
    base_avg = baseline_stats.get(fav_key, float("nan"))
    base_med = baseline_stats.get(med_key, float("nan"))

    if n == 0 or avg != avg:
        return "FAIL", [f"no {direction} signals to evaluate"]

    edge = avg - (base_avg if base_avg == base_avg else 0.0)
    beats_mean = edge > MIN_EDGE
    beats_median = (med == med) and (base_med != base_med or med > base_med)
    # pre-13 adverse move: UP -> pre_13_mae is negative (tolerable >= -TOL);
    # DOWN -> pre_13_mae is the adverse rise, positive (tolerable <= +TOL).
    tol = abs(MAX_TOLERABLE_PRE13_MAE)
    if direction == "DOWN":
        adverse_ok = (mae != mae) or (mae <= tol)
        tol_txt = f"<= +{tol}"
    else:
        adverse_ok = (mae != mae) or (mae >= -tol)
        tol_txt = f">= -{tol}"
    enough = n >= MIN_DAYS

    reasons = [
        f"{direction} avg {label} {avg:+.4f} vs baseline {base_avg:+.4f} (edge {edge:+.4f})",
        f"{direction} median {med:+.4f} vs baseline median {base_med:+.4f}",
        f"avg pre-13 MAE {mae:+.4f} (tolerable {tol_txt})",
        f"{direction} signal days n={n} (min {MIN_DAYS})",
    ]
    if not beats_mean:
        return "FAIL", reasons
    if beats_mean and beats_median and adverse_ok and enough:
        return "PASS", reasons
    return "PARTIAL", reasons
