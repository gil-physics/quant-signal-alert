"""Short-horizon diagnostics on frozen replay signals, never on future labels.

Returns are signed price changes per entry notional (down is a hypothetical
short), not portfolio returns. Minute OHLC cannot establish intrabar order.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)
BRACKETS = ((.003, .003), (.005, .003), (.005, .005))
ENTRY_MODES = ("next_open", "delay1_open", "delay3_open")
AUDIT_MODES = ("decision_close", "legacy_skip_first",
               "delay3_close", "legacy_delay3_close")
ACTIONABLE_TYPES = {"UP_ALERT", "UP_CONTINUATION", "DOWN_ALERT", "DOWN_CONTINUATION"}


def minute_of_day(text):
    return int(text[:2]) * 60 + int(text[3:5])


@dataclass
class Day:
    symbol: str
    date: str
    minutes: np.ndarray
    prices: np.ndarray  # open, high, low, close

    @classmethod
    def from_bundle(cls, bundle):
        rows = bundle["bars"]
        minutes = np.array([minute_of_day(r[0]) for r in rows], dtype=int)
        prices = np.array([r[1:5] for r in rows], dtype=float)
        if len(minutes) == 0 or (np.diff(minutes) <= 0).any():
            raise ValueError("bar times must be unique and strictly increasing")
        if (not np.isfinite(prices).all() or (prices <= 0).any()
                or (prices[:, 1] < prices[:, [0, 2, 3]].max(axis=1)).any()
                or (prices[:, 2] > prices[:, [0, 1, 3]].min(axis=1)).any()):
            raise ValueError("invalid OHLC prices")
        return cls(bundle["meta"]["symbol"], bundle["meta"]["date"], minutes, prices)

    def index(self, minute):
        i = int(np.searchsorted(self.minutes, minute))
        return i if i < len(self.minutes) and self.minutes[i] == minute else None


def signals_from_bundle(bundle):
    """Collapse duplicate same-minute/family/direction alerts before outcomes."""
    signals = {}
    for event in bundle["events"]:
        if event["kind"] == "status":
            continue
        family = "touch" if event["kind"] == "touch" else "trend"
        alerts = ([{"dir": event["dir"], "type": "MA_CROSS"}]
                  if family == "touch" else event.get("alerts", []))
        for alert in alerts:
            if alert["dir"] not in ("up", "down"):
                continue
            direction = 1 if alert["dir"] == "up" else -1
            key = (event["decision_time"], family, direction)
            if key not in signals:
                signals[key] = {
                    "symbol": bundle["meta"]["symbol"], "date": bundle["meta"]["date"],
                    "decision_time": event["decision_time"],
                    "decision_minute": minute_of_day(event["decision_time"][11:16]),
                    "family": family, "direction": direction,
                    "signal_close": float(event["close"]), "types": set(),
                    "actionable": False,
                }
            signals[key]["types"].add(alert["type"])
            signals[key]["actionable"] |= alert["type"] in ACTIONABLE_TYPES
    result = []
    for signal in signals.values():
        signal["types"] = "|".join(sorted(signal["types"]))
        result.append(signal)
    return sorted(result, key=lambda r: (r["decision_time"], r["family"], r["direction"]))


def entry_window(day, signal, mode, horizon):
    """Exact wall-clock windows; never silently jump missing bars or truncate."""
    decision = signal["decision_minute"]
    delay = {"delay1_open": 1, "delay3_open": 3, "delay3_close": 3,
             "legacy_delay3_close": 4}.get(mode, 0)
    start = decision + delay + (mode == "legacy_skip_first")
    i = day.index(start)
    if i is None:
        return None, "missing_entry_bar"
    window = day.prices[i:i + horizon]
    if len(window) != horizon or not np.array_equal(
            day.minutes[i:i + horizon], np.arange(start, start + horizon)):
        return None, "incomplete_or_gapped_window"
    if mode in ("decision_close", "legacy_skip_first"):
        entry = signal["signal_close"]
    elif mode in ("delay3_close", "legacy_delay3_close"):
        prior = day.index(decision + delay - 1)
        if prior is None:
            return None, "missing_delayed_close"
        entry = day.prices[prior, 3]
    elif mode in ENTRY_MODES:
        entry = window[0, 0]
    else:
        raise ValueError(f"unknown entry mode: {mode}")
    return (float(entry), window, start), None


def bracket_path(entry, prices, direction, target, stop):
    """First touch; opening gaps precede intrabar extremes.

    Same-bar target+stop is stop-first. optimistic_gross changes ONLY ambiguous
    stop-first exits; it is an upper-bound sensitivity, not a tradable rule.
    """
    for j, (open_, high, low, close) in enumerate(prices):
        opening = direction * (open_ / entry - 1)
        if opening <= -stop + 1e-12:
            return {"outcome": "stop", "gross": float(opening),
                    "optimistic_gross": float(opening), "ambiguous": False,
                    "exit_minute": j + 1, "gap_stop": True}
        if opening >= target - 1e-12:
            return {"outcome": "target", "gross": target, "optimistic_gross": target,
                    "ambiguous": False, "exit_minute": j + 1, "gap_stop": False}
        favorable = direction * ((high if direction == 1 else low) / entry - 1)
        adverse = direction * ((low if direction == 1 else high) / entry - 1)
        hit_t, hit_s = favorable >= target - 1e-12, adverse <= -stop + 1e-12
        if hit_s or hit_t:
            gross = -stop if hit_s else target
            return {"outcome": "stop" if hit_s else "target", "gross": gross,
                    "optimistic_gross": target if hit_t else gross,
                    "ambiguous": bool(hit_s and hit_t), "exit_minute": j + 1,
                    "gap_stop": False}
    gross = float(direction * (prices[-1, 3] / entry - 1))
    return {"outcome": "time", "gross": gross, "optimistic_gross": gross,
            "ambiguous": False, "exit_minute": len(prices), "gap_stop": False}


def measure_window(entry, prices, direction):
    favorable_col, adverse_col = (1, 2) if direction == 1 else (2, 1)
    favorable = direction * (prices[:, favorable_col] / entry - 1)
    adverse = direction * (prices[:, adverse_col] / entry - 1)
    gross = float(direction * (prices[-1, 3] / entry - 1))
    mfe, mae = max(0., float(favorable.max())), min(0., float(adverse.min()))
    return {"gross": gross, "mfe": mfe, "mae": mae,
            "ever_positive": mfe > 1e-12, "touch_003": mfe >= .003 - 1e-12,
            "touch_005": mfe >= .005 - 1e-12,
            "gave_back_003": mfe >= .003 - 1e-12 and gross <= 0}


def evaluate_signals(signals, days):
    rows, exclusions = [], []
    for signal in signals:
        day = days[(signal["symbol"], signal["date"])]
        for mode in ENTRY_MODES + AUDIT_MODES:
            horizons = HORIZONS if mode in ENTRY_MODES else (10,)
            for horizon in horizons:
                prepared, reason = entry_window(day, signal, mode, horizon)
                base = {**signal, "entry_mode": mode, "horizon": horizon}
                if reason:
                    exclusions.append({**base, "reason": reason})
                    continue
                entry, window, start = prepared
                measured = measure_window(entry, window, signal["direction"])
                rows.append({**base, **measured, "entry": entry, "entry_minute": start,
                             "strategy": f"hold_{horizon}m", "outcome": "time",
                             "ambiguous": False, "optimistic_gross": measured["gross"],
                             "exit_minute": horizon, "gap_stop": False})
                if horizon == 10:
                    brackets = BRACKETS if mode in ENTRY_MODES else ((.005, .005),)
                    for target, stop in brackets:
                        result = bracket_path(entry, window, signal["direction"], target, stop)
                        rows.append({**base, **measured, **result, "entry": entry,
                                     "entry_minute": start,
                                     "strategy": f"tp{target:.3f}_sl{stop:.3f}_10m"})
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def nonoverlap_keys(signals, mode):
    """At most one pending/open position per symbol in this cohort for 10m.

    Opposing simultaneous alerts are skipped. No realized future exit is used.
    """
    delay = {"delay1_open": 1, "delay3_open": 3, "delay3_close": 3,
             "legacy_delay3_close": 4}.get(mode, 0)
    accepted, available = set(), {}
    grouped = {}
    for signal in signals:
        grouped.setdefault((signal["symbol"], signal["date"],
                            signal["decision_minute"]), []).append(signal)
    for (symbol, date, minute), group in sorted(grouped.items()):
        if len({s["direction"] for s in group}) != 1:
            continue
        if minute < available.get((symbol, date), -1):
            continue
        for signal in group:
            accepted.add((symbol, signal["decision_time"], signal["direction"]))
        available[(symbol, date)] = minute + delay + 10
    return accepted


def matched_baselines(frame, signals, days):
    """Other-day, same symbol/month/clock/direction, non-signal baseline.

    Diagnostic only: future prices are used exclusively for outcomes. Baseline
    availability is reported; no-match rows keep NaN rather than a false zero.
    """
    blocked = {(s["symbol"], s["date"], s["family"], s["decision_minute"])
               for s in signals}
    by_month = {}
    for (symbol, date), day in days.items():
        by_month.setdefault((symbol, date[:7]), []).append(day)
    cache, values, counts = {}, [], []
    for row in frame.to_dict("records"):
        if row["entry_mode"] != "next_open" or not row["strategy"].startswith("hold_"):
            values.append(np.nan); counts.append(0)
            continue
        key = (row["symbol"], row["date"][:7], row["decision_minute"],
               row["family"], row["direction"], row["horizon"])
        if key not in cache:
            candidates = {}
            for day in by_month.get(key[:2], []):
                if (day.symbol, day.date, row["family"], row["decision_minute"]) in blocked:
                    continue
                window, reason = entry_window(day, row, "next_open", row["horizon"])
                if reason is None:
                    entry, prices, _ = window
                    candidates[day.date] = row["direction"] * (prices[-1, 3] / entry - 1)
            cache[key] = candidates
        matches = [value for date, value in cache[key].items() if date != row["date"]]
        values.append(float(np.mean(matches)) if matches else np.nan)
        counts.append(len(matches))
    result = frame.copy()
    result["baseline_gross"] = values
    result["baseline_days"] = counts
    result["baseline_edge"] = result["gross"] - result["baseline_gross"]
    return result


def cluster_interval(frame, column, *, seed=20261020, repetitions=2000):
    """Trade-weighted mean CI from resampling whole trading dates."""
    valid = frame[["date", column]].dropna()
    daily = valid.groupby("date")[column].agg(["sum", "count"])
    if len(daily) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(repetitions, len(daily)))
    means = daily["sum"].to_numpy()[picks].sum(axis=1) / daily["count"].to_numpy()[picks].sum(axis=1)
    return tuple(float(v) for v in np.quantile(means, [.025, .975]))


def summarize(frame, signals, curated_dates):
    rows = []
    cohorts = {
        "trend_all": lambda x: x["family"] == "trend",
        "trend_directional": lambda x: x["family"] == "trend" and x["actionable"],
        "touch": lambda x: x["family"] == "touch",
    }
    for cohort, predicate in cohorts.items():
        sig = [s for s in signals if predicate(s)]
        part = frame.loc[frame.apply(predicate, axis=1)].copy()
        part["nonoverlap"] = False
        for mode in part["entry_mode"].unique():
            accepted = nonoverlap_keys(sig, mode)
            mask = part["entry_mode"] == mode
            part.loc[mask, "nonoverlap"] = [
                (r.symbol, r.decision_time, r.direction) in accepted
                for r in part.loc[mask].itertuples()]
        for scope in ("all_dates", "curated_dates", "other_dates"):
            mask = part["date"].isin(curated_dates)
            subset = part if scope == "all_dates" else part.loc[
                mask if scope == "curated_dates" else ~mask]
            for selection in ("all_alerts", "nonoverlap_10m"):
                selected = subset if selection == "all_alerts" else subset[subset["nonoverlap"] == True]
                for direction in (0, 1, -1):
                    selected_dir = selected if direction == 0 else selected[selected["direction"] == direction]
                    for (mode, strategy), group in selected_dir.groupby(["entry_mode", "strategy"]):
                        lo, hi = cluster_interval(group, "gross")
                        edge_lo, edge_hi = cluster_interval(group, "baseline_edge")
                        row = {
                            "cohort": cohort, "scope": scope, "selection": selection,
                            "direction": direction, "entry_mode": mode, "strategy": strategy,
                            "n": len(group), "days": group["date"].nunique(),
                            "gross": group["gross"].mean(),
                            "gross_ci_low": lo, "gross_ci_high": hi,
                            "positive_rate": (group["gross"] > 0).mean(),
                            "ever_positive_rate": group["ever_positive"].mean(),
                            "touch_003_rate": group["touch_003"].mean(),
                            "touch_005_rate": group["touch_005"].mean(),
                            "gave_back_003_rate": group["gave_back_003"].mean(),
                            "mean_mfe": group["mfe"].mean(), "mean_mae": group["mae"].mean(),
                            "target_first_rate": (group["outcome"] == "target").mean(),
                            "stop_first_rate": (group["outcome"] == "stop").mean(),
                            "time_exit_rate": (group["outcome"] == "time").mean(),
                            "ambiguous_rate": group["ambiguous"].mean(),
                            "gap_stop_rate": group["gap_stop"].mean(),
                            "optimistic_gross": group["optimistic_gross"].mean(),
                            "baseline_n": group["baseline_gross"].notna().sum(),
                            "baseline_gross": group["baseline_gross"].mean(),
                            "baseline_edge": group["baseline_edge"].mean(),
                            "edge_ci_low": edge_lo, "edge_ci_high": edge_hi,
                        }
                        for cost in (0, .001, .002, .003):
                            row[f"net_cost_{cost:.3f}"] = row["gross"] - cost
                        rows.append(row)
    return pd.DataFrame(rows)
