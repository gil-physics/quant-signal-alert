"""Step 15: direction-aware forward metrics for SHORT entry alerts.

Evaluation layer only (decision support research, no orders). Mirrors the long
conventions in directional_alert_eval but with the short sign flips:

  * ``future_return_*`` stay RAW close returns -> negative = profit for a short.
  * ``short_mfe`` = max favorable DOWN move = -(min(low)/entry - 1); positive good.
  * ``short_mae`` = max adverse UP move = max(high)/entry - 1; positive bad.
  * ``short_stop_first`` = adverse +1.5% (high) touches before favorable -1.5%
    (low); a bar touching both counts as stop-first (same conservative tie rule
    as directional_alert_eval._stop_first / bracket_eval.simulate_bracket).

Execution model reuses bracket_eval.prepare_entry: entry = close of the bar
``entry_delay_minutes`` after the decision (project standard: 3), and every
metric reads only bars strictly AFTER the entry bar. Selection/gating must
happen upstream (e.g. directional_alert.select_online_alerts) and never reads
these forward columns.

Pure module: pandas/numpy/stdlib + sibling src imports; no sqlite, no model
loading.

Bounce-long addendum (dip-buy study): the module also carries the LONG-direction
mirrors (:func:`evaluate_long_alerts`, :func:`baseline_same_time_long`,
``LONG_*_COLUMNS``) so both directions share one entry convention and one
baseline/t-stat machinery. LONG sign conventions:

  * ``future_return_*`` stay RAW close returns -> positive = profit for a long.
  * ``long_mfe`` = max favorable UP move = max(high)/entry - 1; positive good.
  * ``long_mae`` = max adverse DOWN move = -(min(low)/entry - 1); positive bad.
  * ``long_stop_first`` = adverse -1.5% (low) touches before favorable +1.5%
    (high); a bar touching both counts as stop-first (same conservative tie
    rule as the short side).

:func:`paired_t_stat` is direction-agnostic (it only joins and differences the
requested column); for a LONG gate a POSITIVE ``mean_diff`` on a raw-return
column means the fired alerts beat unconditional same-time entries.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bracket_eval import day_arrays, prepare_entry

# stop-first bracket (mirror of directional_alert_eval UP_TARGET/DOWN_STOP)
SHORT_ADVERSE_STOP = 0.015    # high >= entry*(1+this) -> adverse stop touch
SHORT_FAVOR_TARGET = 0.015    # low  <= entry*(1-this) -> favorable target touch
DOWN2_LEVEL = 0.02            # down2_60m profit event: -2% within 60m
DN15_LEVEL = 0.015            # reach_dn15_eod: -1.5% any time before EOD

# forward metric columns (order = output order appended to the alert rows)
SHORT_METRIC_COLUMNS = [
    "future_return_30m",
    "future_return_60m",
    "future_return_120m",
    "future_return_eod",
    "short_mfe",
    "short_mae",
    "down2_60m",
    "reach_dn15_eod",
    "short_stop_first",
    "direction_hit_30m",
    "direction_hit_60m",
    "direction_hit_120m",
]

SHORT_EVAL_COLUMNS = ["entry_delay_minutes", "entry_price"] + SHORT_METRIC_COLUMNS

BASELINE_COLUMNS = (
    ["symbol", "decision_time", "baseline_n_days"]
    + [f"baseline_{c}" for c in SHORT_METRIC_COLUMNS]
)


def _ensure_date_time(frame):
    df = frame.copy()
    if "date" not in df.columns:
        df["date"] = df["decision_time"].str.slice(0, 10)
    if "time" not in df.columns:
        df["time"] = df["decision_time"].str.slice(11, 19)
    return df


def _window_return(entry_price, closes, offsets, minutes):
    window = closes[offsets <= minutes]
    return float(window[-1] / entry_price - 1) if len(window) else np.nan


def _hit_down(entry_price, lows, offsets, level, minutes):
    window = lows[offsets <= minutes]
    return int((window <= entry_price * (1 - level)).any()) if len(window) else 0


def _short_stop_first(entry_price, highs, lows):
    """1 if the adverse up-touch comes first (or ties). Same-bar tie = stop-first,
    mirroring the long convention (intrabar order unknown -> assume the worst)."""
    up_hits = np.nonzero(highs >= entry_price * (1 + SHORT_ADVERSE_STOP))[0]
    dn_hits = np.nonzero(lows <= entry_price * (1 - SHORT_FAVOR_TARGET))[0]
    up_i = int(up_hits[0]) if up_hits.size else None
    dn_i = int(dn_hits[0]) if dn_hits.size else None
    return int(up_i is not None and (dn_i is None or up_i <= dn_i))


def _short_metrics(entry_price, highs, lows, closes, offsets):
    """All SHORT_METRIC_COLUMNS for one post-entry window (arrays exclude the
    entry bar; offsets are minutes from the entry bar start, first bar = 1.0)."""
    out = {}
    out["future_return_30m"] = _window_return(entry_price, closes, offsets, 30)
    out["future_return_60m"] = _window_return(entry_price, closes, offsets, 60)
    out["future_return_120m"] = _window_return(entry_price, closes, offsets, 120)
    out["future_return_eod"] = float(closes[-1] / entry_price - 1) if len(closes) else np.nan
    out["short_mfe"] = -(float(lows.min()) / entry_price - 1) if len(lows) else np.nan
    out["short_mae"] = float(highs.max()) / entry_price - 1 if len(highs) else np.nan
    out["down2_60m"] = _hit_down(entry_price, lows, offsets, DOWN2_LEVEL, 60)
    out["reach_dn15_eod"] = _hit_down(entry_price, lows, offsets, DN15_LEVEL, float("inf"))
    out["short_stop_first"] = _short_stop_first(entry_price, highs, lows)
    for minutes in (30, 60, 120):
        ret = out[f"future_return_{minutes}m"]
        out[f"direction_hit_{minutes}m"] = 0 if pd.isna(ret) else int(ret < 0)
    return out


def evaluate_short_alerts(alerts, bars, *, entry_delay_minutes=3):
    """Append SHORT_EVAL_COLUMNS to each fired-alert row, computed from ``bars``.

    ``alerts`` needs columns: symbol, decision_time (str), close. ``bars`` needs
    symbol, bar_start, high, low, close. Any forward-looking columns already on
    the input are ignored and overwritten -- metrics are always recomputed from
    the bars strictly after the entry bar (entry = close of the decision +
    ``entry_delay_minutes`` bar via bracket_eval.prepare_entry). Rows without an
    entry bar or without post-entry bars are dropped.
    """
    if alerts is None or len(alerts) == 0:
        base_cols = list(alerts.columns) if alerts is not None else []
        return pd.DataFrame(columns=base_cols + SHORT_EVAL_COLUMNS)

    selected = _ensure_date_time(alerts)
    by_day = day_arrays(bars)
    rows = []
    for row in selected.to_dict("records"):
        arr = by_day.get((row["symbol"], row["date"]))
        if arr is None:
            continue
        prep = prepare_entry(
            arr,
            pd.Timestamp(row["decision_time"]),
            float(row["close"]),
            delay_bars=entry_delay_minutes,
        )
        if prep is None:
            continue
        entry_price, highs, lows, closes, offsets = prep
        out = dict(row)
        out["entry_delay_minutes"] = int(entry_delay_minutes)
        out["entry_price"] = float(entry_price)
        out.update(_short_metrics(entry_price, highs, lows, closes, offsets))
        rows.append(out)
    return pd.DataFrame(rows)


def baseline_same_time(alerts, bars, *, entry_delay_minutes=3):
    """Time-of-day-matched unconditional baseline, one row per alert.

    For each alert, enter at the SAME HH:MM(:SS) on every OTHER trading day of
    that symbol found in ``bars`` (same delay/entry convention as
    :func:`evaluate_short_alerts`) and average each forward metric across those
    days. Returns BASELINE_COLUMNS: the (symbol, decision_time) join keys,
    ``baseline_n_days`` (days actually evaluated) and ``baseline_<metric>``
    per-alert means (NaN-mean; NaN when no day could be evaluated).
    """
    if alerts is None or len(alerts) == 0:
        return pd.DataFrame(columns=BASELINE_COLUMNS)

    selected = _ensure_date_time(alerts)
    by_day = day_arrays(bars)
    dates_by_symbol = {}
    for symbol, date in by_day:
        dates_by_symbol.setdefault(symbol, []).append(date)
    for symbol in dates_by_symbol:
        dates_by_symbol[symbol].sort()

    rows = []
    for row in selected.to_dict("records"):
        symbol, own_date, tod = row["symbol"], row["date"], row["time"]
        per_day = []
        for date in dates_by_symbol.get(symbol, []):
            if date == own_date:
                continue
            arr = by_day[(symbol, date)]
            decision_dt = pd.Timestamp(f"{date} {tod}")
            idx = int(np.searchsorted(arr["dt"], np.datetime64(decision_dt), side="left"))
            if idx >= len(arr["dt"]):
                continue
            # decision close = the bar prepare_entry treats as the decision bar
            # (side="left"); only consumed when entry_delay_minutes == 0.
            prep = prepare_entry(
                arr, decision_dt, float(arr["close"][idx]), delay_bars=entry_delay_minutes,
            )
            if prep is None:
                continue
            entry_price, highs, lows, closes, offsets = prep
            per_day.append(_short_metrics(entry_price, highs, lows, closes, offsets))

        out = {
            "symbol": symbol,
            "decision_time": row["decision_time"],
            "baseline_n_days": int(len(per_day)),
        }
        for col in SHORT_METRIC_COLUMNS:
            vals = np.array([d[col] for d in per_day], dtype=float)
            valid = vals[~np.isnan(vals)]
            out[f"baseline_{col}"] = float(valid.mean()) if valid.size else np.nan
        rows.append(out)
    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


def paired_t_stat(evaluated, baseline, *, column="future_return_60m"):
    """Paired difference (alert value - same-time baseline mean) on ``column``.

    Joins ``evaluated`` (from :func:`evaluate_short_alerts`) with ``baseline``
    (from :func:`baseline_same_time`) on (symbol, decision_time); pairs with a
    NaN on either side are dropped. For a short gate a NEGATIVE ``mean_diff``
    on a raw-return column means the fired alerts beat unconditional same-time
    entries. Returns {"n", "mean_alert", "mean_baseline", "mean_diff",
    "t_stat"}; t_stat is NaN when n < 2 or the diffs are constant.
    """
    out = {
        "n": 0,
        "mean_alert": float("nan"),
        "mean_baseline": float("nan"),
        "mean_diff": float("nan"),
        "t_stat": float("nan"),
    }
    if evaluated is None or baseline is None or len(evaluated) == 0 or len(baseline) == 0:
        return out

    base_col = f"baseline_{column}"
    merged = evaluated[["symbol", "decision_time", column]].merge(
        baseline[["symbol", "decision_time", base_col]].drop_duplicates(
            ["symbol", "decision_time"]
        ),
        on=["symbol", "decision_time"],
        how="inner",
        validate="many_to_one",
    )
    a = merged[column].astype(float).to_numpy()
    b = merged[base_col].astype(float).to_numpy()
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    out["n"] = int(a.size)
    if not a.size:
        return out
    diff = a - b
    out["mean_alert"] = float(a.mean())
    out["mean_baseline"] = float(b.mean())
    out["mean_diff"] = float(diff.mean())
    if a.size >= 2:
        sd = float(diff.std(ddof=1))
        if sd > 0:
            out["t_stat"] = float(diff.mean() / (sd / math.sqrt(a.size)))
    return out


# --------------------------------------------------------------------------- #
# LONG direction (bounce-long / dip-buy study) -- sign mirror of the short side
# --------------------------------------------------------------------------- #
LONG_ADVERSE_STOP = 0.015    # low  <= entry*(1-this) -> adverse stop touch
LONG_FAVOR_TARGET = 0.015    # high >= entry*(1+this) -> favorable target touch
UP2_LEVEL = 0.02             # up2_60m profit event: +2% within 60m
UP15_LEVEL = 0.015           # reach_up15_eod: +1.5% any time before EOD

# forward metric columns (order = output order appended to the alert rows);
# the shared names (future_return_*/direction_hit_*) keep the two directions
# column-compatible, but direction_hit_* here means ret > 0 (long profit).
LONG_METRIC_COLUMNS = [
    "future_return_30m",
    "future_return_60m",
    "future_return_120m",
    "future_return_eod",
    "long_mfe",
    "long_mae",
    "up2_60m",
    "reach_up15_eod",
    "long_stop_first",
    "direction_hit_30m",
    "direction_hit_60m",
    "direction_hit_120m",
]

LONG_EVAL_COLUMNS = ["entry_delay_minutes", "entry_price"] + LONG_METRIC_COLUMNS

LONG_BASELINE_COLUMNS = (
    ["symbol", "decision_time", "baseline_n_days"]
    + [f"baseline_{c}" for c in LONG_METRIC_COLUMNS]
)


def _hit_up(entry_price, highs, offsets, level, minutes):
    window = highs[offsets <= minutes]
    return int((window >= entry_price * (1 + level)).any()) if len(window) else 0


def _long_stop_first(entry_price, highs, lows):
    """1 if the adverse down-touch comes first (or ties). Same-bar tie =
    stop-first, mirroring _short_stop_first (intrabar order unknown -> assume
    the worst)."""
    dn_hits = np.nonzero(lows <= entry_price * (1 - LONG_ADVERSE_STOP))[0]
    up_hits = np.nonzero(highs >= entry_price * (1 + LONG_FAVOR_TARGET))[0]
    dn_i = int(dn_hits[0]) if dn_hits.size else None
    up_i = int(up_hits[0]) if up_hits.size else None
    return int(dn_i is not None and (up_i is None or dn_i <= up_i))


def _long_metrics(entry_price, highs, lows, closes, offsets):
    """All LONG_METRIC_COLUMNS for one post-entry window (arrays exclude the
    entry bar; offsets are minutes from the entry bar start, first bar = 1.0)."""
    out = {}
    out["future_return_30m"] = _window_return(entry_price, closes, offsets, 30)
    out["future_return_60m"] = _window_return(entry_price, closes, offsets, 60)
    out["future_return_120m"] = _window_return(entry_price, closes, offsets, 120)
    out["future_return_eod"] = float(closes[-1] / entry_price - 1) if len(closes) else np.nan
    out["long_mfe"] = float(highs.max()) / entry_price - 1 if len(highs) else np.nan
    out["long_mae"] = -(float(lows.min()) / entry_price - 1) if len(lows) else np.nan
    out["up2_60m"] = _hit_up(entry_price, highs, offsets, UP2_LEVEL, 60)
    out["reach_up15_eod"] = _hit_up(entry_price, highs, offsets, UP15_LEVEL, float("inf"))
    out["long_stop_first"] = _long_stop_first(entry_price, highs, lows)
    for minutes in (30, 60, 120):
        ret = out[f"future_return_{minutes}m"]
        out[f"direction_hit_{minutes}m"] = 0 if pd.isna(ret) else int(ret > 0)
    return out


def evaluate_long_alerts(alerts, bars, *, entry_delay_minutes=3):
    """Append LONG_EVAL_COLUMNS to each fired-alert row, computed from ``bars``.

    LONG mirror of :func:`evaluate_short_alerts` -- same inputs, same entry
    convention (entry = close of the decision + ``entry_delay_minutes`` bar via
    bracket_eval.prepare_entry), same recompute-and-overwrite rule for any
    forward-looking columns already on the input. Rows without an entry bar or
    without post-entry bars are dropped.
    """
    if alerts is None or len(alerts) == 0:
        base_cols = list(alerts.columns) if alerts is not None else []
        return pd.DataFrame(columns=base_cols + LONG_EVAL_COLUMNS)

    selected = _ensure_date_time(alerts)
    by_day = day_arrays(bars)
    rows = []
    for row in selected.to_dict("records"):
        arr = by_day.get((row["symbol"], row["date"]))
        if arr is None:
            continue
        prep = prepare_entry(
            arr,
            pd.Timestamp(row["decision_time"]),
            float(row["close"]),
            delay_bars=entry_delay_minutes,
        )
        if prep is None:
            continue
        entry_price, highs, lows, closes, offsets = prep
        out = dict(row)
        out["entry_delay_minutes"] = int(entry_delay_minutes)
        out["entry_price"] = float(entry_price)
        out.update(_long_metrics(entry_price, highs, lows, closes, offsets))
        rows.append(out)
    return pd.DataFrame(rows)


def baseline_same_time_long(alerts, bars, *, entry_delay_minutes=3):
    """Time-of-day-matched unconditional LONG baseline, one row per alert.

    LONG mirror of :func:`baseline_same_time`: for each alert, enter LONG at
    the SAME HH:MM(:SS) on every OTHER trading day of that symbol found in
    ``bars`` (same delay/entry convention as :func:`evaluate_long_alerts`) and
    average each forward metric across those days. Returns
    LONG_BASELINE_COLUMNS: the (symbol, decision_time) join keys,
    ``baseline_n_days`` (days actually evaluated) and ``baseline_<metric>``
    per-alert means (NaN-mean; NaN when no day could be evaluated).
    """
    if alerts is None or len(alerts) == 0:
        return pd.DataFrame(columns=LONG_BASELINE_COLUMNS)

    selected = _ensure_date_time(alerts)
    by_day = day_arrays(bars)
    dates_by_symbol = {}
    for symbol, date in by_day:
        dates_by_symbol.setdefault(symbol, []).append(date)
    for symbol in dates_by_symbol:
        dates_by_symbol[symbol].sort()

    rows = []
    for row in selected.to_dict("records"):
        symbol, own_date, tod = row["symbol"], row["date"], row["time"]
        per_day = []
        for date in dates_by_symbol.get(symbol, []):
            if date == own_date:
                continue
            arr = by_day[(symbol, date)]
            decision_dt = pd.Timestamp(f"{date} {tod}")
            idx = int(np.searchsorted(arr["dt"], np.datetime64(decision_dt), side="left"))
            if idx >= len(arr["dt"]):
                continue
            # decision close = the bar prepare_entry treats as the decision bar
            # (side="left"); only consumed when entry_delay_minutes == 0.
            prep = prepare_entry(
                arr, decision_dt, float(arr["close"][idx]), delay_bars=entry_delay_minutes,
            )
            if prep is None:
                continue
            entry_price, highs, lows, closes, offsets = prep
            per_day.append(_long_metrics(entry_price, highs, lows, closes, offsets))

        out = {
            "symbol": symbol,
            "decision_time": row["decision_time"],
            "baseline_n_days": int(len(per_day)),
        }
        for col in LONG_METRIC_COLUMNS:
            vals = np.array([d[col] for d in per_day], dtype=float)
            valid = vals[~np.isnan(vals)]
            out[f"baseline_{col}"] = float(valid.mean()) if valid.size else np.nan
        rows.append(out)
    return pd.DataFrame(rows, columns=LONG_BASELINE_COLUMNS)
