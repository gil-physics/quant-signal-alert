"""Step 10: forward-looking label generation.

For each (symbol, decision_time, close_t) we look at FUTURE bars (bar_start >=
decision_time, so the current/just-closed bar is excluded) from bars_1m al_regular
and build, per horizon (60m/120m/180m/eod):

  up{t}_{h}   = future_high >= close_t * (1 + t/100)     for t in {2,3,5}
  down{t}_{h} = future_low  <= close_t * (1 - t/100)

plus has_full_60m/120m/180m flags (whether the full horizon is observable within the
same trading day) and a path-dependent long event over the remaining day:

  up5_event = UP5_SUCCESS  (+5% target reached before the -1.5% stop)
            | UP_FAIL      (stop reached first; same-bar tie is pessimistic -> FAIL)
            | NO_EVENT     (neither within the day)

Labels are stored in their OWN table; no future value ever enters a feature.
Everything is computed within a single trading day (no cross-day reference).
"""

from __future__ import annotations

import numpy as np

THRESHOLDS = (2, 3, 5)
FIXED_HORIZONS = {"60m": 60, "120m": 120, "180m": 180}
HORIZONS = list(FIXED_HORIZONS.keys()) + ["eod"]
TARGET_PCT = 0.05   # +5% long target
STOP_PCT = 0.015    # -1.5% stop (stop_loss = close_t * 0.985)
EVENT_VALUES = ("UP5_SUCCESS", "UP_FAIL", "NO_EVENT")


def binary_label_columns():
    cols = []
    for hz in HORIZONS:
        cols += [f"up{t}_{hz}" for t in THRESHOLDS]
        cols += [f"down{t}_{hz}" for t in THRESHOLDS]
    return cols


LABEL_COLUMNS = binary_label_columns()
FLAG_COLUMNS = [f"has_full_{hz}" for hz in FIXED_HORIZONS]
OUTPUT_COLUMNS = (
    ["symbol", "decision_time", "close"]
    + LABEL_COLUMNS
    + FLAG_COLUMNS
    + ["n_future_bars_eod", "up5_event"]
)


def _minute_of_day(ts: str) -> int:
    return int(ts[11:13]) * 60 + int(ts[14:16])


def _path_event(close_t, sub_high, sub_low):
    if len(sub_high) == 0:
        return "NO_EVENT"
    target = close_t * (1 + TARGET_PCT)
    stop = close_t * (1 - STOP_PCT)
    tgt = np.flatnonzero(sub_high >= target)
    stp = np.flatnonzero(sub_low <= stop)
    first_tgt = tgt[0] if len(tgt) else np.inf
    first_stp = stp[0] if len(stp) else np.inf
    if first_tgt == np.inf and first_stp == np.inf:
        return "NO_EVENT"
    if first_tgt < first_stp:
        return "UP5_SUCCESS"
    return "UP_FAIL"  # stop first, or same bar (pessimistic for a long)


def _empty_row(symbol, decision_time, close_t):
    row = {"symbol": symbol, "decision_time": decision_time, "close": int(close_t)}
    for col in LABEL_COLUMNS:
        row[col] = 0
    for col in FLAG_COLUMNS:
        row[col] = 0
    row["n_future_bars_eod"] = 0
    row["up5_event"] = "NO_EVENT"
    return row


def _label_one(symbol, decision_time, close_t, t, bs, hi, lo, be_last):
    start = int(np.searchsorted(bs, t, side="left"))  # first FUTURE bar (bar_start >= t)
    n_future = len(bs) - start
    row = {"symbol": symbol, "decision_time": decision_time, "close": int(close_t)}

    for hz, horizon in FIXED_HORIZONS.items():
        end = int(np.searchsorted(bs, t + horizon, side="left"))
        if end > start:
            fhigh = hi[start:end].max()
            flow = lo[start:end].min()
        else:
            fhigh = flow = np.nan
        for th in THRESHOLDS:
            row[f"up{th}_{hz}"] = int(fhigh >= close_t * (1 + th / 100)) if fhigh == fhigh else 0
            row[f"down{th}_{hz}"] = int(flow <= close_t * (1 - th / 100)) if flow == flow else 0
        row[f"has_full_{hz}"] = int((t + horizon) <= be_last)

    if n_future > 0:
        ehigh = hi[start:].max()
        elow = lo[start:].min()
    else:
        ehigh = elow = np.nan
    for th in THRESHOLDS:
        row[f"up{th}_eod"] = int(ehigh >= close_t * (1 + th / 100)) if ehigh == ehigh else 0
        row[f"down{th}_eod"] = int(elow <= close_t * (1 - th / 100)) if elow == elow else 0

    row["n_future_bars_eod"] = int(n_future)
    row["up5_event"] = _path_event(close_t, hi[start:], lo[start:])
    return row


def label_symbol(symbol: str, decisions, bars):
    """Label one symbol. decisions: [decision_time, close]; bars: [bar_start, bar_end, high, low]."""
    import pandas as pd

    if decisions is None or len(decisions) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    b = bars.copy()
    b["_date"] = b["bar_start"].str.slice(0, 10)
    b["_bs"] = b["bar_start"].map(_minute_of_day)
    b["_be"] = b["bar_end"].map(_minute_of_day)
    bars_by_date = {}
    for d, g in b.groupby("_date"):
        g = g.sort_values("_bs")
        bars_by_date[d] = (
            g["_bs"].to_numpy(),
            g["high"].to_numpy(dtype=float),
            g["low"].to_numpy(dtype=float),
            int(g["_be"].max()),
        )

    dec = decisions.copy()
    dec["_date"] = dec["decision_time"].str.slice(0, 10)

    rows = []
    for d, dg in dec.groupby("_date"):
        day = bars_by_date.get(d)
        if day is None:
            for _, r in dg.iterrows():
                rows.append(_empty_row(symbol, r["decision_time"], r["close"]))
            continue
        bs, hi, lo, be_last = day
        for _, r in dg.iterrows():
            rows.append(
                _label_one(symbol, r["decision_time"], float(r["close"]),
                           _minute_of_day(r["decision_time"]), bs, hi, lo, be_last)
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def label_distribution(df):
    """Per-label positive rate over eligible rows (has_full for fixed horizons;
    >=1 future bar for eod) + the up5_event distribution."""
    rows = []
    total = len(df)
    for hz in HORIZONS:
        eligible = (df[f"has_full_{hz}"] == 1) if hz in FIXED_HORIZONS else (df["n_future_bars_eod"] > 0)
        n_elig = int(eligible.sum())
        for col in [f"up{t}_{hz}" for t in THRESHOLDS] + [f"down{t}_{hz}" for t in THRESHOLDS]:
            pos = int(df.loc[eligible, col].sum())
            rows.append({"label": col, "eligible": n_elig, "positive": pos,
                         "positive_rate": round(pos / n_elig, 6) if n_elig else 0.0})
    for value in EVENT_VALUES:
        cnt = int((df["up5_event"] == value).sum())
        rows.append({"label": f"up5_event={value}", "eligible": total, "positive": cnt,
                     "positive_rate": round(cnt / total, 6) if total else 0.0})
    return rows
