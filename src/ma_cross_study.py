"""MA5 x MA20 1-minute crossover raw-statistics study -- pure building blocks.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Consumed by
scripts/analyze_ma_cross_1m.py; nothing here touches serving or the model lineages.

User-decided conventions (2026-07-15):
  * P1  (causal):     MAs on OPEN including the current bar; the cross and the entry
                      price are both fixed at bar t's open.
  * P2i (idealized):  MAs on CLOSE (standard chart) -- the cross is only confirmed at
                      bar t's CLOSE, yet entry is assumed at the SAME bar's open, so
                      the crossing bar's whole body is inside the outcome. Look-ahead
                      by construction; reported as a diagnostic upper bound with the
                      crossing-bar body return quantified separately.
  * P2r (realistic):  same close-MA crosses, entry at the NEXT bar's open with the
                      outcome window starting at t+1 -- causal.
  * Warmup variants:  "regular" = intraday reset on the 09:00-15:30 stream;
                      "preopen" = intraday reset on the 08:00-15:30 al_raw stream
                      (premarket bars warm the MAs, signals from 09:00);
                      "carry"   = no daily reset, the full al_raw stream carries the
                      previous trading day (overnight gap enters the MA -- flagged).
  * Cross = flip of the LAST NON-ZERO sign of d = MA5 - MA20 (zeros forward-fill the
    prior regime), so a tie touched and released in the same direction never counts
    as a second cross. A NaN d (warmup / masked bar inside the window) BREAKS the
    regime: no cross can fire across the break until a fresh sign is established.
  * Signal eligibility (user-amended): variant "regular" = regular session only;
    "preopen"/"carry" = premarket AND regular (08:00-15:29). The 15:30
    closing-auction bar is never a signal (kept inside outcome windows, mirroring
    the resampler's incomplete-final-bar convention). Premarket signals carry
    session="premarket" and are aggregated separately from regular ones.
  * Outcome windows are TIMESTAMP-bounded: bars with bar_start in [T, T+W minutes)
    on the same day's 08:00-15:30 stream (premarket minutes without trades are
    simply absent, so a positional window would silently stretch the horizon; on the
    dense regular session the two definitions coincide). complete = the day still
    has a bar at or beyond T+(W-1) minutes; incomplete windows are excluded from
    rate tables and only counted.

Every frame handed to these helpers must be RangeIndex-ed (reset_index(drop=True));
positional numpy indexing is used throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MA_FAST = 5
MA_SLOW = 20
ATR_WINDOW = 20
NEAR_MAX = 0.001          # dist_pct <= 0.10%  -> "near"
FAR_MIN = 0.003           # dist_pct >= 0.30%  -> "far"
ATR_NEAR_MAX = 0.25       # dist_atr bands (auxiliary view)
ATR_FAR_MIN = 0.75
PROFIT_THRESHOLD = 0.007  # +0.7% directional close-touch
MAE_MARKS = (-0.003, -0.007)
PREMARKET_START = "08:00:00"
REGULAR_START = "09:00:00"
REGULAR_END = "15:30:00"  # inclusive session bound; the 15:30 bar = closing auction
VARIANTS = ("regular", "preopen", "carry")
PHASES = ("P1", "P2i", "P2r")
MA_SOURCE_PHASES = {"open": ("P1",), "close": ("P2i", "P2r")}
BAND_ORDER = ("near", "mid", "far")


def invalid_bar_mask(bars):
    """Integrity mask, same conditions as src/quality.py (ohlc_error + price_zero):
    any price <= 0 or a high/low vs open/close relation broken."""
    return (
        (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (bars["high"] < bars["low"])
        | (bars["high"] < bars["open"])
        | (bars["high"] < bars["close"])
        | (bars["low"] > bars["open"])
        | (bars["low"] > bars["close"])
    )


def _with_date_time(bars):
    out = bars.copy()
    s = out["bar_start"].astype(str)
    out["date"] = s.str.slice(0, 10)
    out["time"] = s.str.slice(11, 19)
    return out


def prepare_variant(bars_raw, variant):
    """Slice the al_raw stream into one warmup-variant timeline.

    Returns (frame sorted by symbol+bar_start, group_cols): the rolling groups are
    (symbol, date) for the daily-reset variants and (symbol,) for "carry"
    (continuous across days; after-hours bars stay so the next morning is warm)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choices={VARIANTS}")
    df = _with_date_time(bars_raw)
    if variant == "regular":
        df = df[(df["time"] >= REGULAR_START) & (df["time"] <= REGULAR_END)]
        group_cols = ["symbol", "date"]
    elif variant == "preopen":
        df = df[df["time"] <= REGULAR_END]           # al_raw starts 08:00
        group_cols = ["symbol", "date"]
    else:                                            # carry: full stream, no reset
        group_cols = ["symbol"]
    df = df.sort_values(["symbol", "bar_start"]).reset_index(drop=True)
    return df, group_cols


def _group_rolling(frame, group_cols, col, window, how, *, min_periods=None):
    """Fast per-group rolling via the contiguous-block trick: the frame is sorted by
    group_cols, so groupby(sort=False).rolling emits values in frame order."""
    r = frame.groupby(group_cols, sort=False)[col].rolling(
        int(window), min_periods=min_periods)
    return getattr(r, how)().to_numpy()


def add_mas_and_atr(frame, group_cols, *, price_col, ma_fast=MA_FAST, ma_slow=MA_SLOW,
                    atr_window=ATR_WINDOW):
    """MA fast/slow on ``price_col`` + ATR, rolling within ``group_cols``.

    Bars failing invalid_bar_mask are NaN-poisoned first, so every rolling window
    containing one yields NaN (signals suppressed for that span; the day survives)."""
    df = frame.reset_index(drop=True).copy()
    bad = invalid_bar_mask(df)
    df["bar_invalid"] = bad
    df["_price"] = df[price_col].astype(float).where(~bad)
    df["ma_fast"] = _group_rolling(df, group_cols, "_price", ma_fast, "mean")
    df["ma_slow"] = _group_rolling(df, group_cols, "_price", ma_slow, "mean")
    prev_close = df.groupby(group_cols, sort=False)["close"].shift(1).astype(float)
    high = df["high"].astype(float).where(~bad)
    low = df["low"].astype(float).where(~bad)
    df["_tr"] = pd.concat([(high - low), (high - prev_close).abs(),
                           (low - prev_close).abs()], axis=1).max(axis=1).where(~bad)
    df["atr"] = _group_rolling(df, group_cols, "_tr", atr_window, "mean")
    df["d"] = df["ma_fast"] - df["ma_slow"]
    return df.drop(columns=["_price", "_tr"])


def detect_crossovers(frame, group_cols, *, signal_start=REGULAR_START):
    """Regime-flip crossovers on d = ma_fast - ma_slow (see module docstring).

    Adds: regime (last non-zero sign, NaN-break aware), cross (+1/-1/0), eligible
    (time in [signal_start, 15:30) -- pass PREMARKET_START for the preopen/carry
    variants, whose premarket bars are signal-eligible; the closing-auction bar is
    excluded everywhere), session ("premarket"/"regular")."""
    df = frame.reset_index(drop=True).copy()
    d = df["d"]
    valid = d.notna()
    z = pd.Series(np.sign(d.to_numpy()), index=df.index)      # NaN propagates
    keys = [df[c] for c in group_cols]
    brk = (~valid).groupby(keys, sort=False).cumsum()          # NaN d breaks the regime
    regime = z.replace(0, np.nan).groupby(keys + [brk], sort=False).ffill()
    prev_regime = regime.groupby(keys, sort=False).shift(1)
    up = valid & (z == 1) & (prev_regime == -1)
    down = valid & (z == -1) & (prev_regime == 1)
    df["regime"] = regime
    df["cross"] = np.where(up, 1, np.where(down, -1, 0))
    df["eligible"] = (df["time"] >= signal_start) & (df["time"] < REGULAR_END)
    df["session"] = np.where(df["time"] < REGULAR_START, "premarket", "regular")
    return df


def variant_signal_start(variant):
    """User-amended eligibility: preopen/carry signals may fire in the premarket."""
    return REGULAR_START if variant == "regular" else PREMARKET_START


def classify_bands(dist, *, near_max=NEAR_MAX, far_min=FAR_MIN, labels=BAND_ORDER):
    x = pd.Series(dist, dtype=float)
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    out[x <= near_max] = labels[0]
    out[(x > near_max) & (x < far_min)] = labels[1]
    out[x >= far_min] = labels[2]
    return out


def build_outcome_stream(bars_raw):
    """Outcome/price stream: premarket + regular bars (08:00-15:30) per (symbol,
    date), sorted, with the within-day position ``pos``, day size ``day_n`` and
    ``session``. Premarket bars are part of the stream so premarket signals (preopen/
    carry variants) have outcomes; regular-only anchors are unaffected."""
    out, _ = prepare_variant(bars_raw, "preopen")
    g = out.groupby(["symbol", "date"], sort=False)
    out["pos"] = g.cumcount()
    out["day_n"] = g["symbol"].transform("size")
    out["session"] = np.where(out["time"] < REGULAR_START, "premarket", "regular")
    return out


def per_bar_forward_stats(stream, windows):
    """Per-bar forward-window primitives on the outcome stream, TIMESTAMP-bounded:
    for every W, the max/min close, max high, min low and last ("endpoint") close
    over the bars with bar_start in [T_j, T_j + W minutes) on the same day, plus
    complete_w{W} = the day still has a bar at or beyond T_j + (W-1) minutes.
    Premarket minutes without trades are simply absent (the window never stretches);
    on the dense regular session this equals the classic j..j+W-1 positional window.
    Events at any anchor and both baselines derive their directional metrics from
    these primitives + the anchor bar's open as entry (risk_model's window-gather
    idiom, applied per (symbol, day))."""
    out = stream.reset_index(drop=True).copy()
    n_total = len(out)
    t_min = (pd.to_datetime(out["bar_start"]).to_numpy()
             .astype("datetime64[m]").astype(np.int64))
    close = out["close"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    cols = {}
    for w in windows:
        w = int(w)
        for name in ("fwd_max_close", "fwd_min_close", "fwd_max_high",
                     "fwd_min_low", "endpoint_close"):
            cols[f"{name}_w{w}"] = np.full(n_total, np.nan)
        cols[f"complete_w{w}"] = np.zeros(n_total, dtype=bool)
    for _, gpos in out.groupby(["symbol", "date"], sort=False).indices.items():
        gpos = np.asarray(gpos)
        gt = t_min[gpos]
        n = len(gpos)
        c, h, lo = close[gpos], high[gpos], low[gpos]
        for w in windows:
            w = int(w)
            end = np.searchsorted(gt, gt + w, side="left")     # window = [j, end_j)
            span = int((end - np.arange(n)).max())
            idx = np.arange(n)[:, None] + np.arange(span)[None, :]
            in_win = idx < end[:, None]
            safe = np.where(in_win & (idx < n), idx, 0)
            cols[f"fwd_max_close_w{w}"][gpos] = np.where(in_win, c[safe], -np.inf).max(axis=1)
            cols[f"fwd_min_close_w{w}"][gpos] = np.where(in_win, c[safe], np.inf).min(axis=1)
            cols[f"fwd_max_high_w{w}"][gpos] = np.where(in_win, h[safe], -np.inf).max(axis=1)
            cols[f"fwd_min_low_w{w}"][gpos] = np.where(in_win, lo[safe], np.inf).min(axis=1)
            cols[f"endpoint_close_w{w}"][gpos] = c[end - 1]     # last bar before T+W
            cols[f"complete_w{w}"][gpos] = gt + (w - 1) <= gt[n - 1]
    for name, arr in cols.items():
        out[name] = arr
    return out


def directional_outcomes(anchor_rows, direction, windows, *, threshold=PROFIT_THRESHOLD,
                         entry=None):
    """Directional metrics from per-bar primitives at the anchor rows.

    ``direction`` +1 long / -1 short (scalar or aligned array); ``entry`` defaults
    to the anchor bar's own open. Sign-aligned: positive = favorable. Incomplete
    windows yield NaN metrics (complete flag kept for counting)."""
    e = (anchor_rows["open"].astype(float) if entry is None
         else pd.Series(np.asarray(entry, dtype=float), index=anchor_rows.index))
    d = pd.Series(np.asarray(direction, dtype=float), index=anchor_rows.index) \
        if not np.isscalar(direction) else pd.Series(float(direction), index=anchor_rows.index)
    res = pd.DataFrame(index=anchor_rows.index)
    for w in windows:
        w = int(w)
        long_mask = d > 0
        best_close = anchor_rows[f"fwd_max_close_w{w}"].where(long_mask,
                                                              anchor_rows[f"fwd_min_close_w{w}"])
        fav_extreme = anchor_rows[f"fwd_max_high_w{w}"].where(long_mask,
                                                              anchor_rows[f"fwd_min_low_w{w}"])
        adv_extreme = anchor_rows[f"fwd_min_low_w{w}"].where(long_mask,
                                                             anchor_rows[f"fwd_max_high_w{w}"])
        res[f"w{w}_close_touch"] = (d * (best_close / e - 1) >= threshold).astype(float)
        res[f"w{w}_extreme_touch"] = (d * (fav_extreme / e - 1) >= threshold).astype(float)
        res[f"w{w}_endpoint_ret"] = d * (anchor_rows[f"endpoint_close_w{w}"] / e - 1)
        res[f"w{w}_mae"] = d * (adv_extreme / e - 1)
        res[f"w{w}_mfe"] = d * (fav_extreme / e - 1)
        res[f"w{w}_complete"] = anchor_rows[f"complete_w{w}"].astype(bool)
        bad = ~res[f"w{w}_complete"]
        res.loc[bad, [f"w{w}_close_touch", f"w{w}_extreme_touch",
                      f"w{w}_endpoint_ret", f"w{w}_mae", f"w{w}_mfe"]] = np.nan
    return res


def attach_whipsaw_and_resume(events, *, whipsaw_minutes=5, resume_minutes=15):
    """Whipsaw = opposite-direction cross within (T, T+whipsaw_minutes] (TIMESTAMP
    based -- "5분 안에" means wall clock, not bar count); resume = original-direction
    cross within (T_rev, T_rev+resume_minutes]. Adds whipsaw / tt_whipsaw_min /
    rev_bar_start / resume / tt_resume_min. The caller passes ONE (variant,
    ma_source) cross set at a time; the frame must be RangeIndex-ed."""
    ev = events.reset_index(drop=True).copy()
    ev["whipsaw"] = False
    ev["tt_whipsaw_min"] = np.nan
    ev["rev_bar_start"] = pd.Series(pd.NA, index=ev.index, dtype="object")
    ev["resume"] = False
    ev["tt_resume_min"] = np.nan
    if ev.empty:
        return ev
    t_all = pd.to_datetime(ev["bar_start"]).to_numpy()
    dir_all = ev["direction"].to_numpy()
    for _, pos in ev.groupby("symbol", sort=False).indices.items():
        pos = np.asarray(pos)
        order = np.argsort(t_all[pos], kind="stable")
        pos = pos[order]
        times, dirs = t_all[pos], dir_all[pos]
        up_t, up_p = times[dirs == 1], pos[dirs == 1]
        dn_t, dn_p = times[dirs == -1], pos[dirs == -1]
        for ev_t, ev_p, rev_t_arr, same_t_arr in (
                (up_t, up_p, dn_t, up_t), (dn_t, dn_p, up_t, dn_t)):
            if not len(ev_t) or not len(rev_t_arr):
                continue
            j = np.searchsorted(rev_t_arr, ev_t, side="right")
            has = j < len(rev_t_arr)
            dt_min = np.full(len(ev_t), np.nan)
            dt_min[has] = (rev_t_arr[j[has]] - ev_t[has]) / np.timedelta64(1, "m")
            hit = has & (dt_min <= float(whipsaw_minutes))
            hit_rows = ev_p[hit]
            ev.loc[hit_rows, "whipsaw"] = True
            ev.loc[hit_rows, "tt_whipsaw_min"] = dt_min[hit]
            rev_hit_t = rev_t_arr[j[hit]]
            ev.loc[hit_rows, "rev_bar_start"] = [
                str(pd.Timestamp(t)) for t in rev_hit_t]
            k = np.searchsorted(same_t_arr, rev_hit_t, side="right")
            khas = k < len(same_t_arr)
            rt = np.full(len(rev_hit_t), np.nan)
            rt[khas] = (same_t_arr[k[khas]] - rev_hit_t[khas]) / np.timedelta64(1, "m")
            resumed = khas & (rt <= float(resume_minutes))
            ev.loc[hit_rows[resumed], "resume"] = True
            ev.loc[hit_rows[resumed], "tt_resume_min"] = rt[resumed]
    return ev


def mark_cluster_first(events, *, cluster_minutes=15):
    """First-in-cluster: no SAME-direction cross of the same (symbol, date) within
    the previous ``cluster_minutes`` (rapid re-signals share one swing and one
    overlapping outcome window -- the de-duplicated cut de-weights them)."""
    ev = events.reset_index(drop=True).copy()
    ev["cluster_first"] = True
    if ev.empty:
        return ev
    t = pd.to_datetime(ev["bar_start"])
    for _, pos in ev.groupby(["symbol", "date", "direction"], sort=False).indices.items():
        pos = np.asarray(pos)
        tt = t.iloc[pos].sort_values()
        gap_min = tt.diff().dt.total_seconds().to_numpy() / 60.0
        not_first = np.nan_to_num(gap_min, nan=np.inf) <= float(cluster_minutes)
        ev.loc[tt.index.to_numpy()[not_first], "cluster_first"] = False
    return ev


def matched_baseline_index(events, reg, *, halfwidth_minutes=30, cross_positions=None):
    """Local-regime control: for each event, the same-(symbol, date) regular bars
    within +-halfwidth minutes (timestamp bounds, inclusive), excluding every cross
    bar of the SAME cross set and the event bar itself. Returns (event_row, bar_row)
    positional int arrays; the caller aggregates primitives at bar_row with the
    matched event's direction and the bar's own open as entry."""
    if events.empty:
        return np.array([], dtype=int), np.array([], dtype=int)
    ev = events.reset_index(drop=True)
    reg_t = pd.to_datetime(reg["bar_start"]).to_numpy()
    is_cross = np.zeros(len(reg), dtype=bool)
    if cross_positions is not None and len(cross_positions):
        is_cross[np.asarray(cross_positions, dtype=int)] = True
    reg_groups = reg.groupby(["symbol", "date"], sort=False).indices
    ev_t = pd.to_datetime(ev["bar_start"]).to_numpy()
    half = np.timedelta64(int(halfwidth_minutes), "m")
    ev_rows, bar_rows = [], []
    for key, epos in ev.groupby(["symbol", "date"], sort=False).indices.items():
        gpos = reg_groups.get(key)
        if gpos is None:
            continue
        gpos = np.asarray(gpos)
        gt = reg_t[gpos]
        for row in np.asarray(epos):
            t0 = ev_t[row]
            a = np.searchsorted(gt, t0 - half, side="left")
            b = np.searchsorted(gt, t0 + half, side="right")
            cand = gpos[a:b]
            keep = ~is_cross[cand] & (gt[a:b] != t0)
            cand = cand[keep]
            if len(cand):
                ev_rows.append(np.full(len(cand), row))
                bar_rows.append(cand)
    if not ev_rows:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(ev_rows), np.concatenate(bar_rows)
