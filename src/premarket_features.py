"""v3.0: premarket (08:00-09:00) context features for the risk model.

Pure, leakage-safe builders. The output is **one row per trading date** describing
the premarket window of:

  * the target symbol (``pm_<target>_*`` -- e.g. ``pm_000660_*``),
  * optional anchor symbols (``pm_<anchor>_*`` -- e.g. ``pm_005930_*``), and
  * the active universe (``pm_universe_*`` + the target's relative/rank context).

These daily rows are joined onto same-day **regular-session** decision rows
(``decision_time >= 09:00``) as extra features for the existing risk model. The
premarket window closes *before* the regular open, so every premarket value is
strictly causal for all regular-session rows on the same day (the only cross-day
input is the *previous* completed regular close, used to define the overnight gap).

Leakage rules enforced here:
  * the premarket window uses ``premarket_start <= bar_end < premarket_end`` only;
  * ``pm_ret`` is measured against the **previous** day's regular close (shifted),
    never the current day's;
  * volume z-scores use prior trading days only (``shift(1).rolling``).

No orders. Research / feature-engineering only -- this module never touches Kiwoom
or the DB; the calling script supplies bars and persists the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Premarket window (time-of-day, on bar_end). Regular session bounds are used only to
# pick each day's regular close for the overnight-gap denominator.
PREMARKET_START = "08:00"
PREMARKET_END = "09:00"
REGULAR_START = "09:00:00"
REGULAR_END = "15:30:00"

# Leak-free volume z-score: prior trading days only.
VOLUME_Z_WINDOW = 20
VOLUME_Z_MIN_PERIODS = 5

# Conservative quality flag for a suspicious premarket print (auction spike / bad tick).
OUTLIER_ABS_RET = 0.30

# Provenance tag written into every daily row (the `premarket_feature_set` column).
PREMARKET_FEATURE_SET = "v3_full"


# --------------------------------------------------------------------------- #
# Column-name helpers (names are generated from the resolved codes, never from a
# hard-coded literal -- the target/anchor codes come from config at call time).
# --------------------------------------------------------------------------- #
def pm_target_columns(symbol):
    """Model feature columns for the target symbol's premarket window.

    Excludes the raw price/level columns (``pm_<sym>_vwap``, ``pm_<sym>_volume``)
    which are kept in the daily table for diagnostics but are non-stationary and a
    poor fit as model inputs; the normalised ``close_to_vwap`` / ``volume_z`` are
    used instead.
    """
    return [
        f"pm_{symbol}_ret",
        f"pm_{symbol}_high_ret",
        f"pm_{symbol}_low_ret",
        f"pm_{symbol}_range",
        f"pm_{symbol}_volume_z",
        f"pm_{symbol}_close_to_vwap",
        f"pm_{symbol}_last_5m_ret",
        f"pm_{symbol}_has_data",
        f"pm_{symbol}_bar_count",
        f"pm_{symbol}_outlier_flag",
    ]


def pm_target_table_columns(symbol):
    """All target columns emitted into the daily table (model features + diagnostics)."""
    return pm_target_columns(symbol) + [f"pm_{symbol}_volume", f"pm_{symbol}_vwap"]


def pm_anchor_columns(symbol, target):
    """Anchor feature columns (for v3.0c/v3.0d). ``pm_<target>_minus_<anchor>_ret`` is
    the target's premarket gap relative to the anchor's -- a cross-symbol spread."""
    return [
        f"pm_{symbol}_ret",
        f"pm_{symbol}_range",
        f"pm_{symbol}_volume_z",
        f"pm_{symbol}_close_to_vwap",
        f"pm_{symbol}_last_5m_ret",
        f"pm_{target}_minus_{symbol}_ret",
        f"pm_{symbol}_has_data",
    ]


def pm_universe_columns(target):
    """Universe-wide premarket breadth/relative/rank columns (for v3.0b)."""
    return [
        "pm_universe_breadth_pos",
        "pm_universe_breadth_gt_0_5",
        "pm_universe_mean_ret",
        "pm_universe_median_ret",
        "pm_universe_std_ret",
        "pm_universe_dispersion_ret",
        "pm_universe_mean_range",
        "pm_universe_total_volume_z",
        f"pm_{target}_relative_ret_to_universe",
        f"pm_{target}_rank_ret",
        f"pm_{target}_rank_volume_z",
        "pm_universe_active_symbol_count",
        "pm_universe_missing_symbol_count",
    ]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _pad_tod(tod):
    """'08:00' -> '08:00:00' (string time-of-day comparisons need fixed width)."""
    parts = str(tod).split(":")
    while len(parts) < 3:
        parts.append("00")
    return ":".join(f"{int(p):02d}" for p in parts)


def _premarket_daily_by_symbol(bars, *, premarket_start, premarket_end):
    """Per-(symbol, date) premarket aggregates from the 08:00-09:00 window."""
    if bars is None or len(bars) == 0:
        return pd.DataFrame()
    b = bars.copy()
    end = b["bar_end"] if "bar_end" in b.columns else b["bar_start"]
    b["_date"] = b["bar_start"].str.slice(0, 10)
    b["_end_tod"] = end.str.slice(11, 19)
    b["_end_dt"] = pd.to_datetime(end)
    start_t, end_t = _pad_tod(premarket_start), _pad_tod(premarket_end)
    pm = b[(b["_end_tod"] >= start_t) & (b["_end_tod"] < end_t)].copy()
    if pm.empty:
        return pd.DataFrame()

    def _agg(g):
        g = g.sort_values("_end_dt")
        high = float(g["high"].max())
        low = float(g["low"].min())
        last_close = float(g["close"].iloc[-1])
        total_vol = float(g["volume"].sum())
        typical = (g["high"].astype(float) + g["low"].astype(float) + g["close"].astype(float)) / 3.0
        vwap = float((typical * g["volume"].astype(float)).sum() / total_vol) if total_vol > 0 else np.nan
        last_end = g["_end_dt"].iloc[-1]
        prior = g[g["_end_dt"] <= last_end - pd.Timedelta(minutes=5)]
        ref_close = float(prior["close"].iloc[-1]) if len(prior) else np.nan
        return pd.Series({
            "pm_open": float(g["open"].iloc[0]),
            "pm_high": high,
            "pm_low": low,
            "pm_last_close": last_close,
            "pm_volume": total_vol,
            "pm_bar_count": int(len(g)),
            "pm_vwap": vwap,
            "pm_ref_close_5m": ref_close,
        })

    out = pm.groupby(["symbol", "_date"], sort=False).apply(_agg).reset_index()
    return out.rename(columns={"_date": "date"})


def _regular_close_by_symbol(bars):
    """Per-(symbol, date) regular-session close (close of the last regular bar)."""
    if bars is None or len(bars) == 0:
        return pd.DataFrame(columns=["symbol", "date", "regular_close"])
    b = bars.copy()
    b["_date"] = b["bar_start"].str.slice(0, 10)
    b["_start_tod"] = b["bar_start"].str.slice(11, 19)
    reg = b[(b["_start_tod"] >= REGULAR_START) & (b["_start_tod"] <= REGULAR_END)]
    if reg.empty:
        return pd.DataFrame(columns=["symbol", "date", "regular_close"])
    idx = reg.groupby(["symbol", "_date"])["bar_start"].idxmax()
    out = reg.loc[idx, ["symbol", "_date", "close"]].copy()
    out = out.rename(columns={"_date": "date", "close": "regular_close"})
    out["regular_close"] = out["regular_close"].astype(float)
    return out.reset_index(drop=True)


def _prior_zscore(series, *, window=VOLUME_Z_WINDOW, min_periods=VOLUME_Z_MIN_PERIODS):
    """z-score of ``series`` vs its own PRIOR-day rolling stats (leak-free)."""
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _symbol_daily(bars, symbols, *, premarket_start, premarket_end):
    """Per-(symbol, date) premarket frame with all derived per-symbol features.

    Columns: symbol, date, pm_open/high/low/last_close/volume/bar_count/vwap,
    regular_close, prev_regular_close, pm_ret, pm_high_ret, pm_low_ret, pm_range,
    pm_close_to_vwap, pm_last_5m_ret, pm_volume_z, pm_has_data, pm_outlier_flag.
    """
    keep = bars[bars["symbol"].isin(symbols)] if symbols else bars
    pm = _premarket_daily_by_symbol(keep, premarket_start=premarket_start, premarket_end=premarket_end)
    if pm.empty:
        return pd.DataFrame()

    # "Previous completed regular close" = the last regular close STRICTLY BEFORE the
    # premarket day (a backward as-of lookup, per symbol). allow_exact_matches=False is
    # the leakage guard: day D's own 15:30 close lies AFTER the 08:00-09:00 window, so it
    # must never be the gap base -- only the prior trading day's close. This is also robust
    # to days that have a regular close but no premarket bars (and vice versa).
    pm = pm.copy()
    pm["_d"] = pd.to_datetime(pm["date"])
    reg = _regular_close_by_symbol(keep)
    if reg.empty:
        pm["prev_regular_close"] = np.nan
        df = pm.sort_values(["symbol", "date"]).reset_index(drop=True)
    else:
        reg = reg.copy()
        reg["_d"] = pd.to_datetime(reg["date"])
        right = (reg.sort_values("_d")[["symbol", "_d", "regular_close"]]
                 .rename(columns={"regular_close": "prev_regular_close"}))
        df = pd.merge_asof(pm.sort_values("_d"), right, on="_d", by="symbol",
                           direction="backward", allow_exact_matches=False)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.drop(columns=["_d"])

    pcrev = df["prev_regular_close"]
    df["pm_ret"] = df["pm_last_close"] / pcrev - 1.0
    df["pm_high_ret"] = df["pm_high"] / pcrev - 1.0
    df["pm_low_ret"] = df["pm_low"] / pcrev - 1.0
    df["pm_range"] = df["pm_high"] / df["pm_low"] - 1.0
    df["pm_close_to_vwap"] = df["pm_last_close"] / df["pm_vwap"] - 1.0
    df["pm_last_5m_ret"] = df["pm_last_close"] / df["pm_ref_close_5m"] - 1.0
    df["pm_has_data"] = (df["pm_bar_count"] > 0).astype(float)

    df["pm_volume_z"] = (
        df.groupby("symbol", group_keys=False)["pm_volume"].apply(_prior_zscore)
    )

    bad = (
        df["pm_ret"].abs().gt(OUTLIER_ABS_RET)
        | df["pm_range"].abs().gt(OUTLIER_ABS_RET)
        | df["pm_low"].le(0)
    )
    df["pm_outlier_flag"] = bad.fillna(False).astype(float)
    return df


def _rename_symbol_block(sym_daily, symbol, columns_for):
    """Slice one symbol's rows and rename pm_* -> pm_<symbol>_* for the daily table."""
    base = {
        "pm_ret": f"pm_{symbol}_ret",
        "pm_high_ret": f"pm_{symbol}_high_ret",
        "pm_low_ret": f"pm_{symbol}_low_ret",
        "pm_range": f"pm_{symbol}_range",
        "pm_volume": f"pm_{symbol}_volume",
        "pm_volume_z": f"pm_{symbol}_volume_z",
        "pm_vwap": f"pm_{symbol}_vwap",
        "pm_close_to_vwap": f"pm_{symbol}_close_to_vwap",
        "pm_last_5m_ret": f"pm_{symbol}_last_5m_ret",
        "pm_has_data": f"pm_{symbol}_has_data",
        "pm_bar_count": f"pm_{symbol}_bar_count",
        "pm_outlier_flag": f"pm_{symbol}_outlier_flag",
    }
    block = sym_daily[sym_daily["symbol"] == symbol].copy()
    out = block[["date"]].copy()
    for src, dst in base.items():
        if dst in columns_for and src in block.columns:
            out[dst] = block[src].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #
def build_daily_premarket_features(
    bars,
    *,
    symbols,
    target_symbol,
    anchors=None,
    universe_symbols=None,
    premarket_start=PREMARKET_START,
    premarket_end=PREMARKET_END,
):
    """Build the per-trading-date premarket feature table.

    Parameters
    ----------
    bars : DataFrame
        Long OHLCV frame with ``symbol, bar_start, bar_end, open, high, low, close,
        volume`` (e.g. ``bars_1m`` source=``al_raw``, which spans 08:00-19:59 so it
        carries both the premarket window AND the regular session for the gap base).
    symbols : list[str]
        Symbols to aggregate (the active universe).
    target_symbol : str
        The judged symbol (e.g. 000660). Required -- no hard-coded default.
    anchors : list[str] | None
        Anchor codes (e.g. 005930, 402340) for v3.0c/v3.0d. Missing anchors do not
        crash; their columns are emitted as NaN with ``pm_<anchor>_has_data`` = 0.
    universe_symbols : list[str] | None
        Symbols counted in the universe breadth/rank features (defaults to ``symbols``).

    Returns
    -------
    DataFrame
        One row per trading date with ``trade_date, symbol, premarket_feature_set`` +
        the target / anchor / universe premarket columns. Days with no premarket bars
        for a symbol yield NaN features + a ``has_data`` flag of 0.
    """
    anchors = list(anchors or [])
    universe_symbols = list(universe_symbols or symbols)
    all_symbols = list(dict.fromkeys(list(symbols) + universe_symbols + [target_symbol] + anchors))

    target_table_cols = pm_target_table_columns(target_symbol)
    universe_cols = pm_universe_columns(target_symbol)
    anchor_cols = [c for a in anchors for c in pm_anchor_columns(a, target_symbol)]
    all_feature_cols = target_table_cols + anchor_cols + universe_cols

    sym_daily = _symbol_daily(bars, all_symbols, premarket_start=premarket_start, premarket_end=premarket_end)
    if sym_daily.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "premarket_feature_set"] + all_feature_cols)

    dates = pd.DataFrame({"date": sorted(sym_daily["date"].unique())})

    # --- target block ---------------------------------------------------------
    out = dates.merge(_rename_symbol_block(sym_daily, target_symbol, target_table_cols), on="date", how="left")
    if f"pm_{target_symbol}_has_data" in out.columns:
        out[f"pm_{target_symbol}_has_data"] = out[f"pm_{target_symbol}_has_data"].fillna(0.0)

    # --- anchor blocks (v3.0c/v3.0d) -----------------------------------------
    for anchor in anchors:
        cols = pm_anchor_columns(anchor, target_symbol)
        block = _rename_symbol_block(sym_daily, anchor, cols)
        out = out.merge(block, on="date", how="left")
        spread = f"pm_{target_symbol}_minus_{anchor}_ret"
        if f"pm_{anchor}_ret" in out.columns and f"pm_{target_symbol}_ret" in out.columns:
            out[spread] = out[f"pm_{target_symbol}_ret"] - out[f"pm_{anchor}_ret"]
        else:
            out[spread] = np.nan
        hd = f"pm_{anchor}_has_data"
        if hd in out.columns:
            out[hd] = out[hd].fillna(0.0)
        else:
            out[hd] = 0.0

    # --- universe block (v3.0b) ----------------------------------------------
    uni = _universe_daily(sym_daily, universe_symbols, target_symbol)
    out = out.merge(uni, on="date", how="left")

    out.insert(0, "premarket_feature_set", PREMARKET_FEATURE_SET)
    out.insert(0, "symbol", target_symbol)
    out = out.rename(columns={"date": "trade_date"})
    ordered = ["trade_date", "symbol", "premarket_feature_set"] + [c for c in all_feature_cols if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    return out[ordered + extra].sort_values("trade_date").reset_index(drop=True)


def _universe_daily(sym_daily, universe_symbols, target_symbol):
    """Per-date universe breadth / dispersion / rank features."""
    cols = pm_universe_columns(target_symbol)
    u = sym_daily[sym_daily["symbol"].isin(universe_symbols)].copy()
    if u.empty:
        return pd.DataFrame(columns=["date"] + cols)

    # ranks across the universe at each date (1 = strongest); NaN returns rank last.
    u["_rank_ret"] = u.groupby("date")["pm_ret"].rank(ascending=False, method="min")
    u["_rank_volz"] = u.groupby("date")["pm_volume_z"].rank(ascending=False, method="min")

    have = u[u["pm_has_data"] > 0]
    grp = have.groupby("date")
    rows = pd.DataFrame({"date": sorted(u["date"].unique())}).set_index("date")
    rows["pm_universe_breadth_pos"] = grp["pm_ret"].apply(lambda s: float((s > 0).mean()))
    rows["pm_universe_breadth_gt_0_5"] = grp["pm_ret"].apply(lambda s: float((s >= 0.005).mean()))
    rows["pm_universe_mean_ret"] = grp["pm_ret"].mean()
    rows["pm_universe_median_ret"] = grp["pm_ret"].median()
    rows["pm_universe_std_ret"] = grp["pm_ret"].std()
    rows["pm_universe_dispersion_ret"] = grp["pm_ret"].apply(
        lambda s: float(s.quantile(0.75) - s.quantile(0.25)) if s.notna().any() else np.nan
    )
    rows["pm_universe_mean_range"] = grp["pm_range"].mean()
    rows["pm_universe_active_symbol_count"] = grp.size().astype(float)
    n_universe = len(set(universe_symbols))
    rows["pm_universe_missing_symbol_count"] = (n_universe - rows["pm_universe_active_symbol_count"]).clip(lower=0)

    # universe total premarket volume z-score (prior days only)
    total_vol = grp["pm_volume"].sum().reindex(rows.index)
    rows["pm_universe_total_volume_z"] = _prior_zscore(total_vol)

    # target's relative return + ranks
    tgt = u[u["symbol"] == target_symbol].set_index("date")
    rows[f"pm_{target_symbol}_relative_ret_to_universe"] = tgt["pm_ret"] - rows["pm_universe_mean_ret"]
    rows[f"pm_{target_symbol}_rank_ret"] = tgt["_rank_ret"]
    rows[f"pm_{target_symbol}_rank_volume_z"] = tgt["_rank_volz"]

    return rows.reset_index()[["date"] + cols]
