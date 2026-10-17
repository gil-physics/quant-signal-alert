"""Step 7: per-timeframe feature builders (shared logic).

Every rolling / shift / expanding computation is grouped by (symbol, trade date)
and looks strictly backward, so:
  * no value crosses a day boundary (instr #5/#6), and
  * no feature uses future bars — each bar uses only itself and earlier bars, and
    is usable from its bar_end onward (instr #7).

Feature formulas mirror the legacy minute-feature definitions (ratio-based, so the
denominator is part of the same bar): range = high/low - 1, body = close/open - 1,
shadows vs max/min(open,close), volume z = (v - rollmean)/rollstd, day-anchored
return_from_open / distance_from_day_high/low / vwap_distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS_1M = ["symbol", "source", "decision_time", "close", "bar_end_1m"]

FEATURE_COLUMNS_1M = [
    "ret_1m_1m", "ret_3m_1m", "ret_5m_1m", "ret_10m_1m", "ret_20m_1m", "ret_60m_1m",
    "range_1m", "body_1m", "upper_shadow_1m", "lower_shadow_1m",
    "volume_z_20m_1m", "volume_z_60m_1m",
    "vwap_distance_1m", "return_from_open_1m",
    "distance_from_day_high_1m", "distance_from_day_low_1m",
]

# 5-day moving-average features. These cross day boundaries (need the prior 5
# trading days), so they are NaN during the first few days and are kept OUT of
# FEATURE_COLUMNS_1M: build_features.py drops warmup rows on FEATURE_COLUMNS only,
# so adding these never removes a v0 row. ``ma5_daily`` uses the prior 5 daily
# closes (constant intraday); ``ma5_intraday`` uses the same minute-of-day on the
# prior 5 trading days. ``*_n`` report how many prior days actually contributed.
MA_WINDOW = 5
MA_COLUMNS_1M = [
    "ma5_daily", "ma5_intraday", "dist_ma5_daily", "dist_ma5_intraday",
    "ma5_daily_n", "ma5_intraday_n",
]
COLUMNS_1M = ID_COLUMNS_1M + FEATURE_COLUMNS_1M + MA_COLUMNS_1M


def id_columns_bar(suffix: str):
    return ["symbol", "source", f"bar_start_{suffix}", f"bar_end_{suffix}", f"close_{suffix}"]


def feature_columns_bar(suffix: str):
    return [
        f"ret_1bar_{suffix}", f"ret_2bar_{suffix}", f"ret_3bar_{suffix}",
        f"range_1bar_{suffix}", f"body_1bar_{suffix}",
        f"volume_z_10bar_{suffix}", f"volume_z_20bar_{suffix}",
    ]


def _prep(df):
    df = df.copy()
    df["_date"] = df["bar_start"].str.slice(0, 10)
    return df.sort_values(["symbol", "bar_start"]).reset_index(drop=True)


def _roll(group_volume, window, func):
    """Grouped rolling reduction realigned to the original row index."""
    return getattr(group_volume.rolling(window), func)().reset_index(level=[0, 1], drop=True)


def _volume_z(df, group, window):
    mean = _roll(group["volume"], window, "mean")
    std = _roll(group["volume"], window, "std")
    return (df["volume"] - mean) / std.replace(0, np.nan)


def build_minute_features(df_1m, source: str):
    """Build the 1-minute feature frame (expects bars from source=al_regular).

    ``source`` is recorded as a provenance column so the table self-documents which
    bar source it was built from.
    """
    if df_1m is None or len(df_1m) == 0:
        return pd.DataFrame(columns=COLUMNS_1M)

    df = _prep(df_1m)
    group = df.groupby(["symbol", "_date"], sort=False)
    close_g = group["close"]

    out = pd.DataFrame(index=df.index)
    out["symbol"] = df["symbol"]
    out["source"] = source
    out["decision_time"] = df["bar_end"]
    out["close"] = df["close"].astype("int64")
    out["bar_end_1m"] = df["bar_end"]

    for w in (1, 3, 5, 10, 20, 60):
        out[f"ret_{w}m_1m"] = df["close"] / close_g.shift(w) - 1

    out["range_1m"] = df["high"] / df["low"] - 1
    out["body_1m"] = df["close"] / df["open"] - 1
    out["upper_shadow_1m"] = df["high"] / df[["open", "close"]].max(axis=1) - 1
    out["lower_shadow_1m"] = df[["open", "close"]].min(axis=1) / df["low"] - 1

    out["volume_z_20m_1m"] = _volume_z(df, group, 20)
    out["volume_z_60m_1m"] = _volume_z(df, group, 60)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical * df["volume"]).groupby([df["symbol"], df["_date"]]).cumsum()
    cum_v = df["volume"].groupby([df["symbol"], df["_date"]]).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    out["vwap_distance_1m"] = df["close"] / vwap - 1
    out["return_from_open_1m"] = df["close"] / group["open"].transform("first") - 1
    out["distance_from_day_high_1m"] = df["close"] / group["high"].cummax() - 1
    out["distance_from_day_low_1m"] = df["close"] / group["low"].cummin() - 1

    # --- 5-day moving averages (cross-day; leakage-safe: only PRIOR days used) ---
    df["_tod"] = df["bar_start"].str.slice(11, 19)

    # intraday, time-of-day aligned: mean of the SAME minute's close over the prior
    # MA_WINDOW trading days (today excluded). shift(1) then rolling, within (symbol, tod).
    iv = df[["symbol", "_tod", "_date", "close"]].sort_values(["symbol", "_tod", "_date"])
    iv_prev = iv.groupby(["symbol", "_tod"], sort=False)["close"].shift(1)
    iv_grp = iv_prev.groupby([iv["symbol"], iv["_tod"]], sort=False)
    out["ma5_intraday"] = iv_grp.rolling(MA_WINDOW).mean().reset_index(level=[0, 1], drop=True)
    out["ma5_intraday_n"] = iv_grp.rolling(MA_WINDOW).count().reset_index(level=[0, 1], drop=True)

    # daily: mean of the prior MA_WINDOW trading days' CLOSE (last regular bar of the
    # day), broadcast constant across the trading day.
    daily = df.groupby(["symbol", "_date"], sort=True)["close"].last().reset_index(name="_dclose")
    daily = daily.sort_values(["symbol", "_date"])
    d_prev = daily.groupby("symbol", sort=False)["_dclose"].shift(1)
    d_grp = d_prev.groupby(daily["symbol"], sort=False)
    daily["ma5_daily"] = d_grp.rolling(MA_WINDOW).mean().reset_index(level=0, drop=True)
    daily["ma5_daily_n"] = d_grp.rolling(MA_WINDOW).count().reset_index(level=0, drop=True)
    keyed = df[["symbol", "_date"]].merge(
        daily[["symbol", "_date", "ma5_daily", "ma5_daily_n"]], on=["symbol", "_date"], how="left")
    out["ma5_daily"] = keyed["ma5_daily"].to_numpy()
    out["ma5_daily_n"] = keyed["ma5_daily_n"].to_numpy()

    out["dist_ma5_intraday"] = df["close"] / out["ma5_intraday"] - 1
    out["dist_ma5_daily"] = df["close"] / out["ma5_daily"] - 1

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[COLUMNS_1M].reset_index(drop=True)


def build_bar_features(df_bar, suffix: str, source: str):
    """Build 3m/5m features (expects bars from source=resampled_from_al_regular).

    suffix is "3m" or "5m"; produces ret_1..3bar, range/body, volume_z_10/20bar.
    ``source`` is recorded as a provenance column.
    """
    cols = id_columns_bar(suffix) + feature_columns_bar(suffix)
    if df_bar is None or len(df_bar) == 0:
        return pd.DataFrame(columns=cols)

    df = _prep(df_bar)
    group = df.groupby(["symbol", "_date"], sort=False)
    close_g = group["close"]

    out = pd.DataFrame(index=df.index)
    out["symbol"] = df["symbol"]
    out["source"] = source
    out[f"bar_start_{suffix}"] = df["bar_start"]
    out[f"bar_end_{suffix}"] = df["bar_end"]
    out[f"close_{suffix}"] = df["close"].astype("int64")

    for w in (1, 2, 3):
        out[f"ret_{w}bar_{suffix}"] = df["close"] / close_g.shift(w) - 1

    out[f"range_1bar_{suffix}"] = df["high"] / df["low"] - 1
    out[f"body_1bar_{suffix}"] = df["close"] / df["open"] - 1

    out[f"volume_z_10bar_{suffix}"] = _volume_z(df, group, 10)
    out[f"volume_z_20bar_{suffix}"] = _volume_z(df, group, 20)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[cols].reset_index(drop=True)
