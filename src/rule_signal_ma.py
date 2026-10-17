"""MA-breakout rule variants, compared against the v0 cross-symbol rule.

Versions (each MA-using one has a daily and an intraday variant):
  v0      : the existing rule (src/rule_signal.py) -- unchanged, no MA.
  ma_only : up/down come purely from the MA breakout cross (v0 score ignored).
  v1      : v0 score PLUS a breakout bonus (additive).
  v2      : v0 score, but a candidate is kept ONLY if it also broke out (hard filter).

Breakout is a cross EVENT: the previous (same-day) bar closed on one side of its
MA and the current bar closed on the other side.
  up_cross   = prev_close <= prev_MA  AND  close > MA
  down_cross = prev_close >= prev_MA  AND  close < MA
Only same-day prior bars are used (no leakage). The first bar of a day, or a bar
whose MA is NaN (first ~5 trading days), yields no cross.

ma_only gets a fixed score (MA_ONLY_SCORE) on a cross so it clears the downstream
conservative gate (up_score >= 4); the daily top-k tiebreakers then rank the
breakouts. v2 leaves the v0 score untouched and instead gates is_up/down_candidate
by the cross, so the Step 12 long filter keeps only confirmed breakouts.
"""

from __future__ import annotations

from rule_signal import (
    DEFAULT_DOWN_THRESHOLD,
    DEFAULT_UP_THRESHOLD,
    OUTPUT_COLUMNS,
    build_rule_signals,
)

MA_COLUMN = {"daily": "ma5_daily", "intraday": "ma5_intraday"}
MODES = ("ma_only", "v1", "v2")
DEFAULT_BREAKOUT_WEIGHT = 2.0
MA_ONLY_SCORE = 5.0  # >= the conservative up_threshold (4) so a cross is selectable

EXTRA_COLUMNS = ["ma5_value", "up_cross", "down_cross", "ma_breakout", "rule_version"]
V_OUTPUT_COLUMNS = OUTPUT_COLUMNS + EXTRA_COLUMNS


def breakout_flags(df, ma_col):
    """Return (sorted_df, up_cross, down_cross) for same-day prior-bar crosses.

    ``df`` needs symbol, decision_time, close, and ``ma_col``. The returned frame
    is sorted by (symbol, decision_time) with a contiguous index; up/down are
    boolean numpy arrays aligned to it.
    """
    import numpy as np

    d = df.copy()
    d["_date"] = d["decision_time"].str.slice(0, 10)
    d = d.sort_values(["symbol", "decision_time"]).reset_index(drop=True)
    g = d.groupby(["symbol", "_date"], sort=False)
    prev_close = g["close"].shift(1)
    prev_ma = g[ma_col].shift(1)
    ma = d[ma_col]
    close = d["close"]
    up = ((prev_close <= prev_ma) & (close > ma)).fillna(False).to_numpy()
    down = ((prev_close >= prev_ma) & (close < ma)).fillna(False).to_numpy()
    return d, up, down


def build_ma_rule_signals(features, ma_frame, *, mode, ma_kind,
                          up_threshold=DEFAULT_UP_THRESHOLD, down_threshold=DEFAULT_DOWN_THRESHOLD,
                          breakout_weight=DEFAULT_BREAKOUT_WEIGHT, ma_only_score=MA_ONLY_SCORE,
                          base=None):
    """Build one MA-breakout rule variant from features_relative + MA columns.

    ``ma_frame`` carries (symbol, decision_time, ma5_daily, ma5_intraday). Pass a
    precomputed v0 ``base`` (build_rule_signals output) to avoid recomputing it.
    """
    import numpy as np
    import pandas as pd

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if ma_kind not in MA_COLUMN:
        raise ValueError(f"ma_kind must be one of {tuple(MA_COLUMN)}, got {ma_kind!r}")
    ma_col = MA_COLUMN[ma_kind]

    base = build_rule_signals(features, up_threshold, down_threshold) if base is None else base.copy()
    if len(base) == 0:
        return pd.DataFrame(columns=V_OUTPUT_COLUMNS)

    ma = ma_frame[["symbol", "decision_time", ma_col]].drop_duplicates(["symbol", "decision_time"])
    d = base.merge(ma, on=["symbol", "decision_time"], how="left", validate="one_to_one")
    d, up, down = breakout_flags(d, ma_col)
    up_f = up.astype(float)
    down_f = down.astype(float)

    if mode == "ma_only":
        d["up_score"] = np.round(up_f * ma_only_score, 4)
        d["down_score"] = np.round(down_f * ma_only_score, 4)
    elif mode == "v1":
        d["up_score"] = np.round(d["up_score"].to_numpy() + breakout_weight * up_f, 4)
        d["down_score"] = np.round(d["down_score"].to_numpy() + breakout_weight * down_f, 4)
    # v2: scores unchanged

    if mode == "v2":
        d["is_up_candidate"] = ((d["is_up_candidate"] == 1) & up).astype(int)
        d["is_down_candidate"] = ((d["is_down_candidate"] == 1) & down).astype(int)
    else:
        d["is_up_candidate"] = (d["up_score"] >= up_threshold).astype(int)
        d["is_down_candidate"] = (d["down_score"] >= down_threshold).astype(int)

    iu = d["is_up_candidate"] == 1
    idn = d["is_down_candidate"] == 1
    d["primary_side"] = np.select(
        [iu & ~idn, idn & ~iu, iu & idn & (d["up_score"] > d["down_score"]),
         iu & idn & (d["down_score"] > d["up_score"]), iu & idn],
        ["UP", "DOWN", "UP", "DOWN", "MIXED"], default="NONE")

    tag = f"ma5_{ma_kind}"
    d["reason_1"] = np.where(up, f"{tag}_up_cross", np.where(down, f"{tag}_down_cross", d["reason_1"]))
    d["ma5_value"] = d[ma_col]
    d["up_cross"] = up.astype(int)
    d["down_cross"] = down.astype(int)
    d["ma_breakout"] = np.where(up, "up", np.where(down, "down", "none"))
    d["rule_version"] = f"{mode}_{ma_kind}"
    return d[V_OUTPUT_COLUMNS].reset_index(drop=True)
