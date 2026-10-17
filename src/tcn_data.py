"""Step 16: sequence-dataset construction for the TCN risk-model challenger (v4_tcn).

Torch-free by design: this module is pure numpy/pandas so the leakage invariants are
testable on the dependency-free / 32-bit runtime (torch lives in tcn_net.py, which is
only ever imported lazily from tcn_model.py).

The TCN consumes, per directional-alert decision row, a causal window of the last
``seq_len`` 1-minute bars' per-bar features (from the ``features_1m`` table) plus the
v1 static features (alert scores + cross-symbol relatives) at the decision time.

Causality contract (same convention as the whole repo):
  * ``decision_time`` is the CONFIRMED feature bar's ``bar_end`` (feature_lib
    builds ``features_1m.decision_time = bar_end``), and it is a FULL timestamp
    ("YYYY-MM-DD HH:MM:SS") -- a bare time-of-day would cross-match identical
    clock times on different days, so the anchor join hard-fails on it.
  * The window's newest element is the decision bar itself (bar_end == decision_time);
    the label's forward window starts at ``bar_start >= decision_time`` (risk_model
    ``searchsorted(side="left")``), so window and label never overlap.
  * Windows never cross a day boundary (day-start clamp) and never index past the
    anchor (the gather formula ``idx = j - offsets`` cannot express a future bar).
  * Rows are PADDED, never dropped, so the challenger scores the exact same frozen
    validation frame as v1.1 (no coverage skew).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import risk_model as rm

# Per-bar sequence channels drawn from features_1m (feature_lib.FEATURE_COLUMNS_1M +
# the scale-free MA5 distances from MA_COLUMNS_1M). The multi-window returns
# (ret_3/5/10/20/60m_1m) are deliberately EXCLUDED: within a 60-bar window they are
# (near-)partial sums of ret_1m_1m, i.e. linear functionals the dilated convolutions
# represent natively; keeping them adds five collinear channels. The day-anchored
# channels (vwap/return_from_open/day-high/low) carry the pre-window information.
SEQ_FEATURE_COLUMNS_V4 = [
    "ret_1m_1m",
    "range_1m",
    "body_1m",
    "upper_shadow_1m",
    "lower_shadow_1m",
    "volume_z_20m_1m",
    "volume_z_60m_1m",
    "vwap_distance_1m",
    "return_from_open_1m",
    "distance_from_day_high_1m",
    "distance_from_day_low_1m",
    "dist_ma5_intraday",
    "dist_ma5_daily",
]

# Static head inputs: exactly the v1 alert + cross-symbol features (13), reusing the
# risk_model constants so the lists can never drift apart.
STATIC_FEATURE_COLUMNS_V4 = list(rm.ALERT_FEATURE_COLUMNS) + list(rm.RELATIVE_FEATURE_COLUMNS)

# Session anchor for the time-of-day channel: minutes since 09:00, scaled by the
# 390-minute regular session (09:00-15:30). decision_time = bar_end -> (0, 1].
_SESSION_OPEN_MINUTES = 9 * 60
_SESSION_LENGTH_MINUTES = 390.0

ANCHOR_MIN_MATCH_RATE = 0.999


@dataclass(frozen=True)
class SequenceConfig:
    seq_len: int = 60
    seq_feature_columns: tuple = tuple(SEQ_FEATURE_COLUMNS_V4)
    add_tod_channel: bool = True
    add_mask_channel: bool = True

    @property
    def n_channels(self) -> int:
        return (len(self.seq_feature_columns)
                + int(self.add_tod_channel) + int(self.add_mask_channel))


def _validate_full_timestamps(values: pd.Series, *, where: str) -> None:
    """Hard-fail unless every value is a full 'YYYY-MM-DD HH:MM[:SS]' timestamp.

    A date-less time-of-day would cross-match identical clock times on different
    days in the anchor join, silently corrupting every window.
    """
    s = values.astype(str)
    ok = (
        (s.str.len() >= 16)
        & (s.str.slice(4, 5) == "-")
        & (s.str.slice(7, 8) == "-")
        & (s.str.slice(10, 11).isin([" ", "T"]))
        & (s.str.slice(13, 14) == ":")
    )
    if not bool(ok.all()):
        bad = s[~ok].head(3).tolist()
        raise ValueError(
            f"{where}: decision_time must be a full timestamp with a date component "
            f"('YYYY-MM-DD HH:MM:SS'); got e.g. {bad}"
        )


def _tod_fraction(decision_time: pd.Series) -> np.ndarray:
    """Minutes-since-09:00 of the (confirmed-bar-end) decision_time, / 390."""
    s = decision_time.astype(str)
    minutes = s.str.slice(11, 13).astype(int) * 60 + s.str.slice(14, 16).astype(int)
    return ((minutes - _SESSION_OPEN_MINUTES) / _SESSION_LENGTH_MINUTES).to_numpy(dtype=np.float32)


class SequencePreprocessor:
    """Median-impute + z-score per sequence channel (train-range bars ONLY).

    Mirrors risk_model.FeaturePreprocessor's recipe and to_dict/from_dict schema so
    the stats serialize into the same .meta.json shape.
    """

    def __init__(self, feature_columns):
        self.feature_columns = list(feature_columns)
        self.median_ = None
        self.mean_ = None
        self.scale_ = None
        self.fit_train_end = None

    def fit(self, bar_mat: np.ndarray, *, dates: np.ndarray = None, train_end: str = None):
        """Fit stats on ``bar_mat`` (n_bars, n_features) float rows.

        When ``train_end`` is given, ``dates`` is required and only bars with
        date <= train_end contribute -- the guard that keeps validation-period
        values out of the standardization stats.
        """
        x = np.asarray(bar_mat, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"bar matrix shape {x.shape} does not match "
                f"{len(self.feature_columns)} feature columns")
        if train_end is not None:
            if dates is None:
                raise ValueError("dates is required when train_end is given")
            mask = np.asarray(dates, dtype=str) <= str(train_end)
            if not mask.any():
                raise ValueError(f"no bars on or before train_end={train_end!r} to fit on")
            x = x[mask]
            self.fit_train_end = str(train_end)
        x = np.where(np.isfinite(x), x, np.nan)
        with np.errstate(all="ignore"):
            median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(x), median, x)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
        self.median_ = median.astype(np.float64)
        self.mean_ = mean.astype(np.float64)
        self.scale_ = scale.astype(np.float64)
        return self

    def transform_matrix(self, bar_mat: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("sequence preprocessor is not fitted")
        x = np.asarray(bar_mat, dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        x = np.where(np.isnan(x), self.median_, x)
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def to_dict(self):
        cols = self.feature_columns
        as_map = lambda arr: (dict(zip(cols, [float(v) for v in arr])) if arr is not None else {})  # noqa: E731
        return {
            "feature_columns": cols,
            "median": as_map(self.median_),
            "mean": as_map(self.mean_),
            "scale": as_map(self.scale_),
            "fit_train_end": self.fit_train_end,
        }

    @classmethod
    def from_dict(cls, payload):
        pre = cls(payload.get("feature_columns", []))
        cols = pre.feature_columns

        def as_arr(key, default):
            m = payload.get(key, {})
            return np.asarray([float(m.get(c, default)) for c in cols], dtype=np.float64)

        if cols:
            pre.median_ = as_arr("median", 0.0)
            pre.mean_ = as_arr("mean", 0.0)
            pre.scale_ = as_arr("scale", 1.0)
        pre.fit_train_end = payload.get("fit_train_end")
        return pre


@dataclass
class BarIndex:
    """Sorted shared per-bar matrix + anchor keys for window gathering.

    ``bar_mat_raw`` rows are sorted by (symbol, date, decision_time); ``day_start[i]``
    is the position of the first bar of row i's (symbol, trade date) group, which is
    what clamps windows inside a single day.
    """

    bar_mat_raw: np.ndarray          # (n_bars, n_seq_features) float32
    bar_tod: np.ndarray              # (n_bars,) float32
    dates: np.ndarray                # (n_bars,) str "YYYY-MM-DD"
    keys: pd.DataFrame               # columns: symbol, decision_time, pos, day_start
    seq_config: SequenceConfig = field(default_factory=SequenceConfig)


def build_bar_index(features_1m: pd.DataFrame, *, symbols=None,
                    seq_config: SequenceConfig = SequenceConfig()) -> BarIndex:
    """Build the shared bar matrix from the features_1m table/parquet frame."""
    df = features_1m
    if symbols is not None:
        df = df[df["symbol"].isin(list(symbols))]
    missing = [c for c in seq_config.seq_feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"features_1m is missing sequence channels: {missing}")
    _validate_full_timestamps(df["decision_time"], where="features_1m")

    df = df.copy()
    df["_date"] = df["decision_time"].astype(str).str.slice(0, 10)
    df = df.sort_values(["symbol", "_date", "decision_time"], kind="stable").reset_index(drop=True)
    if bool(df.duplicated(["symbol", "decision_time"]).any()):
        raise ValueError("features_1m has duplicate (symbol, decision_time) bars")

    bar_mat = df[list(seq_config.seq_feature_columns)].to_numpy(dtype=np.float32)
    bar_tod = _tod_fraction(df["decision_time"])
    dates = df["_date"].to_numpy(dtype=str)

    pos = np.arange(len(df), dtype=np.int64)
    # first position of each (symbol, date) group, broadcast to every row of the group
    group_ids = (df["symbol"].astype(str) + "|" + df["_date"]).to_numpy()
    new_group = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        new_group[1:] = group_ids[1:] != group_ids[:-1]
    day_start = pos[new_group][np.cumsum(new_group) - 1] if len(df) else pos

    keys = pd.DataFrame({
        "symbol": df["symbol"].to_numpy(),
        "decision_time": df["decision_time"].astype(str).to_numpy(),
        "pos": pos,
        "day_start": day_start,
    })
    return BarIndex(bar_mat_raw=bar_mat, bar_tod=bar_tod, dates=dates,
                    keys=keys, seq_config=seq_config)


def attach_anchor_positions(model_frame: pd.DataFrame, bar_index: BarIndex, *,
                            min_match_rate: float = ANCHOR_MIN_MATCH_RATE):
    """Anchor each decision row to its confirmed feature bar.

    Join key = (symbol, decision_time) where decision_time is the confirmed bar's
    bar_end as a FULL timestamp (validated on both sides). Returns int64 arrays
    ``(j, day_start)`` aligned to ``model_frame`` rows; unmatched rows get j = -1
    (a fully-padded window -- rows are never dropped). Raises when the match rate
    falls below ``min_match_rate`` (expected: 1.0).
    """
    if len(model_frame) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    _validate_full_timestamps(model_frame["decision_time"], where="model_frame")

    left = pd.DataFrame({
        "symbol": model_frame["symbol"].to_numpy(),
        "decision_time": model_frame["decision_time"].astype(str).to_numpy(),
    })
    merged = left.merge(bar_index.keys, on=["symbol", "decision_time"],
                        how="left", validate="many_to_one")
    j = merged["pos"].fillna(-1).to_numpy(dtype=np.int64)
    day_start = merged["day_start"].fillna(0).to_numpy(dtype=np.int64)

    match_rate = float((j >= 0).mean())
    if match_rate < min_match_rate:
        raise ValueError(
            f"anchor match rate {match_rate:.4%} < {min_match_rate:.4%}: "
            f"{int((j < 0).sum())}/{len(j)} decision rows have no features_1m bar "
            "with bar_end == decision_time")
    return j, day_start


def gather_windows(bar_mat_std: np.ndarray, bar_tod: np.ndarray,
                   j: np.ndarray, day_start: np.ndarray, *,
                   seq_config: SequenceConfig = SequenceConfig()) -> np.ndarray:
    """Vectorized causal window gather -> (n_rows, seq_len, n_channels) float32.

    Window positions are ``idx = j - offsets`` with offsets in [seq_len-1 .. 0], so
    the NEWEST element (last) is the decision bar itself and a future index is
    unrepresentable. Positions before the day's first bar (or before position 0, or
    any position of an unanchored j = -1 row) are left-padded: zeros after
    standardization, mask channel = 0.
    """
    j = np.asarray(j, dtype=np.int64)
    day_start = np.asarray(day_start, dtype=np.int64)
    n = len(j)
    seq_len = int(seq_config.seq_len)
    offsets = np.arange(seq_len - 1, -1, -1, dtype=np.int64)      # oldest .. newest(=0)
    idx = j[:, None] - offsets[None, :]                            # (n, seq_len), idx <= j
    valid = (j[:, None] >= 0) & (idx >= 0) & (idx >= day_start[:, None])
    safe = np.where(valid, idx, 0)

    x = np.asarray(bar_mat_std, dtype=np.float32)[safe]            # (n, seq_len, F)
    x[~valid] = 0.0
    channels = [x]
    if seq_config.add_tod_channel:
        tod = np.asarray(bar_tod, dtype=np.float32)[safe]
        tod[~valid] = 0.0
        channels.append(tod[:, :, None])
    if seq_config.add_mask_channel:
        channels.append(valid.astype(np.float32)[:, :, None])
    out = np.concatenate(channels, axis=2) if len(channels) > 1 else channels[0]
    if n and out.shape != (n, seq_len, seq_config.n_channels):
        raise AssertionError(f"gathered window shape {out.shape} != "
                             f"{(n, seq_len, seq_config.n_channels)}")
    return out


def iter_batch_indices(n_rows: int, batch_size: int, *, rng: np.random.Generator = None):
    """Yield index arrays covering [0, n_rows); shuffled when ``rng`` is given."""
    order = rng.permutation(n_rows) if rng is not None else np.arange(n_rows)
    for start in range(0, n_rows, int(batch_size)):
        yield order[start:start + int(batch_size)]
