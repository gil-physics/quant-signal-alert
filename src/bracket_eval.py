"""Path-dependent bracket evaluation: target / stop / time-exit, first-touch.

Shared primitive for the exit sweep (scripts/bracket_exit_sweep.py) and the
risk-filter diagnostics (scripts/risk_filter_diagnostics.py).

Execution model (no look-ahead on the entry bar):
  * decision_dt is the signal bar's END, while arr["dt"] contains bar STARTs.
    Entry is the close known at decision_dt + delay_bars minutes. For delay 0
    use close_t; for delay N use the bar starting at decision_dt + N - 1.
  * Evaluate bars starting AT/AFTER the entry time, including the first future
    minute. Offsets are elapsed minutes at each evaluated bar's END.
  * First touch wins. A bar that touches BOTH the target and the stop counts as a
    stop (we cannot see intrabar order, so we assume the worst).
  * Target fill = +target_pct, stop fill = -stop_pct (no slippage/gap modelling);
    a time exit realises the close of the last bar within the time budget.
  * MFE/MAE are measured over the realised holding window (entry -> exit bar).
"""

from __future__ import annotations

TARGET_FIRST = "target"
STOP_FIRST = "stop"
TIME_EXIT = "time"

# outcome -> A/B/C label used by the risk diagnostics
OUTCOME_GROUP = {TARGET_FIRST: "A", STOP_FIRST: "B", TIME_EXIT: "C"}

SUMMARY_KEYS = (
    "n", "target_first_rate", "stop_first_rate", "time_exit_rate",
    "win_rate", "avg_win", "avg_loss", "realized_return", "expectancy_r",
    "mfe", "mae", "avg_hold_min",
)


def day_arrays(bars):
    """Map (symbol, date) -> dict of sorted numpy arrays {dt, high, low, close}.

    ``bars`` needs columns: symbol, bar_start, high, low, close.
    """
    import pandas as pd

    b = bars.copy()
    b["_dt"] = pd.to_datetime(b["bar_start"])
    b["_date"] = b["bar_start"].str.slice(0, 10)
    out = {}
    for (symbol, date), g in b.groupby(["symbol", "_date"], sort=False):
        g = g.sort_values("_dt")
        out[(symbol, date)] = {
            "dt": g["_dt"].to_numpy(),
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
        }
    return out


def prepare_entry(arr, decision_dt, close_t, *, delay_bars=0):
    """Return (entry_price, highs, lows, closes, offsets_min) for the post-entry
    window, or None when there is no entry bar or no bar left after it.

    ``arr`` is one value from :func:`day_arrays`. Missing minutes do not shorten
    the delay or holding period. A missing delayed-entry bar returns None.
    """
    import numpy as np

    if isinstance(delay_bars, bool) or int(delay_bars) != delay_bars or delay_bars < 0:
        raise ValueError("delay_bars must be a non-negative integer number of minutes")
    dt = arr["dt"]
    minute = np.timedelta64(1, "m")
    entry_dt = np.datetime64(decision_dt) + int(delay_bars) * minute
    entry_price = float(close_t)
    if delay_bars:
        entry_start = entry_dt - minute
        entry_i = int(np.searchsorted(dt, entry_start, side="left"))
        if entry_i >= len(dt) or dt[entry_i] != entry_start:
            return None
        entry_price = float(arr["close"][entry_i])
    post_lo = int(np.searchsorted(dt, entry_dt, side="left"))
    if post_lo >= len(dt):
        return None
    highs = arr["high"][post_lo:]
    lows = arr["low"][post_lo:]
    closes = arr["close"][post_lo:]
    offsets = ((dt[post_lo:] + minute - entry_dt) / minute).astype(float)
    return entry_price, highs, lows, closes, offsets


def simulate_bracket(entry_price, highs, lows, closes, offsets, *,
                     target_pct, stop_pct, time_exit_min):
    """First-touch bracket outcome within a time budget. Returns None if no
    eligible bar. ``time_exit_min`` may be ``float('inf')`` for an EOD exit."""
    import numpy as np

    if np.isfinite(time_exit_min):
        elig = offsets <= time_exit_min
        if not elig.any():
            return None
        h, l, c, off = highs[elig], lows[elig], closes[elig], offsets[elig]
    else:
        if len(closes) == 0:
            return None
        h, l, c, off = highs, lows, closes, offsets

    target_price = entry_price * (1 + target_pct)
    stop_price = entry_price * (1 - stop_pct)
    t_hits = np.nonzero(h >= target_price)[0]
    s_hits = np.nonzero(l <= stop_price)[0]
    t_idx = int(t_hits[0]) if t_hits.size else None
    s_idx = int(s_hits[0]) if s_hits.size else None

    if s_idx is not None and (t_idx is None or s_idx <= t_idx):
        outcome, exit_idx, realized = STOP_FIRST, s_idx, -stop_pct
    elif t_idx is not None:
        outcome, exit_idx, realized = TARGET_FIRST, t_idx, target_pct
    else:
        exit_idx = len(c) - 1
        outcome, realized = TIME_EXIT, float(c[exit_idx] / entry_price - 1)

    mfe = float(h[:exit_idx + 1].max() / entry_price - 1)
    mae = float(l[:exit_idx + 1].min() / entry_price - 1)
    return {
        "outcome": outcome,
        "realized_return": float(realized),
        "mfe": mfe,
        "mae": mae,
        "exit_offset_min": float(off[exit_idx]),
    }


def classify_outcome(entry_price, highs, lows, closes, offsets, *, target_pct, stop_pct):
    """A/B/C grouping by first-touch of (+target, -stop) held to EOD.

    A = reached target first, B = stopped first, C = neither (no event).
    Returns (group, bracket_result) or None.
    """
    res = simulate_bracket(entry_price, highs, lows, closes, offsets,
                           target_pct=target_pct, stop_pct=stop_pct,
                           time_exit_min=float("inf"))
    if res is None:
        return None
    return OUTCOME_GROUP[res["outcome"]], res


def summarize_bracket(results, *, stop_pct):
    """Aggregate a list of simulate_bracket() dicts into the sweep metrics."""
    import numpy as np

    n = len(results)
    if n == 0:
        return {k: 0.0 for k in SUMMARY_KEYS} | {"n": 0}

    realized = np.array([r["realized_return"] for r in results], dtype=float)
    outcomes = np.array([r["outcome"] for r in results])
    holds = np.array([r["exit_offset_min"] for r in results], dtype=float)
    mfe = np.array([r["mfe"] for r in results], dtype=float)
    mae = np.array([r["mae"] for r in results], dtype=float)
    wins = realized[realized > 0]
    losses = realized[realized < 0]

    return {
        "n": int(n),
        "target_first_rate": float(np.mean(outcomes == TARGET_FIRST)),
        "stop_first_rate": float(np.mean(outcomes == STOP_FIRST)),
        "time_exit_rate": float(np.mean(outcomes == TIME_EXIT)),
        "win_rate": float(np.mean(realized > 0)),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "realized_return": float(realized.mean()),
        "expectancy_r": float(realized.mean() / stop_pct) if stop_pct else float("nan"),
        "mfe": float(mfe.mean()),
        "mae": float(mae.mean()),
        "avg_hold_min": float(holds.mean()),
    }
