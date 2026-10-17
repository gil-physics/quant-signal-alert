"""Causal (online) single-trade-per-day selection.

The Step 12 daily top-k sorts a whole day's candidates by score and keeps the best
-- that needs to see the entire day, so it is NOT tradeable live. The online rule
instead commits to the FIRST candidate that qualifies each day (earliest
decision_time), using only information available at that moment.
"""

from __future__ import annotations


def select_online_top1(candidates):
    """Earliest qualifying candidate per day (1 trade/day, fully causal).

    ``candidates`` must already be gated (the apply_long_filter output) and carry
    ``date`` and ``decision_time``. Returns one row per day, marked daily_rank=1.
    """
    import pandas as pd

    if candidates is None or len(candidates) == 0:
        cols = list(candidates.columns) + ["daily_rank"] if candidates is not None else ["daily_rank"]
        return pd.DataFrame(columns=cols)

    ordered = candidates.sort_values(["date", "decision_time"]).reset_index(drop=True)
    picks = ordered.groupby("date", as_index=False, sort=True).head(1).copy()
    picks["daily_rank"] = 1
    return picks.reset_index(drop=True)
