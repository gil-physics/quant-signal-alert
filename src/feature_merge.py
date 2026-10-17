"""Step 8: leakage-safe merge of 1m/3m/5m features onto the 1-minute decision grid.

For each (symbol, decision_time) row of features_1m, attach the MOST RECENT 3m and
5m feature whose bar_end <= decision_time (backward as-of join, per symbol, equality
allowed). A 3m/5m bar that has not finished by decision_time is never used, so no
not-yet-complete higher-timeframe feature can leak into the present.
"""

from __future__ import annotations


def merge_asof_features(f1, f3, f5):
    """Backward as-of merge of 3m/5m features onto the features_1m decision grid.

    f1: features_1m (has decision_time), f3/f5: features_3m/5m (have bar_end_3m/5m).
    Returns the merged DataFrame (no _dt helper column).
    """
    import pandas as pd

    if f1 is None or len(f1) == 0:
        return f1.copy() if f1 is not None else None

    left = f1.copy()
    left["_dt"] = pd.to_datetime(left["decision_time"])
    left["_date"] = pd.to_datetime(left["decision_time"]).dt.normalize()
    left = left.sort_values("_dt").reset_index(drop=True)

    def _prep_right(fr, end_col):
        # drop the constant provenance source (it would collide with the 1m source)
        r = fr.drop(columns=["source"], errors="ignore").copy()
        r["_dt"] = pd.to_datetime(r[end_col])
        r["_date"] = pd.to_datetime(r[end_col]).dt.normalize()
        return r.sort_values("_dt").reset_index(drop=True)

    def _asof(base, fr, end_col):
        # by=[symbol, date] so the as-of match never crosses a day boundary: at the
        # start of a day, decision times before that day's first 3m/5m feature get
        # NaN rather than borrowing the previous day's last (stale) feature.
        right = _prep_right(fr, end_col)
        if len(right) == 0:
            # no higher-timeframe bars at all -> attach the right columns as NaN
            for col in [c for c in right.columns if c not in ("symbol", "_dt", "_date")]:
                base[col] = pd.NA
            return base
        return pd.merge_asof(
            base, right, on="_dt", by=["symbol", "_date"],
            direction="backward",      # last right row with right_key <= left_key
            allow_exact_matches=True,  # bar_end == decision_time is allowed (<=)
        )

    merged = _asof(left, f3, "bar_end_3m")
    merged = _asof(merged, f5, "bar_end_5m")
    return merged.drop(columns=["_dt", "_date"]).reset_index(drop=True)


def leakage_violations(merged):
    """Count rows where an attached higher-timeframe bar_end is AFTER decision_time.

    Returns (violations_3m, violations_5m). Must both be 0 (NaN attachments — early
    in the day, before the first complete 3m/5m feature — are not violations).
    """
    import pandas as pd

    dt = pd.to_datetime(merged["decision_time"])
    out = []
    for col in ("bar_end_3m", "bar_end_5m"):
        if col not in merged.columns:
            out.append(0)
            continue
        end = pd.to_datetime(merged[col])
        out.append(int((end.notna() & (end > dt)).sum()))
    return tuple(out)
