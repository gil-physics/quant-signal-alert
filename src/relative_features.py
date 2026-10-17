"""Step 9: cross-symbol (relative-strength) features.

At each decision_time, compare the active symbols against each other:
  rank_*            competition rank within the decision_time, rank 1 = highest value
  relative_*        value minus the universe (same-decision_time) mean
  universe_mean_ret_20m / universe_breadth / active_symbol_count

Cross-sectional only: every value used at a decision_time is a feature already
completed and merged for that same decision_time (Step 8), so no future data is used.
"""

from __future__ import annotations

# inputs that must exist on the merged frame (all are 1-minute features, non-NaN)
RANK_INPUTS = ["ret_5m_1m", "ret_20m_1m", "volume_z_20m_1m"]
# close is carried through for downstream labeling (Step 10 reads symbol/decision_time/close)
INPUT_COLUMNS = ["symbol", "decision_time", "close"] + RANK_INPUTS

# v1 cross-sectional columns (the original 8 the v1 risk model trains on).
V1_RELATIVE_COLUMNS = [
    "active_symbol_count",
    "rank_ret_5m_1m",
    "rank_ret_20m_1m",
    "rank_volume_z_20m_1m",
    "universe_mean_ret_20m",
    "relative_ret_20m_to_universe",
    "relative_volume_z_to_universe",
    "universe_breadth",
]
# v2.0 richer universe features (cross-sectional dispersion, multi-horizon breadth,
# distance-to-leader) -- contemporaneous reductions, same leakage class as v1.
V2_RELATIVE_COLUMNS = [
    "universe_disp_ret_20m",
    "universe_breadth_5m",
    "universe_breadth_60m",
    "ret_20m_gap_to_best",
    "universe_disp_volume_z",
]
OUTPUT_COLUMNS = INPUT_COLUMNS + V1_RELATIVE_COLUMNS + V2_RELATIVE_COLUMNS

# v2.1x causal lead-lag: a labeled leader's already-closed returns at decision_time D,
# broadcast onto every symbol's row at D. Generic, label-keyed names (no embedded code)
# -- the leaders' identities come from config (settings.leader_symbols), never
# hard-coded here. Produced ONLY when build_relative_features is called with leaders;
# appended after OUTPUT_COLUMNS so the v1/v2.0 serving + parity paths stay byte-identical.
def leader_columns(label):
    """The lead-lag column pair generated for one leader label."""
    return [f"leader_{label}_ret_5m_1m", f"leader_{label}_ret_20m_1m"]


def build_relative_features(merged, *, leaders=None):
    """Build the cross-symbol relative features from a features_merged-like frame.

    Returns one row per input row (symbol, decision_time), with ranks/relative/universe
    columns computed within each decision_time across the symbols present.

    ``leaders`` is an optional list of ``(label, code)`` pairs. For each leader present
    in the frame, two extra columns ``leader_<label>_ret_5m_1m`` /
    ``leader_<label>_ret_20m_1m`` are appended: that leader's return as of the SAME
    decision_time (already-closed, contemporaneous -> leakage-safe), broadcast onto
    every symbol's row at that decision_time. Omitted entirely when ``leaders`` is
    empty so the v1/v2.0 output schema is unchanged.
    """
    import pandas as pd

    if merged is None or len(merged) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = merged[INPUT_COLUMNS].copy()

    # Rank inputs are 1-minute features and must be complete (no NaN) — ranks are cast
    # to int. Fail loudly with the offending rows rather than an opaque cast error.
    nan_mask = df[RANK_INPUTS].isna().any(axis=1)
    if nan_mask.any():
        sample = df.loc[nan_mask, ["symbol", "decision_time"]].head(5).to_dict("records")
        raise ValueError(
            f"rank inputs contain NaN in {int(nan_mask.sum())} row(s); e.g. {sample}. "
            f"{RANK_INPUTS} must be complete 1-minute features."
        )

    g = df.groupby("decision_time", sort=False)
    out = df.copy()

    out["active_symbol_count"] = g["symbol"].transform("size").astype("int64")
    # rank 1 = highest value (strongest); ties share the lower rank (competition ranking)
    out["rank_ret_5m_1m"] = g["ret_5m_1m"].rank(ascending=False, method="min").astype("int64")
    out["rank_ret_20m_1m"] = g["ret_20m_1m"].rank(ascending=False, method="min").astype("int64")
    out["rank_volume_z_20m_1m"] = g["volume_z_20m_1m"].rank(ascending=False, method="min").astype("int64")

    out["universe_mean_ret_20m"] = g["ret_20m_1m"].transform("mean")
    out["relative_ret_20m_to_universe"] = df["ret_20m_1m"] - out["universe_mean_ret_20m"]
    out["relative_volume_z_to_universe"] = df["volume_z_20m_1m"] - g["volume_z_20m_1m"].transform("mean")

    up = (df["ret_20m_1m"] > 0).astype(float)
    out["universe_breadth"] = up.groupby(df["decision_time"]).transform("mean")

    # --- v2.0 richer universe features (contemporaneous, same leakage class) --------
    # dispersion = cross-sectional spread of returns at this decision_time (NaN when a
    # decision_time has a single symbol -> handled downstream by the median fill).
    out["universe_disp_ret_20m"] = g["ret_20m_1m"].transform("std")
    out["universe_disp_volume_z"] = g["volume_z_20m_1m"].transform("std")
    # multi-horizon breadth (5m available as a rank input; 60m is an OPTIONAL input read
    # from `merged` if present, NaN-skipped so warm-up bars don't bias the share down).
    up5 = (df["ret_5m_1m"] > 0).astype(float)
    out["universe_breadth_5m"] = up5.groupby(df["decision_time"]).transform("mean")
    if "ret_60m_1m" in merged.columns:
        r60 = merged.loc[df.index, "ret_60m_1m"]
        up60 = (r60 > 0).astype(float).where(r60.notna())
        out["universe_breadth_60m"] = up60.groupby(df["decision_time"]).transform("mean")
    else:
        out["universe_breadth_60m"] = float("nan")
    # distance to the universe leader (<= 0; 0 means this symbol IS the strongest).
    out["ret_20m_gap_to_best"] = df["ret_20m_1m"] - g["ret_20m_1m"].transform("max")

    result_columns = list(OUTPUT_COLUMNS)
    # --- v2.1x leader lead-lag (causal: leader's return AS OF this decision_time) ----
    for label, code in (leaders or []):
        c5, c20 = leader_columns(label)
        result_columns += [c5, c20]
        lead = df[df["symbol"] == code].drop_duplicates("decision_time")
        if lead.empty:
            out[c5] = float("nan")
            out[c20] = float("nan")
        else:
            lead = lead.set_index("decision_time")
            out[c5] = df["decision_time"].map(lead["ret_5m_1m"]).to_numpy()
            out[c20] = df["decision_time"].map(lead["ret_20m_1m"]).to_numpy()

    return out[result_columns].reset_index(drop=True)


def partial_decision_times(out, active_total: int):
    """decision_times whose symbol count is below the full active universe (Step 9 #8)."""
    counts = out.groupby("decision_time")["symbol"].size()
    return counts[counts < active_total]
