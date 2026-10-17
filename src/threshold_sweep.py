"""Step 13: threshold sweep for rule-based long signals.

Selection uses only rule-signal scores and current-time fields. Labels are joined
for reporting only, and price-path metrics are recomputed from bars so delayed
entries are evaluated at their actual entry price.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import pandas as pd

from daily_topk_backtest import (
    SELECTION_COLUMNS,
    SIGNAL_KEY_COLUMNS,
    apply_base_filters,
    select_daily_topk,
)
from entry_variants import bars_by_day, evaluate_variant


TRAIN_START = "2025-06-02"
TRAIN_END = "2026-02-28"
VALIDATION_START = "2026-03-01"
VALIDATION_END = "2026-06-19"

DEFAULT_UP_THRESHOLDS = (4.0, 5.0, 6.0)
DEFAULT_DOWN_RISK_LIMITS = (0.0, 1.0, 2.0)
DEFAULT_SCORE_MARGINS = (2.0, 3.0, 4.0, 5.0)
DEFAULT_TOP_KS = (1, 2)
DEFAULT_ENTRY_DELAYS = (0, 3)
DEFAULT_TOP_PERCENTILES = (0.10, 0.05, 0.02, 0.01)
DEFAULT_PERCENTILE_METRICS = ("up_score", "score_margin")

SWEEP_LABEL_COLUMNS = [
    "up2_120m",
    "up3_120m",
    "up2_180m",
    "up3_180m",
    "up5_180m",
    "up5_eod",
    "down2_120m",
    "down2_180m",
    "has_full_120m",
    "has_full_180m",
    "up5_event",
]


@dataclass(frozen=True)
class SweepSpec:
    threshold_name: str
    up_threshold: float
    down_risk_limit: float | None
    score_margin: float
    top_k: int
    entry_delay: int
    score_metric: str = "score_margin"
    top_percentile: float | None = None
    percentile_cutoff: float | None = None

    @property
    def is_percentile(self) -> bool:
        return self.top_percentile is not None

    @property
    def is_baseline(self) -> bool:
        return (
            self.up_threshold == 4.0
            and self.down_risk_limit == 1.0
            and self.score_margin == 3.0
            and self.entry_delay == 3
        )


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "none"
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def _fmt_pct(value: float) -> str:
    pct = value * 100
    return f"{int(pct)}p" if float(pct).is_integer() else str(pct).replace(".", "p") + "p"


def build_score_grid_specs(
    *,
    up_thresholds=DEFAULT_UP_THRESHOLDS,
    down_risk_limits=DEFAULT_DOWN_RISK_LIMITS,
    score_margins=DEFAULT_SCORE_MARGINS,
    top_ks=DEFAULT_TOP_KS,
    entry_delays=DEFAULT_ENTRY_DELAYS,
):
    specs = []
    for up, down, margin, top_k, delay in product(
        up_thresholds, down_risk_limits, score_margins, top_ks, entry_delays
    ):
        name = (
            f"score_u{_fmt_num(up)}_d{_fmt_num(down)}_m{_fmt_num(margin)}"
            f"_top{top_k}_delay{delay}m"
        )
        specs.append(
            SweepSpec(
                threshold_name=name,
                up_threshold=float(up),
                down_risk_limit=float(down) if down is not None else None,
                score_margin=float(margin),
                top_k=int(top_k),
                entry_delay=int(delay),
            )
        )
    return specs


def build_percentile_specs(
    *,
    top_percentiles=DEFAULT_TOP_PERCENTILES,
    metrics=DEFAULT_PERCENTILE_METRICS,
    top_ks=DEFAULT_TOP_KS,
    entry_delays=DEFAULT_ENTRY_DELAYS,
    up_threshold=4.0,
    down_risk_limit=1.0,
    score_margin=3.0,
):
    specs = []
    for metric, pct, top_k, delay in product(metrics, top_percentiles, top_ks, entry_delays):
        if metric not in ("up_score", "score_margin"):
            raise ValueError(f"unsupported percentile metric: {metric}")
        name = (
            f"pct_{metric}_top{_fmt_pct(float(pct))}"
            f"_top{top_k}_delay{delay}m"
        )
        specs.append(
            SweepSpec(
                threshold_name=name,
                up_threshold=float(up_threshold),
                down_risk_limit=float(down_risk_limit) if down_risk_limit is not None else None,
                score_margin=float(score_margin),
                top_k=int(top_k),
                entry_delay=int(delay),
                score_metric=metric,
                top_percentile=float(pct),
            )
        )
    return specs


def build_default_specs(*, include_score_grid=True, include_percentiles=True):
    specs = []
    if include_score_grid:
        specs.extend(build_score_grid_specs())
    if include_percentiles:
        specs.extend(build_percentile_specs())
    return specs


def join_signals_labels_for_sweep(rule_signals, labels, symbols=None):
    missing_signal = [c for c in SELECTION_COLUMNS if c not in rule_signals.columns]
    missing_label = [c for c in SIGNAL_KEY_COLUMNS + SWEEP_LABEL_COLUMNS if c not in labels.columns]
    if missing_signal:
        raise ValueError(f"rule_signals missing required columns: {missing_signal}")
    if missing_label:
        raise ValueError(f"labels missing required columns: {missing_label}")

    sig = rule_signals[SELECTION_COLUMNS].copy()
    lab = labels[SIGNAL_KEY_COLUMNS + SWEEP_LABEL_COLUMNS].copy()
    if symbols is not None:
        sig = sig[sig["symbol"].isin(symbols)]
        lab = lab[lab["symbol"].isin(symbols)]

    joined = sig.merge(lab, on=SIGNAL_KEY_COLUMNS, how="inner", validate="one_to_one")
    joined["date"] = joined["decision_time"].str.slice(0, 10)
    joined["time"] = joined["decision_time"].str.slice(11, 19)
    joined["score_margin"] = joined["up_score"] - joined["down_score"]
    joined["_decision_dt"] = pd.to_datetime(joined["decision_time"])
    return joined.reset_index(drop=True)


def prepare_base_frame(
    rule_signals,
    labels,
    *,
    symbols=None,
    min_time="09:10:00",
    max_time="15:00:00",
    fail_quality_pairs=None,
    outlier_pairs=None,
):
    joined = join_signals_labels_for_sweep(rule_signals, labels, symbols=symbols)
    return apply_base_filters(
        joined,
        min_time=min_time,
        max_time=max_time,
        fail_quality_pairs=fail_quality_pairs,
        outlier_pairs=outlier_pairs,
    )


def date_slice(df, start_date, end_date):
    if df is None or len(df) == 0:
        return df.copy()
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def _score_pool_mask(base, spec: SweepSpec):
    both = (base["is_up_candidate"] == 1) & (base["is_down_candidate"] == 1)
    up_not_both = (base["is_up_candidate"] == 1) & ~both
    mask = up_not_both & (base["up_score"] >= spec.up_threshold)
    if spec.down_risk_limit is not None:
        mask = mask & (base["down_score"] <= spec.down_risk_limit)
    mask = mask & (base["score_margin"] >= spec.score_margin)
    return both, up_not_both, mask


def build_candidate_pool(base, spec: SweepSpec):
    if base is None or len(base) == 0:
        return base.copy(), {
            "both_excluded_count": 0,
            "after_both_exclusion": 0,
            "after_score_filter": 0,
            "after_percentile_filter": 0,
            "long_candidate_count": 0,
        }

    both, up_not_both, score_mask = _score_pool_mask(base, spec)
    final_mask = score_mask.copy()

    if spec.top_percentile is not None:
        cutoff = spec.percentile_cutoff
        if cutoff is None or pd.isna(cutoff):
            final_mask = final_mask & False
        else:
            final_mask = final_mask & (base[spec.score_metric] >= cutoff)

    stats = {
        "both_excluded_count": int(both.sum()),
        "after_both_exclusion": int(up_not_both.sum()),
        "after_score_filter": int(score_mask.sum()),
        "after_percentile_filter": int(final_mask.sum()),
        "long_candidate_count": int(final_mask.sum()),
    }
    return base.loc[final_mask].reset_index(drop=True), stats


def fit_percentile_cutoffs(train_base, specs):
    fitted = []
    for spec in specs:
        if not spec.is_percentile:
            fitted.append(spec)
            continue
        score_only = replace(spec, top_percentile=None, percentile_cutoff=None)
        pool, _ = build_candidate_pool(train_base, score_only)
        if pool.empty:
            cutoff = float("nan")
        else:
            cutoff = float(pool[spec.score_metric].quantile(1 - float(spec.top_percentile)))
        fitted.append(replace(spec, percentile_cutoff=cutoff))
    return fitted


def _rate(df, col, eligible_col=None):
    if df is None or len(df) == 0 or col not in df.columns:
        return 0.0
    work = df
    if eligible_col and eligible_col in work.columns:
        work = work[work[eligible_col] == 1]
    if len(work) == 0:
        return 0.0
    return float(work[col].mean())


def _mean(df, col):
    if df is None or len(df) == 0 or col not in df.columns:
        return 0.0
    value = df[col].mean()
    return float(value) if pd.notna(value) else 0.0


def _count_eligible(df, col):
    if df is None or len(df) == 0 or col not in df.columns:
        return 0
    return int((df[col] == 1).sum())


def summarize_evaluated(
    *,
    spec: SweepSpec,
    split_name: str,
    split_base,
    selected,
    evaluated,
    candidate_daily,
    base_stats,
    candidate_stats,
):
    trading_days = int(split_base["date"].nunique()) if split_base is not None and len(split_base) else 0
    signal_count = int(len(evaluated)) if evaluated is not None else 0
    selected_count = int(len(selected)) if selected is not None else 0
    symbol_concentration = 0.0
    if signal_count:
        symbol_concentration = float(evaluated["symbol"].value_counts().max() / signal_count)

    after_cooldown = 0
    if candidate_daily is not None and len(candidate_daily) and "after_cooldown_count" in candidate_daily:
        after_cooldown = int(candidate_daily["after_cooldown_count"].sum())

    period_start = ""
    period_end = ""
    if split_base is not None and len(split_base):
        period_start = str(split_base["date"].min())
        period_end = str(split_base["date"].max())

    return {
        "threshold_name": spec.threshold_name,
        "train_or_validation": split_name,
        "period_start": period_start,
        "period_end": period_end,
        "trading_days": trading_days,
        "score_metric": spec.score_metric,
        "top_percentile": spec.top_percentile if spec.top_percentile is not None else "",
        "percentile_cutoff": round(float(spec.percentile_cutoff), 6)
        if spec.percentile_cutoff is not None and pd.notna(spec.percentile_cutoff)
        else "",
        "up_threshold": spec.up_threshold,
        "down_risk_limit": "" if spec.down_risk_limit is None else spec.down_risk_limit,
        "score_margin": spec.score_margin,
        "top_k": spec.top_k,
        "entry_delay": spec.entry_delay,
        "is_baseline": int(spec.is_baseline),
        "raw_rows": int(base_stats.get("raw_rows", 0)),
        "after_time_filter": int(base_stats.get("after_time_filter", 0)),
        "after_quality_filter": int(base_stats.get("after_quality_filter", 0)),
        "after_both_exclusion": int(candidate_stats.get("after_both_exclusion", 0)),
        "after_score_filter": int(candidate_stats.get("after_score_filter", 0)),
        "after_percentile_filter": int(candidate_stats.get("after_percentile_filter", 0)),
        "after_cooldown": after_cooldown,
        "selected_count": selected_count,
        "dropped_no_entry_count": selected_count - signal_count,
        "signal_count": signal_count,
        "avg_signals_per_day": round(signal_count / trading_days, 6) if trading_days else 0.0,
        "hit_up2_rate": round(_rate(evaluated, "hit_up2"), 6),
        "hit_up3_rate": round(_rate(evaluated, "hit_up3"), 6),
        "hit_up5_rate": round(_rate(evaluated, "hit_up5"), 6),
        "up2_120m_rate": round(_rate(evaluated, "up2_120m", "has_full_120m"), 6),
        "up3_120m_rate": round(_rate(evaluated, "up3_120m", "has_full_120m"), 6),
        "up2_180m_rate": round(_rate(evaluated, "up2_180m", "has_full_180m"), 6),
        "up3_180m_rate": round(_rate(evaluated, "up3_180m", "has_full_180m"), 6),
        "up5_180m_rate": round(_rate(evaluated, "up5_180m", "has_full_180m"), 6),
        "up5_eod_rate": round(_rate(evaluated, "up5_eod"), 6),
        "down2_120m_rate": round(_rate(evaluated, "down2_120m", "has_full_120m"), 6),
        "down2_180m_rate": round(_rate(evaluated, "down2_180m", "has_full_180m"), 6),
        "eligible_120m_count": _count_eligible(evaluated, "has_full_120m"),
        "eligible_180m_count": _count_eligible(evaluated, "has_full_180m"),
        "average_MFE": round(_mean(evaluated, "mfe"), 6),
        "average_MAE": round(_mean(evaluated, "mae"), 6),
        "stop_first_rate": round(_rate(evaluated, "stop_first"), 6),
        "return_30m": round(_mean(evaluated, "return_30m"), 6),
        "return_60m": round(_mean(evaluated, "return_60m"), 6),
        "return_120m": round(_mean(evaluated, "return_120m"), 6),
        "return_eod": round(_mean(evaluated, "return_eod"), 6),
        "symbol_concentration": round(symbol_concentration, 6),
    }


def run_threshold_sweep(
    base,
    bars,
    specs,
    *,
    base_stats=None,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    validation_start=VALIDATION_START,
    validation_end=VALIDATION_END,
    cooldown_minutes=60,
):
    base_stats = base_stats or {}
    train_base = date_slice(base, train_start, train_end)
    validation_base = date_slice(base, validation_start, validation_end)
    fitted_specs = fit_percentile_cutoffs(train_base, specs)
    by_day = bars_by_day(bars)

    rows = []
    evaluated_frames = []
    for spec in fitted_specs:
        for split_name, split_base in (
            ("train", train_base),
            ("validation", validation_base),
        ):
            candidates, candidate_stats = build_candidate_pool(split_base, spec)
            selected, candidate_daily = select_daily_topk(
                candidates,
                top_k=spec.top_k,
                cooldown_minutes=cooldown_minutes,
            )
            evaluated = evaluate_variant(selected, by_day, delay_bars=spec.entry_delay)
            if evaluated is None:
                evaluated = pd.DataFrame()
            if len(evaluated):
                evaluated.insert(0, "threshold_name", spec.threshold_name)
                evaluated.insert(1, "train_or_validation", split_name)
            evaluated_frames.append(evaluated)
            rows.append(
                summarize_evaluated(
                    spec=spec,
                    split_name=split_name,
                    split_base=split_base,
                    selected=selected,
                    evaluated=evaluated,
                    candidate_daily=candidate_daily,
                    base_stats=base_stats,
                    candidate_stats=candidate_stats,
                )
            )

    evaluated_all = pd.concat(evaluated_frames, ignore_index=True) if evaluated_frames else pd.DataFrame()
    summary = pd.DataFrame(rows)
    return summary, evaluated_all


def stable_validation_view(summary, *, min_validation_signals=10):
    if summary is None or len(summary) == 0:
        return pd.DataFrame()
    train = summary[summary["train_or_validation"] == "train"].copy()
    validation = summary[summary["train_or_validation"] == "validation"].copy()
    if train.empty or validation.empty:
        return pd.DataFrame()

    key = ["threshold_name"]
    keep_cols = [
        "threshold_name",
        "signal_count",
        "avg_signals_per_day",
        "hit_up3_rate",
        "hit_up5_rate",
        "up3_120m_rate",
        "up3_180m_rate",
        "average_MFE",
        "average_MAE",
        "stop_first_rate",
        "return_30m",
        "return_60m",
        "symbol_concentration",
    ]
    merged = validation[keep_cols].merge(
        train[keep_cols],
        on=key,
        how="inner",
        suffixes=("_validation", "_train"),
    )
    if merged.empty:
        return merged
    merged = merged[merged["signal_count_validation"] >= min_validation_signals].copy()
    if merged.empty:
        return merged
    merged["up3_rate_gap"] = (
        merged["hit_up3_rate_validation"] - merged["hit_up3_rate_train"]
    )
    merged["stop_gap"] = (
        merged["stop_first_rate_validation"] - merged["stop_first_rate_train"]
    )
    return merged.sort_values(
        [
            "hit_up3_rate_validation",
            "up3_120m_rate_validation",
            "stop_first_rate_validation",
            "return_60m_validation",
            "symbol_concentration_validation",
            "signal_count_validation",
        ],
        ascending=[False, False, True, False, True, False],
    ).reset_index(drop=True)
