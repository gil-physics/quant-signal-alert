"""Step 12: daily top1/top2 backtest for rule-signal candidates.

Signal selection uses only ``rule_signals`` columns. Labels are merged in the
same frame for evaluation, but the filter/sort/cooldown path must not read them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    name: str
    up_threshold: float
    down_risk_limit: float | None
    score_margin: float


THRESHOLD_CONFIGS = [
    ThresholdConfig("conservative", up_threshold=4.0, down_risk_limit=1.0, score_margin=3.0),
    ThresholdConfig("medium", up_threshold=4.0, down_risk_limit=None, score_margin=2.0),
    ThresholdConfig("aggressive", up_threshold=3.0, down_risk_limit=None, score_margin=1.0),
]

SIGNAL_KEY_COLUMNS = ["symbol", "decision_time"]
SELECTION_COLUMNS = [
    "symbol",
    "source",
    "decision_time",
    "close",
    "ret_5m_1m",
    "ret_20m_1m",
    "volume_z_20m_1m",
    "rank_ret_20m_1m",
    "up_score",
    "down_score",
    "is_up_candidate",
    "is_down_candidate",
    "reason_1",
    "reason_2",
    "reason_3",
    "up_reasons",
    "down_reasons",
]
LABEL_COLUMNS = [
    "up2_eod",
    "up3_eod",
    "up5_eod",
    "down2_eod",
    "down3_eod",
    "down5_eod",
    "up5_event",
]
RETURN_HORIZONS = {"30m": 30, "60m": 60, "120m": 120}


def threshold_config_by_name(name: str) -> ThresholdConfig:
    for cfg in THRESHOLD_CONFIGS:
        if cfg.name == name:
            return cfg
    raise ValueError(f"unknown threshold config: {name}")


def join_signals_labels(rule_signals, labels, symbols=None):
    """Inner join rule_signals and labels by (symbol, decision_time).

    The returned frame keeps signal ``close``/``source`` and appends only the
    label columns needed for evaluation.
    """
    import pandas as pd

    missing_signal = [c for c in SELECTION_COLUMNS if c not in rule_signals.columns]
    missing_label = [c for c in SIGNAL_KEY_COLUMNS + LABEL_COLUMNS if c not in labels.columns]
    if missing_signal:
        raise ValueError(f"rule_signals missing required columns: {missing_signal}")
    if missing_label:
        raise ValueError(f"labels missing required columns: {missing_label}")

    sig = rule_signals[SELECTION_COLUMNS].copy()
    lab = labels[SIGNAL_KEY_COLUMNS + LABEL_COLUMNS].copy()
    if symbols is not None:
        sig = sig[sig["symbol"].isin(symbols)]
        lab = lab[lab["symbol"].isin(symbols)]

    joined = sig.merge(lab, on=SIGNAL_KEY_COLUMNS, how="inner", validate="one_to_one")
    joined["date"] = joined["decision_time"].str.slice(0, 10)
    joined["time"] = joined["decision_time"].str.slice(11, 19)
    joined["score_margin"] = joined["up_score"] - joined["down_score"]
    joined["_decision_dt"] = pd.to_datetime(joined["decision_time"])
    return joined.reset_index(drop=True)


def quality_exclusion_pairs(quality_report, *, exclude_outlier_dates=True):
    """Return (fail_pairs, outlier_pairs) from a Step 6 quality report frame."""
    if quality_report is None or len(quality_report) == 0:
        return set(), set()

    fail = quality_report[quality_report["status"] == "FAIL"]
    fail_pairs = set(zip(fail["symbol"].astype(str), fail["date"].astype(str)))

    outlier_pairs = set()
    if exclude_outlier_dates:
        work = quality_report.copy()
        for col in ("spike", "range_outlier"):
            if col in work.columns:
                work[col] = work[col].astype(int)
            else:
                work[col] = 0
        out = work[(work["spike"] > 0) | (work["range_outlier"] > 0)]
        outlier_pairs = set(zip(out["symbol"].astype(str), out["date"].astype(str)))
    return fail_pairs, outlier_pairs


def apply_base_filters(
    joined,
    *,
    min_time="09:10:00",
    max_time="15:00:00",
    fail_quality_pairs=None,
    outlier_pairs=None,
):
    """Apply Step 12 basic filters except cooldown/top-k."""
    import pandas as pd

    fail_quality_pairs = fail_quality_pairs or set()
    outlier_pairs = outlier_pairs or set()
    df = joined.copy()
    pair_index = list(zip(df["symbol"], df["date"]))

    before_open = df["time"] < min_time
    after_cutoff = df["time"] >= max_time
    fail_quality = pd.Series([pair in fail_quality_pairs for pair in pair_index], index=df.index)
    outlier = pd.Series([pair in outlier_pairs for pair in pair_index], index=df.index)

    time_mask = ~(before_open | after_cutoff)
    quality_mask = ~(fail_quality | outlier)
    mask = time_mask & quality_mask
    stats = {
        "raw_rows": int(len(df)),
        "before_min_time_excluded": int(before_open.sum()),
        "after_max_time_excluded": int(after_cutoff.sum()),
        "after_time_filter": int(time_mask.sum()),
        "fail_quality_excluded": int(sum(fail_quality)),
        "outlier_excluded": int(sum(outlier)),
        "after_quality_filter": int(mask.sum()),
        "base_rows": int(mask.sum()),
    }
    return df.loc[mask].reset_index(drop=True), stats


def apply_long_filter(base, cfg: ThresholdConfig):
    """Conservative/medium/aggressive long candidate filter.

    Labels are deliberately not referenced here.
    """
    both = (base["is_up_candidate"] == 1) & (base["is_down_candidate"] == 1)
    up_not_both = (base["is_up_candidate"] == 1) & ~both
    mask = up_not_both & (base["up_score"] >= cfg.up_threshold)
    if cfg.down_risk_limit is not None:
        mask = mask & (base["down_score"] <= cfg.down_risk_limit)
    mask = mask & ((base["up_score"] - base["down_score"]) >= cfg.score_margin)

    stats = {
        "both_excluded_count": int(both.sum()),
        "after_both_exclusion": int(up_not_both.sum()),
        "after_score_filter": int(mask.sum()),
        "long_candidate_count": int(mask.sum()),
    }
    return base.loc[mask].reset_index(drop=True), stats


def select_daily_topk(candidates, *, top_k=2, cooldown_minutes=60):
    """Select daily top-k candidates with same-symbol cooldown."""
    import pandas as pd

    selected_rows = []
    daily_rows = []
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame(columns=list(candidates.columns) + ["daily_rank"] if candidates is not None else ["daily_rank"]), pd.DataFrame()

    sort_cols = ["score_margin", "up_score", "rank_ret_20m_1m", "volume_z_20m_1m", "_decision_dt"]
    ascending = [False, False, True, False, True]  # rank 1 is strongest, so ascending.

    for date, day in candidates.groupby("date", sort=True):
        ordered = day.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        cooldown_kept = []
        selected_by_symbol = {}
        cooldown_excluded = 0
        for _, row in ordered.iterrows():
            prior = selected_by_symbol.get(row["symbol"], [])
            if any(abs((row["_decision_dt"] - t).total_seconds()) < cooldown_minutes * 60 for t in prior):
                cooldown_excluded += 1
                continue
            out = row.to_dict()
            out["cooldown_rank"] = len(cooldown_kept) + 1
            cooldown_kept.append(out)
            selected_by_symbol.setdefault(row["symbol"], []).append(row["_decision_dt"])

        selected_for_day = []
        for out in cooldown_kept[:top_k]:
            out = dict(out)
            out["daily_rank"] = len(selected_for_day) + 1
            selected_for_day.append(out)

        selected_rows.extend(selected_for_day)
        daily_rows.append({
            "date": date,
            "candidate_count": int(len(ordered)),
            "cooldown_excluded_count": int(cooldown_excluded),
            "after_cooldown_count": int(len(cooldown_kept)),
            "selected_count": int(len(selected_for_day)),
        })

    selected = pd.DataFrame(selected_rows)
    daily = pd.DataFrame(daily_rows)
    return selected.reset_index(drop=True), daily.reset_index(drop=True)


def _prepare_bars(bars):
    b = bars.copy()
    b["_bar_start_dt"] = b["bar_start_dt"] if "bar_start_dt" in b.columns else None
    if b["_bar_start_dt"].isna().all():
        import pandas as pd

        b["_bar_start_dt"] = pd.to_datetime(b["bar_start"])
    b["_date"] = b["bar_start"].str.slice(0, 10)
    return {
        (symbol, date): g.sort_values("_bar_start_dt").reset_index(drop=True)
        for (symbol, date), g in b.groupby(["symbol", "_date"], sort=False)
    }


def _last_close_return(future, decision_dt, close_t, minutes):
    end = decision_dt + __import__("pandas").Timedelta(minutes=minutes)
    sub = future[future["_bar_start_dt"] < end]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["close"] / close_t - 1)


def evaluate_selected(selected, bars):
    """Add label hits plus MFE/MAE and horizon returns for selected signals."""
    import pandas as pd

    if selected is None or len(selected) == 0:
        return selected.copy() if selected is not None else pd.DataFrame()

    bars_by_day = _prepare_bars(bars)
    rows = []
    for _, row in selected.iterrows():
        out = row.to_dict()
        close_t = float(row["close"])
        day_bars = bars_by_day.get((row["symbol"], row["date"]))
        if day_bars is None:
            future = day_bars
        else:
            future = day_bars[day_bars["_bar_start_dt"] >= row["_decision_dt"]]

        if future is None or future.empty:
            out["future_bar_count"] = 0
            out["mfe"] = float("nan")
            out["mae"] = float("nan")
            for hz in RETURN_HORIZONS:
                out[f"return_{hz}"] = float("nan")
            out["return_eod"] = float("nan")
        else:
            out["future_bar_count"] = int(len(future))
            out["mfe"] = float(future["high"].max() / close_t - 1)
            out["mae"] = float(future["low"].min() / close_t - 1)
            for hz, minutes in RETURN_HORIZONS.items():
                out[f"return_{hz}"] = _last_close_return(future, row["_decision_dt"], close_t, minutes)
            out["return_eod"] = float(future.iloc[-1]["close"] / close_t - 1)

        out["hit_up2"] = int(row["up2_eod"])
        out["hit_up3"] = int(row["up3_eod"])
        out["hit_up5"] = int(row["up5_eod"])
        out["stop_first"] = int(row["up5_event"] == "UP_FAIL")
        rows.append(out)

    result = pd.DataFrame(rows)
    drop_cols = [c for c in result.columns if c.startswith("_")]
    return result.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)


def build_daily_summary(all_dates, selected, candidate_daily, threshold_name, filter_stats):
    import pandas as pd

    base = pd.DataFrame({"date": sorted(all_dates)})
    if candidate_daily is None or candidate_daily.empty:
        candidate_daily = pd.DataFrame(columns=[
            "date", "candidate_count", "cooldown_excluded_count",
            "after_cooldown_count", "selected_count",
        ])
    daily = base.merge(candidate_daily, on="date", how="left")
    for col in ("candidate_count", "cooldown_excluded_count", "after_cooldown_count", "selected_count"):
        daily[col] = daily[col].fillna(0).astype("int64")

    if selected is not None and len(selected):
        agg = selected.groupby("date").agg(
            hit_up2_count=("hit_up2", "sum"),
            hit_up3_count=("hit_up3", "sum"),
            hit_up5_count=("hit_up5", "sum"),
            stop_first_count=("stop_first", "sum"),
            average_mfe=("mfe", "mean"),
            average_mae=("mae", "mean"),
            return_30m=("return_30m", "mean"),
            return_60m=("return_60m", "mean"),
            return_120m=("return_120m", "mean"),
            return_eod=("return_eod", "mean"),
        ).reset_index()
        symbols = selected.groupby("date")["symbol"].apply(lambda s: ",".join(s.astype(str))).reset_index(name="selected_symbols")
        agg = agg.merge(symbols, on="date", how="left")
        daily = daily.merge(agg, on="date", how="left")

    fill_zero = ["hit_up2_count", "hit_up3_count", "hit_up5_count", "stop_first_count"]
    for col in fill_zero:
        if col not in daily:
            daily[col] = 0
        daily[col] = daily[col].fillna(0).astype("int64")
    for col in ["average_mfe", "average_mae", "return_30m", "return_60m", "return_120m", "return_eod", "selected_symbols"]:
        if col not in daily:
            daily[col] = "" if col == "selected_symbols" else float("nan")

    daily["threshold_name"] = threshold_name
    daily["both_excluded_count"] = int(filter_stats.get("both_excluded_count", 0))
    return daily[["threshold_name"] + [c for c in daily.columns if c != "threshold_name"]]


def build_threshold_summary(threshold_name, daily, selected, filter_stats):
    import pandas as pd

    signal_count = int(len(selected)) if selected is not None else 0
    trading_days = int(len(daily))
    symbol_concentration = 0.0
    if signal_count:
        symbol_concentration = float(selected["symbol"].value_counts().max() / signal_count)

    def _rate(col):
        return float(selected[col].mean()) if signal_count else 0.0

    def _mean(col):
        return float(selected[col].mean()) if signal_count else float("nan")

    def _rank_rate(rank, col):
        if not signal_count:
            return 0.0
        sub = selected[selected["daily_rank"] == rank]
        return float(sub[col].mean()) if len(sub) else 0.0

    def _rank_mean(rank, col):
        if not signal_count:
            return 0.0
        sub = selected[selected["daily_rank"] == rank]
        return float(sub[col].mean()) if len(sub) else 0.0

    return {
        "threshold_name": threshold_name,
        "period_start": str(daily["date"].min()) if trading_days else "",
        "period_end": str(daily["date"].max()) if trading_days else "",
        "trading_days": trading_days,
        "raw_rows": int(filter_stats.get("raw_rows", 0)),
        "after_time_filter": int(filter_stats.get("after_time_filter", 0)),
        "after_quality_filter": int(filter_stats.get("after_quality_filter", 0)),
        "after_both_exclusion": int(filter_stats.get("after_both_exclusion", 0)),
        "after_score_filter": int(filter_stats.get("after_score_filter", 0)),
        "after_cooldown": int(daily["after_cooldown_count"].sum()) if "after_cooldown_count" in daily else 0,
        "final_topk": signal_count,
        "signal_count": signal_count,
        "daily_top1_count": int((selected["daily_rank"] == 1).sum()) if signal_count else 0,
        "daily_top2_count": int((selected["daily_rank"] == 2).sum()) if signal_count else 0,
        "avg_signals_per_day": round(signal_count / trading_days, 6) if trading_days else 0.0,
        "hit_up2_rate": round(_rate("hit_up2"), 6),
        "hit_up3_rate": round(_rate("hit_up3"), 6),
        "hit_up5_rate": round(_rate("hit_up5"), 6),
        "average_mfe": round(_mean("mfe"), 6) if signal_count else 0.0,
        "average_mae": round(_mean("mae"), 6) if signal_count else 0.0,
        "stop_first_rate": round(_rate("stop_first"), 6),
        "return_30m": round(_mean("return_30m"), 6) if signal_count else 0.0,
        "return_60m": round(_mean("return_60m"), 6) if signal_count else 0.0,
        "return_120m": round(_mean("return_120m"), 6) if signal_count else 0.0,
        "return_eod": round(_mean("return_eod"), 6) if signal_count else 0.0,
        "top1_hit_up2_rate": round(_rank_rate(1, "hit_up2"), 6),
        "top1_hit_up3_rate": round(_rank_rate(1, "hit_up3"), 6),
        "top1_hit_up5_rate": round(_rank_rate(1, "hit_up5"), 6),
        "top1_average_mfe": round(_rank_mean(1, "mfe"), 6),
        "top1_average_mae": round(_rank_mean(1, "mae"), 6),
        "top1_stop_first_rate": round(_rank_rate(1, "stop_first"), 6),
        "top1_return_eod": round(_rank_mean(1, "return_eod"), 6),
        "top2_hit_up2_rate": round(_rank_rate(2, "hit_up2"), 6),
        "top2_hit_up3_rate": round(_rank_rate(2, "hit_up3"), 6),
        "top2_hit_up5_rate": round(_rank_rate(2, "hit_up5"), 6),
        "top2_average_mfe": round(_rank_mean(2, "mfe"), 6),
        "top2_average_mae": round(_rank_mean(2, "mae"), 6),
        "top2_stop_first_rate": round(_rank_rate(2, "stop_first"), 6),
        "top2_return_eod": round(_rank_mean(2, "return_eod"), 6),
        "symbol_concentration": round(symbol_concentration, 6),
        "both_excluded_count": int(filter_stats.get("both_excluded_count", 0)),
        "base_rows": int(filter_stats.get("base_rows", 0)),
        "long_candidate_count": int(filter_stats.get("long_candidate_count", 0)),
        "fail_quality_excluded": int(filter_stats.get("fail_quality_excluded", 0)),
        "outlier_excluded": int(filter_stats.get("outlier_excluded", 0)),
    }


def run_backtests(
    joined,
    bars,
    *,
    threshold_configs=None,
    top_k=2,
    cooldown_minutes=60,
    min_time="09:10:00",
    max_time="15:00:00",
    fail_quality_pairs=None,
    outlier_pairs=None,
):
    """Run all threshold configs and return (signals, daily_summary, threshold_summary)."""
    import pandas as pd

    threshold_configs = threshold_configs or THRESHOLD_CONFIGS
    base, base_stats = apply_base_filters(
        joined,
        min_time=min_time,
        max_time=max_time,
        fail_quality_pairs=fail_quality_pairs,
        outlier_pairs=outlier_pairs,
    )
    all_dates = sorted(base["date"].unique().tolist())
    signals_all = []
    daily_all = []
    threshold_rows = []

    for cfg in threshold_configs:
        candidates, long_stats = apply_long_filter(base, cfg)
        selected, candidate_daily = select_daily_topk(candidates, top_k=top_k, cooldown_minutes=cooldown_minutes)
        evaluated = evaluate_selected(selected, bars)
        if len(evaluated):
            evaluated.insert(0, "threshold_name", cfg.name)
        else:
            evaluated = pd.DataFrame(columns=["threshold_name"])
        stats = {**base_stats, **long_stats}
        daily = build_daily_summary(all_dates, evaluated, candidate_daily, cfg.name, stats)
        summary = build_threshold_summary(cfg.name, daily, evaluated, stats)

        signals_all.append(evaluated)
        daily_all.append(daily)
        threshold_rows.append(summary)

    signals = pd.concat(signals_all, ignore_index=True) if signals_all else pd.DataFrame()
    daily_summary = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()
    threshold_summary = pd.DataFrame(threshold_rows)
    return signals, daily_summary, threshold_summary
