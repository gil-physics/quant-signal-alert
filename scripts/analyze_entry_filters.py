from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

from daily_topk_backtest import (
    THRESHOLD_CONFIGS,
    apply_base_filters,
    apply_long_filter,
    join_signals_labels,
    quality_exclusion_pairs,
    select_daily_topk,
)
from settings import load_active_symbols, load_data_config


def _latest_quality_report():
    reports = sorted(
        (ROOT_DIR / "data" / "processed").glob("quality_report_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return reports[0] if reports else None


SYMBOLS = load_active_symbols()
QUALITY_REPORT = _latest_quality_report()


def _load_inputs():
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        rule_signals = pd.read_sql_query(
            "SELECT * FROM rule_signals WHERE source = ?", conn, params=(model_source,)
        )
        labels = pd.read_sql_query(
            "SELECT * FROM labels WHERE source = ?", conn, params=(model_source,)
        )
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume FROM bars_1m WHERE source = ?",
            conn,
            params=(model_source,),
        )
        extras = pd.read_sql_query(
            "SELECT symbol, decision_time, vwap_distance_1m, upper_shadow_1m "
            "FROM features_merged WHERE source = ?",
            conn,
            params=(model_source,),
        )
    finally:
        conn.close()

    rule_signals = rule_signals[rule_signals["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    labels = labels[labels["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    bars = bars[bars["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    extras = extras[extras["symbol"].isin(SYMBOLS)].reset_index(drop=True)

    joined = join_signals_labels(rule_signals, labels, symbols=SYMBOLS)
    joined = joined.merge(extras, on=["symbol", "decision_time"], how="left", validate="one_to_one")
    quality_report = pd.read_csv(QUALITY_REPORT)
    fail_pairs, outlier_pairs = quality_exclusion_pairs(quality_report)
    base, _ = apply_base_filters(joined, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)
    return base, bars


def _load_selected_inputs():
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        selected = pd.read_sql_query("SELECT * FROM backtest_signals", conn)
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume FROM bars_1m WHERE source = ?",
            conn,
            params=(model_source,),
        )
        extras = pd.read_sql_query(
            "SELECT symbol, decision_time, vwap_distance_1m, upper_shadow_1m "
            "FROM features_merged WHERE source = ?",
            conn,
            params=(model_source,),
        )
    finally:
        conn.close()
    selected = selected[selected["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    bars = bars[bars["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    extras = extras[extras["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    selected = selected.merge(extras, on=["symbol", "decision_time"], how="left", validate="many_to_one")
    selected["date"] = selected["decision_time"].str.slice(0, 10)
    selected["_decision_dt"] = pd.to_datetime(selected["decision_time"])
    return selected, bars


def _summarize(df):
    if df is None or len(df) == 0:
        return {
            "n": 0, "hit2": 0.0, "hit3": 0.0, "hit5": 0.0,
            "mfe": 0.0, "mae": 0.0, "stop": 0.0,
            "ret30": 0.0, "ret60": 0.0, "eod": 0.0,
        }
    return {
        "n": int(len(df)),
        "hit2": float(df["hit_up2"].mean()),
        "hit3": float(df["hit_up3"].mean()),
        "hit5": float(df["hit_up5"].mean()),
        "mfe": float(df["mfe"].mean()),
        "mae": float(df["mae"].mean()),
        "stop": float(df["stop_first"].mean()),
        "ret30": float(df["return_30m"].mean()),
        "ret60": float(df["return_60m"].mean()),
        "eod": float(df["return_eod"].mean()),
    }


def _fmt(row):
    return (
        f"n={row['n']:>4} hit2={row['hit2']*100:>5.2f}% hit3={row['hit3']*100:>5.2f}% "
        f"hit5={row['hit5']*100:>5.2f}% mfe={row['mfe']*100:>5.2f}% "
        f"mae={row['mae']*100:>6.2f}% stop={row['stop']*100:>5.2f}% "
        f"r30={row['ret30']*100:>6.2f}% r60={row['ret60']*100:>6.2f}% eod={row['eod']*100:>6.2f}%"
    )


def _eval_immediate(row, day_bars):
    future = day_bars[day_bars["_dt"] >= row["_decision_dt"]].copy()
    if future.empty:
        return None
    close_t = float(row["close"])
    out = row.to_dict()
    out["future_bar_count"] = int(len(future))
    out["mfe"] = float(future["high"].max() / close_t - 1)
    out["mae"] = float(future["low"].min() / close_t - 1)
    for label, minutes in {"30m": 30, "60m": 60, "120m": 120}.items():
        end = row["_decision_dt"] + pd.Timedelta(minutes=minutes)
        sub = future[future["_dt"] < end]
        out[f"return_{label}"] = float(sub.iloc[-1]["close"] / close_t - 1) if len(sub) else float("nan")
    out["return_eod"] = float(future.iloc[-1]["close"] / close_t - 1)
    out["hit_up2"] = int(out["mfe"] >= 0.02)
    out["hit_up3"] = int(out["mfe"] >= 0.03)
    out["hit_up5"] = int(out["mfe"] >= 0.05)
    target = close_t * 1.05
    stop = close_t * 0.985
    hit_target = future.index[future["high"] >= target]
    hit_stop = future.index[future["low"] <= stop]
    first_target = hit_target[0] if len(hit_target) else None
    first_stop = hit_stop[0] if len(hit_stop) else None
    out["stop_first"] = int(first_stop is not None and (first_target is None or first_stop <= first_target))
    return out


def _evaluate_selected_fast(selected, by_day):
    rows = []
    if selected is None or len(selected) == 0:
        return pd.DataFrame()
    for _, row in selected.iterrows():
        day = by_day.get((row["symbol"], row["date"]))
        if day is None:
            continue
        out = _eval_immediate(row, day)
        if out is not None:
            rows.append(out)
    return pd.DataFrame(rows)


def _select_with_filter(base, by_day, predicate, top_k=2):
    rows = []
    for cfg in THRESHOLD_CONFIGS:
        candidates, _ = apply_long_filter(base, cfg)
        candidates = candidates[predicate(candidates)].reset_index(drop=True)
        selected = _select_daily_topk_fast(candidates, top_k=top_k, cooldown_minutes=60)
        evaluated = _evaluate_selected_fast(selected, by_day)
        if len(evaluated):
            evaluated.insert(0, "threshold_name", cfg.name)
        rows.append(evaluated)
    return pd.concat(rows, ignore_index=True)


def _select_daily_topk_fast(candidates, *, top_k=2, cooldown_minutes=60):
    selected_rows = []
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame()
    sort_cols = ["score_margin", "up_score", "rank_ret_20m_1m", "volume_z_20m_1m", "_decision_dt"]
    ascending = [False, False, True, False, True]
    for _, day in candidates.groupby("date", sort=True):
        ordered = day.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        selected_for_day = []
        selected_by_symbol = {}
        for _, row in ordered.iterrows():
            if len(selected_for_day) >= top_k:
                break
            prior = selected_by_symbol.get(row["symbol"], [])
            if any(abs((row["_decision_dt"] - t).total_seconds()) < cooldown_minutes * 60 for t in prior):
                continue
            out = row.to_dict()
            out["cooldown_rank"] = len(selected_for_day) + 1
            out["daily_rank"] = len(selected_for_day) + 1
            selected_for_day.append(out)
            selected_by_symbol.setdefault(row["symbol"], []).append(row["_decision_dt"])
        selected_rows.extend(selected_for_day)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def overheat_scenarios(base, by_day):
    scenarios = [
        ("base", lambda d: pd.Series(True, index=d.index)),
        ("ret20<=1.5%", lambda d: d["ret_20m_1m"] <= 0.015),
        ("vwap<=1.5%", lambda d: d["vwap_distance_1m"] <= 0.015),
        ("volz<=3.0", lambda d: d["volume_z_20m_1m"] <= 3.0),
        ("upper<=0.5%", lambda d: d["upper_shadow_1m"] <= 0.005),
        (
            "combined_mild",
            lambda d: (
                (d["ret_20m_1m"] <= 0.020)
                & (d["vwap_distance_1m"] <= 0.015)
                & (d["volume_z_20m_1m"] <= 3.0)
                & (d["upper_shadow_1m"] <= 0.005)
            ),
        ),
    ]
    print("\n=== OVERHEAT FILTERS, RESELECT TOP2 ===")
    for name, predicate in scenarios:
        selected = _select_with_filter(base, by_day, predicate, top_k=2)
        for threshold, g in selected.groupby("threshold_name", sort=False):
            print(f"{name:<16} {threshold:<12} {_fmt(_summarize(g))}")


def overheat_scenarios_selected(selected):
    scenarios = [
        ("base", lambda d: pd.Series(True, index=d.index)),
        ("ret20<=1.5%", lambda d: d["ret_20m_1m"] <= 0.015),
        ("vwap<=1.5%", lambda d: d["vwap_distance_1m"] <= 0.015),
        ("volz<=3.0", lambda d: d["volume_z_20m_1m"] <= 3.0),
        ("upper<=0.5%", lambda d: d["upper_shadow_1m"] <= 0.005),
        (
            "combined_mild",
            lambda d: (
                (d["ret_20m_1m"] <= 0.020)
                & (d["vwap_distance_1m"] <= 0.015)
                & (d["volume_z_20m_1m"] <= 3.0)
                & (d["upper_shadow_1m"] <= 0.005)
            ),
        ),
    ]
    print("\n=== OVERHEAT FILTERS ON CURRENT TOP2 SIGNALS ===")
    for name, predicate in scenarios:
        filtered = selected[predicate(selected)].reset_index(drop=True)
        for threshold, g in filtered.groupby("threshold_name", sort=False):
            print(f"{name:<16} {threshold:<12} {_fmt(_summarize(g))}")


def _bars_by_day(bars):
    b = bars.copy()
    b["_dt"] = pd.to_datetime(b["bar_start"])
    b["_date"] = b["bar_start"].str.slice(0, 10)
    typical = (b["high"] + b["low"] + b["close"]) / 3
    b["_cum_pv"] = (typical * b["volume"]).groupby([b["symbol"], b["_date"]]).cumsum()
    b["_cum_v"] = b["volume"].groupby([b["symbol"], b["_date"]]).cumsum()
    b["_vwap"] = b["_cum_pv"] / b["_cum_v"].replace(0, pd.NA)
    return {
        (symbol, date): g.sort_values("_dt").reset_index(drop=True)
        for (symbol, date), g in b.groupby(["symbol", "_date"], sort=False)
    }


def _eval_from_entry(row, day_bars, entry_idx, entry_price):
    future = day_bars.iloc[entry_idx + 1:].copy()
    if future.empty:
        return None
    close_t = float(entry_price)
    out = row.to_dict()
    out["close"] = close_t
    out["future_bar_count"] = int(len(future))
    out["mfe"] = float(future["high"].max() / close_t - 1)
    out["mae"] = float(future["low"].min() / close_t - 1)
    for label, minutes in {"30m": 30, "60m": 60, "120m": 120}.items():
        end = day_bars.iloc[entry_idx]["_dt"] + pd.Timedelta(minutes=minutes)
        sub = future[future["_dt"] < end]
        out[f"return_{label}"] = float(sub.iloc[-1]["close"] / close_t - 1) if len(sub) else float("nan")
    out["return_eod"] = float(future.iloc[-1]["close"] / close_t - 1)
    out["hit_up2"] = int(out["mfe"] >= 0.02)
    out["hit_up3"] = int(out["mfe"] >= 0.03)
    out["hit_up5"] = int(out["mfe"] >= 0.05)
    target = close_t * 1.05
    stop = close_t * 0.985
    hit_target = future.index[future["high"] >= target]
    hit_stop = future.index[future["low"] <= stop]
    first_target = hit_target[0] if len(hit_target) else None
    first_stop = hit_stop[0] if len(hit_stop) else None
    out["stop_first"] = int(first_stop is not None and (first_target is None or first_stop <= first_target))
    return out


def _base_selected(base, by_day):
    selected = _select_with_filter(base, by_day, lambda d: pd.Series(True, index=d.index), top_k=2)
    selected["date"] = selected["decision_time"].str.slice(0, 10)
    selected["_decision_dt"] = pd.to_datetime(selected["decision_time"])
    return selected


def entry_delay_scenarios(selected, by_day):
    print("\n=== ENTRY DELAY, CURRENT TOP2 SIGNALS ===")
    for delay in [0, 1, 3, 5]:
        rows = []
        for _, row in selected.iterrows():
            day = by_day.get((row["symbol"], row["date"]))
            if day is None:
                continue
            if delay == 0:
                eval_row = row.to_dict()
                rows.append(eval_row)
                continue
            future = day[day["_dt"] >= row["_decision_dt"]].reset_index()
            if len(future) < delay:
                continue
            entry_orig_idx = int(future.iloc[delay - 1]["index"])
            entry_price = float(day.iloc[entry_orig_idx]["close"])
            eval_row = _eval_from_entry(row, day, entry_orig_idx, entry_price)
            if eval_row is not None:
                rows.append(eval_row)
        out = pd.DataFrame(rows)
        for threshold, g in out.groupby("threshold_name", sort=False):
            print(f"delay+{delay:<1}m {threshold:<12} {_fmt(_summarize(g))}")


def pullback_scenarios(selected, by_day):
    print("\n=== PULLBACK THEN REBREAK, CURRENT TOP2 SIGNALS ===")
    for pb in [0.002, 0.003, 0.005]:
        rows = []
        for _, row in selected.iterrows():
            day = by_day.get((row["symbol"], row["date"]))
            if day is None:
                continue
            signal_bar_start = row["_decision_dt"] - pd.Timedelta(minutes=1)
            signal_bar = day[day["_dt"] == signal_bar_start]
            signal_high = float(signal_bar.iloc[0]["high"]) if len(signal_bar) else float(row["close"])
            future = day[day["_dt"] >= row["_decision_dt"]].copy()
            first5 = future.head(5)
            if first5.empty:
                continue
            pull = first5[first5["low"] <= float(row["close"]) * (1 - pb)]
            if pull.empty:
                continue
            pull_idx = pull.index[0]
            after_pull = first5.loc[pull_idx:]
            rebreak = after_pull[(after_pull["close"] >= after_pull["_vwap"]) | (after_pull["high"] >= signal_high)]
            if rebreak.empty:
                continue
            entry_idx = int(rebreak.index[0])
            entry_price = float(day.loc[entry_idx, "close"])
            eval_row = _eval_from_entry(row, day, entry_idx, entry_price)
            if eval_row is not None:
                rows.append(eval_row)
        out = pd.DataFrame(rows)
        if out.empty:
            print(f"pullback {pb*100:.1f}% no trades")
            continue
        for threshold, g in out.groupby("threshold_name", sort=False):
            print(f"pullback {pb*100:.1f}% {threshold:<12} {_fmt(_summarize(g))}")


def main():
    selected, bars = _load_selected_inputs()
    by_day = _bars_by_day(bars)
    overheat_scenarios_selected(selected)
    entry_delay_scenarios(selected, by_day)
    pullback_scenarios(selected, by_day)


if __name__ == "__main__":
    main()
