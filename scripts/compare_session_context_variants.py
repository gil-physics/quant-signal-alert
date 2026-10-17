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


def _session_stats(g, prefix):
    if len(g) == 0:
        return {}
    g = g.sort_values("bar_start").copy()
    typical = (g["high"] + g["low"] + g["close"]) / 3
    volume = g["volume"].sum()
    vwap = float((typical * g["volume"]).sum() / volume) if volume else float("nan")
    first_open = float(g.iloc[0]["open"])
    last_close = float(g.iloc[-1]["close"])
    high = float(g["high"].max())
    low = float(g["low"].min())
    return {
        f"{prefix}_bar_count": int(len(g)),
        f"{prefix}_ret": last_close / first_open - 1 if first_open else float("nan"),
        f"{prefix}_range": high / low - 1 if low else float("nan"),
        f"{prefix}_volume": int(volume),
        f"{prefix}_vwap_distance": last_close / vwap - 1 if vwap == vwap and vwap else float("nan"),
    }


def build_session_context(raw_bars, regular_dates):
    raw = raw_bars.copy()
    raw["date"] = raw["bar_start"].str.slice(0, 10)
    raw["time"] = raw["bar_start"].str.slice(11, 19)

    pre_rows = []
    pre = raw[(raw["time"] >= "08:00:00") & (raw["time"] < "09:00:00")]
    for (symbol, date), g in pre.groupby(["symbol", "date"], sort=False):
        pre_rows.append({"symbol": symbol, "date": date, **_session_stats(g, "preopen")})
    pre_df = pd.DataFrame(pre_rows)

    ah_rows = []
    ah = raw[(raw["time"] >= "15:30:00") & (raw["time"] < "20:00:00")]
    for (symbol, date), g in ah.groupby(["symbol", "date"], sort=False):
        ah_rows.append({"symbol": symbol, "afterhours_date": date, **_session_stats(g, "prev_afterhours")})
    ah_df = pd.DataFrame(ah_rows)

    map_rows = []
    for symbol, dates in regular_dates.items():
        dates = sorted(dates)
        for i, date in enumerate(dates):
            map_rows.append({
                "symbol": symbol,
                "date": date,
                "afterhours_date": dates[i - 1] if i > 0 else None,
            })
    day_map = pd.DataFrame(map_rows)
    out = day_map.merge(pre_df, on=["symbol", "date"], how="left")
    out = out.merge(ah_df, on=["symbol", "afterhours_date"], how="left")

    out["preopen_ok"] = (out["preopen_ret"] >= 0) & (out["preopen_vwap_distance"] >= 0)
    out["prev_afterhours_ok"] = (
        (out["prev_afterhours_ret"] >= 0) & (out["prev_afterhours_vwap_distance"] >= 0)
    )
    return out


def _bars_by_day(bars):
    b = bars.copy()
    b["_dt"] = pd.to_datetime(b["bar_start"])
    b["_date"] = b["bar_start"].str.slice(0, 10)
    return {
        (symbol, date): g.sort_values("_dt").reset_index(drop=True)
        for (symbol, date), g in b.groupby(["symbol", "_date"], sort=False)
    }


def _eval_selected(selected, by_day):
    rows = []
    for _, row in selected.iterrows():
        day = by_day.get((row["symbol"], row["date"]))
        if day is None:
            continue
        future = day[day["_dt"] >= row["_decision_dt"]]
        if future.empty:
            continue
        close_t = float(row["close"])
        out = row.to_dict()
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
        rows.append(out)
    return pd.DataFrame(rows)


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
            out["daily_rank"] = len(selected_for_day) + 1
            selected_for_day.append(out)
            selected_by_symbol.setdefault(row["symbol"], []).append(row["_decision_dt"])
        selected_rows.extend(selected_for_day)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _summary(df):
    if len(df) == 0:
        return {
            "n": 0, "hit2": 0.0, "hit3": 0.0, "hit5": 0.0,
            "mfe": 0.0, "mae": 0.0, "stop": 0.0,
            "ret30": 0.0, "ret60": 0.0, "ret120": 0.0, "eod": 0.0,
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
        "ret120": float(df["return_120m"].mean()),
        "eod": float(df["return_eod"].mean()),
    }


def _fmt(s):
    return (
        f"n={s['n']:>4} hit2={s['hit2']*100:>5.2f}% hit3={s['hit3']*100:>5.2f}% "
        f"hit5={s['hit5']*100:>5.2f}% mfe={s['mfe']*100:>5.2f}% "
        f"mae={s['mae']*100:>6.2f}% stop={s['stop']*100:>5.2f}% "
        f"r30={s['ret30']*100:>6.2f}% r60={s['ret60']*100:>6.2f}% "
        f"r120={s['ret120']*100:>6.2f}% eod={s['eod']*100:>6.2f}%"
    )


def main():
    model_source = load_data_config().model_source
    conn = sqlite3.connect(ROOT_DIR / "db" / "market_data.sqlite")
    try:
        rule_signals = pd.read_sql_query("SELECT * FROM rule_signals WHERE source = ?", conn, params=(model_source,))
        labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
        regular_bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn,
            params=(model_source,),
        )
        raw_bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume FROM bars_1m WHERE source = 'al_raw'",
            conn,
        )
    finally:
        conn.close()

    rule_signals = rule_signals[rule_signals["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    labels = labels[labels["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    regular_bars = regular_bars[regular_bars["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    raw_bars = raw_bars[raw_bars["symbol"].isin(SYMBOLS)].reset_index(drop=True)

    joined = join_signals_labels(rule_signals, labels, symbols=SYMBOLS)
    quality_report = pd.read_csv(QUALITY_REPORT)
    fail_pairs, outlier_pairs = quality_exclusion_pairs(quality_report)
    base, base_stats = apply_base_filters(joined, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)
    regular_dates = {
        symbol: sorted(dates.tolist())
        for symbol, dates in base.groupby("symbol")["date"].unique().items()
    }
    session_context = build_session_context(raw_bars, regular_dates)
    base = base.merge(session_context, on=["symbol", "date"], how="left", validate="many_to_one")
    by_day = _bars_by_day(regular_bars)

    variants = [
        ("A_regular_only", lambda d: pd.Series(True, index=d.index)),
        ("B_preopen", lambda d: d["preopen_ok"].fillna(False)),
        ("C_prev_afterhours", lambda d: d["prev_afterhours_ok"].fillna(False)),
        ("D_preopen_afterhours", lambda d: d["preopen_ok"].fillna(False) & d["prev_afterhours_ok"].fillna(False)),
    ]

    print("session context availability:")
    print(
        session_context[["preopen_ok", "prev_afterhours_ok"]]
        .agg(["count", "sum"])
        .to_string()
    )

    print("\n=== SESSION CONTEXT VARIANTS, RESELECT DAILY TOP2 ===")
    rows = []
    for variant_name, predicate in variants:
        for cfg in THRESHOLD_CONFIGS:
            candidates, long_stats = apply_long_filter(base, cfg)
            candidate_mask = predicate(candidates)
            filtered = candidates[candidate_mask].reset_index(drop=True)
            selected = _select_daily_topk_fast(filtered, top_k=2, cooldown_minutes=60)
            evaluated = _eval_selected(selected, by_day)
            s = _summary(evaluated)
            rows.append({
                "variant": variant_name,
                "threshold": cfg.name,
                "after_score_filter": long_stats["after_score_filter"],
                "after_context_filter": int(len(filtered)),
                **s,
            })
            print(
                f"{variant_name:<22} {cfg.name:<12} score={long_stats['after_score_filter']:>6} "
                f"context={len(filtered):>6} {_fmt(s)}"
            )

    out = pd.DataFrame(rows)
    path = ROOT_DIR / "data" / "processed" / "session_context_variant_summary.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {path}")


if __name__ == "__main__":
    main()
