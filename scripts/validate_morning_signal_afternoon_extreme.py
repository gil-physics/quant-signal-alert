"""Validate whether a morning 🔔 signal (before 11:00) for 000660 predicts a useful
afternoon (13:00-15:30) high/low relative to the alert price.

RESEARCH / VALIDATION ONLY — not a trading script. It places NO orders and never calls
any order API. It reads 1-minute bars (bars_1m, source 'al_regular') and an alert feed,
selects one morning signal per day (first / last / strongest / risk-based / all), and
measures the afternoon extreme vs the alert price. Afternoon high/low is HINDSIGHT — this
measures predictive POTENTIAL, not a tradable exit. Pure metric logic lives in
src/morning_signal_validation.py (unit-tested).

Alert feeds (--alerts-source, default auto):
  * realtime_replay      the realtime engine's fired 🔔 replayed over history
                         (build_realtime_replay_alerts.py) — 09:10-capable because it seeds
                         prior-day history, so it reflects the true FIRST morning 🔔. PREFERRED.
  * directional_alerts   offline feed; starts at 10:01 each day (within-day 60-minute feature
                         warmup) so it MISSES the 09:00–10:00 window — do not use for a
                         "first morning 🔔" question.
  * realtime_alert_log   the sparse live log (only the few days actually run live).

Run (cwd = quant, 64-bit):
    ..\\Python311-64\\python.exe scripts\\validate_morning_signal_afternoon_extreme.py --code 000660 --selection last
"""

from __future__ import annotations

import argparse
import random
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

import numpy as np
import pandas as pd

from config import DB_PATH, PROCESSED_DIR, ensure_directories
from morning_signal_validation import (
    DAILY_COLUMNS, SELECTION_MODES, aggregate, classify_direction, compute_windows,
    daily_comparison, decide_verdict, risk_band, select_signals, trend_state_from_alert,
)
from settings import load_data_config, normalize_symbol_code, realtime_default_symbol

RANDOM_SEED = 20260628          # fixed -> the random-morning-bar baseline is reproducible
SAMPLE_PER_DAY = 5              # all_morning_bars_sample: deterministic bars/day (bounded)

TRADE_COLUMNS = [
    "trade_date", "symbol", "selection", "signal_time", "signal_direction", "alert_type",
    "alert_grade", "trend_state", "risk_prob", "risk_bucket", "alert_price",
    "price_13_00", "signal_to_13_return",
    "afternoon_high", "afternoon_high_time", "afternoon_high_return",
    "afternoon_low", "afternoon_low_time", "afternoon_low_return", "short_potential",
    "afternoon_close", "afternoon_close_time", "afternoon_close_return",
    "pre_13_high", "pre_13_low", "pre_13_mfe", "pre_13_mae",
    "full_day_mfe_after_signal", "full_day_mae_after_signal",
    "bars_before_13", "bars_in_afternoon", "notes",
]
BASELINE_COLUMNS = [
    "baseline", "trade_date", "symbol", "signal_time", "signal_direction", "alert_price",
    "price_13_00", "signal_to_13_return", "afternoon_high", "afternoon_high_return",
    "afternoon_low", "afternoon_low_return", "short_potential",
    "afternoon_close", "afternoon_close_return", "pre_13_mfe", "pre_13_mae",
    "full_day_mfe_after_signal", "full_day_mae_after_signal", "bars_in_afternoon", "notes",
]
GROUP_KEYS = ["selection", "signal_direction", "alert_type", "alert_grade", "risk_bucket"]


def _norm_tod(text):
    parts = str(text).strip().split(":")
    if len(parts) == 2:
        parts.append("00")
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _table_exists(conn, name):
    return bool(conn.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') "
                             "AND name=?", (name,)).fetchone()[0])


def _date_clause(start, end, col):
    clause, params = "", []
    if start:
        clause += f" AND substr({col},1,10) >= ?"; params.append(start)
    if end:
        clause += f" AND substr({col},1,10) <= ?"; params.append(end)
    return clause, params


def resolve_alerts_source(conn, requested, symbol, source, start, end):
    """auto -> prefer the realtime-replay feed (09:10-capable, the alerts a user actually
    sees) when it exists; else whichever of the offline directional_alerts / sparse live
    realtime_alert_log has more distinct alert-days. The offline directional_alerts feed
    only starts at 10:01 (within-day 60m feature warmup), so it MISSES the 09:00-10:00 🔔."""
    if requested != "auto":
        return requested
    if _table_exists(conn, "realtime_replay_alerts"):
        rc, rp = _date_clause(start, end, "decision_time")
        n_rep = conn.execute(
            f"SELECT count(DISTINCT substr(decision_time,1,10)) FROM realtime_replay_alerts "
            f"WHERE symbol=? AND alert_type!=''{rc}", (symbol, *rp)).fetchone()[0]
        if n_rep:
            return "realtime_replay"
    have_dir = _table_exists(conn, "directional_alerts")
    have_rt = _table_exists(conn, "realtime_alert_log")
    if not have_rt:
        return "directional_alerts"
    if not have_dir:
        return "realtime_alert_log"
    dc, dp = _date_clause(start, end, "decision_time")
    n_dir = conn.execute(
        f"SELECT count(DISTINCT substr(decision_time,1,10)) FROM directional_alerts "
        f"WHERE symbol=? AND source=? AND alert_type!=''{dc}", (symbol, source, *dp)).fetchone()[0]
    rc, rp = _date_clause(start, end, "ts")
    n_rt = conn.execute(
        f"SELECT count(DISTINCT substr(ts,1,10)) FROM realtime_alert_log WHERE symbol=?{rc}",
        (symbol, *rp)).fetchone()[0]
    return "directional_alerts" if n_dir >= n_rt else "realtime_alert_log"


def load_alerts(conn, alerts_source, symbol, source, start, end, *, risk_as_short=False):
    """Unified alert frame: symbol, decision_time, date, tod, alert_price, alert_type,
    alert_grade, alert_score, direction_score, risk_prob (NaN if unavailable)."""
    if alerts_source == "realtime_replay":
        # the realtime engine's fired 🔔 replayed over history (build_realtime_replay_alerts.py):
        # 09:10-capable (prior-day-seeded features), the alerts a user actually sees — unlike the
        # offline directional_alerts feed, which starts at 10:01 (within-day 60m warmup).
        rc, rp = _date_clause(start, end, "decision_time")
        df = pd.read_sql_query(
            f"SELECT decision_time, symbol, alert_type, alert_grade, alert_score, "
            f"direction_score, risk_prob, close AS alert_price FROM realtime_replay_alerts "
            f"WHERE symbol=? AND alert_type!=''{rc} ORDER BY decision_time",
            conn, params=(symbol, *rp))
    elif alerts_source == "realtime_alert_log":
        rc, rp = _date_clause(start, end, "ts")
        df = pd.read_sql_query(
            f"SELECT ts AS decision_time, symbol, alert_type, alert_grade, risk_prob, "
            f"close AS alert_price FROM realtime_alert_log WHERE symbol=?{rc} ORDER BY ts",
            conn, params=(symbol, *rp))
        df["alert_score"] = np.nan
        df["direction_score"] = np.nan
    else:
        dc, dp = _date_clause(start, end, "decision_time")
        df = pd.read_sql_query(
            f"SELECT decision_time, symbol, alert_type, alert_grade, alert_score, "
            f"direction_score, close AS alert_price FROM directional_alerts "
            f"WHERE symbol=? AND source=? AND alert_type!=''{dc} ORDER BY decision_time",
            conn, params=(symbol, source, *dp))
        df["risk_prob"] = _risk_prob_map(conn, symbol, source).reindex(df["decision_time"]).to_numpy() \
            if len(df) else np.nan
    if df.empty:
        return df
    df["date"] = df["decision_time"].str.slice(0, 10)
    df["tod"] = df["decision_time"].str.slice(11, 19)
    df["alert_price"] = pd.to_numeric(df["alert_price"], errors="coerce")
    df["risk_prob"] = pd.to_numeric(df["risk_prob"], errors="coerce")
    df["alert_score"] = pd.to_numeric(df["alert_score"], errors="coerce")
    df["signal_direction"] = df["alert_type"].map(
        lambda t: classify_direction(t, risk_as_down=risk_as_short))
    df["trend_state"] = df["alert_type"].map(trend_state_from_alert)
    df["risk_bucket"] = df["risk_prob"].map(risk_band)
    return df


def _risk_prob_map(conn, symbol, source):
    if not _table_exists(conn, "risk_model_predictions"):
        return pd.Series(dtype=float)
    rp = pd.read_sql_query(
        "SELECT decision_time, risk_prob FROM risk_model_predictions WHERE symbol=? AND source=?",
        conn, params=(symbol, source))
    return rp.drop_duplicates("decision_time").set_index("decision_time")["risk_prob"]


def load_bars_by_day(conn, symbol, source, start, end):
    dc, dp = _date_clause(start, end, "bar_end")
    bars = pd.read_sql_query(
        f"SELECT bar_end, high, low, close FROM bars_1m WHERE symbol=? AND source=?{dc} "
        f"ORDER BY bar_end", conn, params=(symbol, source, *dp))
    if bars.empty:
        return {}
    for c in ("high", "low", "close"):
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    bars["tod"] = bars["bar_end"].str.slice(11, 19)
    bars["date"] = bars["bar_end"].str.slice(0, 10)
    return {d: g.reset_index(drop=True) for d, g in bars.groupby("date", sort=True)}


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
def build_trade_row(sig, day_bars, selection, *, afternoon_start, afternoon_end):
    alert_price = sig["alert_price"]
    if (alert_price is None or alert_price != alert_price) and day_bars is not None:
        at = day_bars[day_bars["bar_end"] == sig["decision_time"]]
        if not at.empty:
            alert_price = float(at["close"].iloc[0])
    w = compute_windows(day_bars, alert_price=alert_price, signal_tod=sig["tod"],
                        direction=sig["signal_direction"], afternoon_start=afternoon_start,
                        afternoon_end=afternoon_end)
    row = {
        "trade_date": sig["date"], "symbol": sig["symbol"], "selection": selection,
        "signal_time": sig["decision_time"], "signal_direction": sig["signal_direction"],
        "alert_type": sig["alert_type"], "alert_grade": sig["alert_grade"],
        "trend_state": sig.get("trend_state"), "risk_prob": sig.get("risk_prob"),
        "risk_bucket": sig.get("risk_bucket"),
    }
    notes = "|".join(w.pop("notes"))
    row.update(w)
    row["notes"] = notes
    return {c: row.get(c) for c in TRADE_COLUMNS}


def build_baseline_rows(bars_by_day, *, symbol, afternoon_start, afternoon_end, cutoff_tod):
    """Reference morning prices (no alert): last-before-11, first-after-10:30, a fixed
    random morning bar, and a small deterministic morning sample — same afternoon
    metrics, so the report can ask 'does the 🔔 beat any morning price?'."""
    rng = random.Random(RANDOM_SEED)
    rows = []

    def add(baseline, date, bar):
        if bar is None:
            rows.append({**{c: None for c in BASELINE_COLUMNS},
                         "baseline": baseline, "trade_date": date, "symbol": symbol,
                         "notes": "no_morning_bar"})
            return
        w = compute_windows(bars_by_day[date], alert_price=float(bar["close"]),
                            signal_tod=bar["tod"], direction="UP",
                            afternoon_start=afternoon_start, afternoon_end=afternoon_end)
        rows.append({
            "baseline": baseline, "trade_date": date, "symbol": symbol,
            "signal_time": bar["bar_end"], "signal_direction": "UP",
            "alert_price": float(bar["close"]), "price_13_00": w["price_13_00"],
            "signal_to_13_return": w["signal_to_13_return"], "afternoon_high": w["afternoon_high"],
            "afternoon_high_return": w["afternoon_high_return"], "afternoon_low": w["afternoon_low"],
            "afternoon_low_return": w["afternoon_low_return"], "short_potential": w["short_potential"],
            "afternoon_close": w["afternoon_close"], "afternoon_close_return": w["afternoon_close_return"],
            "pre_13_mfe": w["pre_13_mfe"], "pre_13_mae": w["pre_13_mae"],
            "full_day_mfe_after_signal": w["full_day_mfe_after_signal"],
            "full_day_mae_after_signal": w["full_day_mae_after_signal"],
            "bars_in_afternoon": w["bars_in_afternoon"], "notes": "|".join(w["notes"]),
        })

    for date in sorted(bars_by_day):
        day = bars_by_day[date]
        morning = day[day["tod"] < cutoff_tod]
        after_1030 = day[day["tod"] >= "10:30:00"]
        add("last_bar_before_11", date, morning.iloc[-1] if not morning.empty else None)
        add("first_bar_after_10_30", date, after_1030.iloc[0] if not after_1030.empty else None)
        if not morning.empty:
            add("random_morning_bar", date, morning.iloc[rng.randrange(len(morning))])
            idx = np.linspace(0, len(morning) - 1, min(SAMPLE_PER_DAY, len(morning))).round().astype(int)
            for i in sorted(set(idx.tolist())):
                add("all_morning_bars_sample", date, morning.iloc[i])
        else:
            add("random_morning_bar", date, None)
    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


# --------------------------------------------------------------------------- #
# Summary + report
# --------------------------------------------------------------------------- #
def summarize(trades: pd.DataFrame, selection: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in trades.groupby(GROUP_KEYS, dropna=False, sort=True):
        rows.append({**dict(zip(GROUP_KEYS, keys)), **aggregate(g)})
    return pd.DataFrame(rows)


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x*100:+.2f}%" if pct else f"{x:.3f}"


def write_report(path, *, symbol, start, end, n_days, valid_days, selection, alerts_source,
                 up_stats, down_stats, base_stats_by_name, verdict, reasons,
                 down_verdict, down_reasons, risk_as_short, missing_notes):
    base_last = base_stats_by_name.get("last_bar_before_11", {})
    up_beats = (up_stats.get("avg_afternoon_high_return", float("nan"))
                > base_last.get("avg_afternoon_high_return", float("nan")))
    lines = [
        f"# Morning 🔔 signal → afternoon extreme validation ({symbol})",
        "",
        f"- **Date range:** {start or '(min)'} → {end or '(max)'}",
        f"- **Trading days (with bars):** {n_days}",
        f"- **Days with a valid morning signal:** {valid_days}",
        f"- **Selection mode:** `{selection}`   **Alerts source:** `{alerts_source}`",
        f"- **Afternoon window:** 13:00–15:30   **Signal cutoff:** before 11:00",
        "",
        "Afternoon high/low is hindsight — this measures predictive POTENTIAL, not a tradable exit.",
        "",
        "## Main UP result",
        f"- n = {up_stats.get('n', 0)} signal-days",
        f"- avg afternoon_high_return = {_fmt(up_stats.get('avg_afternoon_high_return'))} "
        f"(median {_fmt(up_stats.get('median_afternoon_high_return'))})",
        f"- hit ≥0.5% / 1.0% / 1.5% / 2.0% = {_fmt(up_stats.get('hit_up_0_5'), False)} / "
        f"{_fmt(up_stats.get('hit_up_1_0'), False)} / {_fmt(up_stats.get('hit_up_1_5'), False)} / "
        f"{_fmt(up_stats.get('hit_up_2_0'), False)}",
        f"- avg afternoon_close_return = {_fmt(up_stats.get('avg_afternoon_close_return'))}",
        f"- avg pre-13 MFE = {_fmt(up_stats.get('avg_pre_13_mfe'))}, "
        f"avg pre-13 MAE = {_fmt(up_stats.get('avg_pre_13_mae'))} (adverse move before 13:00)",
        f"- avg full-day MFE after signal = {_fmt(up_stats.get('avg_full_day_mfe_after_signal'))}, "
        f"MAE = {_fmt(up_stats.get('avg_full_day_mae_after_signal'))}",
        "",
        "## Main DOWN result"
        + ("  _(RISK_ALERT treated as SHORT — `--risk-as-short`)_" if risk_as_short else ""),
    ]
    base_short = base_stats_by_name.get("last_bar_before_11", {}).get("avg_short_potential", float("nan"))
    if down_stats.get("n", 0):
        lines += [
            f"- n = {down_stats['n']} signal-days",
            f"- avg short_potential = {_fmt(down_stats.get('avg_short_potential'))} "
            f"(median {_fmt(down_stats.get('median_short_potential'))}); "
            f"baseline last_bar_before_11 = {_fmt(base_short)}",
            f"- hit ≥0.5%/1.0%/1.5%/2.0% = {_fmt(down_stats.get('hit_down_0_5'), False)} / "
            f"{_fmt(down_stats.get('hit_down_1_0'), False)} / {_fmt(down_stats.get('hit_down_1_5'), False)} / "
            f"{_fmt(down_stats.get('hit_down_2_0'), False)}",
            f"- avg pre-13 adverse (price rose vs short) = {_fmt(down_stats.get('avg_pre_13_mae'))}",
        ]
    elif risk_as_short:
        lines.append("- No RISK_ALERT short signals selected in the morning window.")
    else:
        lines.append("- No DOWN/SHORT alerts in this feed yet (use --risk-as-short to test RISK_ALERT as short).")
    lines += [
        "",
        "## Baseline comparison (avg afternoon_high_return)",
        "| source | n | avg high-return | median | avg pre-13 MAE |",
        "|---|---|---|---|---|",
        f"| **🔔 UP signal** | {up_stats.get('n',0)} | {_fmt(up_stats.get('avg_afternoon_high_return'))} | "
        f"{_fmt(up_stats.get('median_afternoon_high_return'))} | {_fmt(up_stats.get('avg_pre_13_mae'))} |",
    ]
    for name in ("last_bar_before_11", "first_bar_after_10_30", "random_morning_bar", "all_morning_bars_sample"):
        s = base_stats_by_name.get(name, {})
        lines.append(f"| {name} | {s.get('n',0)} | {_fmt(s.get('avg_afternoon_high_return'))} | "
                     f"{_fmt(s.get('median_afternoon_high_return'))} | {_fmt(s.get('avg_pre_13_mae'))} |")
    lines += [
        "",
        f"- **UP signals beat baseline (last_bar_before_11) on afternoon_high_return?** "
        f"{'YES' if up_beats else 'NO'}",
        f"- **DOWN signals beat baseline on short_potential?** "
        f"{'n/a (no DOWN alerts)' if not down_stats.get('n', 0) else 'see CSV'}",
        f"- **Pre-13 adverse move tolerable?** avg pre-13 MAE = {_fmt(up_stats.get('avg_pre_13_mae'))}",
        f"- **Rows with missing-data notes:** {missing_notes}",
        "",
        f"## Verdict (UP / long): **{verdict}**",
        "",
    ] + [f"- {r}" for r in reasons]
    if down_stats.get("n", 0):
        lines += [
            "",
            f"## Verdict (DOWN / short): **{down_verdict}**",
            "",
        ] + [f"- {r}" for r in down_reasons]
    lines += [
        "",
        "_PASS = beats baseline on mean+median favorable return, enough days, pre-13 adverse "
        "tolerable. PARTIAL = potential exists but one criterion fails. FAIL = does not beat baseline._",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    default_source = load_data_config().model_source
    p = argparse.ArgumentParser(description="Validate morning 🔔 signals vs afternoon extremes (research only).")
    p.add_argument("--code", default=None, help="symbol (default: realtime default 000660).")
    p.add_argument("--source", default=default_source, help="bars_1m source (default al_regular).")
    p.add_argument("--signal-cutoff", default="11:00", help="only signals before this time (HH:MM).")
    p.add_argument("--afternoon-start", default="13:00")
    p.add_argument("--afternoon-end", default="15:30")
    p.add_argument("--selection", choices=list(SELECTION_MODES), default="last",
                   help="per-day morning signal pick (default last).")
    p.add_argument("--start-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--end-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--alerts-source",
                   choices=["realtime_replay", "realtime_alert_log", "directional_alerts", "auto"],
                   default="auto",
                   help="alert feed. realtime_replay = the engine's fired 🔔 replayed over history "
                        "(09:10-capable, faithful to what you see); directional_alerts = offline "
                        "feed (starts 10:01 due to within-day 60m warmup); auto prefers realtime_replay.")
    p.add_argument("--risk-as-short", action="store_true",
                   help="treat RISK_ALERT (하방 위험) as a SHORT/DOWN signal so the short side "
                        "is validated too (no real DOWN/SHORT alerts exist yet).")
    p.add_argument("--output-dir", default=str(PROCESSED_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    symbol = normalize_symbol_code(args.code or realtime_default_symbol())
    cutoff = _norm_tod(args.signal_cutoff)
    aft_start, aft_end = _norm_tod(args.afternoon_start), _norm_tod(args.afternoon_end)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        alerts_source = resolve_alerts_source(conn, args.alerts_source, symbol, args.source,
                                              args.start_date, args.end_date)
        alerts = load_alerts(conn, alerts_source, symbol, args.source, args.start_date, args.end_date,
                             risk_as_short=args.risk_as_short)
        bars_by_day = load_bars_by_day(conn, symbol, args.source, args.start_date, args.end_date)
    finally:
        conn.close()

    if not bars_by_day:
        raise SystemExit(f"no bars_1m for {symbol} source={args.source} in the given date range.")

    selected = select_signals(alerts, selection=args.selection, cutoff_tod=cutoff)
    trades = pd.DataFrame([
        build_trade_row(sig, bars_by_day.get(sig["date"]), args.selection,
                        afternoon_start=aft_start, afternoon_end=aft_end)
        for sig in selected.to_dict("records")
    ], columns=TRADE_COLUMNS) if len(selected) else pd.DataFrame(columns=TRADE_COLUMNS)

    baselines = build_baseline_rows(bars_by_day, symbol=symbol, afternoon_start=aft_start,
                                    afternoon_end=aft_end, cutoff_tod=cutoff)
    summary = summarize(trades, args.selection)

    # per-day comparison (one row per day; direction set by the day's single signal,
    # so a day is long XOR short — never both). Signal return vs same-direction baseline.
    base_last_rows = baselines[baselines["baseline"] == "last_bar_before_11"]
    daily = daily_comparison(trades, base_last_rows)

    trades_path = out_dir / "morning_signal_afternoon_extreme_trades.csv"
    summary_path = out_dir / "morning_signal_afternoon_extreme_summary.csv"
    baseline_path = out_dir / "morning_signal_afternoon_extreme_baseline.csv"
    daily_path = out_dir / "morning_signal_afternoon_extreme_daily.csv"
    report_path = out_dir / "morning_signal_afternoon_extreme_report.md"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    baselines.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    up_stats = aggregate(trades[trades["signal_direction"] == "UP"]) if len(trades) else aggregate(trades)
    down_stats = aggregate(trades[trades["signal_direction"] == "DOWN"]) if len(trades) else {"n": 0}
    base_stats = {name: aggregate(g) for name, g in baselines.groupby("baseline")}
    base_last = base_stats.get("last_bar_before_11", {})
    verdict, reasons = decide_verdict(up_stats, base_last, direction="UP")
    down_verdict, down_reasons = decide_verdict(down_stats, base_last, direction="DOWN")
    n_days = len(bars_by_day)
    valid_days = int(trades["trade_date"].nunique()) if len(trades) else 0
    missing_notes = int((trades["notes"].astype(str).str.len() > 0).sum()) if len(trades) else 0
    eff_start = args.start_date or min(bars_by_day)
    eff_end = args.end_date or max(bars_by_day)

    write_report(report_path, symbol=symbol, start=eff_start, end=eff_end,
                 n_days=n_days, valid_days=valid_days, selection=args.selection,
                 alerts_source=alerts_source, up_stats=up_stats, down_stats=down_stats,
                 base_stats_by_name=base_stats, verdict=verdict, reasons=reasons,
                 down_verdict=down_verdict, down_reasons=down_reasons,
                 risk_as_short=args.risk_as_short, missing_notes=missing_notes)

    dr = f"{min(bars_by_day)}..{max(bars_by_day)}"
    print("Morning signal afternoon extreme validation")
    print(f"  symbol={symbol}  date_range={dr}  alerts_source={alerts_source}")
    print(f"  selection={args.selection}  trading_days={n_days}  valid_signal_days={valid_days}")
    print(f"  UP n={up_stats.get('n',0)}  avg afternoon_high_return={_fmt(up_stats.get('avg_afternoon_high_return'))}  "
          f"hit_up_1_0={_fmt(up_stats.get('hit_up_1_0'), False)}  median={_fmt(up_stats.get('median_afternoon_high_return'))}")
    print(f"  UP avg pre-13 MAE={_fmt(up_stats.get('avg_pre_13_mae'))}  "
          f"avg full-day MFE={_fmt(up_stats.get('avg_full_day_mfe_after_signal'))}")
    if down_stats.get("n", 0):
        print(f"  DOWN n={down_stats['n']}  avg short_potential={_fmt(down_stats.get('avg_short_potential'))}  "
              f"median={_fmt(down_stats.get('median_short_potential'))}  "
              f"hit_down_1_0={_fmt(down_stats.get('hit_down_1_0'), False)}"
              + ("  (RISK_ALERT as short)" if args.risk_as_short else ""))
    else:
        print("  DOWN n=0 (no DOWN/SHORT alerts; use --risk-as-short to test RISK_ALERT as short)")
    bl = base_stats.get("last_bar_before_11", {})
    print(f"  baseline(last_bar_before_11) avg afternoon_high_return={_fmt(bl.get('avg_afternoon_high_return'))}  "
          f"avg short_potential={_fmt(bl.get('avg_short_potential'))}")
    print(f"  verdict(UP/long)={verdict}" + (f"  verdict(DOWN/short)={down_verdict}" if down_stats.get('n',0) else ""))
    if len(daily):
        n_long = int((daily["direction"] == "long").sum())
        n_short = int((daily["direction"] == "short").sum())
        unique_days = bool(daily["trade_date"].is_unique)
        print(f"  per-day rows: {len(daily)}  (long days={n_long}, short days={n_short})  "
              f"each day single-direction: {unique_days}")
        if not unique_days:
            print("  WARNING: a trade_date appears more than once — day direction is not unique!")
    print(f"  -> {trades_path}")
    print(f"  -> {summary_path}")
    print(f"  -> {baseline_path}")
    print(f"  -> {daily_path}   <-- per-day long/short returns vs baseline")
    print(f"  -> {report_path}")


if __name__ == "__main__":
    main()
