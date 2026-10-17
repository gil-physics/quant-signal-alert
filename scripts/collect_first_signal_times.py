"""Collect the first replay signal time per trading day.

This is a narrow integrity-check helper for the morning-signal validation flow.
It does not compute afternoon returns and it never places orders.  It can either
read the saved realtime replay alert feed, or recompute the replay engine's
per-minute signal rows directly.  It writes one row per day:

* first_any_*: the first alert of any type in the configured time window.
* signal_*: the first long/short candidate in the same window.

Direction follows the validation tests:
UP_ALERT/UP_CONTINUATION -> long, RISK_ALERT -> short.

Run (cwd = quant):
    ..\Python311-64\python.exe scripts\collect_first_signal_times.py --code 000660
    ..\Python311-64\python.exe scripts\collect_first_signal_times.py --code 000660 --feed engine_replay --cross-symbol
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
for _p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

from config import DB_PATH, PROCESSED_DIR, ensure_directories
from directional_alert import AlertThresholds
from morning_signal_validation import classify_direction, risk_band
from risk_model import load_model
from settings import (
    load_active_symbols,
    load_data_config,
    normalize_symbol_code,
    realtime_default_symbol,
)

import realtime_trend_alert as rta
from build_realtime_replay_alerts import _score_day

OUTPUT_COLUMNS = [
    "trade_date",
    "symbol",
    "feed",
    "cross_symbol",
    "time_basis",
    "direction",
    "signal_direction",
    "signal_time",
    "signal_decision_time",
    "signal_bar_start",
    "signal_tod",
    "alert_type",
    "alert_grade",
    "risk_prob",
    "risk_bucket",
    "alert_price",
    "alert_score",
    "direction_score",
    "first_any_time",
    "first_any_decision_time",
    "first_any_bar_start",
    "first_any_tod",
    "first_any_alert_type",
    "first_any_alert_grade",
    "first_any_risk_prob",
    "first_any_risk_bucket",
    "first_any_alert_price",
    "notes",
]


def _norm_tod(text: str) -> str:
    parts = str(text).strip().split(":")
    if len(parts) == 2:
        parts.append("00")
    if len(parts) != 3:
        raise ValueError(f"invalid time of day: {text!r}")
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"


def _date_clause(start: str | None, end: str | None, col: str) -> tuple[str, list[str]]:
    clause, params = "", []
    if start:
        clause += f" AND substr({col},1,10) >= ?"
        params.append(start)
    if end:
        clause += f" AND substr({col},1,10) <= ?"
        params.append(end)
    return clause, params


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    return bool(row and row[0])


def _trading_days(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    source: str,
    start: str | None,
    end: str | None,
) -> list[str]:
    dc, dp = _date_clause(start, end, "bar_start")
    rows = conn.execute(
        f"SELECT DISTINCT substr(bar_start,1,10) d FROM bars_1m "
        f"WHERE symbol=? AND source=?{dc} ORDER BY d",
        (symbol, source, *dp),
    ).fetchall()
    return [r[0] for r in rows]


def load_realtime_replay_alerts(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    time_basis: str,
) -> pd.DataFrame:
    if not _table_exists(conn, "realtime_replay_alerts"):
        raise SystemExit(
            "realtime_replay_alerts table is missing. "
            "Run scripts/build_realtime_replay_alerts.py first."
        )
    dc, dp = _date_clause(start, end, "decision_time")
    df = pd.read_sql_query(
        f"SELECT decision_time, symbol, alert_type, alert_grade, alert_score, "
        f"direction_score, risk_prob, risk_band, close AS alert_price "
        f"FROM realtime_replay_alerts WHERE symbol=? AND alert_type!=''{dc} "
        f"ORDER BY decision_time",
        conn,
        params=(symbol, *dp),
    )
    df["bar_start"] = None
    return _finalize_alerts(df, time_basis=time_basis)


def _reset_engine(engine) -> None:
    engine.fired = []
    engine._last_fire_at = None
    engine._last_fire_type = None
    engine.today = pd.DataFrame(columns=rta.BAR_COLUMNS)
    engine.history = pd.DataFrame(columns=rta.BAR_COLUMNS)
    engine.peer_history = {}
    engine._cross_cache = {}


def load_engine_replay_alerts(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    source: str,
    start: str | None,
    end: str | None,
    model_path: str,
    cross_symbol: bool,
    time_basis: str,
) -> pd.DataFrame:
    dates = _trading_days(conn, symbol=symbol, source=source, start=start, end=end)
    if not dates:
        return _finalize_alerts(pd.DataFrame(), time_basis=time_basis)

    symbols = load_active_symbols()
    peers = [normalize_symbol_code(s) for s in symbols]
    model, pre, _meta = load_model(model_path)
    engine = rta.TrendAlertEngine(
        symbol,
        model,
        pre,
        active_count=len(symbols),
        cooldown_min=rta.DEFAULT_COOLDOWN_MIN,
        thresholds=AlertThresholds(),
        cross_symbol=cross_symbol,
        peers=peers,
    )

    frames = []
    for date in dates:
        _reset_engine(engine)
        engine.seed_history(conn, date)
        scored = _score_day(engine, conn, date)
        if scored.empty:
            continue
        day_bars = pd.read_sql_query(
            f"SELECT bar_start, bar_end FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn,
            params=(source, symbol, date),
        )
        bar_start_by_end = day_bars.set_index("bar_end")["bar_start"] if len(day_bars) else {}
        scored = scored.copy()
        scored["bar_start"] = scored["decision_time"].map(bar_start_by_end)
        scored["alert_price"] = scored["close"]
        scored = scored[scored["alert_type"].fillna("").astype(str) != ""]
        keep = [
            "decision_time",
            "bar_start",
            "symbol",
            "alert_type",
            "alert_grade",
            "alert_score",
            "direction_score",
            "risk_prob",
            "alert_price",
        ]
        frames.append(scored[[c for c in keep if c in scored.columns]])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _finalize_alerts(df, time_basis=time_basis)


def _finalize_alerts(df: pd.DataFrame, *, time_basis: str) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(
            columns=[
                "decision_time",
                "bar_start",
                "symbol",
                "alert_type",
                "alert_grade",
                "alert_score",
                "direction_score",
                "risk_prob",
                "risk_band",
                "alert_price",
            ]
        )
    if time_basis == "bar_start" and "bar_start" not in df.columns:
        raise SystemExit("time_basis=bar_start requires engine_replay feed.")
    if "bar_start" not in df.columns:
        df["bar_start"] = None
    if "risk_band" not in df.columns:
        df["risk_band"] = ""
    clock_col = "bar_start" if time_basis == "bar_start" else "decision_time"
    df["signal_clock"] = df[clock_col].fillna(df["decision_time"])
    df["trade_date"] = df["signal_clock"].str.slice(0, 10)
    df["tod"] = df["signal_clock"].str.slice(11, 19)
    for col in ("alert_score", "direction_score", "risk_prob", "alert_price"):
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["signal_direction"] = df["alert_type"].map(
        lambda t: classify_direction(t, risk_as_down=True)
    )
    df["direction"] = df["signal_direction"].map({"UP": "long", "DOWN": "short"})

    def bucket(row):
        band = str(row.get("risk_band") or "").strip()
        if band and band.lower() != "nan":
            return band
        return risk_band(row["risk_prob"])

    df["risk_bucket"] = df.apply(
        bucket,
        axis=1,
    )
    return df.sort_values("signal_clock", kind="stable").reset_index(drop=True)


def _pick_first(rows: pd.DataFrame) -> dict | None:
    if rows is None or rows.empty:
        return None
    return rows.sort_values("signal_clock", kind="stable").iloc[0].to_dict()


def collect_first_signals(
    alerts: pd.DataFrame,
    dates: list[str],
    *,
    symbol: str,
    feed: str,
    cross_symbol: bool,
    time_basis: str,
    signal_start: str,
    signal_cutoff: str,
) -> pd.DataFrame:
    rows = []
    alert_dates = set(alerts["trade_date"]) if alerts is not None and len(alerts) else set()
    for date in sorted(set(dates) | alert_dates):
        notes: list[str] = []
        day = alerts[alerts["trade_date"] == date] if len(alerts) else alerts
        window = day[(day["tod"] >= signal_start) & (day["tod"] < signal_cutoff)].copy()
        first_any = _pick_first(window)
        tradable = window[window["signal_direction"].isin(("UP", "DOWN"))].copy()
        first_signal = _pick_first(tradable)

        if first_any is None:
            notes.append("no_alert_in_window")
        if first_signal is None:
            notes.append("no_long_short_signal_in_window")
        elif first_any is not None and first_any["signal_clock"] != first_signal["signal_clock"]:
            notes.append("first_any_non_directional")

        row = {c: None for c in OUTPUT_COLUMNS}
        row.update(
            {
                "trade_date": date,
                "symbol": symbol,
                "feed": feed,
                "cross_symbol": bool(cross_symbol),
                "time_basis": time_basis,
                "notes": "|".join(notes),
            }
        )

        if first_signal is not None:
            row.update(
                {
                    "direction": first_signal.get("direction"),
                    "signal_direction": first_signal.get("signal_direction"),
                    "signal_time": first_signal.get("signal_clock"),
                    "signal_decision_time": first_signal.get("decision_time"),
                    "signal_bar_start": first_signal.get("bar_start"),
                    "signal_tod": first_signal.get("tod"),
                    "alert_type": first_signal.get("alert_type"),
                    "alert_grade": first_signal.get("alert_grade"),
                    "risk_prob": first_signal.get("risk_prob"),
                    "risk_bucket": first_signal.get("risk_bucket"),
                    "alert_price": first_signal.get("alert_price"),
                    "alert_score": first_signal.get("alert_score"),
                    "direction_score": first_signal.get("direction_score"),
                }
            )

        if first_any is not None:
            row.update(
                {
                    "first_any_time": first_any.get("signal_clock"),
                    "first_any_decision_time": first_any.get("decision_time"),
                    "first_any_bar_start": first_any.get("bar_start"),
                    "first_any_tod": first_any.get("tod"),
                    "first_any_alert_type": first_any.get("alert_type"),
                    "first_any_alert_grade": first_any.get("alert_grade"),
                    "first_any_risk_prob": first_any.get("risk_prob"),
                    "first_any_risk_bucket": first_any.get("risk_bucket"),
                    "first_any_alert_price": first_any.get("alert_price"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args():
    default_source = load_data_config().model_source
    p = argparse.ArgumentParser(
        description="Collect first realtime-replay signal times per day."
    )
    p.add_argument("--code", default=None, help="symbol (default: realtime default 000660).")
    p.add_argument("--source", default=default_source, help="bars_1m source for trading days.")
    p.add_argument(
        "--feed",
        choices=["realtime_replay_table", "engine_replay"],
        default="realtime_replay_table",
        help="data source: saved fired-alert table, or direct per-minute replay engine.",
    )
    p.add_argument(
        "--cross-symbol",
        action="store_true",
        help="engine_replay only: use realtime_trend_signal.py's cross-symbol feature path.",
    )
    p.add_argument(
        "--time-basis",
        choices=["decision_time", "bar_start"],
        default=None,
        help="signal_time clock. Default: decision_time for table feed, bar_start for engine_replay.",
    )
    p.add_argument("--model-path", default=str(ROOT_DIR / "models" / "risk_model_lgbm.txt"))
    p.add_argument("--start-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--end-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument(
        "--signal-start",
        default="09:20",
        help="inclusive signal window start (default 09:20).",
    )
    p.add_argument(
        "--signal-cutoff",
        default="11:00",
        help="exclusive signal window cutoff (default 11:00, matching morning validation).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="output CSV path.",
    )
    p.add_argument("--db", default=str(DB_PATH))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    symbol = normalize_symbol_code(args.code or realtime_default_symbol())
    signal_start = _norm_tod(args.signal_start)
    signal_cutoff = _norm_tod(args.signal_cutoff)
    if signal_start >= signal_cutoff:
        raise SystemExit("--signal-start must be earlier than --signal-cutoff")
    if args.cross_symbol and args.feed != "engine_replay":
        raise SystemExit("--cross-symbol requires --feed engine_replay")

    time_basis = args.time_basis or ("bar_start" if args.feed == "engine_replay" else "decision_time")
    if time_basis == "bar_start" and args.feed != "engine_replay":
        raise SystemExit("--time-basis bar_start requires --feed engine_replay")

    if args.output:
        output = Path(args.output)
    else:
        suffix = "_engine_cross" if args.feed == "engine_replay" and args.cross_symbol else (
            "_engine" if args.feed == "engine_replay" else ""
        )
        output = PROCESSED_DIR / f"morning_first_signal_times{suffix}.csv"

    conn = sqlite3.connect(args.db)
    try:
        dates = _trading_days(
            conn,
            symbol=symbol,
            source=args.source,
            start=args.start_date,
            end=args.end_date,
        )
        if args.feed == "engine_replay":
            alerts = load_engine_replay_alerts(
                conn,
                symbol=symbol,
                source=args.source,
                start=args.start_date,
                end=args.end_date,
                model_path=args.model_path,
                cross_symbol=args.cross_symbol,
                time_basis=time_basis,
            )
        else:
            alerts = load_realtime_replay_alerts(
                conn,
                symbol=symbol,
                start=args.start_date,
                end=args.end_date,
                time_basis=time_basis,
            )
    finally:
        conn.close()

    if not dates and alerts.empty:
        raise SystemExit(f"no bars or replay alerts for {symbol} in the requested range.")

    out = collect_first_signals(
        alerts,
        dates,
        symbol=symbol,
        feed=args.feed,
        cross_symbol=args.cross_symbol,
        time_basis=time_basis,
        signal_start=signal_start,
        signal_cutoff=signal_cutoff,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")

    signal_days = out["signal_time"].notna()
    long_days = int((out["direction"] == "long").sum())
    short_days = int((out["direction"] == "short").sum())
    first_any_only = int((out["signal_time"].isna() & out["first_any_time"].notna()).sum())
    print("First signal time collection")
    print(
        f"  symbol={symbol} source={args.source} feed={args.feed} cross_symbol={args.cross_symbol} "
        f"time_basis={time_basis} window=[{signal_start}, {signal_cutoff}) "
        f"days={len(out)} signal_days={int(signal_days.sum())}"
    )
    print(f"  long_days={long_days} short_days={short_days} first_any_only_days={first_any_only}")
    print(f"  -> {output}")


if __name__ == "__main__":
    main()
