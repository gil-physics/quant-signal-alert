"""Capture the REALTIME engine's fired 🔔 alerts across history, into a feed that the
morning-signal validation can use.

Why this exists: the offline ``directional_alerts`` feed is built from WITHIN-DAY features,
so its 60-minute features (ret_60m_1m, volume_z_60m) only become defined ~60 minutes after
the open -> its earliest alert each day is 10:01, and the entire 09:00-10:00 window is
missing. The live trend engine (``realtime_trend_alert.TrendAlertEngine``) instead seeds
``HISTORY_DAYS`` prior trading days, so its 60-minute features are valid right after the
open and it fires 🔔 from 09:10 (the engine's session gate). This script replays that exact
engine over every trading day and records the cooldown-gated 🔔 it fires -- i.e. the alerts a
user actually SEES -- so the validation reflects the true first morning 🔔, not a feed that
starts an hour late.

For each day it builds the day's serving features ONCE over (seeded history + the day) and
replays only the engine's cooldown gate (``_maybe_fire``); because the features are
backward-looking this yields the identical fired set to bar-by-bar streaming, far faster.

Research only -- no orders, no Kiwoom, no SendOrder. Writes the ``realtime_replay_alerts``
table + data/processed/realtime_replay_alerts.csv.

    ..\\Python311-64\\python.exe scripts\\build_realtime_replay_alerts.py --code 000660
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

import realtime_trend_alert as rta
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from directional_alert import AlertThresholds
from realtime_features import (build_serving_features, build_serving_features_cross,
                               score_serving)
from risk_model import load_model
from settings import (load_active_symbols, load_data_config, normalize_symbol_code,
                      realtime_default_symbol)
from utils import now_kst_text

OUTPUT_COLUMNS = ["symbol", "decision_time", "date", "time", "close", "alert_type",
                  "alert_grade", "alert_score", "direction_score", "risk_prob",
                  "risk_band", "message"]


def parse_args():
    p = argparse.ArgumentParser(description="Capture the realtime engine's fired 🔔 over history (research only).")
    p.add_argument("--code", default=None, help="target symbol (default: realtime default 000660).")
    p.add_argument("--cooldown-min", type=int, default=rta.DEFAULT_COOLDOWN_MIN,
                   help=f"engine cooldown (default {rta.DEFAULT_COOLDOWN_MIN}, same as the live feed).")
    p.add_argument("--cross-symbol", action="store_true",
                   help="judge with REAL universe cross-symbol features (v1.1 adopted config). "
                        "Slower; default off matches the basic single-symbol feed.")
    p.add_argument("--model-path", default=str(ROOT_DIR / "models" / "risk_model_lgbm.txt"))
    p.add_argument("--start-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--end-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--db", default=str(DB_PATH))
    return p.parse_args()


def _trading_days(conn, source, symbol, start, end):
    clause, params = "", [source, symbol]
    if start:
        clause += " AND substr(bar_start,1,10) >= ?"; params.append(start)
    if end:
        clause += " AND substr(bar_start,1,10) <= ?"; params.append(end)
    rows = conn.execute(
        f"SELECT DISTINCT substr(bar_start,1,10) d FROM bars_1m WHERE source=? AND symbol=?{clause} "
        "ORDER BY d", params).fetchall()
    return [r[0] for r in rows]


def _score_day(engine, conn, date):
    """Per-minute scored rows for ``date`` (rule alert + risk_prob), built ONCE over the
    seeded history + the day. Mirrors what the streaming engine computes bar-by-bar."""
    day = pd.read_sql_query(
        f"SELECT {', '.join(rta.BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
        "AND substr(bar_start,1,10)=? ORDER BY bar_start",
        conn, params=(engine.source, engine.symbol, date))
    if day.empty:
        return pd.DataFrame()
    if engine.cross_symbol:
        engine.precompute_cross(rta.load_universe_day(conn, engine.peers, engine.source, date), date)
        if not engine._cross_cache:
            return pd.DataFrame()
        scored = pd.DataFrame(list(engine._cross_cache.values()))
    else:
        window = pd.concat([engine.history, day], ignore_index=True)
        serv = build_serving_features(window, source=engine.source, active_count=engine.active_count)
        if serv.empty:
            return pd.DataFrame()
        serv = serv[serv["decision_time"].str.slice(0, 10) == date]
        scored = score_serving(serv, engine.model, engine.pre, thresholds=engine.thresholds)
    if scored.empty:
        return pd.DataFrame()
    return scored.sort_values("decision_time").reset_index(drop=True)


def capture(engine, conn, dates):
    """Replay the engine's cooldown gate over each day; collect every fired 🔔."""
    records = []
    for i, date in enumerate(dates, 1):
        # fresh per-day engine state (cooldown memory + history) -- same as a new session.
        engine.fired = []
        engine._last_fire_at = None
        engine._last_fire_type = None
        engine.today = pd.DataFrame(columns=rta.BAR_COLUMNS)
        engine.history = pd.DataFrame(columns=rta.BAR_COLUMNS)
        engine.peer_history = {}
        engine._cross_cache = {}
        engine.seed_history(conn, date)

        scored = _score_day(engine, conn, date)
        n_fired = 0
        for row in scored.to_dict("records"):
            alert = engine._maybe_fire(pd.Series(row))   # reuse the EXACT live gate
            if alert is None:
                continue
            n_fired += 1
            records.append({
                "symbol": alert["symbol"],
                "decision_time": alert["time"],
                "date": alert["time"][:10],
                "time": alert["time"][11:19],
                "close": alert["close"],
                "alert_type": alert["alert_type"],
                "alert_grade": alert["alert_grade"],
                "alert_score": row.get("alert_score"),
                "direction_score": row.get("direction_score"),
                "risk_prob": alert["risk_prob"],
                "risk_band": alert["risk_band"],
                "message": alert["message"],
            })
        if i % 25 == 0 or i == len(dates):
            print(f"  [{i}/{len(dates)}] {date}: {n_fired} fired (cumulative {len(records)})")
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def main():
    args = parse_args()
    ensure_directories()
    symbol = normalize_symbol_code(args.code or realtime_default_symbol())
    source = load_data_config().model_source
    symbols = load_active_symbols()
    peers = [normalize_symbol_code(s) for s in symbols]

    model, pre, meta = load_model(args.model_path)
    engine = rta.TrendAlertEngine(symbol, model, pre, active_count=len(symbols),
                                  cooldown_min=args.cooldown_min, thresholds=AlertThresholds(),
                                  cross_symbol=args.cross_symbol, peers=peers)

    conn = sqlite3.connect(args.db)
    try:
        dates = _trading_days(conn, source, symbol, args.start_date, args.end_date)
        print(f"[{now_kst_text()}] realtime-replay alert capture symbol={symbol} source={source} "
              f"days={len(dates)} cross_symbol={args.cross_symbol} cooldown={args.cooldown_min}m "
              f"model={getattr(model,'model_type','?')}")
        out = capture(engine, conn, dates)
        out.to_sql("realtime_replay_alerts", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    csv_path = PROCESSED_DIR / "realtime_replay_alerts.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if len(out):
        per_day_first = out.sort_values("decision_time").groupby("date")["time"].first().astype(str)
        print(f"  total fired={len(out)} over {out['date'].nunique()} days; "
              f"earliest-of-day tod: min={per_day_first.min()} max={per_day_first.max()}")
        before10 = int((per_day_first < '10:00:00').sum())
        print(f"  days whose FIRST fired 🔔 is before 10:00: {before10}/{len(per_day_first)} "
              f"(the 09:xx window the offline feed misses)")
        print(f"  alert_type counts: {out['alert_type'].value_counts().to_dict()}")
    print(f"[{now_kst_text()}] realtime_replay_alerts -> db + {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
