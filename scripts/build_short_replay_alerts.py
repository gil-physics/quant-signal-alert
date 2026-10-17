"""Step 15: build the DOWNTREND trend-alert would-have-seen replay feed over history.

Mirror of scripts/build_realtime_replay_alerts.py for the conservative down-alert
layer: for every historical trading day it seeds the SAME engine history a live
session would hold, builds the day's serving frame ONCE (single-symbol or REAL
cross-symbol), scores it with ``short_entry.score_serving_short``, joins the long
engine's per-minute context (alert_type/grade/risk_prob from ``bra._score_day``),
and streams the rows through the SAME ``short_entry.ShortFireGate`` the realtime
runner uses -- so the replayed 🔻 feed records ALL five down-side alert types
(SHORT_RISK_ALERT / DOWN_WEAKENING_ALERT / DOWN_CONTINUATION / DOWN_ALERT /
VOLATILITY_ALERT) exactly as a live session would have fired them.

Because the features are backward-looking, the one-shot day build yields the
identical fired set to bar-by-bar streaming, far faster.

Two outputs per run:
  1. ``short_replay_alerts``      (table, if_exists="replace") + CSV -- the
     cooldown/veto-gated fires a user would actually have SEEN.
  2. ``short_replay_candidates``  (table) + parquet -- ALL pre-gate scored rows
     with down_score >= 3 (the sweep's lowest boundary) carrying every sweep
     column plus ``alert_type`` (additive), so the threshold sweep re-gates
     offline without recomputing features.

Research only -- no orders, no Kiwoom, no order API. Build with --cross-symbol so
the bottom-rank gate sees REAL universe ranks (single-symbol serving fills the
neutral mid rank and the knob is auto-disabled).

    ..\\Python311-64\\python.exe scripts\\build_short_replay_alerts.py --cross-symbol
    ..\\Python311-64\\python.exe scripts\\build_short_replay_alerts.py --start-date 2026-06-01 --end-date 2026-06-26 --no-short-model
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import replace
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

import numpy as np
import pandas as pd

import build_realtime_replay_alerts as bra
import realtime_trend_alert as rta
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from directional_alert import AlertThresholds
from realtime_features import build_serving_features, build_serving_features_cross
from risk_model import load_model, resolve_model_path
from settings import (load_active_symbols, load_data_config, normalize_symbol_code,
                      realtime_default_symbol)
from short_entry import (SHORT_RESULT_COLUMNS, ShortFireGate, ShortThresholds,
                         score_serving_short)
from utils import now_kst_text

# sweep floor: candidates keep every scored row down_score >= 3 so the offline
# sweep (down_threshold grid starts at 4) can re-gate without feature recompute.
CANDIDATE_MIN_DOWN_SCORE = 3.0

ALERT_COLUMNS = [
    "symbol", "decision_time", "date", "time", "close", "alert_type", "short_grade",
    "up_score", "down_score", "direction_score", "short_score", "upside_risk_score",
    "short_adverse_prob", "short_risk_band", "rank_ret_20m_1m", "active_symbol_count",
    "long_alert_type", "long_alert_grade", "long_risk_prob",
    "reason_1", "reason_2", "reason_3", "thresholds_name", "message",
]
CANDIDATE_COLUMNS = SHORT_RESULT_COLUMNS + [
    "long_alert_type", "long_alert_grade", "long_risk_prob",
    "date", "time", "cross_symbol", "thresholds_name",
]

_D = ShortThresholds()   # dataclass defaults drive the CLI defaults


def _opt_float(text):
    """Float CLI value where the literal 'none' disables the knob."""
    return None if str(text).strip().lower() in ("none", "nan", "") else float(text)


def _opt_int(text):
    """Int CLI value where the literal 'none' disables the knob."""
    return None if str(text).strip().lower() in ("none", "nan", "") else int(text)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build the DOWNTREND trend-alert would-have-seen replay feed "
                    "+ pre-gate candidates (research only).")
    p.add_argument("--code", default=None, help="target symbol (default: realtime default symbol).")
    p.add_argument("--cross-symbol", action="store_true",
                   help="judge with REAL universe cross-symbol features (needed for the bottom-rank "
                        "gate; single-symbol serving fills the NEUTRAL mid rank).")
    p.add_argument("--start-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--end-date", default=None, help="optional YYYY-MM-DD.")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--model-path", default=str(ROOT_DIR / "models" / "risk_model_lgbm.txt"),
                   help="LONG-context risk model (display join only, never gates the short feed).")
    p.add_argument("--short-model-path", default=str(resolve_model_path("v1_short")),
                   help="v1_short model scoring short_adverse_prob (annotated on every alert; "
                        "feeds the optional DOWN-type veto).")
    p.add_argument("--no-short-model", action="store_true",
                   help="rule-only feed: short_adverse_prob all-NaN and any configured model veto "
                        "is disabled (short_adverse_prob_max=None), else NaN would veto every "
                        "DOWN-type row.")
    # ShortThresholds overrides (mirror names shared with the realtime runner)
    p.add_argument("--down-threshold", type=float, default=_D.down_threshold,
                   help=f"down_score >= (default {_D.down_threshold}).")
    p.add_argument("--up-risk-limit", type=_opt_float, default=_D.up_risk_limit,
                   help=f"up_score <= (default {_D.up_risk_limit}; 'none' skips).")
    p.add_argument("--score-margin", type=float, default=_D.score_margin,
                   help=f"direction_score <= -margin (default {_D.score_margin}).")
    p.add_argument("--bottom-rank-limit", type=_opt_int, default=_D.bottom_rank_limit,
                   help=f"rank_ret_20m_1m > active_symbol_count - limit (default {_D.bottom_rank_limit}; "
                        "'none' skips; auto-disabled without --cross-symbol).")
    p.add_argument("--require-below-ma5", action="store_true",
                   help="optional confirm: dist_ma5_intraday < 0.")
    p.add_argument("--require-rel-weakness", action="store_true",
                   help="optional confirm: relative_ret_20m_to_universe < 0.")
    p.add_argument("--short-adverse-prob-max", type=_opt_float, default=_D.short_adverse_prob_max,
                   help="optional model veto on the DOWN types (DOWN_ALERT/DOWN_CONTINUATION): "
                        "short_adverse_prob >= max -> no fire (NaN vetoed too). Default "
                        f"{_D.short_adverse_prob_max} = annotation-only; pass a float to enable.")
    p.add_argument("--daily-limit", "--daily-alert-limit", dest="daily_limit", type=_opt_int,
                   default=_D.daily_alert_limit,
                   help="max fires per day; 'none'/'off' (the default) = no cap, so the per-type "
                        "cooldown is the only throttle. --daily-alert-limit is the runner alias.")
    p.add_argument("--cooldown-min", type=int, default=_D.cooldown_minutes,
                   help=f"minutes between fires (default {_D.cooldown_minutes}).")
    p.add_argument("--min-time", default=_D.min_time, help=f"session start (default {_D.min_time}).")
    p.add_argument("--max-time", default=_D.max_time, help=f"session end, exclusive (default {_D.max_time}).")
    p.add_argument("--thresholds-name", default=None,
                   help="explicit provenance tag; default 'mirror_baseline', or 'cli_override' "
                        "when any threshold arg differs from the dataclass defaults.")
    return p.parse_args()


def build_thresholds(args):
    """ShortThresholds from the CLI. USER overrides rename the combination to
    'cli_override'; forced auto-adjustments (--no-short-model veto off, non-cross
    rank off) keep the name -- they are traceable via the candidates' cross_symbol
    flag and the all-NaN short_adverse_prob column."""
    fields = dict(
        down_threshold=args.down_threshold,
        up_risk_limit=args.up_risk_limit,
        score_margin=args.score_margin,
        bottom_rank_limit=args.bottom_rank_limit,
        require_below_ma5=args.require_below_ma5,
        require_rel_weakness=args.require_rel_weakness,
        short_adverse_prob_max=args.short_adverse_prob_max,
        daily_alert_limit=args.daily_limit,
        cooldown_minutes=args.cooldown_min,
        min_time=args.min_time,
        max_time=args.max_time,
    )
    user_overrode = any(getattr(_D, k) != v for k, v in fields.items())
    notes = []
    if args.no_short_model and fields["short_adverse_prob_max"] is not None:
        fields["short_adverse_prob_max"] = None
        notes.append("model veto disabled: --no-short-model serves all-NaN short_adverse_prob "
                     "and a configured veto would suppress every fire")
    if not args.cross_symbol and fields["bottom_rank_limit"] is not None:
        fields["bottom_rank_limit"] = None
        notes.append("bottom_rank_limit disabled: single-symbol serving fills the NEUTRAL mid "
                     "rank (build with --cross-symbol for the rank gate)")
    name = args.thresholds_name or ("cli_override" if user_overrode else _D.name)
    return replace(_D, name=name, **fields), notes


def _short_serving_frame(engine, conn, date, day):
    """The day's serving frame for the SHORT scorer, built ONCE over the seeded
    history (+ universe when cross); mirrors the window the engine itself scores
    (precompute_cross / on_new_bar) so the replay matches a live session."""
    if engine.cross_symbol:
        peer_today = rta.load_universe_day(conn, engine.peers, engine.source, date)
        frames = []
        for s in engine.peers:
            h = engine.peer_history.get(s)
            if h is not None and len(h):
                frames.append(h[rta.BAR_COLUMNS])
            d = peer_today.get(s)
            if d is not None and len(d):
                frames.append(d[rta.BAR_COLUMNS])
        if not frames:
            return pd.DataFrame()
        window = pd.concat(frames, ignore_index=True)
        serv = build_serving_features_cross(window, target=engine.symbol, source=engine.source,
                                            active_count=len(engine.peers))
    else:
        window = pd.concat([engine.history, day], ignore_index=True)
        serv = build_serving_features(window, source=engine.source, active_count=engine.active_count)
    if serv.empty:
        return serv
    return serv[serv["decision_time"].str.slice(0, 10) == date].reset_index(drop=True)


def _attach_long_context(scored, long_ctx):
    """Left-join the long engine's per-minute context (display only, never gates)."""
    scored = scored.sort_values("decision_time").reset_index(drop=True)
    if long_ctx is not None and not long_ctx.empty:
        lc = long_ctx[["decision_time", "alert_type", "alert_grade", "risk_prob"]].rename(
            columns={"alert_type": "long_alert_type", "alert_grade": "long_alert_grade",
                     "risk_prob": "long_risk_prob"})
        scored = scored.merge(lc, on="decision_time", how="left", validate="one_to_one")
    for col, default in (("long_alert_type", ""), ("long_alert_grade", ""),
                         ("long_risk_prob", np.nan)):
        if col not in scored.columns:
            scored[col] = default
    # keep the string columns parquet-safe ("" = no long alert / no aligned long row)
    scored["long_alert_type"] = scored["long_alert_type"].fillna("")
    scored["long_alert_grade"] = scored["long_alert_grade"].fillna("")
    return scored


def capture(engine, gate, conn, dates, *, short_model, short_pre):
    """Replay the short gate over each day; collect fires + pre-gate candidates."""
    thr = gate.thresholds
    records, cand_frames = [], []
    for i, date in enumerate(dates, 1):
        # fresh per-day engine state (cooldown memory + history) -- same as a new
        # session (mirror of bra.capture); the short gate resets independently.
        engine.fired = []
        engine._last_fire_at = None
        engine._last_fire_type = None
        engine.today = pd.DataFrame(columns=rta.BAR_COLUMNS)
        engine.history = pd.DataFrame(columns=rta.BAR_COLUMNS)
        engine.peer_history = {}
        engine._cross_cache = {}
        engine.seed_history(conn, date)
        gate.reset_day(date)

        day = pd.read_sql_query(
            f"SELECT {', '.join(rta.BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn, params=(engine.source, engine.symbol, date))
        n_fired = 0
        if not day.empty:
            serv = _short_serving_frame(engine, conn, date, day)
            scored = score_serving_short(serv, short_model, short_pre, thresholds=thr)
            if not scored.empty:
                scored = _attach_long_context(scored, bra._score_day(engine, conn, date))

                cand = scored[scored["down_score"] >= CANDIDATE_MIN_DOWN_SCORE].copy()
                if len(cand):
                    cand["date"] = cand["decision_time"].str.slice(0, 10)
                    cand["time"] = cand["decision_time"].str.slice(11, 19)
                    cand["cross_symbol"] = int(bool(engine.cross_symbol))
                    cand["thresholds_name"] = thr.name
                    cand_frames.append(cand[CANDIDATE_COLUMNS])

                for row in scored.to_dict("records"):
                    # the SAME stateful gate the realtime runner uses (replay parity)
                    alert = gate.maybe_fire(row, short_adverse_prob=row.get("short_adverse_prob",
                                                                            float("nan")))
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
                        "short_grade": alert["short_grade"],
                        "up_score": alert["up_score"],
                        "down_score": alert["down_score"],
                        "direction_score": alert["direction_score"],
                        "short_score": alert["short_score"],
                        "upside_risk_score": alert["upside_risk_score"],
                        "short_adverse_prob": alert["short_adverse_prob"],
                        "short_risk_band": alert["short_risk_band"],
                        "rank_ret_20m_1m": row.get("rank_ret_20m_1m"),
                        "active_symbol_count": row.get("active_symbol_count"),
                        "long_alert_type": row.get("long_alert_type", ""),
                        "long_alert_grade": row.get("long_alert_grade", ""),
                        "long_risk_prob": row.get("long_risk_prob", float("nan")),
                        "reason_1": alert["reason_1"],
                        "reason_2": alert["reason_2"],
                        "reason_3": alert["reason_3"],
                        "thresholds_name": alert["thresholds_name"],
                        "message": alert["message"],
                    })
        if i % 25 == 0 or i == len(dates):
            n_cand = sum(len(f) for f in cand_frames)
            print(f"  [{i}/{len(dates)}] {date}: {n_fired} fired "
                  f"(cumulative fires={len(records)} candidates={n_cand})")
    alerts = pd.DataFrame(records, columns=ALERT_COLUMNS)
    cands = (pd.concat(cand_frames, ignore_index=True) if cand_frames
             else pd.DataFrame(columns=CANDIDATE_COLUMNS))
    return alerts, cands


def main():
    args = parse_args()
    ensure_directories()
    symbol = normalize_symbol_code(args.code or realtime_default_symbol())
    source = load_data_config().model_source
    symbols = load_active_symbols()
    peers = [normalize_symbol_code(s) for s in symbols]

    thr, notes = build_thresholds(args)
    for note in notes:
        print(f"  note: {note}")

    long_model, long_pre, long_meta = load_model(args.model_path)
    if args.no_short_model:
        short_model, short_pre = None, None
        short_tag = "rule-only (no short model)"
    else:
        short_model, short_pre, short_meta = load_model(args.short_model_path)
        short_tag = f"{getattr(short_model, 'model_type', '?')} target={short_meta.get('target', '?')}"
        if short_meta.get("target") not in (None, "target_up_risk"):
            print(f"  warn: short model target is {short_meta.get('target')!r}, expected 'target_up_risk'")

    # engine supplies seeded history + the LONG display context only; its cooldown
    # state never gates the short feed (ShortFireGate owns the short state).
    engine = rta.TrendAlertEngine(symbol, long_model, long_pre, active_count=len(symbols),
                                  cooldown_min=rta.DEFAULT_COOLDOWN_MIN, thresholds=AlertThresholds(),
                                  cross_symbol=args.cross_symbol, peers=peers)
    gate = ShortFireGate(symbol, thr)

    conn = sqlite3.connect(args.db)
    try:
        dates = bra._trading_days(conn, source, symbol, args.start_date, args.end_date)
        print(f"[{now_kst_text()}] down-trend alert replay feed symbol={symbol} source={source} "
              f"days={len(dates)} cross_symbol={args.cross_symbol}")
        print(f"  thresholds={thr.name} (down>={thr.down_threshold:g}, up<={thr.up_risk_limit}, "
              f"margin>={thr.score_margin:g}, rank_limit={thr.bottom_rank_limit}, "
              f"veto_max={thr.short_adverse_prob_max}, cooldown={thr.cooldown_minutes}m, "
              f"daily_limit={'off' if thr.daily_alert_limit is None else thr.daily_alert_limit}, "
              f"session={thr.min_time}-{thr.max_time})")
        print(f"  long_model={getattr(long_model, 'model_type', '?')} short_model={short_tag}")
        alerts, cands = capture(engine, gate, conn, dates,
                                short_model=short_model, short_pre=short_pre)
        alerts.to_sql("short_replay_alerts", conn, if_exists="replace", index=False)
        cands.to_sql("short_replay_candidates", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    csv_path = PROCESSED_DIR / "short_replay_alerts.csv"
    alerts.to_csv(csv_path, index=False, encoding="utf-8-sig")
    parquet_path = PROCESSED_DIR / "short_replay_candidates.parquet"
    cands.to_parquet(parquet_path, index=False)

    if len(alerts):
        per_day_first = alerts.sort_values("decision_time").groupby("date")["time"].first().astype(str)
        print(f"  total fired={len(alerts)} over {alerts['date'].nunique()} days; "
              f"earliest-of-day tod: min={per_day_first.min()} max={per_day_first.max()}")
        before10 = int((per_day_first < "10:00:00").sum())
        print(f"  days whose FIRST fired 🔻 is before 10:00: {before10}/{len(per_day_first)}")
        print(f"  alert_type counts: {alerts['alert_type'].value_counts().to_dict()}")
        print(f"  short_grade counts: {alerts['short_grade'].value_counts().to_dict()}")
        print(f"  short_risk_band counts: {alerts['short_risk_band'].value_counts().to_dict()}")
    else:
        print("  no down-trend alerts fired under the current thresholds")
    n_cand_days = int(cands["date"].nunique()) if len(cands) else 0
    print(f"  candidates (down_score>={CANDIDATE_MIN_DOWN_SCORE:g}, pre-gate): "
          f"{len(cands)} rows over {n_cand_days} days")
    print(f"[{now_kst_text()}] short_replay_alerts -> db + {csv_path}")
    print(f"[{now_kst_text()}] short_replay_candidates -> db + {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
