"""Export replay bundles (static JSON) for the replay app in app/.

For each requested trading day this runs the realtime runners' OWN replay code
and saves what they would have printed, minute by minute, together with the
chart bars and each alert's forward outcome:

  * long+short trend lines -- scripts/realtime_long_short_trend_alert.py with
    --cross-symbol and its default per-type cooldown: the same objects
    (TrendAlertEngine with cooldown 0, ShortFireGate, LongFireGate) and the same
    per-bar _process_minute call, captured from stdout so the raw line is exactly
    the runner's;
  * MA5x20 touch lines -- scripts/realtime_touch_alert.py's TouchAlertRunner for
    the breakout records, with the grade taken from the 5-seed OUT-OF-FOLD
    predictions (data/processed/touch_predictions_cnn_5seed_fold*.parquet)
    instead of the prod ensemble, which was trained on these days.

The cross-section is pinned to the universe the models were trained on (the
symbol list frozen in the touch prod metadata), not today's config, so a symbol
activated later does not change past features. Pass --universe active to use the
current config instead (that is what the parity check compares against the
runners' own replay output).

Output: app/public/data/<date>/<symbol>.json + app/public/data/index.json.
Education/decision-support only -- no orders are placed or sent.

Run (cwd = quant, 64-bit):
    ..\\Python311-64\\python.exe scripts\\export_replay_bundle.py --dates 2026-06-19
    ..\\Python311-64\\python.exe scripts\\export_replay_bundle.py            # every date in app/replay_dates.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
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

import ma_cross_study as mcs
import realtime_long_short_trend_alert as lsta
import realtime_touch_alert as rtt
import realtime_trend_alert as rta
import replay_export as rx
import short_entry as se
import touch_serving as ts
from config import DB_PATH, PROCESSED_DIR, connect
from directional_alert import AlertThresholds
from risk_model import load_model, resolve_model_path
from settings import (load_active_symbols, load_data_config, load_symbols,
                      normalize_symbol_code, realtime_default_symbol)

APP_DIR = ROOT_DIR / "app"
DATA_DIR = APP_DIR / "public" / "data"
DATES_FILE = APP_DIR / "replay_dates.json"
TOUCH_PROD_META = ROOT_DIR / "models" / "touch_model_cnn_prod_seed42.pt.meta.json"
OOF_PATTERN = "touch_predictions_cnn_5seed_fold{k}.parquet"
OOF_FOLDS = (1, 2, 3, 4, 5)
BUNDLE_VERSION = 1


# --------------------------------------------------------------------------- #
# universe + data loading (indexed range queries on the bars_1m primary key)
# --------------------------------------------------------------------------- #
def trained_universe():
    """The symbol list the models were trained on (frozen in the touch prod
    metadata); falls back to the active config when the metadata is absent."""
    if TOUCH_PROD_META.exists():
        meta = json.loads(TOUCH_PROD_META.read_text(encoding="utf-8"))
        syms = (meta.get("metadata") or {}).get("symbols") or meta.get("symbols") or []
        if syms:
            return [normalize_symbol_code(s) for s in syms]
    return [normalize_symbol_code(s) for s in load_active_symbols()]


def prior_days(conn, source, symbol, date, n):
    days = pd.read_sql_query(
        "SELECT DISTINCT substr(bar_start,1,10) d FROM bars_1m "
        "WHERE symbol=? AND source=? AND bar_start < ? ORDER BY d",
        conn, params=(symbol, source, f"{date} 00:00:00"))["d"].tolist()
    return days[-n:]


def load_bars(conn, source, symbols, lo_date, hi_date):
    """All bars of ``symbols`` with lo_date <= date <= hi_date (one source)."""
    ph = ",".join("?" * len(symbols))
    hi = (pd.Timestamp(hi_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return pd.read_sql_query(
        f"SELECT {', '.join(rta.BAR_COLUMNS)} FROM bars_1m WHERE source=? "
        f"AND symbol IN ({ph}) AND bar_start >= ? AND bar_start < ? "
        "ORDER BY symbol, bar_start",
        conn, params=(source, *symbols, f"{lo_date} 00:00:00", f"{hi} 00:00:00"))


def _split_by_day(frame, date):
    day_mask = frame["bar_start"].str.slice(0, 10) == date
    return frame[~day_mask].reset_index(drop=True), frame[day_mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# long+short trend lines (the unified runner, captured)
# --------------------------------------------------------------------------- #
def runner_thresholds():
    """ShortThresholds exactly as the runner builds them for `--cross-symbol`."""
    saved = sys.argv
    sys.argv = [saved[0], "--cross-symbol"]
    try:
        args = lsta.parse_args()
    finally:
        sys.argv = saved
    with contextlib.redirect_stdout(io.StringIO()):
        thresholds = lsta.build_thresholds(args)
    return args, thresholds


def run_long_short(conn, date, symbol, universe, models):
    source = load_data_config().model_source
    long_model, long_pre, short_model, short_pre = models
    args, thresholds = runner_thresholds()
    engine = rta.TrendAlertEngine(symbol, long_model, long_pre, active_count=len(universe),
                                  cooldown_min=0, thresholds=AlertThresholds(),
                                  cross_symbol=True, peers=universe)
    hist_days = prior_days(conn, source, symbol, date, rta.HISTORY_DAYS)
    frame = load_bars(conn, source, engine.peers, hist_days[0] if hist_days else date, date)
    hist, today = _split_by_day(frame, date)
    engine.history = hist[hist["symbol"] == symbol].reset_index(drop=True)
    engine.peer_history = {s: g.reset_index(drop=True) for s, g in hist.groupby("symbol")}
    peer_today = {s: g.reset_index(drop=True) for s, g in today.groupby("symbol")}
    day = peer_today.get(symbol)
    if day is None or day.empty:
        raise SystemExit(f"no {source} bars for {symbol} on {date}")
    engine.precompute_cross(peer_today, date)

    scope = args.cooldown_scope.replace("-", "_")
    gate = se.ShortFireGate(symbol, thresholds, cooldown_scope=scope)
    long_gate = lsta.LongFireGate(args.cooldown_min, cooldown_scope=scope)
    cache = lsta._score_short_cache(engine, day, peer_today, date,
                                    short_model, short_pre, thresholds)

    ledger, per_bar = [], []
    for bar in day.to_dict("records"):
        long_alert_raw = engine.on_new_bar(bar)
        engine.fired.clear()
        srow = cache.get(bar["bar_end"])
        before = len(ledger)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lsta._process_minute(engine, gate, long_gate, bar, srow, ledger,
                                 long_alert_raw=long_alert_raw, alerts_only=False,
                                 persist_db=False, strict=False)
        per_bar.append({
            "bar": bar, "raw": buf.getvalue().rstrip("\n"), "srow": srow,
            "ctx": lsta._long_context(engine),
            "latest": None if engine.latest is None else engine.latest.copy(),
            "fired": ledger[before:],
        })
    return day, engine.history, per_bar


# --------------------------------------------------------------------------- #
# MA5x20 touch lines (TouchAlertRunner records, out-of-fold grades)
# --------------------------------------------------------------------------- #
_OOF_CACHE = {}


def oof_grades():
    """(symbol, decision_time, direction) -> 5-seed mean OOF prob."""
    if "map" not in _OOF_CACHE:
        frames = []
        for k in OOF_FOLDS:
            p = PROCESSED_DIR / OOF_PATTERN.format(k=k)
            if p.exists():
                frames.append(pd.read_parquet(p, columns=["symbol", "decision_time",
                                                           "direction", "prob"]))
        oof = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["symbol", "decision_time", "direction", "prob"])
        _OOF_CACHE["map"] = {
            (normalize_symbol_code(s), str(dt), int(d)): float(p)
            for s, dt, d, p in zip(oof["symbol"], oof["decision_time"],
                                   oof["direction"], oof["prob"])}
    return _OOF_CACHE["map"]


def run_touch(conn, date, symbol, universe):
    """Touch records for ``symbol`` with raw runner lines and OOF grades, plus the
    carry-stream MA5/MA20 used for the chart (the same MAs the cross is read on)."""
    # a missing grade-map path keeps the runner in breakout-only mode (no prod
    # model is loaded); the out-of-fold grades are attached below instead
    runner = rtt.TouchAlertRunner(universe, [symbol],
                                  grade_map_path=str(ROOT_DIR / "nonexistent_grade_map.json"))
    grade_map = ts.load_grade_map()
    runner.grade_map = grade_map
    runner.date = date
    raw_days = prior_days(conn, "al_raw", symbol, date, rtt.TAIL_DAYS)
    reg_days = prior_days(conn, runner.source, symbol, date, rtt.FEATURE_HISTORY_DAYS)
    lo = min(raw_days[0] if raw_days else date, reg_days[0] if reg_days else date)
    raw_all = load_bars(conn, "al_raw", runner.symbols, lo, date)
    raw_hist, raw_today = _split_by_day(raw_all, date)
    runner.ma_tail = raw_hist[raw_hist["bar_start"].str.slice(0, 10).isin(raw_days)] \
        .reset_index(drop=True)
    reg_all = load_bars(conn, runner.source, runner.symbols,
                        reg_days[0] if reg_days else date, date)
    reg_hist, _ = _split_by_day(reg_all, date)
    runner.feat_history = reg_hist.reset_index(drop=True)
    if raw_today.empty:
        raise SystemExit(f"no al_raw bars on {date}")

    # one batch == the per-minute path: detection, the re-cross flag and
    # cluster_first only look backward, and the outcome backfill runs on the
    # full day either way (the parity check compares against the real runner)
    runner.ingest(raw_today.sort_values(["bar_start", "symbol"]), emit=False)
    runner.finalize()

    covered, _rel = runner._serving_frames()
    cov_keys = set(zip(covered["symbol"], covered["decision_time"])) if len(covered) else set()
    oof = oof_grades()
    out, misses = [], 0
    for key in runner.order:
        rec = runner.records[key]
        if rec["symbol"] != symbol:
            continue
        dt = rec["decision_time"]
        prob = oof.get((symbol, dt, int(rec["direction"])))
        if prob is not None:
            g = str(rtt.tmm.apply_grade([prob], grade_map)[0])
            rate, n = ts.grade_display(grade_map, g)
            rec.update(score=prob, grade=g, grade_oof_touch_rate=rate, grade_oof_n=n,
                       grade_status=rtt.GRADE_STATUS_OK)
        elif dt[11:19] > rtt.GRADE_LAST_DECISION:
            rec.update(score=None, grade=None, grade_status="창 불완전(마감 임박) — 점수 없음")
        elif (symbol, dt) not in cov_keys:
            rec.update(score=None, grade=None, grade_status="워밍업(피처 미커버) — 점수 없음")
        else:
            misses += 1
            rec.update(score=None, grade=None, grade_status="점수 없음")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runner._emit(rec)
        rec["_raw"] = buf.getvalue().rstrip("\n")
        out.append(rec)

    vf, gcols = mcs.prepare_variant(pd.concat([runner.ma_tail, raw_today], ignore_index=True),
                                    "carry")
    mf = mcs.add_mas_and_atr(vf, gcols, price_col="close")
    mf = mf[(mf["symbol"] == symbol) & (mf["bar_start"].str.slice(0, 10) == date)]
    ma = {str(b)[11:16]: (rx.finite_or_none(f, 2), rx.finite_or_none(s, 2))
          for b, f, s in zip(mf["bar_start"], mf["ma_fast"], mf["ma_slow"])}
    return out, ma, misses


# --------------------------------------------------------------------------- #
# bundle assembly
# --------------------------------------------------------------------------- #
def _long_alert(fired_row, latest):
    typ = fired_row["alert_type"]
    prob = fired_row.get("long_risk_prob")
    reasons = rx.parse_message_reasons(fired_row.get("message", ""))
    return {"side": "long", "type": typ, "type_ko": rx.type_ko("long", typ),
            "dir": rx.type_dir("long", typ), "grade": fired_row.get("long_alert_grade", ""),
            "prob": rx.finite_or_none(prob, 3), "band": rta.risk_band(float(prob)),
            "reasons": rx.reasons_ko(reasons), "message": fired_row.get("message", "")}


def _down_alert(fired_row):
    typ = fired_row["alert_type"]
    prob = fired_row.get("short_adverse_prob")
    reasons = rx.parse_message_reasons(fired_row.get("message", ""))
    return {"side": "down", "type": typ, "type_ko": rx.type_ko("down", typ),
            "dir": rx.type_dir("down", typ), "grade": fired_row.get("short_grade", ""),
            "prob": rx.finite_or_none(prob, 3), "band": fired_row.get("short_risk_band", ""),
            "reasons": rx.reasons_ko(reasons), "message": fired_row.get("message", "")}


def _status(ctx, srow):
    st = {}
    if ctx is not None:
        p = ctx["long_risk_prob"]
        st["long"] = {"type": ctx["long_alert_type"], "grade": ctx["long_alert_grade"],
                      "prob": rx.finite_or_none(p, 3),
                      "band": rta.risk_band(p) if p == p else "NA"}
    if srow is not None:
        p = float(srow["short_adverse_prob"])
        st["short"] = {"type": str(srow["alert_type"] or ""), "grade": str(srow["short_grade"]),
                       "dn": rx.finite_or_none(srow["down_score"], 1),
                       "up": rx.finite_or_none(srow["up_score"], 1),
                       "dir": rx.finite_or_none(srow["direction_score"], 1),
                       "prob": rx.finite_or_none(p, 3), "band": se.short_risk_band(p)}
    return st


def build_bundle(conn, date, symbol, name, universe, models):
    t0 = time.time()
    day, history, per_bar = run_long_short(conn, date, symbol, universe, models)
    touch, ma, misses = run_touch(conn, date, symbol, universe)

    bars = [[str(b["bar_start"])[11:16], float(b["open"]), float(b["high"]),
             float(b["low"]), float(b["close"]), float(b["volume"])]
            for b in day.to_dict("records")]
    labels = [b[0] for b in bars]
    prev_close = float(history["close"].iloc[-1]) if len(history) else None

    events = []
    for pb in per_bar:
        if not pb["raw"]:
            continue
        bar = pb["bar"]
        label = str(bar["bar_start"])[11:16]
        alerts = []
        for row in pb["fired"]:
            if row.get("side") == "down":
                alerts.append(_down_alert(row))
            elif row.get("side") == "long":
                alerts.append(_long_alert(row, pb["latest"]))
        if alerts:
            sides = {a["side"] for a in alerts}
            kind = "both" if len(sides) == 2 else alerts[0]["side"]
        else:
            kind = "status"
        ev = {"bar": label, "decision_time": str(bar["bar_end"]), "kind": kind,
              "raw": pb["raw"], "close": float(bar["close"])}
        if alerts:
            ev["alerts"] = alerts
            ev["dir"] = alerts[0]["dir"]
            ev["outcome"] = {
                "h60": rx.forward_outcome(day, str(bar["bar_end"]), float(bar["close"]),
                                          rx.FORWARD_MINUTES),
                "eod": rx.forward_outcome(day, str(bar["bar_end"]), float(bar["close"]), None),
            }
        else:
            ev["status"] = _status(pb["ctx"], pb["srow"])
        events.append(ev)

    for rec in touch:
        label = rec["bar_start"][11:16]
        if label not in labels:
            continue
        graded = rec["grade_status"] == rtt.GRADE_STATUS_OK
        ev = {"bar": label, "decision_time": rec["decision_time"], "kind": "touch",
              "raw": rec["_raw"], "close": float(rec["price_at_alert"]),
              "dir": "up" if int(rec["direction"]) > 0 else "down",
              "touch": {"direction": int(rec["direction"]), "band": rec["band"],
                        "dist_pct": rx.finite_or_none(rec["dist_pct"], 5),
                        "whipsaw": bool(rec["whipsaw_recross"]),
                        "grade": rec["grade"] if graded else None,
                        "score": rx.finite_or_none(rec["score"], 4) if graded else None,
                        "grade_rate": rx.finite_or_none(rec["grade_oof_touch_rate"], 4)
                        if graded else None,
                        "grade_n": rec["grade_oof_n"] if graded else None,
                        "status": rec["grade_status"]}}
        if not rec.get("window_incomplete") and rec.get("realized_touch_05") is not None:
            ev["outcome"] = {"touch": {
                "hit": bool(rec["realized_touch_05"]),
                "mae": rx.finite_or_none(rec["realized_mae"], 5),
                "reveal": rx.touch_reveal_label(rec["decision_time"])}}
        events.append(ev)

    order = {"status": 0, "long": 0, "down": 0, "both": 0, "touch": 1}
    events.sort(key=lambda e: (e["bar"], order[e["kind"]]))
    for i, ev in enumerate(events):
        ev["id"] = i

    n_long = sum(1 for e in events if e["kind"] in ("long", "both"))
    n_down = sum(1 for e in events if e["kind"] in ("down", "both"))
    n_touch = sum(1 for e in events if e["kind"] == "touch")
    n_graded = sum(1 for e in events if e["kind"] == "touch" and e["touch"]["grade"])
    ma5 = [ma.get(lb, (None, None))[0] for lb in labels]
    ma20 = [ma.get(lb, (None, None))[1] for lb in labels]
    open_, close = bars[0][1], bars[-1][4]
    bundle = {
        "version": BUNDLE_VERSION,
        "meta": {
            "date": date, "symbol": symbol, "name": name,
            "open": open_, "close": close, "prev_close": prev_close,
            "change_pct": rx.finite_or_none(close / prev_close - 1.0, 5) if prev_close else None,
            "high": max(b[2] for b in bars), "low": min(b[3] for b in bars),
            "counts": {"long": n_long, "down": n_down, "touch": n_touch,
                       "touch_graded": n_graded},
            "universe": universe,
            "models": {
                "long": "v1 LightGBM 위험 모델 (2026-02-28까지 학습, 이후는 표본 외)",
                "down": "v1_short 반등 위험 모델 (2026-02-28까지 학습)",
                "touch": "CNN 5시드, 그 달을 학습하지 않은 폴드의 예측(OOF)",
            },
            "touch_grade_misses": misses,
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "bars": bars,
        "ma": {"ma5": ma5, "ma20": ma20},
        "events": events,
    }
    problems = rx.validate_bundle(bundle)
    if problems:
        raise SystemExit(f"{date}: bundle validation failed: {problems[:5]}")
    print(f"  {date}: {len(bars)} bars, {len(events)} lines "
          f"(long {n_long}, down {n_down}, touch {n_touch}, graded {n_graded}, "
          f"grade misses {misses}) in {time.time() - t0:.0f}s")
    return bundle


def write_index(symbol, name, date_labels):
    entries = []
    for date in sorted({d for d in date_labels} | {
            p.parent.name for p in DATA_DIR.glob(f"*/{symbol}.json")}):
        path = DATA_DIR / date / f"{symbol}.json"
        if not path.exists():
            continue
        b = json.loads(path.read_text(encoding="utf-8"))
        closes = [bar[4] for bar in b["bars"]]
        step = max(1, len(closes) // 48)
        entries.append({
            "date": date, "label": date_labels.get(date, ""),
            "change_pct": b["meta"]["change_pct"], "counts": b["meta"]["counts"],
            "open": b["meta"]["open"], "close": b["meta"]["close"],
            "prev_close": b["meta"]["prev_close"],
            "spark": closes[::step] + ([closes[-1]] if (len(closes) - 1) % step else []),
            "file": f"data/{date}/{symbol}.json",
        })
    index = {"version": BUNDLE_VERSION, "symbol": symbol, "name": name, "dates": entries}
    (DATA_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  index -> {DATA_DIR / 'index.json'} ({len(entries)} dates)")


def parse_args():
    p = argparse.ArgumentParser(description="Export replay bundles for the replay app.")
    p.add_argument("--dates", default="",
                   help="comma list of YYYY-MM-DD (default: every date in app/replay_dates.json).")
    p.add_argument("--code", default=None, help="symbol (default: realtime.default_symbol).")
    p.add_argument("--universe", choices=["trained", "active"], default="trained",
                   help="cross-section: the models' training universe (default) or the "
                        "current active config (used by the parity check).")
    p.add_argument("--out-dir", default=None,
                   help="write bundles here instead of app/public/data (parity runs).")
    p.add_argument("--no-index", action="store_true", help="skip rewriting index.json.")
    return p.parse_args()


def main():
    global DATA_DIR
    args = parse_args()
    symbol = normalize_symbol_code(args.code or realtime_default_symbol())
    names = {s.code: s.name for s in load_symbols()}
    name = names.get(symbol, symbol)
    date_labels = {}
    if DATES_FILE.exists():
        spec = json.loads(DATES_FILE.read_text(encoding="utf-8"))
        date_labels = {d["date"]: d.get("label", "") for d in spec.get("dates", [])}
    dates = [d.strip() for d in args.dates.split(",") if d.strip()] or sorted(date_labels)
    if not dates:
        raise SystemExit("no dates: pass --dates or list them in app/replay_dates.json")
    if args.out_dir:
        DATA_DIR = Path(args.out_dir)

    universe = trained_universe() if args.universe == "trained" else \
        [normalize_symbol_code(s) for s in load_active_symbols()]
    if symbol not in universe:
        universe = [symbol] + universe
    long_model, long_pre, _ = load_model(str(resolve_model_path("v1")))
    short_model, short_pre, _ = load_model(lsta.DEFAULT_SHORT_MODEL_PATH)
    models = (long_model, long_pre, short_model, short_pre)
    print(f"export replay bundles: {symbol} {name} | universe={args.universe} "
          f"({len(universe)} symbols) | dates={', '.join(dates)}")

    conn = connect(DB_PATH)
    try:
        for date in dates:
            bundle = build_bundle(conn, date, symbol, name, universe, models)
            out = DATA_DIR / date / f"{symbol}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
    finally:
        conn.close()
    if not args.no_index:
        write_index(symbol, name, date_labels)


if __name__ == "__main__":
    main()
