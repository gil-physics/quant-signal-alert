"""Step 15: chartless realtime+replay DOWNTREND trend-alert runner (64-bit).

Composes the EXISTING long-side trend/risk engine (scripts/realtime_trend_alert.py:
TrendAlertEngine, unmodified) with the Step 15 pure down-alert module
(src/short_entry.py) -- the symmetric mirror of the long side's directional-alert
taxonomy. The engine supplies the per-minute LONG context (alert type/grade +
risk_prob) for display only; the short layer runs its OWN scoring path
(score_serving_short) because engine.latest exposes RESULT_COLUMNS only, not the
gate/model inputs. Per finished 1m bar it prints one status line (down alert
type/grade/scores/short_adverse_prob + long context). Each line is one bar; leading
markers show what fired that bar -- 🔻 when the stateful ShortFireGate fires one of the
five down-side trend alerts (priority: SHORT_RISK_ALERT > DOWN_WEAKENING_ALERT >
DOWN_CONTINUATION > DOWN_ALERT > VOLATILITY_ALERT), and a blue ▲ when the long-side
engine fires its own alert (상방 가능성 등) on the same bar.

Decision-support ONLY: it prints, writes a CSV ledger on stop, and (live mode) logs
fired alerts to the short_alert_log table. It NEVER places, cancels, or transmits
any order.

Scoring path (batch-per-day; rows are causal so the batch row at decision_time t
equals what an incremental rebuild at t would produce -- decision_time == bar_end):
  * replay : seed engine history -> build the day's serving frame ONCE
             (build_serving_features / build_serving_features_cross) ->
             score_serving_short -> cache keyed by bar["bar_end"] -> stream bars
             through engine.on_new_bar + ShortFireGate. Nothing is written to
             short_alert_log (CSV ledger + console summary only).
  * live   : poll realtime_minute_bars (32-bit collector), serve the regular session
             only (drop premarket bars before REGULAR_OPEN, train/serve parity),
             rebuild the day short frame per poll, stream fresh bars. The startup
             backlog is replayed with printing but NOT persisted to short_alert_log
             (a restart must not re-log the morning); post-backlog fires are
             inserted best-effort at fire time.

engine.enable_persistence is deliberately NEVER called: realtime_alert_log belongs
to the long runners, and a concurrently running realtime_trend_signal.py must not
see duplicate rows. The long engine's cooldown state and the ShortFireGate's
cooldown/daily-limit state are fully separate.

The v1_short model is ANNOTATION-FIRST (mirror of the long system's risk model):
short_adverse_prob + short_risk_band are displayed on every alert, and DOWN types
carry a Korean model note (LOW band "✓ 저위험 하방" / HIGH band "⚠ 모델 상승위험").
--short-adverse-prob-max defaults to None (annotation only); passing a float
enables the model veto on the DOWN types (DOWN_ALERT/DOWN_CONTINUATION) only.
Trend alerts carry no GO/NO-GO verdict; the historical entry-sweep result is
documented in README, not enforced here.

Run (cwd = quant, 64-bit; the default symbol comes from config.yaml realtime.default_symbol):
    ..\\Python311-64\\python.exe scripts\\realtime_short_trend_alert.py --mode replay --replay-date 2026-06-19 --no-short-model
    ..\\Python311-64\\python.exe scripts\\realtime_short_trend_alert.py --mode replay --cross-symbol
    ..\\Python311-64\\python.exe scripts\\realtime_short_trend_alert.py --mode live --poll-sec 5
"""

from __future__ import annotations

import argparse
import sqlite3
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
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

# reuse the existing trend/risk engine for the LONG context -- never duplicated here
import realtime_trend_alert as rta
import short_entry as se
from config import DB_PATH, PROCESSED_DIR, connect, ensure_directories
from directional_alert import AlertThresholds
from realtime_features import build_serving_features, build_serving_features_cross
from risk_model import load_model, model_version_config, resolve_model_path
from settings import load_active_symbols, normalize_symbol_code, realtime_default_symbol

LEDGER_PATH = PROCESSED_DIR / "short_entry_ledger.csv"
DEFAULT_CODE = realtime_default_symbol()
DEFAULT_SHORT_MODEL_PATH = str(resolve_model_path("v1_short"))

# CSV ledger schema (one row per FIRED alert: the ShortFireGate dict + long context)
LEDGER_COLUMNS = [
    "time", "symbol", "close", "alert_type", "short_grade",
    "up_score", "down_score", "direction_score", "short_score", "upside_risk_score",
    "short_adverse_prob", "short_risk_band",
    "long_alert_type", "long_alert_grade", "long_risk_prob",
    "reason_1", "reason_2", "reason_3", "thresholds_name", "message",
]

# DB table for LIVE fired alerts (best-effort; the CSV ledger is the source of truth)
_SHORT_ALERT_LOG_SCHEMA = [
    ("ts", "TEXT"), ("symbol", "TEXT"), ("alert_type", "TEXT"), ("short_grade", "TEXT"),
    ("up_score", "REAL"), ("down_score", "REAL"), ("direction_score", "REAL"),
    ("short_score", "REAL"), ("upside_risk_score", "REAL"),
    ("short_adverse_prob", "REAL"), ("short_risk_band", "TEXT"),
    ("long_alert_type", "TEXT"), ("long_alert_grade", "TEXT"), ("long_risk_prob", "REAL"),
    ("close", "REAL"), ("reason_1", "TEXT"), ("reason_2", "TEXT"), ("reason_3", "TEXT"),
    ("thresholds_name", "TEXT"), ("message", "TEXT"), ("created_at", "TEXT"),
]


# --------------------------------------------------------------------------- #
# Long context from the reused engine (display only)
# --------------------------------------------------------------------------- #
def _long_context(engine):
    """engine.latest -> {long_alert_type, long_alert_grade, long_risk_prob}; None
    while the engine is still warming up (no scored row yet)."""
    row = engine.latest
    if row is None:
        return None
    rp = row.get("risk_prob")
    return {
        "long_alert_type": row.get("alert_type") or "",
        "long_alert_grade": row.get("alert_grade") or "",
        "long_risk_prob": float(rp) if rp is not None else float("nan"),
    }


_EMPTY_LONG = {"long_alert_type": "", "long_alert_grade": "", "long_risk_prob": float("nan")}


def _fmt_long(ctx):
    if ctx is None:
        return "long=(warming up)"
    prob = ctx["long_risk_prob"]
    typ = ctx["long_alert_type"] or "-"
    grade = ctx["long_alert_grade"] or "-"
    return f"long={typ}({grade}) risk={prob:.2f}[{rta.risk_band(prob)}]"


# line-leading direction markers: 🔻 = a DOWN-side (this system) alert fired this bar,
# blue ▲ = the LONG-side engine fired this bar. Both can show together (🔻▲).
_DOWN_MARK = "🔻"
_BLUE, _RESET = "\033[94m", "\033[0m"


def _up_mark():
    """Blue up-arrow for a long-side fire; plain ▲ when stdout is not a TTY (so a
    redirected log stays free of ANSI escapes)."""
    return f"{_BLUE}▲{_RESET}" if sys.stdout.isatty() else "▲"


def _status_body(srow):
    """The compact per-minute down-state body (no timestamp/markers/long context).
    Returns (body, note); note flags a fire-gate pass that did not fire."""
    if srow is None:
        return "short=(warming up)", ""
    p = float(srow["short_adverse_prob"])
    typ = str(srow["alert_type"] or "-")
    body = (f"short={typ + '(' + str(srow['short_grade']) + ')':<26} "
            f"dn={float(srow['down_score']):.1f} "
            f"up={float(srow['up_score']):.1f} dir={float(srow['direction_score']):+.1f} "
            f"bounce={float(srow['upside_risk_score']):.1f} "
            f"adverse={p:.2f}[{se.short_risk_band(p)}]")
    note = "  *fire-gate pass (not fired)" if bool(srow["short_fires_rule"]) else ""
    return body, note


# --------------------------------------------------------------------------- #
# Persistence (live fires -> short_alert_log; all fires -> CSV ledger on stop)
# --------------------------------------------------------------------------- #
def _persist_alert(ledger_row):
    """Best-effort insert of one FIRED alert into short_alert_log (live mode only).
    A transient lock (collector mid-write) must not crash the feed -- the alert is
    still printed and CSV-ledgered on stop."""
    payload = dict(ledger_row)
    payload["ts"] = payload.pop("time")
    payload["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = [c for c, _ in _SHORT_ALERT_LOG_SCHEMA]
    try:
        conn = connect(DB_PATH)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS short_alert_log ("
                         + ", ".join(f"{c} {t}" for c, t in _SHORT_ALERT_LOG_SCHEMA) + ")")
            conn.execute(
                "INSERT INTO short_alert_log (" + ", ".join(cols) + ") VALUES ("
                + ",".join("?" * len(cols)) + ")",
                tuple(payload[c] for c in cols))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"  (warn: could not persist alert to short_alert_log: {exc})")


def _print_ledger(ledger):
    """CSV ledger + console summary on stop (mirror of the long runners' _print_ledger)."""
    if not ledger:
        print("\n  (no down-trend alerts fired)")
        return
    led = pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
    led.to_csv(LEDGER_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  fired {len(led)} down-trend alerts  "
          f"types={led['alert_type'].value_counts().to_dict()}  "
          f"grades={led['short_grade'].value_counts().to_dict()}  "
          f"bands={led['short_risk_band'].value_counts().to_dict()}")
    print("  last alerts:")
    for a in ledger[-8:]:
        print(f"    {a['time'][11:16]} {str(a['alert_type']):<21} {str(a['short_grade']):<10} "
              f"short_adverse_prob={float(a['short_adverse_prob']):.2f}[{a['short_risk_band']}] "
              f"close={a['close']:,.0f} reasons={a['reason_1']}/{a['reason_2']}")
    print(f"  ledger -> {LEDGER_PATH}")


# --------------------------------------------------------------------------- #
# Short-layer scoring (batch per day / per poll)
# --------------------------------------------------------------------------- #
def _build_short_serving(engine, day_bars, peer_today, date):
    """The SHORT layer's day serving frame, built ONCE per day (replay) or per poll
    (live). Single-symbol serving mirrors engine.on_new_bar's window (history + today,
    NEUTRAL cross-section); cross-symbol serving mirrors engine.precompute_cross
    (peer history + peer today, REAL cross-section). Strictly causal, so caching the
    batch rows by decision_time (== bar_end) is leak-free."""
    if engine.cross_symbol:
        frames = []
        for s in engine.peers:
            h = engine.peer_history.get(s)
            if h is not None and len(h):
                frames.append(h[rta.BAR_COLUMNS])
            d = (peer_today or {}).get(s)
            if d is not None and len(d):
                frames.append(d[rta.BAR_COLUMNS])
        if not frames:
            return pd.DataFrame()
        window = pd.concat(frames, ignore_index=True)
        serv = build_serving_features_cross(window, target=engine.symbol,
                                            source=engine.source,
                                            active_count=len(engine.peers))
    else:
        window = pd.concat([engine.history[rta.BAR_COLUMNS], day_bars[rta.BAR_COLUMNS]],
                           ignore_index=True)
        serv = build_serving_features(window, source=engine.source,
                                      active_count=engine.active_count)
    if serv.empty:
        return serv
    return serv[serv["decision_time"].str.slice(0, 10) == date].reset_index(drop=True)


def _score_short_cache(engine, day_bars, peer_today, date, short_model, short_pre, thresholds):
    """score_serving_short over the day frame -> {decision_time: scored row}."""
    serv = _build_short_serving(engine, day_bars, peer_today, date)
    scored = se.score_serving_short(serv, short_model, short_pre, thresholds=thresholds)
    return {r["decision_time"]: r for _, r in scored.iterrows()}


def _process_minute(engine, gate, bar, srow, ledger, *, long_alert, alerts_only, persist_db):
    """Emit ONE line for this finished bar and ledger a fired DOWN alert.

    A single line carries everything for the bar, labeled with bar_start -- the chart
    candle's label, the same convention as realtime_trend_signal.py -- so the printed
    prices line up with the HTS minute chart and the down feed / long context never
    drift apart. (The DECISION still happens at that bar's close: gates run on
    decision_time == bar_end, and the ledger/DB keep decision_time.) Leading markers:
    🔻 a DOWN-side fire, blue ▲ a LONG-side fire; then the fired down message or the
    compact status body, then the long context. ``srow`` is the cached score row for
    bar["bar_end"] (None while warming up); ``long_alert`` is the long engine's fired
    alert this bar (None otherwise). Returns the fired down alert dict or None."""
    alert = gate.maybe_fire(srow, short_adverse_prob=srow["short_adverse_prob"]) \
        if srow is not None else None
    ctx = _long_context(engine)
    down_fired = alert is not None
    long_fired = long_alert is not None
    # alerts-only: show only bars where something fired; otherwise skip pure warm-up
    if alerts_only and not (down_fired or long_fired):
        return alert
    if not down_fired and not long_fired and srow is None and ctx is None:
        return alert

    markers = (_DOWN_MARK if down_fired else "") + (_up_mark() if long_fired else "")
    prefix = markers if markers else "  "
    hhmm = str(bar["bar_start"])[11:16]   # chart-candle label (decision is at this bar's close)
    if down_fired:
        body, note = alert["message"], ""
    else:
        body, note = _status_body(srow)
    print(f"{prefix} {hhmm} {engine.symbol}  close={float(bar['close']):,.0f}  "
          f"{body}  | {_fmt_long(ctx)}{note}")

    if down_fired:
        row = {**alert, **(ctx if ctx is not None else _EMPTY_LONG)}
        ledger.append(row)
        if persist_db:
            _persist_alert(row)
    return alert


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_replay(engine, gate, short_model, short_pre, thresholds, ledger, *,
               date, alerts_only):
    """Stream a historical day: engine per bar (long context) + one-shot short cache."""
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, date)
        day = pd.read_sql_query(
            f"SELECT {', '.join(rta.BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn, params=(engine.source, engine.symbol, date))
        peer_today = None
        if engine.cross_symbol:
            peer_today = rta.load_universe_day(conn, engine.peers, engine.source, date)
            engine.precompute_cross(peer_today, date)   # long-context cache
    finally:
        conn.close()
    if day.empty:
        raise SystemExit(f"no bars_1m for {engine.symbol} on {date} (source={engine.source}).")

    cache = _score_short_cache(engine, day, peer_today, date, short_model, short_pre, thresholds)
    print(f"replay-short {engine.symbol} {date}: {len(day)} bars, {len(cache)} scorable minutes "
          f"(printing {'🔻/▲ fires only' if alerts_only else 'every scored minute + 🔻/▲ fires'})\n")
    for bar in day.to_dict("records"):
        long_alert = engine.on_new_bar(bar)
        _process_minute(engine, gate, bar, cache.get(bar["bar_end"]), ledger,
                        long_alert=long_alert, alerts_only=alerts_only, persist_db=False)


def run_live(engine, gate, short_model, short_pre, thresholds, ledger, *,
             today, poll_sec, alerts_only):
    """Poll realtime_minute_bars (regular session only) and stream fresh bars. The
    startup backlog is replayed with printing but not persisted to short_alert_log."""
    print(f"live-short {engine.symbol}: polling realtime_minute_bars every {poll_sec}s "
          f"(Ctrl+C to stop). Run the 32-bit collector to populate it.")
    print("  fired down-trend alerts -> DB table short_alert_log (+ CSV ledger on stop); "
          "backlog fires are printed + CSV-ledgered only.\n")
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, today)
    finally:
        conn.close()
    seen = set()
    rt_bars = []       # engine-format bars streamed so far (regular session only)
    primed = False
    while True:
        conn = connect(DB_PATH)
        try:
            new = pd.read_sql_query(
                "SELECT symbol, ts_kst, open, high, low, close, volume FROM realtime_minute_bars "
                "WHERE symbol=? AND substr(ts_kst,1,10)=? ORDER BY ts_kst",
                conn, params=(engine.symbol, today))
        finally:
            conn.close()
        # serve the regular session only (drop premarket 08:00-09:00), like al_regular
        records = [r for r in new.to_dict("records") if r["ts_kst"][11:19] >= rta.REGULAR_OPEN]
        fresh = [r for r in records if r["ts_kst"] not in seen]
        for r in fresh:
            seen.add(r["ts_kst"])
        if fresh:
            peer_today = None
            if engine.cross_symbol:   # refresh the universe cross-section cache
                conn = connect(DB_PATH)
                try:
                    peer_today = rta.load_universe_day(conn, engine.peers, engine.source,
                                                       today, realtime=True)
                finally:
                    conn.close()
                engine.precompute_cross(peer_today, today)   # long-context cache
            bars = [rta._bar_from_rt(r) for r in fresh]
            rt_bars.extend(bars)
            day_df = pd.DataFrame(rt_bars, columns=rta.BAR_COLUMNS)
            cache = _score_short_cache(engine, day_df, peer_today, today,
                                       short_model, short_pre, thresholds)
            backlog = not primed and len(fresh) > 1   # first poll carries the day so far
            if backlog:
                print(f"  replaying {len(fresh)} backlog bars (earlier signals, open→now; "
                      "fires below are NOT written to short_alert_log):")
            for bar in bars:
                long_alert = engine.on_new_bar(bar)
                _process_minute(engine, gate, bar, cache.get(bar["bar_end"]), ledger,
                                long_alert=long_alert, alerts_only=alerts_only,
                                persist_db=not backlog)
            if backlog:
                print("  --- caught up; now streaming one signal per minute ---")
        primed = True
        time.sleep(poll_sec)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _float_or_none(text):
    if str(text).strip().lower() in ("none", "off"):
        return None
    return float(text)


def _int_or_none(text):
    if str(text).strip().lower() in ("none", "off"):
        return None
    return int(text)


def parse_args():
    thr = se.ShortThresholds()
    p = argparse.ArgumentParser(
        description="Chartless realtime+replay DOWNTREND trend-alert runner "
                    "(Step 15; decision-support only).")
    p.add_argument("--code", default=None, help=f"6-digit symbol (default {DEFAULT_CODE}).")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--replay-date", default=None,
                   help="replay mode: YYYY-MM-DD (default: latest bars_1m day for the symbol).")
    p.add_argument("--today", default=None,
                   help="live mode trading date (default: latest realtime_minute_bars day).")
    p.add_argument("--poll-sec", type=float, default=5.0, help="live mode poll interval.")
    p.add_argument("--cross-symbol", action="store_true",
                   help="REAL cross-symbol (universe) features for BOTH the long context and the "
                        "short gate; required for the bottom-rank knob (auto-disabled otherwise).")
    p.add_argument("--alerts-only", action="store_true",
                   help="print only 🔻/▲ fire bars (default: one status line every scored minute).")
    # long-context risk model (display only)
    p.add_argument("--risk-model-version", default="v1",
                   help="long-context risk model version (default v1; v2.x needs --cross-symbol).")
    p.add_argument("--model-path", default=None,
                   help="override the long-context model file (default: the version's path).")
    # short-adverse model (annotation on every alert; optional DOWN-type veto)
    p.add_argument("--short-model-path", default=DEFAULT_SHORT_MODEL_PATH,
                   help=f"v1_short model scoring short_adverse_prob (default {DEFAULT_SHORT_MODEL_PATH}).")
    p.add_argument("--no-short-model", action="store_true",
                   help="rule-only mode: short_adverse_prob=NaN on every row and any configured "
                        "model veto is disabled (short_adverse_prob_max=None).")
    # per-field ShortThresholds overrides (defaults = the mirrored baseline)
    p.add_argument("--thresholds-name", default=None,
                   help="tag recorded on fired alerts (default: the baseline name, or "
                        "'cli_override' when any knob below deviates).")
    p.add_argument("--down-threshold", type=float, default=thr.down_threshold,
                   help="down_score floor (fires need down_score >= this).")
    p.add_argument("--up-risk-limit", type=_float_or_none, default=thr.up_risk_limit,
                   help="up_score ceiling; 'none' disables.")
    p.add_argument("--score-margin", type=float, default=thr.score_margin,
                   help="fires need direction_score <= -margin.")
    p.add_argument("--bottom-rank-limit", type=_int_or_none, default=thr.bottom_rank_limit,
                   help="bottom-rank bucket size; 'none' disables "
                        "(auto-disabled without --cross-symbol).")
    p.add_argument("--require-below-ma5", action="store_true",
                   help="optional confirm: dist_ma5_intraday < 0.")
    p.add_argument("--require-rel-weakness", action="store_true",
                   help="optional confirm: relative_ret_20m_to_universe < 0.")
    p.add_argument("--short-adverse-prob-max", type=_float_or_none,
                   default=thr.short_adverse_prob_max,
                   help="optional model veto on the DOWN types (DOWN_ALERT/DOWN_CONTINUATION): "
                        "no fire when short_adverse_prob >= max (NaN is vetoed too). Default "
                        "None = annotation-only mode; pass a float to enable the veto.")
    p.add_argument("--daily-alert-limit", type=_int_or_none, default=thr.daily_alert_limit,
                   help="cap on fired alerts per day; 'none'/'off' (the default) = no cap, so the "
                        "per-type cooldown is the only throttle. Pass an int to cap.")
    p.add_argument("--cooldown-min", type=int, default=thr.cooldown_minutes,
                   help="per-type ShortFireGate cooldown minutes (keep 5-15; the only throttle "
                        "when --daily-alert-limit is off). The long engine keeps its own default.")
    p.add_argument("--min-time", default=thr.min_time, help="session window start (HH:MM:SS).")
    p.add_argument("--max-time", default=thr.max_time, help="session window end (HH:MM:SS).")
    return p.parse_args()


def build_thresholds(args):
    """ShortThresholds from the CLI overrides. The name tags explicit USER deviations
    only; forced adjustments (bottom-rank auto-disable without --cross-symbol,
    --no-short-model veto-off) are printed instead, so a default non-cross run still
    reads as the baseline combination."""
    base = se.ShortThresholds()
    overrides = dict(
        down_threshold=args.down_threshold,
        up_risk_limit=args.up_risk_limit,
        score_margin=args.score_margin,
        bottom_rank_limit=args.bottom_rank_limit,
        require_below_ma5=args.require_below_ma5,
        require_rel_weakness=args.require_rel_weakness,
        short_adverse_prob_max=args.short_adverse_prob_max,
        daily_alert_limit=args.daily_alert_limit,
        cooldown_minutes=args.cooldown_min,
        min_time=args.min_time,
        max_time=args.max_time,
    )
    name = args.thresholds_name
    if name is None:
        deviates = any(getattr(base, k) != v for k, v in overrides.items())
        name = "cli_override" if deviates else base.name
    if not args.cross_symbol and overrides["bottom_rank_limit"] is not None:
        print("  note: bottom_rank_limit auto-disabled — single-symbol serving fills NEUTRAL "
              "mid ranks (use --cross-symbol for real cross-sectional ranks).")
        overrides["bottom_rank_limit"] = None
    if args.no_short_model and overrides["short_adverse_prob_max"] is not None:
        print("  note: --no-short-model forces short_adverse_prob_max=None "
              "(rule-only, no veto input).")
        overrides["short_adverse_prob_max"] = None
    return se.ShortThresholds(name=name, **overrides)


def _thr_summary(thr):
    up = "-" if thr.up_risk_limit is None else f"{thr.up_risk_limit:g}"
    rank = "off" if thr.bottom_rank_limit is None else f"bottom{thr.bottom_rank_limit}"
    veto = "off" if thr.short_adverse_prob_max is None else f"prob<{thr.short_adverse_prob_max:g}"
    confirms = "".join(
        f" +{label}" for label, on in
        (("below_ma5", thr.require_below_ma5), ("rel_weakness", thr.require_rel_weakness)) if on)
    limit = "off" if thr.daily_alert_limit is None else f"{thr.daily_alert_limit}/day"
    return (f"{thr.name}: down>={thr.down_threshold:g} up<={up} margin>={thr.score_margin:g} "
            f"rank={rank} veto={veto}{confirms} cooldown={thr.cooldown_minutes}m "
            f"limit={limit} session={thr.min_time}-{thr.max_time}")


def main():
    args = parse_args()
    ensure_directories()
    symbols = load_active_symbols()
    symbol = normalize_symbol_code(args.code or DEFAULT_CODE)
    peers = [normalize_symbol_code(s) for s in symbols]
    if symbol not in set(peers):
        print(f"  note: {symbol} is not in the active symbols list; analysing it anyway.")

    version = args.risk_model_version
    vcfg = model_version_config(version)   # validates the version
    if vcfg["requires_cross_symbol"] and not args.cross_symbol:
        raise SystemExit(f"risk model {version} requires real cross-symbol features; add --cross-symbol.")
    if vcfg["requires_premarket"]:
        raise SystemExit(f"risk model {version} needs premarket features, which this runner does not serve.")
    long_model_path = args.model_path or str(resolve_model_path(version))
    long_model, long_pre, _long_meta = load_model(long_model_path)
    print(f"loaded long-context risk model: version={version} "
          f"type={getattr(long_model, 'model_type', '?')} features={len(long_pre.feature_columns)}")

    short_model = short_pre = None
    if args.no_short_model:
        print("  ⚠ rule-only mode (--no-short-model): short_adverse_prob=NaN on every row; "
              "model veto disabled.")
    else:
        try:
            short_model, short_pre, short_meta = load_model(args.short_model_path)
        except FileNotFoundError:
            raise SystemExit(
                f"short risk model not found at {args.short_model_path}; train it with "
                "scripts/train_risk_model.py --target target_up_risk --risk-model-version v1_short "
                "or run with --no-short-model (rule-only).")
        print(f"loaded short-adverse model: type={getattr(short_model, 'model_type', '?')} "
              f"target={short_meta.get('target', '?')} features={len(short_pre.feature_columns)}")

    thresholds = build_thresholds(args)
    gate = se.ShortFireGate(symbol, thresholds)

    fires_label = "🔻/▲ fires only" if args.alerts_only else "every scored minute + 🔻/▲ fires"
    print(f"down-trend alert runner: {symbol} | mode={args.mode} | "
          f"cross-symbol: {f'on (peers={len(peers)})' if args.cross_symbol else 'off'} | "
          f"feed: {fires_label}")
    print(f"  thresholds [{_thr_summary(thresholds)}]")
    print("  each line = one bar, labeled like the chart candle (bar start; the decision "
          "itself happens at that bar's close). leading markers:")
    print(f"    {_DOWN_MARK} = a DOWN-side alert fired (상승 위험/하락 추세 약화/하방 유지/하방 가능성/변동성 확대)"
          f"    {_up_mark()} = the LONG-side engine fired (상방 가능성 등)")
    print("    dn = down_score (하락 점수)   up = up_score (상승 점수)   "
          "dir = up-down 방향 점수 (음수일수록 하방 우위)")
    print("    bounce = upside_risk_score (반등/상승위험 점수)   "
          "adverse = short_adverse_prob[band] = 모델이 본 상승 역행(반등) 확률")
    print("    | long=... = the long engine's current classification (its own fire adds the ▲ marker).")
    print("  decision-support only — NO orders are sent.\n")

    engine = rta.TrendAlertEngine(symbol, long_model, long_pre, active_count=len(symbols),
                                  cooldown_min=rta.DEFAULT_COOLDOWN_MIN,
                                  thresholds=AlertThresholds(),
                                  cross_symbol=args.cross_symbol, peers=peers)
    # engine.enable_persistence is deliberately NOT called (see module docstring)

    ledger = []   # fired alerts (+ long context); flushed to the CSV ledger on stop
    if args.mode == "replay":
        date = args.replay_date
        if not date:
            conn = connect(DB_PATH)
            try:
                date = rta._latest_day(conn, "bars_1m", "bar_start", symbol)
            finally:
                conn.close()
        try:
            run_replay(engine, gate, short_model, short_pre, thresholds, ledger,
                       date=date, alerts_only=args.alerts_only)
        finally:
            _print_ledger(ledger)
    else:
        today = args.today
        if not today:
            conn = connect(DB_PATH)
            try:
                today = rta._latest_day(conn, "realtime_minute_bars", "ts_kst", symbol) or \
                    pd.Timestamp.now().strftime("%Y-%m-%d")
            finally:
                conn.close()
        try:
            run_live(engine, gate, short_model, short_pre, thresholds, ledger,
                     today=today, poll_sec=args.poll_sec, alerts_only=args.alerts_only)
        except KeyboardInterrupt:
            print("\n  stopped.")
        finally:
            _print_ledger(ledger)


if __name__ == "__main__":
    main()
