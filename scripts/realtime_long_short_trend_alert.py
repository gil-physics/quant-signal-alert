"""Step 15: unified LONG+SHORT realtime+replay trend-alert runner (64-bit).

Copy-derivative of scripts/realtime_short_trend_alert.py (which stays available
and unchanged): the whole short-side pipeline is identical, but the LONG uptrend
alerts from the EXISTING engine (scripts/realtime_trend_alert.py:
TrendAlertEngine, reused UNMODIFIED) are promoted from display-only context to
FIRST-CLASS fires on equal footing with the down-trend alerts.

How the long side is re-gated: the engine is instantiated with cooldown_min=0,
which bypasses its internal single-slot cooldown (``(t - last) < Timedelta(0)``
is never true), so ``engine._maybe_fire`` returns the fired alert dict for EVERY
session-gated (09:10-15:00) typed bar. Note ``_maybe_fire`` never reads the
``fires`` column that score_serving computes -- its only gates are the session
window, a non-empty alert_type, and the (bypassed) cooldown. The runner then
applies its OWN long-side cooldown policy (LongFireGate below, mirroring
ShortFireGate's policy switch) so BOTH sides share one cooldown policy:

  * --cooldown-scope per-type (default): the same alert type is suppressed
    within --cooldown-min while a different type fires immediately, each type
    on its own independent timer.
  * --cooldown-scope single-slot: the long engine's historical rule -- one slot
    remembering the last fired type+time; a different type fires immediately
    and REPLACES the slot, so the earlier type may re-fire before its own
    cooldown elapses.

Per finished 1m bar it prints ONE line, labeled with bar_start (chart-candle
convention, same as realtime_trend_signal.py; the decision happens at that
bar's close; ledger/DB keep decision_time). Leading markers: 🔻 = a DOWN-side
alert fired this bar, blue ▲ = a LONG-side alert fired this bar (both can
combine, 🔻▲). Every ALERT line ends with the score tail
``dn/up/dir/bounce/adverse`` taken from the short layer's scored row -- the
up/down/direction scores are the SAME rule_signal values both systems consume,
so one tail serves both sides (omitted while the short layer is warming up).

--strict raises the alert standards on BOTH sides (exact rules are printed in
the banner):
  * short: stricter ShortThresholds (down_threshold +1, score_margin +1) gate
    the DOWN types; the informational types get a strict filter -- SHORT_RISK
    needs upside_risk_score>=5 & direction_score>=+4, DOWN_WEAKENING needs
    down_score>=4, VOLATILITY needs volume_z_20m_1m>=3.0 (the emitted short
    schema carries volume_z_20m_1m but NOT range_1m, so volume_z is the only
    strict volatility filter).
  * long: since _maybe_fire ignores the 'fires' column (verified above), the
    engine's fired alerts are post-filtered on engine.latest fields -- grade C
    suppressed for all types; UP types additionally need up_score>=5 &
    direction_score>=+4; RISK_ALERT needs risk_score>=4 & direction_score<=-3;
    WEAKENING_ALERT needs up_score>=4; VOLATILITY_ALERT has only the C-grade
    rule (engine rows carry no volume_z).
Strict filters run BEFORE the stateful gates, so a strict-suppressed bar never
consumes cooldown or daily-limit state.

Decision-support ONLY: it prints, writes ONE CSV ledger
(data/processed/long_short_trend_ledger.csv, leading "side" column with
"down"/"long" rows) on stop, and (live mode) logs fired DOWN alerts to the
short_alert_log table exactly like realtime_short_trend_alert.py. LONG fires
are console+CSV ONLY: this runner NEVER writes realtime_alert_log -- that table
belongs to a concurrently running long runner (realtime_trend_alert.py /
realtime_trend_signal.py), and engine.enable_persistence is deliberately never
called. It NEVER places, cancels, or transmits any order.

Scoring path (batch-per-day; rows are causal so the batch row at decision_time t
equals what an incremental rebuild at t would produce -- decision_time == bar_end):
  * replay : seed engine history -> build the day's serving frame ONCE
             (build_serving_features / build_serving_features_cross) ->
             score_serving_short -> cache keyed by bar["bar_end"] -> stream bars
             through engine.on_new_bar + both fire gates. Nothing is written to
             short_alert_log (CSV ledger + console summary only).
  * live   : poll realtime_minute_bars (32-bit collector), serve the regular
             session only (drop premarket bars before REGULAR_OPEN), rebuild the
             day short frame per poll, stream fresh bars. The startup backlog is
             replayed with printing but NOT persisted to short_alert_log.

Run (cwd = quant, 64-bit; the default symbol comes from config.yaml realtime.default_symbol):
    ..\\Python311-64\\python.exe scripts\\realtime_long_short_trend_alert.py --mode replay --cross-symbol
    ..\\Python311-64\\python.exe scripts\\realtime_long_short_trend_alert.py --mode replay --cooldown-scope single-slot
    ..\\Python311-64\\python.exe scripts\\realtime_long_short_trend_alert.py --mode live --strict
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

# reuse the existing trend/risk engine for the LONG side -- never duplicated here
import realtime_trend_alert as rta
import short_entry as se
from config import DB_PATH, PROCESSED_DIR, connect, ensure_directories
from directional_alert import AlertThresholds, UP_TYPES
from realtime_features import build_serving_features, build_serving_features_cross
from risk_model import load_model, model_version_config, resolve_model_path
from settings import load_active_symbols, normalize_symbol_code, realtime_default_symbol

LEDGER_PATH = PROCESSED_DIR / "long_short_trend_ledger.csv"
DEFAULT_CODE = realtime_default_symbol()
DEFAULT_SHORT_MODEL_PATH = str(resolve_model_path("v1_short"))

# CSV ledger schema: ONE file for both sides, leading "side" column ("down"/"long").
# "down" rows carry the ShortFireGate dict + long context (same content as the short
# runner's ledger); "long" rows fill time/symbol/close/alert_type (the LONG type),
# long_alert_grade, long_risk_prob and message, leaving the short-only fields empty.
LEDGER_COLUMNS = [
    "side",
    "time", "symbol", "close", "alert_type", "short_grade",
    "up_score", "down_score", "direction_score", "short_score", "upside_risk_score",
    "short_adverse_prob", "short_risk_band",
    "long_alert_type", "long_alert_grade", "long_risk_prob",
    "reason_1", "reason_2", "reason_3", "thresholds_name", "message",
]

# DB table for LIVE fired DOWN alerts (best-effort; the CSV ledger is the source of
# truth). LONG fires are console+CSV only -- realtime_alert_log belongs to the long
# runners and is never written here.
_SHORT_ALERT_LOG_SCHEMA = [
    ("ts", "TEXT"), ("symbol", "TEXT"), ("alert_type", "TEXT"), ("short_grade", "TEXT"),
    ("up_score", "REAL"), ("down_score", "REAL"), ("direction_score", "REAL"),
    ("short_score", "REAL"), ("upside_risk_score", "REAL"),
    ("short_adverse_prob", "REAL"), ("short_risk_band", "TEXT"),
    ("long_alert_type", "TEXT"), ("long_alert_grade", "TEXT"), ("long_risk_prob", "REAL"),
    ("close", "REAL"), ("reason_1", "TEXT"), ("reason_2", "TEXT"), ("reason_3", "TEXT"),
    ("thresholds_name", "TEXT"), ("message", "TEXT"), ("created_at", "TEXT"),
]

# --- FEATURE 3 (--strict) rule constants -------------------------------------- #
# short side: informational-type strict filter (+1 over the module fire constants)
STRICT_SHORT_RISK_FIRE_SCORE = se.SHORT_RISK_FIRE_SCORE + 1.0   # upside_risk_score >= 5.0
STRICT_SHORT_RISK_FIRE_DIR = se.SHORT_RISK_FIRE_DIR + 1.0       # direction_score >= +4.0
STRICT_WEAK_FIRE_DOWN = se.WEAK_FIRE_DOWN + 1.0                 # down_score >= 4.0
STRICT_VOL_FIRE_Z = 3.0     # volume_z_20m_1m >= 3.0 (baseline gate: 2.5 AND range>=0.006;
                            # SHORT_RESULT_COLUMNS has no range_1m, so volume_z only)
# long side: mirror numbers, applied to engine.latest fields (see module docstring)
STRICT_LONG_UP_SCORE = 5.0      # UP types: up_score >= 5.0 ...
STRICT_LONG_UP_DIR = 4.0        # ... and direction_score >= +4.0
STRICT_LONG_RISK_SCORE = 4.0    # RISK_ALERT: risk_score >= 4.0 ...
STRICT_LONG_RISK_DIR = -3.0     # ... and direction_score <= -3.0
STRICT_LONG_WEAK_UP = 4.0       # WEAKENING_ALERT: up_score >= 4.0


# --------------------------------------------------------------------------- #
# Long-side fire gate (the runner's own cooldown policy; the engine's is bypassed)
# --------------------------------------------------------------------------- #
class LongFireGate:
    """Runner-owned cooldown gate for the long engine's fires. The engine itself
    runs with cooldown_min=0 (bypass), so every session-gated typed bar reaches
    this gate and the cooldown policy is decided HERE, with the same policy
    switch as ShortFireGate:

    * ``per_type`` (default): each alert type keeps its own independent timer.
    * ``single_slot``: the engine's historical rule -- one slot with the last
      fired type+time; a different type fires immediately and replaces the slot.

    State is per calendar day (auto-reset on date roll) and fully separate from
    the ShortFireGate's state.
    """

    def __init__(self, cooldown_minutes, *, cooldown_scope="per_type"):
        if cooldown_scope not in se.ShortFireGate.COOLDOWN_SCOPES:
            raise ValueError(f"cooldown_scope must be one of "
                             f"{se.ShortFireGate.COOLDOWN_SCOPES}, got {cooldown_scope!r}")
        self.cooldown_scope = cooldown_scope
        self._cooldown = pd.Timedelta(minutes=cooldown_minutes)
        self._day = None
        self._last_fire_at = {}   # per_type: one entry per type; single_slot: at most ONE

    def allow(self, alert_type, decision_time):
        """True when a long fire of ``alert_type`` at ``decision_time`` passes the
        cooldown policy (and record it); False to suppress."""
        dt = str(decision_time)
        if dt[:10] != self._day:
            self._day = dt[:10]
            self._last_fire_at = {}
        t = pd.Timestamp(dt)
        last = self._last_fire_at.get(alert_type)
        if last is not None and (t - last) < self._cooldown:
            return False
        if self.cooldown_scope == "single_slot":
            self._last_fire_at = {alert_type: t}
        else:
            self._last_fire_at[alert_type] = t
        return True


# --------------------------------------------------------------------------- #
# --strict filters (run BEFORE the stateful gates: a suppressed bar consumes no
# cooldown/daily-limit state)
# --------------------------------------------------------------------------- #
def _strict_short_ok(srow):
    """Strict filter for the short layer's informational types, evaluated on the
    scored row (the same fields the fired alert would carry, plus volume_z).
    DOWN types pass through -- they are already raised via the strict
    ShortThresholds (down_threshold +1, score_margin +1)."""
    typ = str(srow["alert_type"] or "")
    if typ == "SHORT_RISK_ALERT":
        return (float(srow["upside_risk_score"]) >= STRICT_SHORT_RISK_FIRE_SCORE
                and float(srow["direction_score"]) >= STRICT_SHORT_RISK_FIRE_DIR)
    if typ == "DOWN_WEAKENING_ALERT":
        return float(srow["down_score"]) >= STRICT_WEAK_FIRE_DOWN
    if typ == "VOLATILITY_ALERT":
        return float(srow["volume_z_20m_1m"]) >= STRICT_VOL_FIRE_Z
    return True


def _strict_long_ok(row):
    """Strict filter for the long engine's fires, on engine.latest (RESULT_COLUMNS:
    alert_type/alert_grade/up_score/risk_score/direction_score...). _maybe_fire
    ignores the 'fires' column, so this post-filter is where the strict long
    standards live. Grade C is suppressed for every type; VOLATILITY_ALERT has
    only that rule (engine rows carry no volume_z)."""
    if row is None:
        return False
    if str(row["alert_grade"]) == "C":
        return False
    typ = str(row["alert_type"])
    if typ in UP_TYPES:
        return (float(row["up_score"]) >= STRICT_LONG_UP_SCORE
                and float(row["direction_score"]) >= STRICT_LONG_UP_DIR)
    if typ == "RISK_ALERT":
        return (float(row["risk_score"]) >= STRICT_LONG_RISK_SCORE
                and float(row["direction_score"]) <= STRICT_LONG_RISK_DIR)
    if typ == "WEAKENING_ALERT":
        return float(row["up_score"]) >= STRICT_LONG_WEAK_UP
    return True


# --------------------------------------------------------------------------- #
# Long context from the reused engine
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


def _fmt_short_ctx(srow):
    """Compact short-side context for a LONG-only fire line: short=TYPE(grade)."""
    if srow is None:
        return "short=(warming up)"
    typ = str(srow["alert_type"] or "-")
    return f"short={typ}({srow['short_grade']})"


# line-leading direction markers: 🔻 = a DOWN-side alert fired this bar, blue ▲ = a
# LONG-side alert fired this bar (first-class here). Both can show together (🔻▲).
_DOWN_MARK = "🔻"
_BLUE, _RESET = "\033[94m", "\033[0m"


def _up_mark():
    """Blue up-arrow for a long-side fire; plain ▲ when stdout is not a TTY (so a
    redirected log stays free of ANSI escapes)."""
    return f"{_BLUE}▲{_RESET}" if sys.stdout.isatty() else "▲"


def _scores_tail(srow):
    """FEATURE 2: the score tail appended to every ALERT line (empty during the
    short layer's warm-up). up/down/direction come from the shared rule_signals,
    so this one tail serves BOTH sides; bounce/adverse are the short layer's."""
    if srow is None:
        return ""
    p = float(srow["short_adverse_prob"])
    return (f"dn={float(srow['down_score']):.1f} up={float(srow['up_score']):.1f} "
            f"dir={float(srow['direction_score']):+.1f} "
            f"bounce={float(srow['upside_risk_score']):.1f} "
            f"adverse={p:.2f}[{se.short_risk_band(p)}]")


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
# Persistence (live DOWN fires -> short_alert_log; all fires -> CSV ledger on stop)
# --------------------------------------------------------------------------- #
def _persist_alert(ledger_row):
    """Best-effort insert of one FIRED DOWN alert into short_alert_log (live mode
    only; long fires never reach this function). A transient lock (collector
    mid-write) must not crash the feed -- the alert is still printed and
    CSV-ledgered on stop."""
    payload = dict(ledger_row)
    payload.pop("side", None)
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
    """CSV ledger + console summary on stop: per-side counts + last alerts."""
    if not ledger:
        print("\n  (no trend alerts fired on either side)")
        return
    led = pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
    led.to_csv(LEDGER_PATH, index=False, encoding="utf-8-sig")
    down = led[led["side"] == "down"]
    lng = led[led["side"] == "long"]
    print(f"\n  fired {len(down)} down-side (🔻) + {len(lng)} long-side (▲) alerts")
    if len(down):
        print(f"    down: types={down['alert_type'].value_counts().to_dict()}  "
              f"grades={down['short_grade'].value_counts().to_dict()}  "
              f"bands={down['short_risk_band'].value_counts().to_dict()}")
    if len(lng):
        print(f"    long: types={lng['alert_type'].value_counts().to_dict()}  "
              f"grades={lng['long_alert_grade'].value_counts().to_dict()}")
    print("  last alerts:")
    for a in ledger[-8:]:
        if a["side"] == "down":
            print(f"    🔻 {a['time'][11:16]} {str(a['alert_type']):<21} {str(a['short_grade']):<10} "
                  f"short_adverse_prob={float(a['short_adverse_prob']):.2f}[{a['short_risk_band']}] "
                  f"close={a['close']:,.0f}")
        else:
            print(f"    ▲ {a['time'][11:16]} {str(a['alert_type']):<21} "
                  f"{str(a['long_alert_grade']):<10} "
                  f"risk_prob={float(a['long_risk_prob']):.2f} close={a['close']:,.0f}")
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


def _process_minute(engine, gate, long_gate, bar, srow, ledger, *, long_alert_raw,
                    alerts_only, persist_db, strict):
    """Emit ONE line for this finished bar and ledger the fires of BOTH sides.

    A single line carries everything for the bar, labeled with bar_start -- the chart
    candle's label, the same convention as realtime_trend_signal.py -- so the printed
    prices line up with the HTS minute chart and the two feeds never drift apart.
    (The DECISION still happens at that bar's close: gates run on decision_time ==
    bar_end, and the ledger/DB keep decision_time.)

    Line formats (leading markers: 🔻 DOWN-side fire, blue ▲ LONG-side fire):
      short fire only:  "🔻 HH:MM SYM  close=...  {short message}  | long=...  ‖ {scores}"
      long fire only:   "▲ HH:MM SYM  close=...  {long message}  | short=TYPE(grade)  ‖ {scores}"
      both fire:        "🔻▲ HH:MM SYM  close=...  {short message} ‖ {long message}  ‖ {scores}"
      no fire:          the compact status line (unchanged from the short runner)
    {scores} is _scores_tail(srow), omitted while srow is None (short warm-up).

    ``srow`` is the cached score row for bar["bar_end"] (None while warming up);
    ``long_alert_raw`` is the engine's cooldown-bypassed fire this bar (None
    otherwise) -- it becomes a real long fire only after the strict filter (when
    --strict) and the LongFireGate cooldown. There is NO separate echo line for a
    long fire: the engine's own message rides in this bar line. Returns the fired
    down alert dict or None."""
    alert = None
    if srow is not None and (not strict or _strict_short_ok(srow)):
        alert = gate.maybe_fire(srow, short_adverse_prob=srow["short_adverse_prob"])
    long_alert = None
    if long_alert_raw is not None and (not strict or _strict_long_ok(engine.latest)):
        if long_gate.allow(long_alert_raw["alert_type"], long_alert_raw["time"]):
            long_alert = long_alert_raw
    ctx = _long_context(engine)
    down_fired = alert is not None
    long_fired = long_alert is not None
    # alerts-only: show only bars where something fired; otherwise skip pure warm-up
    if alerts_only and not (down_fired or long_fired):
        return alert
    if not down_fired and not long_fired and srow is None and ctx is None:
        return alert

    hhmm = str(bar["bar_start"])[11:16]   # chart-candle label (decision at this bar's close)
    head = f"{hhmm} {engine.symbol}  close={float(bar['close']):,.0f}"
    tail = _scores_tail(srow)
    tail = f"  ‖ {tail}" if tail else ""
    if down_fired and long_fired:
        print(f"{_DOWN_MARK}{_up_mark()} {head}  {alert['message']} ‖ {long_alert['message']}{tail}")
    elif down_fired:
        print(f"{_DOWN_MARK} {head}  {alert['message']}  | {_fmt_long(ctx)}{tail}")
    elif long_fired:
        print(f"{_up_mark()} {head}  {long_alert['message']}  | {_fmt_short_ctx(srow)}{tail}")
    else:
        body, note = _status_body(srow)
        print(f"   {head}  {body}  | {_fmt_long(ctx)}{note}")

    if down_fired:
        row = {"side": "down", **alert, **(ctx if ctx is not None else _EMPTY_LONG)}
        ledger.append(row)
        if persist_db:
            _persist_alert(row)
    if long_fired:
        # long fires are console+CSV only (realtime_alert_log belongs to the long runners)
        ledger.append({
            "side": "long", "time": long_alert["time"], "symbol": long_alert["symbol"],
            "close": float(long_alert["close"]), "alert_type": long_alert["alert_type"],
            "long_alert_grade": long_alert["alert_grade"],
            "long_risk_prob": float(long_alert["risk_prob"]),
            "message": long_alert["message"],
        })
    return alert


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_replay(engine, gate, long_gate, short_model, short_pre, thresholds, ledger, *,
               date, alerts_only, strict):
    """Stream a historical day: engine per bar (long fires) + one-shot short cache."""
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
            engine.precompute_cross(peer_today, date)   # long-side cache
    finally:
        conn.close()
    if day.empty:
        raise SystemExit(f"no bars_1m for {engine.symbol} on {date} (source={engine.source}).")

    cache = _score_short_cache(engine, day, peer_today, date, short_model, short_pre, thresholds)
    print(f"replay-long-short {engine.symbol} {date}: {len(day)} bars, "
          f"{len(cache)} scorable minutes "
          f"(printing {'🔻/▲ fires only' if alerts_only else 'every scored minute + 🔻/▲ fires'})\n")
    for bar in day.to_dict("records"):
        long_alert_raw = engine.on_new_bar(bar)   # cooldown-bypassed; re-gated in _process_minute
        engine.fired.clear()   # cooldown 0 appends every typed bar; nothing here reads it
        _process_minute(engine, gate, long_gate, bar, cache.get(bar["bar_end"]), ledger,
                        long_alert_raw=long_alert_raw, alerts_only=alerts_only,
                        persist_db=False, strict=strict)


def run_live(engine, gate, long_gate, short_model, short_pre, thresholds, ledger, *,
             today, poll_sec, alerts_only, strict):
    """Poll realtime_minute_bars (regular session only) and stream fresh bars. The
    startup backlog is replayed with printing but not persisted to short_alert_log."""
    print(f"live-long-short {engine.symbol}: polling realtime_minute_bars every {poll_sec}s "
          f"(Ctrl+C to stop). Run the 32-bit collector to populate it.")
    print("  fired down-trend alerts -> DB table short_alert_log (+ CSV ledger on stop); "
          "backlog fires are printed + CSV-ledgered only.")
    print("  fired LONG alerts -> console + CSV ledger ONLY (realtime_alert_log is never "
          "written; it belongs to a concurrently running long runner).\n")
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
                engine.precompute_cross(peer_today, today)   # long-side cache
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
                long_alert_raw = engine.on_new_bar(bar)
                engine.fired.clear()   # cooldown 0 appends every typed bar; nothing here reads it
                _process_minute(engine, gate, long_gate, bar, cache.get(bar["bar_end"]), ledger,
                                long_alert_raw=long_alert_raw, alerts_only=alerts_only,
                                persist_db=not backlog, strict=strict)
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
        description="Unified LONG+SHORT realtime+replay trend-alert runner "
                    "(Step 15; decision-support only).")
    p.add_argument("--code", default=None, help=f"6-digit symbol (default {DEFAULT_CODE}).")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--replay-date", default=None,
                   help="replay mode: YYYY-MM-DD (default: latest bars_1m day for the symbol).")
    p.add_argument("--today", default=None,
                   help="live mode trading date (default: latest realtime_minute_bars day).")
    p.add_argument("--poll-sec", type=float, default=5.0, help="live mode poll interval.")
    p.add_argument("--cross-symbol", action="store_true",
                   help="REAL cross-symbol (universe) features for BOTH the long side and the "
                        "short gate; required for the bottom-rank knob (auto-disabled otherwise).")
    p.add_argument("--alerts-only", action="store_true",
                   help="print only 🔻/▲ fire bars (default: one status line every scored minute).")
    # FEATURE 1: one cooldown policy shared by BOTH sides
    p.add_argument("--cooldown-scope", choices=["per-type", "single-slot"], default="per-type",
                   help="cooldown policy for BOTH the short gate and the long gate. "
                        "per-type (default) = 같은 알림만 15분 억제(타입별 독립); "
                        "single-slot = realtime_trend_alert.py의 규칙"
                        "(다른 알림이 울리면 15분 전이라도 재울림).")
    p.add_argument("--cooldown-min", type=int, default=thr.cooldown_minutes,
                   help="cooldown minutes applied to BOTH sides (short gate and long gate; "
                        "default 15). The engine's own cooldown stays bypassed (0).")
    # FEATURE 3: stricter alert standards on BOTH sides
    p.add_argument("--strict", action="store_true",
                   help="raise the alert standards on BOTH sides: short DOWN types via "
                        "down_threshold+1/score_margin+1, short informational types via "
                        "bounce>=5&dir>=+4 / dn>=4 / volz>=3.0; long side suppresses grade C "
                        "and requires up>=5&dir>=+4 (UP types), risk>=4&dir<=-3 (RISK), "
                        "up>=4 (WEAKENING). Exact rules are printed in the banner.")
    # long-side risk model (annotates the long alerts; same engine as the long runners)
    p.add_argument("--risk-model-version", default="v1",
                   help="long-side risk model version (default v1; v2.x needs --cross-symbol).")
    p.add_argument("--model-path", default=None,
                   help="override the long-side model file (default: the version's path).")
    # short-adverse model (annotation on every alert; optional DOWN-type veto)
    p.add_argument("--short-model-path", default=DEFAULT_SHORT_MODEL_PATH,
                   help=f"v1_short model scoring short_adverse_prob (default {DEFAULT_SHORT_MODEL_PATH}).")
    p.add_argument("--no-short-model", action="store_true",
                   help="rule-only mode: short_adverse_prob=NaN on every row and any configured "
                        "model veto is disabled (short_adverse_prob_max=None).")
    # per-field ShortThresholds overrides (defaults = the mirrored baseline)
    p.add_argument("--thresholds-name", default=None,
                   help="tag recorded on fired alerts (default: the baseline name, or "
                        "'cli_override' when any knob below deviates; --strict appends '_strict').")
    p.add_argument("--down-threshold", type=float, default=thr.down_threshold,
                   help="down_score floor (fires need down_score >= this; --strict adds +1).")
    p.add_argument("--up-risk-limit", type=_float_or_none, default=thr.up_risk_limit,
                   help="up_score ceiling; 'none' disables.")
    p.add_argument("--score-margin", type=float, default=thr.score_margin,
                   help="fires need direction_score <= -margin (--strict adds +1).")
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
                   help="cap on fired SHORT-side alerts per day; 'none'/'off' (the default) = no "
                        "cap, so the cooldown is the only throttle. Pass an int to cap.")
    p.add_argument("--min-time", default=thr.min_time, help="session window start (HH:MM:SS).")
    p.add_argument("--max-time", default=thr.max_time, help="session window end (HH:MM:SS).")
    return p.parse_args()


def build_thresholds(args):
    """ShortThresholds from the CLI overrides. The name tags explicit USER deviations
    only; forced adjustments (bottom-rank auto-disable without --cross-symbol,
    --no-short-model veto-off) are printed instead, so a default non-cross run still
    reads as the baseline combination. --strict bumps down_threshold/score_margin by
    +1 AFTER the deviation check and appends '_strict' to the provenance tag (e.g.
    mirror_baseline_strict)."""
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
    if args.strict:
        overrides["down_threshold"] = float(overrides["down_threshold"]) + 1.0
        overrides["score_margin"] = float(overrides["score_margin"]) + 1.0
        name = f"{name}_strict"
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
    print(f"loaded long-side risk model: version={version} "
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
    scope = args.cooldown_scope.replace("-", "_")   # per-type/single-slot -> gate scope names
    gate = se.ShortFireGate(symbol, thresholds, cooldown_scope=scope)
    long_gate = LongFireGate(args.cooldown_min, cooldown_scope=scope)

    fires_label = "🔻/▲ fires only" if args.alerts_only else "every scored minute + 🔻/▲ fires"
    print(f"long+short trend alert runner: {symbol} | mode={args.mode} | "
          f"cross-symbol: {f'on (peers={len(peers)})' if args.cross_symbol else 'off'} | "
          f"feed: {fires_label}")
    print(f"  short thresholds [{_thr_summary(thresholds)}]")
    print(f"  cooldown: scope={args.cooldown_scope} min={args.cooldown_min} on BOTH sides "
          "(per-type = 같은 알림만 억제, 타입별 독립 / single-slot = 다른 알림이 울리면 "
          "15분 전이라도 재울림 — realtime_trend_alert.py의 규칙); "
          "the long engine's own cooldown is bypassed (0) and re-gated here.")
    if args.strict:
        print("  strict: ON — 강화된 알림 기준")
        print(f"    short: DOWN types via thresholds (down>={thresholds.down_threshold:g}, "
              f"dir<=-{thresholds.score_margin:g}); SHORT_RISK bounce>="
              f"{STRICT_SHORT_RISK_FIRE_SCORE:g} & dir>=+{STRICT_SHORT_RISK_FIRE_DIR:g}; "
              f"WEAKENING dn>={STRICT_WEAK_FIRE_DOWN:g}; VOLATILITY volz>={STRICT_VOL_FIRE_Z:g} "
              "(no range_1m in the emitted schema — volume_z only)")
        print(f"    long : grade C suppressed; UP types up>={STRICT_LONG_UP_SCORE:g} & "
              f"dir>=+{STRICT_LONG_UP_DIR:g}; RISK risk>={STRICT_LONG_RISK_SCORE:g} & "
              f"dir<={STRICT_LONG_RISK_DIR:g}; WEAKENING up>={STRICT_LONG_WEAK_UP:g} "
              "(VOLATILITY: C-grade rule only)")
    print("  each line = one bar, labeled like the chart candle (bar start; the decision "
          "itself happens at that bar's close). leading markers:")
    print(f"    {_DOWN_MARK} = a DOWN-side alert fired (상승 위험/하락 추세 약화/하방 유지/하방 가능성/변동성 확대)")
    print(f"    {_up_mark()} = a LONG-side alert fired (상방 가능성 등) — FIRST-CLASS here, "
          "not just context; both markers can combine (🔻▲)")
    print("    dn = down_score (하락 점수)   up = up_score (상승 점수)   "
          "dir = up-down 방향 점수 (음수일수록 하방 우위)")
    print("    bounce = upside_risk_score (반등/상승위험 점수)   "
          "adverse = short_adverse_prob[band] = 모델이 본 상승 역행(반등) 확률")
    print("    every ALERT line ends with the ‖ dn/up/dir/bounce/adverse score tail "
          "(shared rule_signal values, one tail for both sides).")
    print("  decision-support only — NO orders are sent.\n")

    # cooldown_min=0 BYPASSES the engine's internal single-slot cooldown so every
    # session-gated typed bar surfaces; LongFireGate above re-applies the shared policy.
    engine = rta.TrendAlertEngine(symbol, long_model, long_pre, active_count=len(symbols),
                                  cooldown_min=0,
                                  thresholds=AlertThresholds(),
                                  cross_symbol=args.cross_symbol, peers=peers)
    # engine.enable_persistence is deliberately NOT called (see module docstring:
    # realtime_alert_log belongs to the long runners; long fires stay console+CSV)

    ledger = []   # fired alerts of BOTH sides; flushed to the CSV ledger on stop
    if args.mode == "replay":
        date = args.replay_date
        if not date:
            conn = connect(DB_PATH)
            try:
                date = rta._latest_day(conn, "bars_1m", "bar_start", symbol)
            finally:
                conn.close()
        try:
            run_replay(engine, gate, long_gate, short_model, short_pre, thresholds, ledger,
                       date=date, alerts_only=args.alerts_only, strict=args.strict)
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
            run_live(engine, gate, long_gate, short_model, short_pre, thresholds, ledger,
                     today=today, poll_sec=args.poll_sec, alerts_only=args.alerts_only,
                     strict=args.strict)
        except KeyboardInterrupt:
            print("\n  stopped.")
        finally:
            _print_ledger(ledger)


if __name__ == "__main__":
    main()
