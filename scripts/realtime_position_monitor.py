"""Position EXIT-WARNING monitor for an already-open MANUAL position (long-only v1).

Companion to scripts/realtime_trend_signal.py. That feed answers "what is the
market/trend doing?"; THIS monitor answers "I already bought 000660 manually — should
I keep holding, or is a sell warning warranted?". It reuses the SAME real-time trend/
risk engine (scripts/realtime_trend_alert.py: TrendAlertEngine) for the trend/risk
snapshot and layers src/position_exit.py's exit rules (hard stop, profit giveback,
trailing drawdown, MA5/VWAP break, risk + weakening, opposite signal, time stop) on
top of the position's price excursions since entry.

This is decision-support ONLY. It prints / logs HOLD / WEAK_SELL_WARNING /
SELL_WARNING / STRONG_SELL_WARNING. It NEVER places, cancels, or sends any order
(no order API is imported or called).

Run (cwd = quant, 64-bit), after you manually entered a position:
    ..\\Python311-64\\python.exe scripts\\realtime_position_monitor.py --mode live \\
        --code 000660 --side long --entry-price 123000 --entry-time 10:14 --cross-symbol --all-bars
    ..\\Python311-64\\python.exe scripts\\realtime_position_monitor.py --mode replay \\
        --replay-date 2026-06-19 --code 000660 --side long --entry-price 123000 \\
        --entry-time 10:14 --cross-symbol --all-bars
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
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

# reuse the existing trend/risk engine — do NOT duplicate the feature/model pipeline
import realtime_trend_alert as rta
from config import DB_PATH, PROCESSED_DIR, connect, ensure_directories
from directional_alert import AlertThresholds
from position_exit import HOLD, ExitConfig, PositionState, evaluate_exit
from risk_model import load_model, model_version_config, resolve_model_path
from settings import load_active_symbols, normalize_symbol_code, realtime_default_symbol

LEDGER_PATH = PROCESSED_DIR / "realtime_position_ledger.csv"
DEFAULT_CODE = realtime_default_symbol()

LEDGER_COLUMNS = [
    "symbol", "side", "entry_time", "entry_price", "decision_time", "current_price",
    "current_return", "mfe", "mae", "drawdown_from_peak", "bars_since_entry",
    "exit_state", "exit_reason", "reason_details", "trend_state", "alert_type",
    "alert_grade", "risk_prob", "risk_bucket", "close", "ma5", "vwap",
    "above_ma5", "above_vwap",
]


# --------------------------------------------------------------------------- #
# Trend snapshot from the reused engine
# --------------------------------------------------------------------------- #
def _ma5(today, window=rta.MA_VIS):
    """The chart's 5-minute MA = mean of the last ``window`` completed 1m closes."""
    if today is None or len(today) < window:
        return None
    return float(today["close"].astype(float).iloc[-window:].mean())


def _session_vwap(today):
    """Intraday VWAP as of the latest bar (cumulative typical-price * volume)."""
    if today is None or len(today) == 0:
        return None
    h = today["high"].astype(float)
    l = today["low"].astype(float)
    c = today["close"].astype(float)
    v = today["volume"].astype(float).fillna(0.0)
    vol = float(v.sum())
    if vol <= 0:
        return float(c.iloc[-1])
    tp = (h + l + c) / 3.0
    return float((tp * v).sum() / vol)


def derive_trend_state(alert_type, direction_score, above_ma5, above_vwap):
    """Map the engine's alert vocabulary + structure to the position monitor's
    trend_state ("UP"/"RISK"/"WEAKENING"/"CHOPPY"/"DOWN"). The engine has no explicit
    DOWN alert, so a clean downtrend is inferred from broken structure + a down lean.
    """
    if alert_type in ("UP_ALERT", "UP_CONTINUATION"):
        return "UP"
    if alert_type == "RISK_ALERT":
        return "RISK"
    if alert_type == "WEAKENING_ALERT":
        return "WEAKENING"
    if alert_type == "VOLATILITY_ALERT":
        return "CHOPPY"
    # no directional alert -> infer from MA5/VWAP structure and the directional lean
    below_both = (above_ma5 is False) and (above_vwap is False)
    lean = direction_score if isinstance(direction_score, (int, float)) else None
    if below_both and (lean is not None and lean < 0):
        return "DOWN"
    if (above_ma5 and above_vwap) and (lean is None or lean >= 0):
        return "UP"
    return "CHOPPY"


def build_trend_snapshot(engine, bar):
    """Build the trend_snapshot dict for evaluate_exit from the reused engine's latest
    scored row plus MA5/VWAP computed off the same bar history."""
    row = engine.latest
    today = engine.today
    close = float(bar["close"])
    ma5 = _ma5(today)
    vwap = _session_vwap(today)
    above_ma5 = (close > ma5) if ma5 is not None else None
    above_vwap = (close > vwap) if vwap is not None else None

    alert_type = alert_grade = ""
    risk_prob = float("nan")
    direction_score = None
    if row is not None:
        alert_type = row.get("alert_type") or ""
        alert_grade = row.get("alert_grade") or ""
        rp = row.get("risk_prob")
        risk_prob = float(rp) if rp is not None else float("nan")
        ds = row.get("direction_score")
        direction_score = float(ds) if ds is not None and ds == ds else None

    return {
        "trend_state": derive_trend_state(alert_type, direction_score, above_ma5, above_vwap),
        "alert_type": alert_type,
        "alert_grade": alert_grade,
        "risk_prob": risk_prob,
        "risk_bucket": rta.risk_band(risk_prob),
        "close": close,
        "ma5": ma5,
        "vwap": vwap,
        "above_ma5": above_ma5,
        "above_vwap": above_vwap,
    }


# --------------------------------------------------------------------------- #
# Position tracking
# --------------------------------------------------------------------------- #
class PositionTracker:
    """Tracks an open position's price excursions across finished bars since entry."""

    def __init__(self, symbol, side, entry_time: datetime, entry_price: float):
        self.symbol = symbol
        self.side = side
        self.entry_time = entry_time
        self.entry_price = float(entry_price)
        self.highest = float(entry_price)
        self.lowest = float(entry_price)
        self.bars = 0

    def update(self, bar) -> PositionState:
        hi = float(bar["high"])
        lo = float(bar["low"])
        close = float(bar["close"])
        self.highest = max(self.highest, hi)
        self.lowest = min(self.lowest, lo)
        self.bars += 1
        current_time = pd.Timestamp(bar["bar_end"]).to_pydatetime()
        return PositionState.from_prices(
            symbol=self.symbol, side=self.side, entry_time=self.entry_time,
            entry_price=self.entry_price, current_time=current_time, current_price=close,
            highest_price_since_entry=self.highest, lowest_price_since_entry=self.lowest,
            bars_since_entry=self.bars)


def _should_print(sig, prev_key, all_bars):
    """On-change display: with ``all_bars`` print every bar; otherwise print only when
    the EXIT STATE changes (HOLD -> WEAK -> SELL -> STRONG and back), so a run of bars
    in the same state collapses to its first line — the state is the actionable signal,
    and the reason that triggered it is still shown on that line. The first monitored
    bar always prints because ``prev_key`` starts as None. Returns (show?, new_key)."""
    key = sig.exit_state
    return (all_bars or key != prev_key), key


def _fmt_pct(x):
    return f"{x * 100:+.2f}%"


def format_line(pos: PositionState, sig, snap):
    t = pos.current_time.strftime("%H:%M")
    reason = f" reason={sig.exit_reason}" if sig.exit_state != HOLD else ""
    return (f"{t} {pos.symbol} {pos.side.upper()} {sig.exit_state}{reason} "
            f"ret={_fmt_pct(pos.current_return)} MFE={_fmt_pct(pos.mfe)} "
            f"MAE={_fmt_pct(pos.mae)} DD={_fmt_pct(pos.drawdown_from_peak)} "
            f"trend={snap.get('trend_state') or '-'} risk={snap.get('risk_bucket') or '-'}")


def ledger_row(pos: PositionState, sig, snap):
    return {
        "symbol": pos.symbol,
        "side": pos.side,
        "entry_time": pos.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "entry_price": pos.entry_price,
        "decision_time": pos.current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": pos.current_price,
        "current_return": pos.current_return,
        "mfe": pos.mfe,
        "mae": pos.mae,
        "drawdown_from_peak": pos.drawdown_from_peak,
        "bars_since_entry": pos.bars_since_entry,
        "exit_state": sig.exit_state,
        "exit_reason": sig.exit_reason,
        "reason_details": "|".join(sig.reason_details),
        "trend_state": snap.get("trend_state"),
        "alert_type": snap.get("alert_type"),
        "alert_grade": snap.get("alert_grade"),
        "risk_prob": snap.get("risk_prob"),
        "risk_bucket": snap.get("risk_bucket"),
        "close": snap.get("close"),
        "ma5": snap.get("ma5"),
        "vwap": snap.get("vwap"),
        "above_ma5": snap.get("above_ma5"),
        "above_vwap": snap.get("above_vwap"),
    }


# --------------------------------------------------------------------------- #
# Ledger persistence (CSV always; DB position_exit_log for warnings, best-effort)
# --------------------------------------------------------------------------- #
def write_ledger(ledger):
    if not ledger:
        print("\n  (no monitored bars — entry-time after the last bar?)")
        return
    df = pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
    df.to_csv(LEDGER_PATH, index=False, encoding="utf-8-sig")
    warns = df[df["exit_state"] != HOLD]
    print(f"\n  monitored {len(df)} bars; warnings={len(warns)} "
          f"{warns['exit_state'].value_counts().to_dict()}")
    print(f"  ledger -> {LEDGER_PATH}")


def persist_warnings(db_path, ledger):
    """Best-effort insert of WARNING rows into position_exit_log (CSV is the source of
    truth; never fail the run on a DB hiccup)."""
    rows = [r for r in ledger if r["exit_state"] != HOLD]
    if not rows:
        return
    created = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS position_exit_log ("
                + ", ".join(f"{c} TEXT" for c in LEDGER_COLUMNS) + ", created_at TEXT)")
            conn.executemany(
                "INSERT INTO position_exit_log (" + ", ".join(LEDGER_COLUMNS) + ", created_at) "
                "VALUES (" + ",".join("?" * (len(LEDGER_COLUMNS) + 1)) + ")",
                [tuple(str(r[c]) for c in LEDGER_COLUMNS) + (created,) for r in rows])
            conn.commit()
        finally:
            conn.close()
        print(f"  warnings -> DB position_exit_log ({len(rows)} rows)")
    except Exception as exc:   # noqa: BLE001 - logging-only tool, never crash on DB
        print(f"  (could not write position_exit_log: {exc}; CSV ledger still saved)")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_replay_monitor(engine, tracker, config, ledger, *, date, entry_hhmm, all_bars):
    """Stream a historical day; monitor the position from entry_hhmm onward, appending
    rows to ``ledger``. Feeds EVERY bar to the engine (so MA5/VWAP and the cross-section
    stay correct) but only evaluates the position at/after entry."""
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, date)
        day = pd.read_sql_query(
            f"SELECT {', '.join(rta.BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn, params=(engine.source, engine.symbol, date))
        if engine.cross_symbol:
            engine.precompute_cross(
                rta.load_universe_day(conn, engine.peers, engine.source, date), date)
    finally:
        conn.close()
    if day.empty:
        raise SystemExit(f"no bars_1m for {engine.symbol} on {date} (source={engine.source}).")

    print(f"replay-position {engine.symbol} {date}: {len(day)} bars; entry {entry_hhmm} "
          f"@ {tracker.entry_price:,.0f}  ({'every bar' if all_bars else 'on change'})\n")
    last_key = None
    for bar in day.to_dict("records"):
        engine.on_new_bar(bar)
        if bar["bar_start"][11:16] < entry_hhmm:
            continue
        pos = tracker.update(bar)
        snap = build_trend_snapshot(engine, bar)
        sig = evaluate_exit(pos, snap, config)
        ledger.append(ledger_row(pos, sig, snap))
        show, last_key = _should_print(sig, last_key, all_bars)
        if show:
            print(format_line(pos, sig, snap))
    return ledger


def run_live_monitor(engine, tracker, config, ledger, *, today, entry_hhmm, poll_sec, all_bars):
    """Poll realtime_minute_bars; monitor the position from entry_hhmm onward, appending
    rows to ``ledger`` (passed in so a Ctrl+C in the caller can still flush it). The
    initial backlog (bars already present at launch) is processed to seed tracking but
    printed only as its latest line; subsequent bars stream one per minute."""
    print(f"live-position {engine.symbol}: polling realtime_minute_bars every {poll_sec}s "
          f"(Ctrl+C to stop). Run the 32-bit collector to populate it.")
    print(f"  entry {entry_hhmm} @ {tracker.entry_price:,.0f}  decision-support only — NO orders are sent.\n")
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, today)
    finally:
        conn.close()
    seen = set()
    primed = False
    last_key = None   # last shown (state, reason); drives on-change printing across polls
    while True:
        conn = connect(DB_PATH)
        try:
            new = pd.read_sql_query(
                "SELECT symbol, ts_kst, open, high, low, close, volume FROM realtime_minute_bars "
                "WHERE symbol=? AND substr(ts_kst,1,10)=? ORDER BY ts_kst",
                conn, params=(engine.symbol, today))
        finally:
            conn.close()
        fresh = [r for r in new.to_dict("records") if r["ts_kst"] not in seen]
        for r in fresh:
            seen.add(r["ts_kst"])
        if engine.cross_symbol and fresh:   # refresh the universe cross-section cache
            conn = connect(DB_PATH)
            try:
                engine.precompute_cross(
                    rta.load_universe_day(conn, engine.peers, engine.source, today, realtime=True), today)
            finally:
                conn.close()

        last_line = None
        for r in fresh:
            bar = rta._bar_from_rt(r)
            engine.on_new_bar(bar)
            if bar["bar_start"][11:16] < entry_hhmm:
                continue
            pos = tracker.update(bar)
            snap = build_trend_snapshot(engine, bar)
            sig = evaluate_exit(pos, snap, config)
            ledger.append(ledger_row(pos, sig, snap))
            show, last_key = _should_print(sig, last_key, all_bars)
            last_line = format_line(pos, sig, snap)
            if primed and show:
                print(last_line)
        if not primed and fresh:
            if last_line is not None:   # show the current position state at launch
                print(last_line)
            print(f"  primed {len(ledger)} bars since entry; now streaming on change")
        primed = True
        time.sleep(poll_sec)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_entry_time(text):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(text.strip(), fmt)
            return f"{t.hour:02d}:{t.minute:02d}"
        except ValueError:
            continue
    raise SystemExit(f"--entry-time must be HH:MM (got {text!r})")


def parse_args():
    cfg = ExitConfig()
    p = argparse.ArgumentParser(description="Position exit-warning monitor for an open manual position (long-only).")
    p.add_argument("--mode", choices=["live", "replay"], default="replay")
    p.add_argument("--code", default=None, help=f"6-digit symbol (default {DEFAULT_CODE}).")
    p.add_argument("--side", choices=["long"], default="long", help="position side (long-only for now).")
    p.add_argument("--entry-price", type=float, required=True, help="your manual entry price.")
    p.add_argument("--entry-time", required=True, help="entry time HH:MM (on the trading date).")
    p.add_argument("--replay-date", default=None, help="replay mode: YYYY-MM-DD (default latest bars day).")
    p.add_argument("--today", default=None, help="live mode trading date (default latest realtime day).")
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--all-bars", action="store_true",
                   help="print one line per bar (default: only when the exit state/reason changes).")
    p.add_argument("--cross-symbol", action="store_true",
                   help="use REAL cross-symbol (universe) features for the trend/risk snapshot (v1.1).")
    p.add_argument("--risk-model-version", default="v1",
                   help="risk model version (selects default model path; v2.x needs --cross-symbol).")
    p.add_argument("--model-path", default=None, help="override the model file (default: version's path).")
    # exit-rule threshold overrides (defaults from ExitConfig)
    p.add_argument("--hard-stop", type=float, default=cfg.hard_stop)
    p.add_argument("--breakeven-mfe-trigger", type=float, default=cfg.breakeven_mfe_trigger)
    p.add_argument("--breakeven-floor", type=float, default=cfg.breakeven_floor)
    p.add_argument("--trailing-mfe-trigger", type=float, default=cfg.trailing_mfe_trigger)
    p.add_argument("--trailing-drawdown", type=float, default=cfg.trailing_drawdown)
    p.add_argument("--time-stop-minutes", type=int, default=cfg.time_stop_minutes)
    p.add_argument("--time-stop-min-mfe", type=float, default=cfg.time_stop_min_mfe)
    p.add_argument("--time-stop-floor", type=float, default=cfg.time_stop_floor)
    p.add_argument("--risk-high-threshold", type=float, default=cfg.risk_high_threshold)
    p.add_argument("--risk-med-threshold", type=float, default=cfg.risk_med_threshold)
    return p.parse_args()


def build_config(args) -> ExitConfig:
    return ExitConfig(
        hard_stop=args.hard_stop,
        breakeven_mfe_trigger=args.breakeven_mfe_trigger,
        breakeven_floor=args.breakeven_floor,
        trailing_mfe_trigger=args.trailing_mfe_trigger,
        trailing_drawdown=args.trailing_drawdown,
        time_stop_minutes=args.time_stop_minutes,
        time_stop_min_mfe=args.time_stop_min_mfe,
        time_stop_floor=args.time_stop_floor,
        risk_high_threshold=args.risk_high_threshold,
        risk_med_threshold=args.risk_med_threshold,
    )


def main():
    args = parse_args()
    ensure_directories()
    symbols = load_active_symbols()
    symbol = normalize_symbol_code(args.code or DEFAULT_CODE)
    peers = [normalize_symbol_code(s) for s in symbols]
    if symbol not in set(peers):
        print(f"  note: {symbol} is not in the active symbols list; monitoring it anyway.")

    version = args.risk_model_version
    vcfg = model_version_config(version)   # validates the version
    model_path = args.model_path or str(resolve_model_path(version))
    if vcfg["requires_cross_symbol"] and not args.cross_symbol:
        raise SystemExit(f"risk model {version} requires real cross-symbol features; add --cross-symbol.")

    model, pre, meta = load_model(model_path)
    config = build_config(args)
    entry_hhmm = _parse_entry_time(args.entry_time)

    print(f"loaded risk model: version={version} type={getattr(model,'model_type','?')} "
          f"features={len(pre.feature_columns)}")
    print(f"position monitor: {symbol} {args.side.upper()} entry={args.entry_price:,.0f} @ {entry_hhmm}  | "
          f"cross-symbol: {'on (peers=%d)' % len(peers) if args.cross_symbol else 'off'}  | "
          f"per-bar: {'every bar' if args.all_bars else 'on change'}")
    print("  exit rules: hard_stop / profit_giveback / trailing_drawdown / ma5_vwap_break / "
          "risk+weakening / opposite_signal / time_stop  (warnings only — no orders)")

    engine = rta.TrendAlertEngine(symbol, model, pre, active_count=len(symbols),
                                  cooldown_min=rta.DEFAULT_COOLDOWN_MIN,
                                  thresholds=AlertThresholds(),
                                  cross_symbol=args.cross_symbol, peers=peers)

    if args.mode == "replay":
        date = args.replay_date
        if not date:
            conn = connect(DB_PATH)
            try:
                date = rta._latest_day(conn, "bars_1m", "bar_start", symbol)
            finally:
                conn.close()
        entry_dt = datetime.strptime(f"{date} {entry_hhmm}", "%Y-%m-%d %H:%M")
        tracker = PositionTracker(symbol, args.side, entry_dt, args.entry_price)
        ledger = []
        try:
            run_replay_monitor(engine, tracker, config, ledger, date=date,
                               entry_hhmm=entry_hhmm, all_bars=args.all_bars)
        finally:
            write_ledger(ledger)
            persist_warnings(str(DB_PATH), ledger)
    else:
        today = args.today
        if not today:
            conn = connect(DB_PATH)
            try:
                today = rta._latest_day(conn, "realtime_minute_bars", "ts_kst", symbol) or \
                    pd.Timestamp.now().strftime("%Y-%m-%d")
            finally:
                conn.close()
        entry_dt = datetime.strptime(f"{today} {entry_hhmm}", "%Y-%m-%d %H:%M")
        tracker = PositionTracker(symbol, args.side, entry_dt, args.entry_price)
        ledger = []   # passed by reference so a Ctrl+C still flushes what was collected
        try:
            run_live_monitor(engine, tracker, config, ledger, today=today,
                             entry_hhmm=entry_hhmm, poll_sec=args.poll_sec,
                             all_bars=args.all_bars)
        except KeyboardInterrupt:
            print("\n  stopped.")
        finally:
            write_ledger(ledger)
            persist_warnings(str(DB_PATH), ledger)


if __name__ == "__main__":
    main()
