"""Step 14 capstone (chartless): streaming TREND-SIGNAL alerter, no chart.

Same engine as scripts/realtime_trend_alert.py but with NO matplotlib / no chart
window -- it just consumes 1-minute bars as they stream in and prints a signal
line per update: the current directional alert (type/grade), the LightGBM
risk_prob and its band, and a prominent 🔔 line whenever a (cooldown-gated) trend
alert fires. Use this for a lightweight terminal/headless signal feed.

Data granularity is the 1-minute bar (the rule/alert/risk features are defined on
closed bars); each finished bar = one signal update. Two sources, like the chart
version:
  * --mode replay : stream a historical day from bars_1m (testable without Kiwoom)
  * --mode live   : poll realtime_minute_bars (written by the 32-bit collector).
                    On startup it REPLAYS the day's backlog so far (market open ->
                    now) so the morning's signals print too, then streams one signal
                    per minute from now on. Backlog fires are printed + CSV-ledgered
                    but not re-written to realtime_alert_log. Use --no-backlog to skip
                    the replay (fast silent prime, as before).

DEFAULT run = the per-minute TREND JUDGMENT for one stock (000660): one signal
line EVERY minute (time, close, alert type/grade, risk_prob band), plus a 🔔 line
when a cooldown-gated trend alert fires. Everything else is opt-in via flags:
  --tick-cross   add the 5-minute MA crossover (📈/📉) signal: fires when a
                 finished bar's CLOSE crosses the 5m MA (see TickCrossMonitor),
                 in both live and replay. `--replay-ticks ohlc` (replay only)
                 instead approximates an intra-bar cross from each bar's OHLC.
  --alerts-only  print only minutes that carry a directional alert (+🔔 fires).
  --code         analyse a different symbol (default 000660).

Run (cwd = quant, 64-bit):
    ..\\Python311-64\\python.exe scripts\\realtime_trend_signal.py --mode replay --replay-date 2026-06-19
    ..\\Python311-64\\python.exe scripts\\realtime_trend_signal.py --mode replay --tick-cross
    ..\\Python311-64\\python.exe scripts\\realtime_trend_signal.py --mode live --code 000660
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
except (AttributeError, ValueError):
    pass

import pandas as pd

# the chartless version reuses the engine core; importing this module does NOT pull
# in matplotlib (that is imported lazily inside its LiveChart only).
import realtime_trend_alert as rta
from config import DB_PATH, PROCESSED_DIR, connect, ensure_directories
from directional_alert import AlertThresholds
from risk_model import load_model
from settings import load_active_symbols, normalize_symbol_code, realtime_default_symbol

LEDGER_PATH = PROCESSED_DIR / "realtime_signal_ledger.csv"

# Default single-stock target for this trend-signal feed (overridable with --code).
# Loaded from config.yaml (realtime.default_symbol = 000660) rather than hard-coded in
# source, per the project's no-hardcoded-symbol rule; membership is checked in main().
DEFAULT_CODE = realtime_default_symbol()


def _emit_signal(engine, bar, alert, *, all_bars):
    """One concise signal line per bar update; 🔔 for a fired trend alert."""
    row = engine.latest
    tod = bar["bar_start"][11:16]
    if row is None:
        return
    typ = row["alert_type"]
    band = rta.risk_band(float(row["risk_prob"]))
    line = (f"  {tod}  close={float(row['close']):>11,.0f}  {typ or '-':<16} "
            f"{row['alert_grade'] or '-':<4} risk={row['risk_prob']:.2f}[{band}]")
    if alert:
        print(f"🔔{line[1:]}   →  {alert['message']}")
    elif typ:
        print(line)
    elif all_bars:
        print(line)


# --------------------------------------------------------------------------- #
# 5-minute MA crossover (intra-bar tick signal)
# --------------------------------------------------------------------------- #
# The signal lines above print once per FINISHED 1m bar. This adds a second,
# finer signal: it watches the price *between* bar completions and fires the
# moment the price crosses the 5-minute moving-average line -- the same MA the
# chart draws (the mean of the last N completed 1m closes). One alert per cross
# (deduped), for both up-crosses (golden 📈) and down-crosses (dead 📉).
#
# The crossover is checked on the FINISHED bar's close (the 64-bit engine never
# talks to Kiwoom directly, so there is no live tick stream):
#   * live   -> the bar close crosses the 5m MA (close-to-close cross).
#   * replay -> same close-to-close cross by default; `--replay-ticks ohlc` instead
#               walks each bar's OHLC path (open->extreme->extreme->close) to
#               approximate an intra-bar cross (flagged as approximate).

DEFAULT_MA_WINDOW = rta.MA_VIS          # 5 -> the chart's "5-minute MA"
CROSS_UP_EMOJI, CROSS_DN_EMOJI = "📈", "📉"
_KIND_NOTE = {
    "ohlc": "intra-bar approx (OHLC path)",
    "bar_close": "at bar close",
}


class TickCrossMonitor:
    """Detect when the live price crosses the N-minute MA line.

    The MA line (``ma``) is the mean of the last ``window`` *completed* 1m closes
    and is held fixed while the current minute forms; the caller refreshes it via
    set_reference() whenever a new bar completes. update(price) is called for each
    incoming price (a real tick, an OHLC-path point, or a bar close) and returns a
    crossover event the first time the price moves to the OTHER side of the line --
    so a steady run of same-side ticks never re-fires. An optional hysteresis band
    (``eps_bp`` basis points) suppresses chatter right at the line.
    """

    def __init__(self, *, window, symbol, eps_bp=0.0):
        self.window = int(window)
        self.symbol = symbol
        self.eps_bp = float(eps_bp)
        self.ma = None          # current reference (5m MA), None until `window` closes
        self.side = None        # +1 above the line, -1 below, None = unknown
        self.last_price = None
        self.events = []        # crossover ledger (dicts)

    def _band(self):
        return (self.ma * self.eps_bp / 1e4) if (self.ma and self.eps_bp) else 0.0

    def _side_of(self, price):
        eps = self._band()
        if price > self.ma + eps:
            return 1
        if price < self.ma - eps:
            return -1
        return self.side        # inside the band -> unchanged

    def set_reference(self, ma, *, seed_price=None):
        """Move the MA line (a new bar completed). Re-seats ``side`` from the latest
        price against the NEW line WITHOUT emitting -- a line move is not a price
        cross, only a price moving across the line is."""
        self.ma = float(ma)
        if seed_price is not None:
            self.last_price = float(seed_price)
        if self.last_price is not None:
            self.side = self._side_of(self.last_price)

    def update(self, price, *, ts, kind="bar_close"):
        """Feed one price. Return a crossover event dict on a fresh cross, else None."""
        price = float(price)
        if self.ma is None:                 # no MA line yet (< window closes)
            self.last_price = price
            return None
        new_side = self._side_of(price)
        self.last_price = price
        if self.side is None:               # first observation against a known line
            self.side = new_side
            return None
        if new_side == self.side:           # same side (or inside the band): no cross
            return None
        self.side = new_side
        ev = {
            "ts": ts, "symbol": self.symbol, "price": price, "ma": self.ma,
            "window": self.window, "direction": "UP" if new_side == 1 else "DOWN",
            "kind": kind,
        }
        self.events.append(ev)
        return ev


def _ohlc_tick_path(bar):
    """Approximate a minute's intra-bar tick path from its OHLC.

    Standard heuristic: an up bar (close>=open) is assumed to dip to the low first
    then rally to the high; a down bar spikes to the high first then sells off.
    """
    o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
    first, second = (l, h) if c >= o else (h, l)
    return [o, first, second, c]


def _emit_cross(ev):
    up = ev["direction"] == "UP"
    emoji = CROSS_UP_EMOJI if up else CROSS_DN_EMOJI
    arrow = "↑" if up else "↓"
    ko = "상향 돌파" if up else "하향 이탈"
    diff = ev["price"] - ev["ma"]
    print(f"  {emoji} {ev['ts']}  MA{ev['window']}(5m) {ko}{arrow}  "
          f"price={ev['price']:>11,.0f}  MA={ev['ma']:>11,.1f}  "
          f"({diff:+,.0f})  [{_KIND_NOTE.get(ev['kind'], ev['kind'])}]")


def _feed_bar_crosses(monitor, engine, bar, *, kind):
    """Refresh the MA line from the completed closes, then feed this forming bar's
    price path and print each crossover. MUST be called BEFORE engine.on_new_bar so
    the line uses only PRIOR completed bars (the bar itself is not yet closed)."""
    tod = bar["bar_start"][11:19]
    if not (rta.MIN_TIME <= tod < rta.MAX_TIME):   # session-hours gate, like _maybe_fire
        return
    closes = engine.today["close"]
    if len(closes) < monitor.window:               # MA undefined for the first bars
        return
    prior = closes.astype(float).tolist()
    monitor.set_reference(sum(prior[-monitor.window:]) / monitor.window,
                          seed_price=prior[-1])
    label = bar["bar_start"][11:16]
    points = _ohlc_tick_path(bar) if kind == "ohlc" else [float(bar["close"])]
    for px in points:
        ev = monitor.update(px, ts=label, kind=kind)
        if ev is not None:
            _emit_cross(ev)


def _process_bar(engine, monitor, r, *, all_bars):
    """Feed one realtime row through the engine (crossover -> rule/alert/risk) and
    print its signal line. Shared by the live stream and the startup backlog replay.

    The crossover is checked BEFORE on_new_bar so the 5m MA uses only prior completed
    bars (same as replay's default close-cross)."""
    bar = rta._bar_from_rt(r)
    if monitor is not None:
        _feed_bar_crosses(monitor, engine, bar, kind="bar_close")
    alert = engine.on_new_bar(bar)
    _emit_signal(engine, bar, alert, all_bars=all_bars)
    return alert


def run_replay_signal(engine, monitor, *, date, speed, all_bars, tick_path):
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, date)
        day = pd.read_sql_query(
            f"SELECT {', '.join(rta.BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn, params=(engine.source, engine.symbol, date))
        if engine.cross_symbol:   # one-shot universe cross-section for the whole day
            engine.precompute_cross(
                rta.load_universe_day(conn, engine.peers, engine.source, date), date)
    finally:
        conn.close()
    if day.empty:
        raise SystemExit(f"no bars_1m for {engine.symbol} on {date} (source={engine.source}).")
    # "close" = one confirmed cross per bar (close on the other side of the MA, like
    # rule_signal_ma — the same rule live mode uses); "ohlc" walks each bar's OHLC
    # path to approximate an intra-bar cross (noisier).
    kind = "ohlc" if tick_path == "ohlc" else "bar_close"
    extra = (f" + 📈📉 MA{monitor.window}(5m) crossovers ({_KIND_NOTE[kind]})"
             if monitor is not None else "")
    print(f"replay-signal {engine.symbol} {date}: {len(day)} bars "
          f"(printing {'every bar' if all_bars else 'bars with a signal'} + 🔔 fires{extra})\n")
    for bar in day.to_dict("records"):
        if monitor is not None:
            _feed_bar_crosses(monitor, engine, bar, kind=kind)
        alert = engine.on_new_bar(bar)
        _emit_signal(engine, bar, alert, all_bars=all_bars)
        if speed > 0:
            time.sleep(speed)


def run_live_signal(engine, monitor, *, today, poll_sec, all_bars, emit_backlog=True):
    engine.enable_persistence(DB_PATH)
    print(f"live-signal {engine.symbol}: polling realtime_minute_bars every {poll_sec}s "
          f"(Ctrl+C to stop). Run the 32-bit collector to populate it.")
    print("  fired alerts -> DB table realtime_alert_log (+ CSV ledger on stop)")
    if monitor is not None:
        print(f"  MA{monitor.window}(5m) crossover signal: 📈/📉 on the bar close (same as replay)")
    print(f"  startup backlog: {'replay every earlier bar' if emit_backlog else 'primed silently (--no-backlog)'}")
    print()
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, today)
    finally:
        conn.close()
    seen = set()
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
        if engine.cross_symbol and fresh:   # refresh the universe cross-section cache
            conn = connect(DB_PATH)
            try:
                engine.precompute_cross(
                    rta.load_universe_day(conn, engine.peers, engine.source, today, realtime=True), today)
            finally:
                conn.close()
        if not primed and len(fresh) > 1:   # first poll carries the day's backlog so far
            if emit_backlog:
                # Replay every earlier bar so the morning's signals (and 🔔/📈📉)
                # print just like the live stream, then continue from now. Historical
                # fires are printed + kept in the CSV ledger but NOT written to the
                # realtime_alert_log DB table (else a restart re-logs the whole day).
                print(f"  replaying {len(fresh)} backlog bars (earlier signals, open→now):")
                saved_persist = engine.persist
                engine.persist = False
                try:
                    for r in fresh:
                        _process_bar(engine, monitor, r, all_bars=all_bars)
                finally:
                    engine.persist = saved_persist
                print("  --- caught up; now streaming one signal per minute ---")
            else:
                alert = engine.prime(pd.DataFrame([rta._bar_from_rt(r) for r in fresh]))
                print(f"  primed {len(fresh)} backlog bars; now streaming one signal per minute")
                _emit_signal(engine, rta._bar_from_rt(fresh[-1]), alert, all_bars=True)
        else:
            for r in fresh:
                _process_bar(engine, monitor, r, all_bars=all_bars)
        primed = True
        time.sleep(poll_sec)


def _print_ledger(engine):
    if not engine.fired:
        print("\n  (no trend alerts fired)")
        return
    led = pd.DataFrame(engine.fired)
    led.to_csv(LEDGER_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  fired {len(engine.fired)} trend alerts  types={led['alert_type'].value_counts().to_dict()}  "
          f"bands={led['risk_band'].value_counts().to_dict()}")
    print(f"  ledger -> {LEDGER_PATH}")


def _print_cross_summary(monitor):
    if monitor is None:
        return
    n = len(monitor.events)
    if not n:
        print(f"  (no MA{monitor.window}(5m) crossovers)")
        return
    ups = sum(1 for e in monitor.events if e["direction"] == "UP")
    print(f"  MA{monitor.window}(5m) crossovers: {n}  (📈{ups} / 📉{n - ups})")


def parse_args():
    p = argparse.ArgumentParser(description="Chartless streaming trend-signal alerter (Step 14 capstone).")
    p.add_argument("--code", default=None, help=f"6-digit symbol to analyse (default {DEFAULT_CODE}).")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--replay-date", default=None, help="YYYY-MM-DD (default: latest day with bars).")
    p.add_argument("--replay-speed", type=float, default=0.0, help="seconds between bars (0 = fast).")
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--today", default=None, help="live trading date (default: latest realtime_minute_bars date).")
    p.add_argument("--cooldown-min", type=int, default=rta.DEFAULT_COOLDOWN_MIN)
    p.add_argument("--no-backlog", action="store_true",
                   help="live mode: skip replaying earlier bars at startup. Default REPLAYS the day's "
                        "backlog (open→now) so the morning's signals print, then streams per minute. "
                        "Backlog fires are NOT written to realtime_alert_log (only printed + CSV-ledgered).")
    p.add_argument("--alerts-only", action="store_true",
                   help="print only minutes carrying a directional alert (+🔔 fires); "
                        "default streams the trend judgment EVERY minute.")
    p.add_argument("--all-bars", action="store_true", help=argparse.SUPPRESS)  # back-compat: now the default
    p.add_argument("--tick-cross", action="store_true",
                   help="add the intra-bar 5-minute MA crossover (📈/📉) signal (off by default).")
    p.add_argument("--ma5-window", type=int, default=DEFAULT_MA_WINDOW,
                   help="MA window in 1m bars for the crossover line (default 5 = 5-minute MA).")
    p.add_argument("--replay-ticks", choices=["close", "ohlc"], default="close",
                   help="replay crossover granularity: 'close' = confirmed close-to-close cross "
                        "(clean, default); 'ohlc' = simulate an intra-bar tick path (mid-bar, noisier).")
    p.add_argument("--cross-eps-bp", type=float, default=0.0,
                   help="hysteresis band around the MA in basis points (suppresses chatter at the line).")
    p.add_argument("--cross-symbol", action="store_true",
                   help="judge the target using REAL cross-symbol (universe) features instead of "
                        "neutral fills (v1.1 train/serve skew fix). Needs all active symbols' bars.")
    p.add_argument("--model-path", default=str(ROOT_DIR / "models" / "risk_model_lgbm.txt"))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    symbols = load_active_symbols()
    symbol = normalize_symbol_code(args.code or DEFAULT_CODE)
    if symbol not in {normalize_symbol_code(s) for s in symbols}:
        print(f"  note: {symbol} is not in the active symbols list; analysing it anyway.")
    print_every = not args.alerts_only   # default: one trend-judgment line every minute

    model, pre, meta = load_model(args.model_path)
    print(f"loaded risk model: type={getattr(model,'model_type','?')} "
          f"target={meta.get('target','?')} features={len(pre.feature_columns)}")
    peers = [normalize_symbol_code(s) for s in symbols]
    print(f"target symbol: {symbol}  | per-minute feed: "
          f"{'alerts only' if args.alerts_only else 'every minute'}  | "
          f"5m crossover: {'on' if args.tick_cross else 'off (--tick-cross to enable)'}  | "
          f"cross-symbol: {f'on (peers={len(peers)})' if args.cross_symbol else 'off (--cross-symbol to enable)'}")
    engine = rta.TrendAlertEngine(symbol, model, pre, active_count=len(symbols),
                                  cooldown_min=args.cooldown_min, thresholds=AlertThresholds(),
                                  cross_symbol=args.cross_symbol, peers=peers)
    monitor = None
    if args.tick_cross:
        monitor = TickCrossMonitor(window=args.ma5_window, symbol=symbol, eps_bp=args.cross_eps_bp)

    if args.mode == "replay":
        date = args.replay_date
        if not date:
            conn = connect(DB_PATH)
            try:
                date = rta._latest_day(conn, "bars_1m", "bar_start", symbol)
            finally:
                conn.close()
        run_replay_signal(engine, monitor, date=date, speed=args.replay_speed,
                          all_bars=print_every, tick_path=args.replay_ticks)
        _print_ledger(engine)
        _print_cross_summary(monitor)
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
            run_live_signal(engine, monitor, today=today, poll_sec=args.poll_sec,
                            all_bars=print_every, emit_backlog=not args.no_backlog)
        except KeyboardInterrupt:
            _print_ledger(engine)
            _print_cross_summary(monitor)


if __name__ == "__main__":
    main()
