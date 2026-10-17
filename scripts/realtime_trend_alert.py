"""Step 14 capstone: real-time TREND-ALERT engine + live chart (64-bit).

Consumes 1-minute bars for one symbol, reproduces the full feature pipeline at
serving time (src/realtime_features.py), and on every finished bar runs:
    rule signal (Step 11) -> directional alert type/grade (Step 13_5)
                          -> LightGBM risk_prob (Step 14, risk filter).
It then emits a TREND ALERT whose direction comes from the alert type/grade and
whose risk is CORRECTED by the model's risk_prob, and draws a live candlestick
chart (5-min MA, alert markers, a risk_prob gauge).

Two data sources (the 32-bit Kiwoom feed cannot host LightGBM, so they are split):
  * --mode live    : poll ``realtime_minute_bars`` (written by the 32-bit
                     scripts/realtime_alert_collector.py) for newly finished bars.
  * --mode replay  : stream a historical day from ``bars_1m`` as if it were live,
                     so the whole engine is testable without a Kiwoom session.

Run (cwd = quant, 64-bit):
    ..\\Python311-64\\python.exe scripts\\realtime_trend_alert.py --mode replay --replay-date 2026-06-19
    ..\\Python311-64\\python.exe scripts\\realtime_trend_alert.py --mode live --code 000660
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
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

from config import DB_PATH, PROCESSED_DIR, connect, ensure_directories
from directional_alert import AlertThresholds, UP_TYPES
from realtime_features import build_serving_features, build_serving_features_cross, score_serving
from risk_model import load_model
from settings import load_active_symbols, load_data_config, normalize_symbol_code

BAR_COLUMNS = ["symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume"]
HISTORY_DAYS = 8                 # prior trading days kept for the 5-day MA features
CHART_BARS = 90                  # bars shown on the chart
MA_VIS = 5                       # visual 5-minute MA on the 1-min chart
MIN_TIME, MAX_TIME = "09:10:00", "15:00:00"
# Regular-session open (al_regular = 09:00-15:30). The live feed reads
# realtime_minute_bars, which the collector fills from the integrated (_AL) snapshot
# INCLUDING premarket (08:00-09:00); training/replay use al_regular (regular session
# only). So the live serving path drops premarket bars (time-of-day < REGULAR_OPEN)
# to keep train/serve consistent -- premarket is strictly separated.
REGULAR_OPEN = "09:00:00"
RISK_HIGH, RISK_MED = 0.60, 0.45  # risk_prob bands (0.60 ~ training top-decile)
DEFAULT_COOLDOWN_MIN = 15

# Korean candle colours: up = red, down = blue.
UP_COLOR, DOWN_COLOR = "#d62728", "#1f77b4"
TYPE_MARK = {
    "UP_ALERT": ("^", "#d62728"), "UP_CONTINUATION": ("^", "#ff7f0e"),
    "RISK_ALERT": ("v", "#9467bd"), "WEAKENING_ALERT": ("x", "#8c564b"),
    "VOLATILITY_ALERT": ("s", "#7f7f7f"),
}


def risk_band(p):
    if p != p:
        return "NA"
    return "HIGH" if p >= RISK_HIGH else ("MED" if p >= RISK_MED else "LOW")


def trend_message(row):
    """Human-readable trend alert: direction from the alert, risk from the model."""
    typ, grade, prob = row["alert_type"], row["alert_grade"], float(row["risk_prob"])
    band = risk_band(prob)
    head = {
        "UP_ALERT": "상방 가능성", "UP_CONTINUATION": "상방 유지 가능성",
        "RISK_ALERT": "하방 위험", "WEAKENING_ALERT": "추세 약화",
        "VOLATILITY_ALERT": "변동성 확대",
    }.get(typ, typ)
    note = ""
    if typ in UP_TYPES and band == "HIGH":
        note = "  ⚠ 모델 고위험: 상방이지만 위험 (risk filter)"
    elif typ in UP_TYPES and band == "LOW":
        note = "  ✓ 저위험 상방"
    elif typ == "RISK_ALERT" and band == "HIGH":
        note = "  ⚠ 하방+고위험 일치"
    reasons = "/".join(r for r in (row.get("reason_1"), row.get("reason_2")) if r)
    return f"{head} [{typ} {grade}] risk_prob={prob:.2f}({band}){note}  reasons={reasons}"


class TrendAlertEngine:
    def __init__(self, symbol, model, pre, *, active_count, cooldown_min, thresholds,
                 cross_symbol=False, peers=None):
        self.symbol = symbol
        self.model = model
        self.pre = pre
        self.active_count = active_count
        self.cooldown = pd.Timedelta(minutes=cooldown_min)
        self.thresholds = thresholds
        self.source = load_data_config().model_source
        self.history = pd.DataFrame(columns=BAR_COLUMNS)   # prior days (fixed)
        self.today = pd.DataFrame(columns=BAR_COLUMNS)      # current day, growing
        self.latest = None                                  # latest scored row (Series)
        self.fired = []                                     # alert ledger (dicts)
        self._last_fire_at = None
        self._last_fire_type = None
        self.db_path = None                                 # set by enable_persistence()
        self.persist = False
        # v1.1 cross-symbol serving: judge the target using REAL universe features.
        self.cross_symbol = bool(cross_symbol)
        self.peers = list(dict.fromkeys(peers)) if peers else [symbol]
        if symbol not in self.peers:                        # target must be in the cross-section
            self.peers = [symbol] + self.peers
        self.peer_history = {}                              # sym -> prior-days DataFrame (fixed)
        self._cross_cache = {}                              # decision_time -> scored Series

    def enable_persistence(self, db_path):
        """Persist each fired alert to the realtime_alert_log table (live mode)."""
        self.db_path = str(db_path)
        self.persist = True
        conn = connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS realtime_alert_log ("
                "ts TEXT, symbol TEXT, alert_type TEXT, alert_grade TEXT, "
                "risk_prob REAL, risk_band TEXT, close REAL, message TEXT, created_at TEXT)")
            conn.commit()
        finally:
            conn.close()

    def _persist_alert(self, alert):
        # Best-effort: a transient lock (collector mid-write) must NOT crash a live
        # session -- the alert is still printed, kept in self.fired, and CSV-ledgered
        # on stop. WAL + busy_timeout makes this rare; we just warn and move on.
        try:
            conn = connect(self.db_path)
            try:
                conn.execute(
                    "INSERT INTO realtime_alert_log (ts, symbol, alert_type, alert_grade, risk_prob, "
                    "risk_band, close, message, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (alert["time"], alert["symbol"], alert["alert_type"], alert["alert_grade"],
                     alert["risk_prob"], alert["risk_band"], alert["close"], alert["message"],
                     pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            print(f"  (warn: could not persist alert to realtime_alert_log: {exc})")

    # -- data -------------------------------------------------------------- #
    def seed_history(self, conn, before_date):
        days = pd.read_sql_query(
            "SELECT DISTINCT substr(bar_start,1,10) d FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10) < ? ORDER BY d", conn, params=(self.source, self.symbol, before_date))
        keep = days["d"].tolist()[-HISTORY_DAYS:]
        if not keep:
            return
        self.history = pd.read_sql_query(
            f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10) BETWEEN ? AND ? ORDER BY bar_start",
            conn, params=(self.source, self.symbol, keep[0], keep[-1]))
        if self.cross_symbol:
            self._seed_peer_history(conn, keep[0], keep[-1])

    def _seed_peer_history(self, conn, lo, hi):
        """Load all peers' prior-day bars (fixed) for the cross-section (v1.1)."""
        ph = ",".join("?" * len(self.peers))
        allh = pd.read_sql_query(
            f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol IN ({ph}) "
            "AND substr(bar_start,1,10) BETWEEN ? AND ? ORDER BY bar_start",
            conn, params=(self.source, *self.peers, lo, hi))
        self.peer_history = {s: g.reset_index(drop=True) for s, g in allh.groupby("symbol")}

    def precompute_cross(self, peer_today, date):
        """Build REAL cross-symbol serving features + score for the whole ``date`` in
        ONE recompute and cache by decision_time (replay one-shot / live per-poll
        refresh). ``peer_today``: dict[symbol] -> that symbol's current-day bars
        (BAR_COLUMNS). Strictly causal: build_relative_features only compares rows
        sharing a decision_time."""
        frames = []
        for s in self.peers:
            h = self.peer_history.get(s)
            if h is not None and len(h):
                frames.append(h[BAR_COLUMNS])
            d = peer_today.get(s)
            if d is not None and len(d):
                frames.append(d[BAR_COLUMNS])
        if not frames:
            self._cross_cache = {}
            return
        window = pd.concat(frames, ignore_index=True)
        serv = build_serving_features_cross(window, target=self.symbol, source=self.source,
                                            active_count=len(self.peers))
        if serv.empty:
            self._cross_cache = {}
            return
        serv = serv[serv["decision_time"].str.slice(0, 10) == date]
        scored = score_serving(serv, self.model, self.pre, thresholds=self.thresholds)
        self._cross_cache = {r["decision_time"]: r for _, r in scored.iterrows()}

    def _score_cached(self, decision_time):
        """Cross-symbol path: look up the precomputed scored row for this bar."""
        row = self._cross_cache.get(decision_time)
        if row is None:
            return None
        self.latest = row
        return self._maybe_fire(row)

    def on_new_bar(self, bar):
        """Ingest one finished 1m bar (dict-like with BAR_COLUMNS); rescore + alert."""
        self.today = pd.concat([self.today, pd.DataFrame([{c: bar[c] for c in BAR_COLUMNS}])],
                               ignore_index=True)
        if self.cross_symbol:   # scored row comes from the precomputed universe cross-section
            return self._score_cached(bar["bar_end"])
        window = pd.concat([self.history, self.today], ignore_index=True)
        serv = build_serving_features(window, source=self.source, active_count=self.active_count)
        if serv.empty:
            return None
        cur_day = self.today["bar_start"].iloc[-1][:10]
        serv = serv[serv["decision_time"].str.slice(0, 10) == cur_day]
        scored = score_serving(serv, self.model, self.pre, thresholds=self.thresholds)
        if scored.empty:
            return None
        self.latest = scored.iloc[-1]
        return self._maybe_fire(self.latest)

    def prime(self, bars_df):
        """Warm up today's backlog (e.g. the collector's snapshot seed) in ONE
        recompute instead of replaying every bar through on_new_bar — so the live
        engine starts fast. Sets state + scores the latest bar; backlog bars are not
        re-emitted (a live alerter cares about NOW). Returns the current fire or None.
        """
        if bars_df is None or len(bars_df) == 0:
            return None
        self.today = bars_df[BAR_COLUMNS].reset_index(drop=True).copy()
        if self.cross_symbol:   # cache already built by the driver's precompute_cross
            return self._score_cached(str(self.today["bar_end"].iloc[-1]))
        window = pd.concat([self.history, self.today], ignore_index=True)
        serv = build_serving_features(window, source=self.source, active_count=self.active_count)
        if serv.empty:
            return None
        cur_day = self.today["bar_start"].iloc[-1][:10]
        serv = serv[serv["decision_time"].str.slice(0, 10) == cur_day]
        scored = score_serving(serv, self.model, self.pre, thresholds=self.thresholds)
        if scored.empty:
            return None
        self.latest = scored.iloc[-1]
        return self._maybe_fire(self.latest)

    def _maybe_fire(self, row):
        t = pd.Timestamp(row["decision_time"])
        tod = row["decision_time"][11:19]
        if not (MIN_TIME <= tod < MAX_TIME):
            return None
        if not row["alert_type"]:
            return None
        # de-spam: same type within cooldown is suppressed; a new type fires immediately
        if (self._last_fire_at is not None and row["alert_type"] == self._last_fire_type
                and (t - self._last_fire_at) < self.cooldown):
            return None
        self._last_fire_at = t
        self._last_fire_type = row["alert_type"]
        alert = {
            "time": row["decision_time"], "symbol": self.symbol, "close": float(row["close"]),
            "alert_type": row["alert_type"], "alert_grade": row["alert_grade"],
            "risk_prob": float(row["risk_prob"]), "risk_band": risk_band(float(row["risk_prob"])),
            "message": trend_message(row),
        }
        self.fired.append(alert)
        if self.persist and self.db_path:
            self._persist_alert(alert)
        return alert


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
class LiveChart:
    def __init__(self, engine, *, window, png_path):
        self.engine = engine
        self.png_path = png_path
        import matplotlib
        matplotlib.use("TkAgg" if window else "Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.window = window
        self.fig, (self.ax, self.ax_r) = plt.subplots(
            2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        if window:
            plt.ion()
            self.fig.show()

    def _candles(self, df):
        ax = self.ax
        ax.clear()
        x = np.arange(len(df))
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        up = c >= o
        ax.vlines(x, l, h, color=np.where(up, UP_COLOR, DOWN_COLOR), linewidth=0.8)
        ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.6, color=UP_COLOR)
        ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.6, color=DOWN_COLOR)
        if len(df) >= MA_VIS:
            ma = df["close"].rolling(MA_VIS).mean().to_numpy()
            ax.plot(x, ma, color="#222", linewidth=1.1, label=f"MA{MA_VIS}(5m)")
            ax.legend(loc="upper left", fontsize=8)
        # alert markers within the visible window
        tmap = {t[11:19]: i for i, t in enumerate(df["bar_start"])}
        for a in self.engine.fired:
            i = tmap.get(a["time"][11:19])
            if i is None:
                continue
            mark, col = TYPE_MARK.get(a["alert_type"], ("o", "#000"))
            ax.scatter(i, df["high"].iloc[i] * 1.001, marker=mark, color=col, s=70, zorder=5)
        return x, df

    def update(self):
        eng = self.engine
        if eng.today.empty:
            return
        df = eng.today.tail(CHART_BARS).reset_index(drop=True)
        self._candles(df)
        row = eng.latest
        prob = float(row["risk_prob"]) if row is not None else float("nan")
        band = risk_band(prob)
        last_t = df["bar_start"].iloc[-1][11:16]
        atype = row["alert_type"] if row is not None and row["alert_type"] else "-"
        agrade = row["alert_grade"] if row is not None else "-"
        self.ax.set_title(
            f"{eng.symbol}  {last_t}  close={df['close'].iloc[-1]:,.0f}   "
            f"alert={atype}({agrade})  risk_prob={prob:.2f} [{band}]", fontsize=11)
        self.ax.set_ylabel("price")
        # risk_prob gauge / recent history
        self.ax_r.clear()
        probs = [f["risk_prob"] for f in eng.fired][-CHART_BARS:]
        self.ax_r.axhline(RISK_HIGH, color="#d62728", lw=0.8, ls="--")
        self.ax_r.axhline(RISK_MED, color="#ff7f0e", lw=0.8, ls="--")
        if row is not None and prob == prob:
            self.ax_r.bar([0], [prob], width=0.5,
                          color="#d62728" if band == "HIGH" else ("#ff7f0e" if band == "MED" else "#2ca02c"))
        self.ax_r.set_ylim(0, 1)
        self.ax_r.set_ylabel("risk_prob")
        self.ax_r.set_xticks([])
        self.fig.tight_layout()
        if self.png_path:
            self.fig.savefig(self.png_path, dpi=90)
        if self.window:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def _emit(alert):
    print(f"  🔔 ALERT {alert['time'][11:16]} {alert['symbol']}  {alert['message']}")


def _bar_from_rt(r):
    """realtime_minute_bars row -> engine bar dict."""
    return {"symbol": r["symbol"], "bar_start": r["ts_kst"],
            "bar_end": (pd.Timestamp(r["ts_kst"]) + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": r["open"], "high": r["high"], "low": r["low"],
            "close": r["close"], "volume": r["volume"]}


def load_universe_day(conn, peers, source, date, *, realtime=False):
    """All peers' bars for ``date`` -> dict[symbol]->DataFrame(BAR_COLUMNS), for the
    cross-symbol cross-section. ``realtime=True`` reads realtime_minute_bars (live)."""
    ph = ",".join("?" * len(peers))
    if realtime:
        rt = pd.read_sql_query(
            f"SELECT symbol, ts_kst, open, high, low, close, volume FROM realtime_minute_bars "
            f"WHERE symbol IN ({ph}) AND substr(ts_kst,1,10)=? ORDER BY ts_kst",
            conn, params=(*peers, date))
        # drop premarket (08:00-09:00) so the cross-section matches al_regular
        rows = [r for r in rt.to_dict("records") if r["ts_kst"][11:19] >= REGULAR_OPEN]
        df = (pd.DataFrame([_bar_from_rt(r) for r in rows], columns=BAR_COLUMNS)
              if rows else pd.DataFrame(columns=BAR_COLUMNS))
    else:
        df = pd.read_sql_query(
            f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol IN ({ph}) "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start", conn, params=(source, *peers, date))
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}


def run_replay(engine, chart, *, date, speed):
    conn = connect(DB_PATH)
    try:
        engine.seed_history(conn, date)
        day = pd.read_sql_query(
            f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source=? AND symbol=? "
            "AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn, params=(engine.source, engine.symbol, date))
        if engine.cross_symbol:
            engine.precompute_cross(load_universe_day(conn, engine.peers, engine.source, date), date)
    finally:
        conn.close()
    if day.empty:
        raise SystemExit(f"no bars_1m for {engine.symbol} on {date} (source={engine.source}).")
    print(f"replay {engine.symbol} {date}: {len(day)} bars, history_days={engine.history['bar_start'].str.slice(0,10).nunique()}")
    for bar in day.to_dict("records"):
        alert = engine.on_new_bar(bar)
        if alert:
            _emit(alert)
        if chart is not None and engine.today.shape[0] % 5 == 0:
            chart.update()
        if speed > 0:
            time.sleep(speed)
    if chart is not None:
        chart.update()
    return day


def run_live(engine, chart, *, poll_sec, today):
    engine.enable_persistence(DB_PATH)
    print(f"live {engine.symbol}: polling realtime_minute_bars every {poll_sec}s "
          f"(Ctrl+C to stop). Run the 32-bit collector to populate it.")
    print("  fired alerts -> DB table realtime_alert_log (+ CSV ledger on stop)")
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
        records = [r for r in new.to_dict("records") if r["ts_kst"][11:19] >= REGULAR_OPEN]
        fresh = [r for r in records if r["ts_kst"] not in seen]
        for r in fresh:
            seen.add(r["ts_kst"])
        if engine.cross_symbol and fresh:   # refresh the universe cross-section cache
            conn = connect(DB_PATH)
            try:
                engine.precompute_cross(
                    load_universe_day(conn, engine.peers, engine.source, today, realtime=True), today)
            finally:
                conn.close()
        if not primed and len(fresh) > 1:   # warm up the backlog in one recompute
            alert = engine.prime(pd.DataFrame([_bar_from_rt(r) for r in fresh]))
            print(f"  primed {len(fresh)} backlog bars; now streaming one per minute")
            if alert:
                _emit(alert)
        else:
            for r in fresh:
                alert = engine.on_new_bar(_bar_from_rt(r))
                if alert:
                    _emit(alert)
        primed = True
        if chart is not None:
            chart.update()
        time.sleep(poll_sec)


def parse_args():
    p = argparse.ArgumentParser(description="Real-time trend-alert engine + chart (Step 14 capstone).")
    p.add_argument("--code", default=None, help="6-digit symbol; default = first active symbol.")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--replay-date", default=None, help="YYYY-MM-DD (default: latest day with bars).")
    p.add_argument("--replay-speed", type=float, default=0.0, help="seconds between bars (0 = as fast as possible).")
    p.add_argument("--poll-sec", type=float, default=5.0, help="live mode poll interval.")
    p.add_argument("--today", default=None, help="live mode trading date (default: latest realtime_minute_bars date).")
    p.add_argument("--cooldown-min", type=int, default=DEFAULT_COOLDOWN_MIN)
    p.add_argument("--cross-symbol", action="store_true",
                   help="judge the target using REAL cross-symbol (universe) features instead of "
                        "neutral fills (v1.1 train/serve skew fix). Needs all active symbols' bars.")
    p.add_argument("--model-path", default=str(ROOT_DIR / "models" / "risk_model_lgbm.txt"))
    p.add_argument("--no-window", action="store_true", help="headless: save PNG only, no GUI window.")
    p.add_argument("--png", default=str(PROCESSED_DIR / "realtime_trend_chart.png"))
    return p.parse_args()


def _latest_day(conn, table, col, symbol):
    q = f"SELECT max(substr({col},1,10)) FROM {table} WHERE symbol=?"
    r = conn.execute(q, (symbol,)).fetchone()[0]
    return r


def main():
    args = parse_args()
    ensure_directories()
    symbols = load_active_symbols()
    symbol = normalize_symbol_code(args.code) if args.code else symbols[0]
    active_count = len(symbols)

    model, pre, meta = load_model(args.model_path)
    print(f"loaded risk model: type={getattr(model,'model_type','?')} "
          f"target={meta.get('target','?')} features={len(pre.feature_columns)}")
    peers = [normalize_symbol_code(s) for s in symbols]
    if args.cross_symbol:
        print(f"  cross-symbol: ON (peers={len(peers)}) — real universe features")
    engine = TrendAlertEngine(symbol, model, pre, active_count=active_count,
                              cooldown_min=args.cooldown_min, thresholds=AlertThresholds(),
                              cross_symbol=args.cross_symbol, peers=peers)

    chart = None
    try:
        chart = LiveChart(engine, window=not args.no_window, png_path=args.png)
    except Exception as exc:   # no display / backend -> fall back to headless PNG
        print(f"  (chart window unavailable: {exc}; saving PNG only)")
        chart = LiveChart(engine, window=False, png_path=args.png)

    if args.mode == "replay":
        date = args.replay_date
        if not date:
            conn = connect(DB_PATH)
            try:
                date = _latest_day(conn, "bars_1m", "bar_start", symbol)
            finally:
                conn.close()
        run_replay(engine, chart, date=date, speed=args.replay_speed)
        print(f"\nreplay done: {len(engine.fired)} trend alerts fired.")
        _print_ledger(engine)
        print(f"chart -> {args.png}")
    else:
        today = args.today
        if not today:
            conn = connect(DB_PATH)
            try:
                today = _latest_day(conn, "realtime_minute_bars", "ts_kst", symbol) or \
                    pd.Timestamp.now().strftime("%Y-%m-%d")
            finally:
                conn.close()
        try:
            run_live(engine, chart, poll_sec=args.poll_sec, today=today)
        except KeyboardInterrupt:
            print(f"\nstopped: {len(engine.fired)} trend alerts fired.")
            _print_ledger(engine)


def _print_ledger(engine):
    if not engine.fired:
        print("  (no alerts)")
        return
    led = pd.DataFrame(engine.fired)
    path = PROCESSED_DIR / "realtime_alert_ledger.csv"
    led.to_csv(path, index=False, encoding="utf-8-sig")
    by_type = led["alert_type"].value_counts().to_dict()
    print(f"  alert types: {by_type}")
    print(f"  risk bands : {led['risk_band'].value_counts().to_dict()}")
    print("  last alerts:")
    for a in engine.fired[-8:]:
        print(f"    {a['time'][11:16]} {a['alert_type']:<16} {a['alert_grade']:<4} "
              f"risk={a['risk_prob']:.2f}[{a['risk_band']}] close={a['close']:,.0f}")
    print(f"  ledger -> {path}")


if __name__ == "__main__":
    main()
