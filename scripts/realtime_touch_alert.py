"""Step 22 capstone: realtime MA5x20 touch alert (chartless, two-tier).

Live decision-support ONLY -- prints and logs breakout alerts + CNN touch grades;
it NEVER places, cancels, or sends any order (no order API is imported or called).

Tier 1 (돌파, unconditional): every carry-variant MA5x20 close-cross CONFIRMED in
the regular session fires a breakout line (direction/band/dist + a 재교차(휩쏘)
flag when an opposite cross sits within the previous RECROSS_WINDOW_MIN minutes --
the raw study found the re-cross itself is an exit signal, so it is flagged, never
suppressed; there is NO cooldown). A scoring failure or missing artifact only
disables the grade line (user modification #7) -- tier 1 always survives.

Tier 2 (등급): only for feature-covered crosses (the features_1m stack starts
~10:01 -- earlier crosses tag "워밍업") whose FULL 10-minute outcome window fits
inside the regular session (cutoff derived from REGULAR_END - (window-1) minutes,
never a hard-coded clock literal). The prod CNN ensemble's raw score maps through
the FROZEN OOF grade map and the line cites that grade's HISTORICAL OOF touch
rate -- scores are class-weighted and UNCALIBRATED, so they are NEVER quoted as a
probability. The July gate confirmation is supporting evidence, NOT final proof:
every alert's realized outcome is backfilled into the ledger/DB so live-forward
evidence accumulates (per-grade realized rates print in the session summary).

Bar flow: the previous TAIL_DAYS al_raw days seed the carry MA stream (premarket
AND after-hours bars included -- the MA carries across days exactly like the
study); today's bars come from realtime_minute_bars (live -- premarket INCLUDED
for the MA stream; the collector must run with --all-symbols) or bars_1m al_raw
(replay). The feature engine consumes the regular session only (al_regular
train/serve parity). Every poll only CHECKS for new completed bars (user
modification #5): no new bar -> no computation at all; new bars -> ONE batch
detection recompute over all symbols (offline code path, so parity holds by
construction and bridge bar rewrites self-heal) and CNN inference for the NEW
crosses only. The relative features need every active symbol, so bars are always
ingested for the whole active set; --symbols only filters which symbols ALERT.

Run (cwd = quant, 64-bit):
  ..\\Python311-64\\python.exe scripts\\realtime_touch_alert.py --mode replay --replay-date 2026-07-10
  ..\\Python311-64\\python.exe scripts\\realtime_touch_alert.py --mode live
  ..\\Python311-64\\python.exe scripts\\realtime_touch_alert.py --mode live --symbols "000660,005930"
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

import ma_cross_study as mcs
import touch_model as tmm
import touch_serving as ts
from config import DB_PATH, PROCESSED_DIR, connect, ensure_directories
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text

BAR_COLUMNS = ["symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume"]
TAIL_DAYS = 2                  # prior al_raw days seeding the carry MA stream
FEATURE_HISTORY_DAYS = 8       # prior al_regular days for the 5-day MA features (house)
RECROSS_WINDOW_MIN = 5         # 재교차(휩쏘) display flag window (raw-study convention)
LEDGER_PATH = PROCESSED_DIR / "touch_alert_ledger.csv"
ALERT_TABLE = "touch_alert_log"

# Tier-2 completeness cutoff, DERIVED from the pre-registered constants (user
# modification #4 -- no clock literal): the offline population keeps a cross only
# when the day still has a bar at/beyond entry + (window-1) minutes, and the last
# regular bar is the closing auction at REGULAR_END, so the latest gradable
# decision_time is REGULAR_END - (window-1) minutes.
GRADE_LAST_DECISION = (
    pd.Timestamp(f"2000-01-01 {mcs.REGULAR_END}")
    - pd.Timedelta(minutes=tmm.TOUCH_WINDOW_MIN - 1)
).strftime("%H:%M:%S")

GRADE_STATUS_OK = "graded"


def _fmt_pct(x, digits=1):
    return f"{100.0 * float(x):.{digits}f}%"


class TouchAlertRunner:
    """State + emit/backfill logic shared by replay and live (bars in, alerts out)."""

    def __init__(self, symbols, alert_symbols, *, device="cpu", grade_map_path=None,
                 models_dir=None, artifact_suffix="", verbose=False):
        self.symbols = [normalize_symbol_code(s) for s in symbols]
        self.alert_symbols = {normalize_symbol_code(s)
                              for s in (alert_symbols or self.symbols)}
        self.source = load_data_config().model_source
        self.verbose = verbose
        # grade layer is OPTIONAL at startup (user modification #7): any load
        # problem -> breakout-only mode with the reason surfaced, never a crash.
        self.grade_map = None
        self.adapter = None
        self.grade_disabled_reason = None
        try:
            self.grade_map = ts.load_grade_map(grade_map_path)
            self.adapter = ts.TouchModelAdapter(
                models_dir=models_dir, artifact_suffix=artifact_suffix,
                device=device, expected_symbols=self.symbols)
        except Exception as exc:                     # noqa: BLE001 -- degrade, don't die
            self.grade_disabled_reason = f"{type(exc).__name__}: {exc}"
        self.ma_tail = pd.DataFrame(columns=BAR_COLUMNS)
        self.feat_history = pd.DataFrame(columns=BAR_COLUMNS)
        self.today = pd.DataFrame(columns=BAR_COLUMNS)
        self.date = None
        self.records = {}          # (symbol, decision_time) -> alert record (dict)
        self.order = []            # emit order for the CSV ledger
        self.persist = False
        self.db_path = None
        self.live = False          # run_live sets True; latency is live-only

    # -- persistence -------------------------------------------------------- #
    def enable_persistence(self, db_path):
        self.persist = True
        self.db_path = str(db_path)
        conn = connect(self.db_path)
        try:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {ALERT_TABLE} ("
                "symbol TEXT, decision_time TEXT, date TEXT, direction INTEGER, "
                "band TEXT, dist_pct REAL, dist_atr REAL, cluster_first INTEGER, "
                "whipsaw_recross INTEGER, cross_confirmed_time TEXT, "
                "alert_emitted_time TEXT, alert_latency_ms REAL, price_at_alert REAL, "
                "score REAL, grade TEXT, grade_oof_touch_rate REAL, grade_oof_n INTEGER, "
                "grade_status TEXT, offline_reference_entry REAL, "
                "entry_slippage_vs_open_t1 REAL, realized_touch_05 INTEGER, "
                "realized_mae REAL, window_incomplete INTEGER, backlog INTEGER, "
                "created_at TEXT, PRIMARY KEY (symbol, decision_time))")
            conn.commit()
        finally:
            conn.close()

    _DB_COLS = ("symbol", "decision_time", "date", "direction", "band", "dist_pct",
                "dist_atr", "cluster_first", "whipsaw_recross", "cross_confirmed_time",
                "alert_emitted_time", "alert_latency_ms", "price_at_alert", "score",
                "grade", "grade_oof_touch_rate", "grade_oof_n", "grade_status",
                "offline_reference_entry", "entry_slippage_vs_open_t1",
                "realized_touch_05", "realized_mae", "window_incomplete", "backlog",
                "created_at")

    def _persist_record(self, rec):
        """Best-effort INSERT OR REPLACE (a collector mid-write lock must never
        crash the session -- the record stays in memory and the CSV ledger)."""
        if not (self.persist and not rec["backlog"]):
            return
        try:
            conn = connect(self.db_path)
            try:
                conn.execute(
                    f"INSERT OR REPLACE INTO {ALERT_TABLE} "
                    f"({', '.join(self._DB_COLS)}) VALUES "
                    f"({', '.join('?' * len(self._DB_COLS))})",
                    tuple(rec.get(c) for c in self._DB_COLS))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            print(f"  (warn: could not persist to {ALERT_TABLE}: {exc})")

    # -- seeding ------------------------------------------------------------ #
    def seed(self, conn, date):
        """Prior-day bars: al_raw tail for the carry MA stream + al_regular history
        for the 5-day MA features."""
        self.date = str(date)
        ph = ",".join("?" * len(self.symbols))

        def _load(source, n_days):
            days = pd.read_sql_query(
                "SELECT DISTINCT substr(bar_start,1,10) d FROM bars_1m "
                "WHERE source=? AND substr(bar_start,1,10) < ? ORDER BY d",
                conn, params=(source, self.date))["d"].tolist()[-n_days:]
            if not days:
                return pd.DataFrame(columns=BAR_COLUMNS)
            return pd.read_sql_query(
                f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source=? "
                f"AND symbol IN ({ph}) AND substr(bar_start,1,10) BETWEEN ? AND ? "
                "ORDER BY symbol, bar_start",
                conn, params=(source, *self.symbols, days[0], days[-1]))

        self.ma_tail = _load("al_raw", TAIL_DAYS)
        self.feat_history = _load(self.source, FEATURE_HISTORY_DAYS)
        tail_days = self.ma_tail["bar_start"].str.slice(0, 10).nunique() \
            if len(self.ma_tail) else 0
        if tail_days < TAIL_DAYS:
            print(f"  (warn: only {tail_days}/{TAIL_DAYS} al_raw tail days available "
                  "-- the carry MA warmup may differ from the offline study)")

    # -- core: one batch of new completed bars ------------------------------- #
    def ingest(self, new_bars, *, backlog=False, emit=True):
        """Process new COMPLETED bars (any symbols). Detection is one batch
        recompute over tail+today via the offline code path (parity by
        construction; a bridge rewrite replaces the keyed rows, so the next batch
        self-heals). CNN inference runs for the NEW crosses only."""
        if new_bars is None or len(new_bars) == 0:
            return []
        add = new_bars[[c for c in BAR_COLUMNS if c in new_bars.columns]].copy()
        keys = set(zip(add["symbol"], add["bar_start"]))
        if len(self.today):
            keep = ~pd.Series(list(zip(self.today["symbol"], self.today["bar_start"])),
                              index=self.today.index).isin(keys)
            self.today = self.today[keep]
        self.today = pd.concat([self.today, add], ignore_index=True) \
            .sort_values(["symbol", "bar_start"]).reset_index(drop=True)

        frame = pd.concat([self.ma_tail, self.today], ignore_index=True)
        setups, _stats = tmm.build_touch_setups(
            frame[["symbol", "bar_start", "open", "high", "low", "close"]])
        day = setups[setups["date"] == self.date].reset_index(drop=True)
        # dtype=bool guards the empty case (df[[]] would select COLUMNS, not rows)
        mask = np.asarray([(s, dt) not in self.records
                           for s, dt in zip(day["symbol"], day["decision_time"])],
                          dtype=bool)
        fresh = day[mask].sort_values(["bar_start", "symbol"]).reset_index(drop=True)

        new_records = []
        if len(fresh):
            graded = self._grade_events(fresh, day)
            for i, row in fresh.iterrows():
                rec = self._build_record(row, day, graded.get(i), backlog=backlog)
                self.records[(rec["symbol"], rec["decision_time"])] = rec
                self.order.append((rec["symbol"], rec["decision_time"]))
                new_records.append(rec)
                if emit and rec["symbol"] in self.alert_symbols:
                    self._emit(rec)
                self._persist_record(rec)
        self._backfill()
        return new_records

    # -- tier-2 grading ------------------------------------------------------ #
    def _grade_events(self, fresh, day_setups):
        """{fresh positional index -> grade payload}. Never raises: any scoring
        problem becomes a per-batch failure payload (tier 1 already fired)."""
        out = {}
        tod = fresh["decision_time"].str.slice(11, 19)
        window_ok = tod <= GRADE_LAST_DECISION
        if self.adapter is None:
            reason = f"등급 비활성: {self.grade_disabled_reason}"
            return {i: {"status": reason} for i in fresh.index}
        for i in fresh.index[~window_ok]:
            out[i] = {"status": "창 불완전(마감 임박) — 점수 없음"}
        cand = fresh[window_ok]
        if cand.empty:
            return out
        try:
            covered, rel = self._serving_frames()
            cov_keys = set(zip(covered["symbol"], covered["decision_time"])) \
                if len(covered) else set()
            in_cov = [ (s, dt) in cov_keys
                       for s, dt in zip(cand["symbol"], cand["decision_time"])]
            for i in cand.index[~np.asarray(in_cov)]:
                out[i] = {"status": "워밍업(피처 미커버) — 점수 없음"}
            scorable = cand[np.asarray(in_cov)]
            if len(scorable):
                probs = self.adapter.score_events(scorable, covered, rel)
                grades = tmm.apply_grade(probs, self.grade_map)
                for pos, i in enumerate(scorable.index):
                    g = grades[pos]
                    rate, n = ts.grade_display(self.grade_map, g)
                    out[i] = {"status": GRADE_STATUS_OK, "score": float(probs[pos]),
                              "grade": str(g), "rate": rate, "n": n}
        except Exception as exc:                    # noqa: BLE001 -- isolate (mod #7)
            for i in cand.index:
                if i not in out:
                    out[i] = {"status": f"점수 실패: {type(exc).__name__}: {exc}"}
        return out

    def _serving_frames(self):
        """Feature-covered merged frame + relatives, rebuilt ONLY when a gradable
        event exists (crosses are sparse; the per-minute path never pays this)."""
        today_reg = self.today[
            (self.today["bar_start"].str.slice(11, 19) >= mcs.REGULAR_START)
            & (self.today["bar_start"].str.slice(11, 19) <= mcs.REGULAR_END)]
        window = pd.concat([self.feat_history, today_reg], ignore_index=True)
        return ts.build_serving_frames(window, source=self.source)

    # -- record/emit ---------------------------------------------------------- #
    def _build_record(self, row, day_setups, grade_payload, *, backlog):
        sym, dt = str(row["symbol"]), str(row["decision_time"])
        t_conf = pd.Timestamp(row["bar_start"])
        same = day_setups[(day_setups["symbol"] == sym)
                          & (day_setups["direction"] == -int(row["direction"]))]
        t_all = pd.to_datetime(same["bar_start"])
        recross = bool(((t_all >= t_conf - pd.Timedelta(minutes=RECROSS_WINDOW_MIN))
                        & (t_all < t_conf)).any()) if len(same) else False
        now = pd.Timestamp.now()
        latency = None
        if self.live and not backlog:   # meaningless against historical replay bars
            latency = float((now - pd.Timestamp(dt)).total_seconds() * 1000.0)
        payload = grade_payload or {"status": "점수 없음"}
        rec = {
            "symbol": sym, "decision_time": dt, "date": str(row["date"]),
            "direction": int(row["direction"]), "band": str(row["band"]),
            "dist_pct": float(row["dist_pct"]), "dist_atr": float(row["dist_atr"]),
            "cluster_first": int(bool(row["cluster_first"])),
            "whipsaw_recross": int(recross),
            "bar_start": str(row["bar_start"]),
            "cross_confirmed_time": dt,
            "alert_emitted_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            # latency includes the collector write + poll delay (documented; the
            # 64-bit side only ever sees COMPLETED bars, there is no tick stream)
            "alert_latency_ms": latency,
            "price_at_alert": float(row["close"]),   # confirm-bar close = last known
            "score": payload.get("score"), "grade": payload.get("grade"),
            "grade_oof_touch_rate": payload.get("rate"),
            "grade_oof_n": payload.get("n"),
            "grade_status": payload["status"],
            "offline_reference_entry": None, "entry_slippage_vs_open_t1": None,
            "realized_touch_05": None, "realized_mae": None,
            "window_incomplete": None,
            "backlog": int(bool(backlog)),
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "_backfill_done": False,
        }
        return rec

    def _emit(self, rec):
        tod = rec["decision_time"][11:16]
        arrow = "상향돌파" if rec["direction"] > 0 else "하향돌파"
        head = (f"🔔 {tod} {rec['symbol']} {arrow} [{rec['band']} "
                f"d={_fmt_pct(rec['dist_pct'], 2)} atr={rec['dist_atr']:.2f}]")
        if rec["whipsaw_recross"]:
            head += " 재교차(휩쏘: 직전 반대신호 청산 시그널)"
        if rec["grade_status"] == GRADE_STATUS_OK:
            tail = (f" | score={rec['score']:.3f} 등급={rec['grade']} | "
                    f"동일 등급 OOF 과거 터치율 {_fmt_pct(rec['grade_oof_touch_rate'])} "
                    f"(N={rec['grade_oof_n']:,})")
        else:
            tail = f" | {rec['grade_status']}"
        prefix = "(지난 신호) " if rec["backlog"] else ""
        print(f"  {prefix}{head}{tail}")

    # -- realized-outcome backfill (user directive #3 / #1) ------------------- #
    def _backfill(self):
        """entry open[t+1] + slippage as soon as the next bar lands; realized
        touch/MAE when the 10-minute window closes. Mirrors the offline
        definitions: entry bar = the NEXT bar in the 08:00-REGULAR_END stream,
        window = bars with bar_start in [entry_ts, entry_ts + window min),
        complete = a bar exists at/beyond entry_ts + (window-1) min."""
        pending = [k for k in self.order if not self.records[k]["_backfill_done"]]
        if not pending:
            return
        stream = self.today[self.today["bar_start"].str.slice(11, 19)
                            <= mcs.REGULAR_END]
        for key in pending:
            rec = self.records[key]
            bars = stream[stream["symbol"] == rec["symbol"]]
            after = bars[bars["bar_start"] > rec["bar_start"]]
            if after.empty:
                continue
            entry_bar = after.iloc[0]
            if rec["offline_reference_entry"] is None:
                entry_open = float(entry_bar["open"])
                rec["offline_reference_entry"] = entry_open
                rec["entry_slippage_vs_open_t1"] = float(
                    rec["direction"] * (entry_open / rec["price_at_alert"] - 1.0))
                self._persist_record(rec)
            t_entry = pd.Timestamp(entry_bar["bar_start"])
            t_hi = t_entry + pd.Timedelta(minutes=tmm.TOUCH_WINDOW_MIN)
            ats = pd.to_datetime(after["bar_start"])
            if not bool((ats >= t_entry
                         + pd.Timedelta(minutes=tmm.TOUCH_WINDOW_MIN - 1)).any()):
                continue                       # window still open
            win = after[(ats >= t_entry) & (ats < t_hi)]
            entry = float(rec["offline_reference_entry"])
            d = float(rec["direction"])
            close = win["close"].astype(float)
            best_close = close.max() if d > 0 else close.min()
            adv = (win["low"].astype(float).min() if d > 0
                   else win["high"].astype(float).max())
            rec["realized_touch_05"] = int(
                d * (best_close / entry - 1.0) >= tmm.TOUCH_THRESHOLD)
            rec["realized_mae"] = float(d * (adv / entry - 1.0))
            rec["window_incomplete"] = 0
            rec["_backfill_done"] = True
            self._persist_record(rec)
            if self.verbose and rec["symbol"] in self.alert_symbols:
                hit = "터치 ✓" if rec["realized_touch_05"] else "미터치"
                print(f"    ({rec['decision_time'][11:16]} {rec['symbol']} 결과: "
                      f"{hit}, MAE={_fmt_pct(rec['realized_mae'], 2)})")

    def finalize(self):
        """Session end: any window that could not complete is marked incomplete
        (offline drops these from the population; live keeps them tagged so the
        realized stats can exclude them)."""
        for key in self.order:
            rec = self.records[key]
            if not rec["_backfill_done"]:
                rec["window_incomplete"] = 1
                rec["_backfill_done"] = True
                self._persist_record(rec)

    # -- session summary ------------------------------------------------------ #
    def summary(self):
        if not self.order:
            print("\n  (no crosses today)")
            return
        rows = pd.DataFrame([self.records[k] for k in self.order])
        rows.drop(columns=["_backfill_done"]).to_csv(
            LEDGER_PATH, index=False, encoding="utf-8-sig")
        n_alert = int(rows["symbol"].isin(self.alert_symbols).sum())
        print(f"\n  crosses={len(rows)} (alerted symbols: {n_alert})  "
              f"ledger -> {LEDGER_PATH}")
        graded = rows[rows["grade_status"] == GRADE_STATUS_OK]
        if len(graded):
            # grade-share drift monitor: OOF thresholds vs the prod score scale
            names = [g for g, _ in tmm.GRADE_BOUNDARIES]
            expected = {}
            prev = 1.0
            for g, p in tmm.GRADE_BOUNDARIES:
                expected[g] = prev - p
                prev = p
            share = graded["grade"].value_counts(normalize=True)
            line = "  ".join(
                f"{g}:{_fmt_pct(share.get(g, 0.0))}(기대 {_fmt_pct(expected[g])})"
                for g in names)
            print(f"  grade share  {line}")
            done = graded[(graded["window_incomplete"] == 0)
                          & graded["realized_touch_05"].notna()]
            if len(done):
                by = done.groupby("grade")["realized_touch_05"].agg(["mean", "count"])
                parts = []
                for g in names:
                    if g in by.index:
                        oof, _n = ts.grade_display(self.grade_map, g) \
                            if self.grade_map else (float("nan"), 0)
                        parts.append(f"{g}:{_fmt_pct(by.loc[g, 'mean'])}"
                                     f"/{int(by.loc[g, 'count'])}건"
                                     f"(OOF {_fmt_pct(oof)})")
                print("  realized(0.5% 터치)  " + "  ".join(parts))
        else:
            reason = self.grade_disabled_reason or "gradable cross 없음"
            print(f"  (등급 라인 없음 — {reason})")


# --------------------------------------------------------------------------- #
# replay / live drivers
# --------------------------------------------------------------------------- #
def _latest_day(conn, table, col):
    row = conn.execute(
        f"SELECT MAX(substr({col},1,10)) FROM {table}").fetchone()
    return row[0] if row and row[0] else None


def run_replay(runner, *, date, speed=0.0):
    conn = connect(DB_PATH)
    try:
        runner.seed(conn, date)
        ph = ",".join("?" * len(runner.symbols))
        day = pd.read_sql_query(
            f"SELECT {', '.join(BAR_COLUMNS)} FROM bars_1m WHERE source='al_raw' "
            f"AND symbol IN ({ph}) AND substr(bar_start,1,10)=? ORDER BY bar_start",
            conn, params=(*runner.symbols, date))
    finally:
        conn.close()
    if day.empty:
        raise SystemExit(f"no al_raw bars_1m on {date}.")
    print(f"replay {date}: {len(day):,} bars over {day['symbol'].nunique()} symbols "
          f"(minute-by-minute, alerts for {len(runner.alert_symbols)} symbols)\n")
    for _, minute in day.groupby("bar_start", sort=True):
        runner.ingest(minute)
        if speed > 0:
            time.sleep(speed)
    runner.finalize()
    runner.summary()


def run_live(runner, *, today, poll_sec, emit_backlog=True):
    runner.live = True
    runner.enable_persistence(DB_PATH)
    print(f"live touch-alert: polling realtime_minute_bars every {poll_sec}s "
          "(Ctrl+C to stop). Run the 32-bit collector with --all-symbols.")
    print(f"  alerts -> DB {ALERT_TABLE} (+ CSV ledger on stop); premarket bars "
          "feed the MA stream; a poll with no new completed bar does nothing.")
    if runner.grade_disabled_reason:
        print(f"  ⚠ 등급 라인 비활성(돌파 알림 전용): {runner.grade_disabled_reason}")
    print()
    conn = connect(DB_PATH)
    try:
        runner.seed(conn, today)
    finally:
        conn.close()
    seen = set()
    primed = False
    ph = ",".join("?" * len(runner.symbols))
    while True:
        conn = connect(DB_PATH)
        try:
            snap = pd.read_sql_query(
                "SELECT symbol, ts_kst, open, high, low, close, volume "
                f"FROM realtime_minute_bars WHERE symbol IN ({ph}) "
                "AND substr(ts_kst,1,10)=? ORDER BY ts_kst",
                conn, params=(*runner.symbols, today))
        finally:
            conn.close()
        if len(snap):
            # full-tuple keys: a collector rewrite (DELETE+INSERT) changes the
            # tuple, so the bar re-ingests and the batch recompute self-heals.
            snap["_key"] = list(zip(snap["symbol"], snap["ts_kst"], snap["open"],
                                    snap["high"], snap["low"], snap["close"],
                                    snap["volume"]))
            fresh = snap[~snap["_key"].isin(seen)]
            if len(fresh):
                seen.update(fresh["_key"])
                bars = pd.DataFrame({
                    "symbol": fresh["symbol"].astype(str),
                    "bar_start": fresh["ts_kst"].astype(str),
                    "bar_end": (pd.to_datetime(fresh["ts_kst"])
                                + pd.Timedelta(minutes=1))
                    .dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": fresh["open"], "high": fresh["high"],
                    "low": fresh["low"], "close": fresh["close"],
                    "volume": fresh["volume"],
                })
                if not primed:
                    n_backlog = len(bars)
                    tag = ("replaying" if emit_backlog else "priming silently")
                    print(f"  {tag} {n_backlog} backlog bars (open→now):")
                    runner.ingest(bars, backlog=True, emit=emit_backlog)
                    print("  --- caught up; watching for new completed bars ---")
                else:
                    runner.ingest(bars)
        primed = True
        time.sleep(poll_sec)


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-tier realtime touch alert (breakout + frozen-OOF grade).")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--replay-date", default=None,
                   help="YYYY-MM-DD (default: latest al_raw day).")
    p.add_argument("--replay-speed", type=float, default=0.0,
                   help="seconds between replayed minutes (0 = fast).")
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--today", default=None,
                   help="live trading date (default: latest realtime_minute_bars day).")
    p.add_argument("--symbols", default="",
                   help="comma list of symbols to ALERT on (default: every active "
                        "symbol). Bars are always ingested for the WHOLE active set "
                        "(the relative features need the cross-section).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--grade-map", default=None,
                   help="grade-map JSON path (default models/touch_grade_map.json).")
    p.add_argument("--models-dir", default=None)
    p.add_argument("--artifact-suffix", default="",
                   help="prod-weight namespace for smoke runs (e.g. _smoke).")
    p.add_argument("--no-backlog", action="store_true",
                   help="live mode: register the morning's earlier crosses silently "
                        "instead of printing them as (지난 신호).")
    p.add_argument("--verbose", action="store_true",
                   help="also print each alert's realized-outcome line when its "
                        "window closes.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    symbols = [normalize_symbol_code(s) for s in load_active_symbols()]
    alert_symbols = ([normalize_symbol_code(s) for s in args.symbols.split(",") if s]
                     if args.symbols else symbols)
    unknown = [s for s in alert_symbols if s not in symbols]
    if unknown:
        raise SystemExit(f"--symbols not in the active set: {unknown}")

    runner = TouchAlertRunner(symbols, alert_symbols, device=args.device,
                              grade_map_path=args.grade_map,
                              models_dir=args.models_dir,
                              artifact_suffix=args.artifact_suffix,
                              verbose=args.verbose)
    mode_note = ("등급+돌파" if runner.adapter is not None
                 else f"돌파 전용 (등급 비활성: {runner.grade_disabled_reason})")
    print(f"[{now_kst_text()}] touch alert  symbols={len(symbols)} "
          f"alerts={len(alert_symbols)}  mode={mode_note}")
    print(f"  tier-2 gate: feature-covered AND decision_time <= {GRADE_LAST_DECISION} "
          f"(= REGULAR_END - {tmm.TOUCH_WINDOW_MIN - 1}m, derived)")

    if args.mode == "replay":
        date = args.replay_date
        if not date:
            conn = connect(DB_PATH)
            try:
                date = _latest_day(conn, "bars_1m", "bar_start")
            finally:
                conn.close()
        run_replay(runner, date=date, speed=args.replay_speed)
    else:
        today = args.today
        if not today:
            conn = connect(DB_PATH)
            try:
                today = _latest_day(conn, "realtime_minute_bars", "ts_kst") \
                    or pd.Timestamp.now().strftime("%Y-%m-%d")
            finally:
                conn.close()
        try:
            run_live(runner, today=today, poll_sec=args.poll_sec,
                     emit_backlog=not args.no_backlog)
        except KeyboardInterrupt:
            runner.finalize()
            runner.summary()


if __name__ == "__main__":
    main()
