"""Step 14 capstone: 32-bit Kiwoom real-time 1-minute collector -> DB bridge.

Reference-derived (test_realtime_integrated_chart.py): logs in to Kiwoom OpenAPI+,
seeds today's bars from one opt10080 snapshot, then subscribes real-time ticks and
aggregates them into 1-minute OHLCV bars. Each FINISHED bar is written to the
shared ``realtime_minute_bars`` table so the 64-bit engine
(scripts/realtime_trend_alert.py --mode live) can read it and score alerts. The
split exists because Kiwoom needs 32-bit Python while LightGBM/sklearn need 64-bit.

The DB symbol is the BASE 6-digit code (e.g. 000660), even when subscribing the
integrated quote code (000660_AL), so it matches bars_1m / the engine's lookups.

Run with 32-bit Python (cwd = quant):
    ..\\Python311-32\\python.exe scripts\\realtime_alert_collector.py --code 000660_AL
"""

from __future__ import annotations

import argparse
import platform
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QAxContainer import QAxWidget
except ImportError as exc:  # pragma: no cover - depends on 32-bit PyQt5
    raise SystemExit(
        "PyQt5 with QAxContainer is required in 32-bit Python:\n"
        "  C:\\OpenAPI\\Python311-32\\python.exe -m pip install PyQt5"
    ) from exc

from config import DB_PATH, connect              # noqa: E402
from settings import load_active_symbols, normalize_symbol_code  # noqa: E402

TR_CODE = "opt10080"
RQ_NAME = "integrated_minute_snapshot"
REALTIME_SOURCE = "realtime"

FIELD_CODE = "종목코드"
FIELD_TICK_RANGE = "틱범위"
FIELD_ADJUSTED = "수정주가구분"
FIELD_CLOSE_AUCTION_BAR = "종가매매분봉"
FIELD_DATETIME = "체결시간"
FIELD_OPEN = "시가"
FIELD_HIGH = "고가"
FIELD_LOW = "저가"
FIELD_CLOSE = "현재가"
FIELD_VOLUME = "거래량"

FID_TRADE_TIME = 20
FID_PRICE = 10
FID_TRADE_VOLUME = 15
FID_ACC_VOLUME = 13
REALTIME_FIDS = ";".join(str(f) for f in (FID_TRADE_TIME, FID_PRICE, FID_TRADE_VOLUME, FID_ACC_VOLUME))


def _to_int(value, *, absolute=False):
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return abs(number) if absolute else number


def _minute_to_ts_kst(minute14: str) -> str:
    """'YYYYMMDDHHMM00' -> 'YYYY-MM-DD HH:MM:SS' (the realtime_minute_bars key)."""
    m = minute14
    return f"{m[0:4]}-{m[4:6]}-{m[6:8]} {m[8:10]}:{m[10:12]}:00"


class MinuteBar:
    __slots__ = ("minute", "open", "high", "low", "close", "volume")

    def __init__(self, minute, price):
        self.minute = minute
        self.open = self.high = self.low = self.close = price
        self.volume = 0

    def update(self, price, volume):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += abs(volume)


class RealtimeCollector(QAxWidget):
    """Subscribes one or many Kiwoom codes; writes each finished 1m bar to the DB.

    Multi-symbol support is what makes the same-timestamp alignment check possible:
    every symbol's finished minute lands in realtime_minute_bars keyed by (symbol,
    ts_kst), so rows for the same ts_kst across symbols line up.
    """

    def __init__(self, *, codes, screen_no, db_path):
        super().__init__()
        self.codes = list(codes)
        self.db_symbol = {c: normalize_symbol_code(c.split("_")[0]) for c in self.codes}
        self.screen_no = screen_no
        self.db_path = str(db_path)

        if self.setControl("KHOPENAPI.KHOpenAPICtrl.1") is False or self.isNull():
            raise RuntimeError("Failed to load KHOPENAPI.KHOpenAPICtrl.1")
        self.OnEventConnect.connect(self._on_event_connect)
        self.OnReceiveTrData.connect(self._on_receive_tr_data)
        self.OnReceiveRealData.connect(self._on_receive_real_data)

        self._login_loop: Optional[QEventLoop] = None
        self._snapshot_loop: Optional[QEventLoop] = None
        self._snapshot_rows = []
        self._pending_code: Optional[str] = None
        self._connected = False
        self._bars = {}        # kiwoom code -> in-progress MinuteBar
        self._written = 0

    # -- DB --------------------------------------------------------------- #
    def _write_bar(self, db_symbol, ts_kst, o, h, l, c, v):
        conn = connect(self.db_path)
        try:
            conn.execute("DELETE FROM realtime_minute_bars WHERE symbol=? AND ts_kst=?",
                         (db_symbol, ts_kst))
            conn.execute(
                "INSERT INTO realtime_minute_bars (symbol, ts_kst, open, high, low, close, volume, source, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (db_symbol, ts_kst, o, h, l, c, v, REALTIME_SOURCE,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        finally:
            conn.close()
        self._written += 1

    # -- Kiwoom flow ------------------------------------------------------ #
    def login(self):
        if int(self.dynamicCall("GetConnectState()")) == 1:
            self._connected = True
            return
        self._login_loop = QEventLoop()
        if int(self.dynamicCall("CommConnect()")) != 0:
            raise RuntimeError("CommConnect failed")
        self._login_loop.exec_()
        if not self._connected:
            raise RuntimeError("Kiwoom login failed or cancelled.")

    def seed_today_snapshots(self):
        """Backfill today's already-formed 1m bars for every code (sequential TRs)."""
        today = datetime.now().strftime("%Y%m%d")
        for code in self.codes:
            self._snapshot_rows = []
            self._pending_code = code
            self.dynamicCall("SetInputValue(QString, QString)", FIELD_CODE, code)
            self.dynamicCall("SetInputValue(QString, QString)", FIELD_TICK_RANGE, "1")
            self.dynamicCall("SetInputValue(QString, QString)", FIELD_ADJUSTED, "1")
            self.dynamicCall("SetInputValue(QString, QString)", FIELD_CLOSE_AUCTION_BAR, "0")
            self._snapshot_loop = QEventLoop()
            if int(self.dynamicCall("CommRqData(QString, QString, int, QString)",
                                    RQ_NAME, TR_CODE, 0, self.screen_no)) != 0:
                print(f"  seed CommRqData failed for {code}")
                continue
            self._snapshot_loop.exec_()
            seeded = 0
            for dt14, o, h, l, c, v in sorted(self._snapshot_rows, key=lambda r: r[0]):
                if dt14[:8] != today:
                    continue
                self._write_bar(self.db_symbol[code], _minute_to_ts_kst(dt14), o, h, l, c, v or 0)
                seeded += 1
            print(f"seeded_today {self.db_symbol[code]} bars={seeded}")
            self._sleep_ms(250)   # brief gap to respect Kiwoom's TR rate limit

    def _sleep_ms(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec_()

    def subscribe(self):
        # one SetRealReg with all codes; OnReceiveRealData carries the code per tick
        joined = ";".join(self.codes)
        if int(self.dynamicCall("SetRealReg(QString, QString, QString, QString)",
                                self.screen_no, joined, REALTIME_FIDS, "0")) != 0:
            raise RuntimeError("SetRealReg failed")
        print(f"subscribed {len(self.codes)} codes -> {list(self.db_symbol.values())} screen={self.screen_no}")
        print("waiting_for_realtime_ticks (writes each FINISHED minute per symbol to realtime_minute_bars)")

    def stop(self):
        for code, bar in self._bars.items():   # flush in-progress bars
            if bar is not None:
                self._write_bar(self.db_symbol[code], _minute_to_ts_kst(bar.minute),
                                bar.open, bar.high, bar.low, bar.close, bar.volume)
        self.dynamicCall("DisconnectRealData(QString)", self.screen_no)
        print(f"stopped: wrote {self._written} bars to realtime_minute_bars")
        QApplication.instance().quit()

    # -- callbacks -------------------------------------------------------- #
    def _on_event_connect(self, err_code):
        self._connected = err_code == 0
        if self._login_loop is not None:
            self._login_loop.exit()

    def _on_receive_tr_data(self, screen_no, rq_name, tr_code, record_name, prev_next, *_):
        if rq_name != RQ_NAME:
            return
        n = int(self.dynamicCall("GetRepeatCnt(QString, QString)", tr_code, rq_name))
        rows = []
        for i in range(n):
            def g(field):
                return str(self.dynamicCall("GetCommData(QString, QString, int, QString)",
                                            tr_code, rq_name, i, field)).strip()
            dt = g(FIELD_DATETIME)[:14]
            rows.append((dt, _to_int(g(FIELD_OPEN), absolute=True), _to_int(g(FIELD_HIGH), absolute=True),
                         _to_int(g(FIELD_LOW), absolute=True), _to_int(g(FIELD_CLOSE), absolute=True),
                         _to_int(g(FIELD_VOLUME), absolute=True)))
        self._snapshot_rows = rows
        if self._snapshot_loop is not None:
            self._snapshot_loop.exit()

    def _on_receive_real_data(self, code, real_type, real_data):
        if code not in self.db_symbol:
            return
        price = _to_int(self.dynamicCall("GetCommRealData(QString, int)", code, FID_PRICE), absolute=True)
        trade_time = str(self.dynamicCall("GetCommRealData(QString, int)", code, FID_TRADE_TIME)).strip()
        volume = _to_int(self.dynamicCall("GetCommRealData(QString, int)", code, FID_TRADE_VOLUME)) or 0
        if price is None or len(trade_time) < 4:
            return
        minute = datetime.now().strftime("%Y%m%d") + trade_time[:4] + "00"
        bar = self._bars.get(code)
        if bar is None:
            self._bars[code] = MinuteBar(minute, price)
        elif bar.minute != minute:
            self._write_bar(self.db_symbol[code], _minute_to_ts_kst(bar.minute),
                            bar.open, bar.high, bar.low, bar.close, bar.volume)
            print(f"bar {self.db_symbol[code]} {_minute_to_ts_kst(bar.minute)} "
                  f"O{bar.open} H{bar.high} L{bar.low} C{bar.close} V{bar.volume}")
            self._bars[code] = MinuteBar(minute, price)
        self._bars[code].update(price, volume)


def _resolve_codes(args):
    if args.codes:
        raw = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.all_symbols:
        raw = [s + args.suffix for s in load_active_symbols()]
    elif args.code:
        raw = [args.code]
    else:
        actives = load_active_symbols()
        raw = [actives[0] + args.suffix] if actives else []
    return raw


def parse_args():
    p = argparse.ArgumentParser(description="32-bit Kiwoom realtime 1m collector -> realtime_minute_bars.")
    p.add_argument("--code", default=None, help="single Kiwoom code, e.g. 000660_AL.")
    p.add_argument("--codes", default=None, help="comma-separated Kiwoom codes (multi-symbol).")
    p.add_argument("--all-symbols", action="store_true",
                   help="subscribe ALL active symbols (enables the same-timestamp alignment check).")
    p.add_argument("--suffix", default="_AL", help="quote suffix for --all-symbols (_AL integrated, _NX NXT).")
    p.add_argument("--screen-no", default="2150")
    p.add_argument("--duration-sec", type=int, default=0, help="0 = run until closed.")
    p.add_argument("--no-seed", action="store_true", help="skip the opt10080 today-backfill snapshot.")
    p.add_argument("--db", default=str(DB_PATH))
    return p.parse_args()


def main():
    args = parse_args()
    if platform.architecture()[0] != "32bit":
        raise SystemExit("Kiwoom OpenAPI+ requires 32-bit Python:\n"
                         "  C:\\OpenAPI\\Python311-32\\python.exe scripts\\realtime_alert_collector.py")
    codes = _resolve_codes(args)
    if not codes:
        raise SystemExit("no codes resolved (set --code/--codes/--all-symbols or configure active symbols).")

    app = QApplication.instance() or QApplication(sys.argv)
    collector = RealtimeCollector(codes=codes, screen_no=args.screen_no, db_path=args.db)
    collector.login()
    if not args.no_seed:
        collector.seed_today_snapshots()
    collector.subscribe()
    if args.duration_sec > 0:
        QTimer.singleShot(args.duration_sec * 1000, collector.stop)
    app.exec_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
