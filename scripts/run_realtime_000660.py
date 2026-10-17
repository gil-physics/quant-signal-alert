from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from config import DB_PATH, REALTIME_LOG_PATH, ensure_directories
from minute_builder import MinuteBuilder
from realtime_receiver import RealtimeReceiver
from settings import default_symbol
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kiwoom realtime 1-minute builder.")
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol code. Defaults to the first active symbol in config/symbols_test.yaml.",
    )
    parser.add_argument("--screen-no", default="2000")
    parser.add_argument("--duration-sec", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument("--quiet-ticks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(REALTIME_LOG_PATH, name="realtime")
    symbol = args.symbol or default_symbol()

    app = QApplication.instance() or QApplication(sys.argv)
    builder = MinuteBuilder(db_path=DB_PATH, logger=logger)

    def on_tick(tick: dict) -> None:
        if not args.quiet_ticks:
            print(
                "tick "
                f"symbol={tick['symbol']} time={tick['trade_time']} "
                f"price={tick['price']} acc_volume={tick['acc_volume']} "
                f"volume_delta={tick['volume_delta']}"
            )
        builder.on_tick(tick)

    receiver = RealtimeReceiver(
        logger=logger,
        on_tick=on_tick,
        screen_no=args.screen_no,
    )
    stopping = {"value": False}

    def shutdown() -> None:
        if stopping["value"]:
            return
        stopping["value"] = True
        bar = builder.flush_current(reason="shutdown")
        if bar is not None:
            print(
                "bar_final "
                f"symbol={bar.symbol} ts_kst={bar.ts_kst} open={bar.open} high={bar.high} "
                f"low={bar.low} close={bar.close} volume={bar.volume}"
            )
        receiver.disconnect()
        print(
            "summary "
            f"finalized={builder.finalized_count} inserted={builder.inserted_count} "
            f"duplicates={builder.duplicate_count}"
        )
        app.quit()

    signal.signal(signal.SIGINT, lambda signum, frame: shutdown())
    signal.signal(signal.SIGTERM, lambda signum, frame: shutdown())

    if not receiver.login():
        print("[FAIL] Login failed")
        return 1
    print("[OK] Login success")

    receiver.register(symbol=symbol)
    print(f"[OK] SetRealReg success: symbol={symbol}")

    keep_signal_timer = QTimer()
    keep_signal_timer.timeout.connect(lambda: None)
    keep_signal_timer.start(500)

    if args.duration_sec > 0:
        QTimer.singleShot(args.duration_sec * 1000, shutdown)

    app.exec_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
