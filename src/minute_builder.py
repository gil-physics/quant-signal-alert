from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from config import REALTIME_MINUTE_BARS_DDL, REALTIME_SOURCE
from utils import now_kst_text


@dataclass
class RealtimeMinuteBar:
    symbol: str
    ts_kst: str
    open: int
    high: int
    low: int
    close: int
    volume: int
    source: str
    created_at: str

    def as_db_tuple(self) -> tuple:
        return (
            self.symbol,
            self.ts_kst,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.source,
            self.created_at,
        )


class MinuteBuilder:
    def __init__(self, *, db_path: Path, logger: logging.Logger) -> None:
        self.db_path = db_path
        self.logger = logger
        self.current_bar: Optional[RealtimeMinuteBar] = None
        self.finalized_count = 0
        self.inserted_count = 0
        self.duplicate_count = 0
        self.initialize_storage()

    def initialize_storage(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(REALTIME_MINUTE_BARS_DDL)
            conn.commit()

    def on_tick(self, tick: Dict[str, object]) -> None:
        symbol = str(tick["symbol"])
        ts_kst = str(tick["minute_ts_kst"])
        price = int(tick["price"])
        volume_delta = int(tick.get("volume_delta") or 0)

        if self.current_bar is None:
            self.current_bar = self._new_bar(symbol=symbol, ts_kst=ts_kst, price=price)
            self.logger.info("minute bar started symbol=%s ts_kst=%s price=%s", symbol, ts_kst, price)

        if self.current_bar.ts_kst != ts_kst:
            self.flush_current(reason="minute_changed")
            self.current_bar = self._new_bar(symbol=symbol, ts_kst=ts_kst, price=price)
            self.logger.info("minute bar started symbol=%s ts_kst=%s price=%s", symbol, ts_kst, price)

        self.current_bar.high = max(self.current_bar.high, price)
        self.current_bar.low = min(self.current_bar.low, price)
        self.current_bar.close = price
        self.current_bar.volume += max(0, volume_delta)

        self.logger.info(
            "minute bar updated symbol=%s ts_kst=%s open=%s high=%s low=%s close=%s volume=%s",
            self.current_bar.symbol,
            self.current_bar.ts_kst,
            self.current_bar.open,
            self.current_bar.high,
            self.current_bar.low,
            self.current_bar.close,
            self.current_bar.volume,
        )

    def flush_current(self, *, reason: str = "flush") -> Optional[RealtimeMinuteBar]:
        if self.current_bar is None:
            return None

        bar = self.current_bar
        self.current_bar = None
        inserted = self._save_bar(bar)
        self.finalized_count += 1
        if inserted:
            self.inserted_count += 1
        else:
            self.duplicate_count += 1

        self.logger.info(
            "minute bar finalized reason=%s inserted=%s symbol=%s ts_kst=%s "
            "open=%s high=%s low=%s close=%s volume=%s",
            reason,
            inserted,
            bar.symbol,
            bar.ts_kst,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        return bar

    def _save_bar(self, bar: RealtimeMinuteBar) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(REALTIME_MINUTE_BARS_DDL)
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO realtime_minute_bars (
                    symbol, ts_kst, open, high, low, close, volume, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                bar.as_db_tuple(),
            )
            conn.commit()
        return int(cursor.rowcount or 0) == 1

    @staticmethod
    def _new_bar(*, symbol: str, ts_kst: str, price: int) -> RealtimeMinuteBar:
        return RealtimeMinuteBar(
            symbol=symbol,
            ts_kst=ts_kst,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0,
            source=REALTIME_SOURCE,
            created_at=now_kst_text(),
        )
