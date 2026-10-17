from __future__ import annotations

import logging
import platform
from typing import Callable, Dict, Optional

from PyQt5.QtCore import QEventLoop
from PyQt5.QAxContainer import QAxWidget

from utils import clean_non_negative_int, clean_price, kst_minute_from_hhmmss


REAL_TYPE_STOCK_TRADE = "\uc8fc\uc2dd\uccb4\uacb0"

FID_TRADE_TIME = 20
FID_PRICE = 10
FID_ACC_VOLUME = 13
FID_ACC_VALUE = 14
FID_OPEN = 16
FID_HIGH = 17
FID_LOW = 18
FID_BEST_ASK = 27
FID_BEST_BID = 28

REALTIME_FIDS = ";".join(
    str(fid)
    for fid in [
        FID_TRADE_TIME,
        FID_PRICE,
        FID_ACC_VOLUME,
        FID_ACC_VALUE,
        FID_OPEN,
        FID_HIGH,
        FID_LOW,
        FID_BEST_ASK,
        FID_BEST_BID,
    ]
)


class RealtimeReceiver(QAxWidget):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        on_tick: Callable[[Dict[str, object]], None],
        screen_no: str = "2000",
    ) -> None:
        if platform.architecture()[0] != "32bit":
            raise RuntimeError("Kiwoom OpenAPI+ realtime receiver requires 32-bit Python.")

        super().__init__()
        self.logger = logger
        self.on_tick = on_tick
        self.screen_no = screen_no
        self.symbol = ""
        self._last_acc_volume_by_symbol: Dict[str, int] = {}
        self._login_loop: Optional[QEventLoop] = None
        self._connected = False

        ok = self.setControl("KHOPENAPI.KHOpenAPICtrl.1")
        if ok is False or self.isNull():
            raise RuntimeError("Failed to load KHOPENAPI.KHOpenAPICtrl.1")

        self.OnEventConnect.connect(self._on_event_connect)
        self.OnReceiveRealData.connect(self._on_receive_real_data)

    def login(self) -> bool:
        if int(self.dynamicCall("GetConnectState()")) == 1:
            self._connected = True
            self.logger.info("login success: already connected")
            return True

        self.logger.info("login start")
        self._login_loop = QEventLoop()
        ret = int(self.dynamicCall("CommConnect()"))
        if ret != 0:
            self.logger.error("login request failed ret=%s", ret)
            return False

        self._login_loop.exec_()
        if self._connected:
            self.logger.info("login success")
        else:
            self.logger.error("login failed or cancelled")
        return self._connected

    def register(self, *, symbol: str) -> None:
        self.symbol = symbol
        ret = int(
            self.dynamicCall(
                "SetRealReg(QString, QString, QString, QString)",
                self.screen_no,
                symbol,
                REALTIME_FIDS,
                "0",
            )
        )
        if ret != 0:
            self.logger.error("SetRealReg failed symbol=%s ret=%s", symbol, ret)
            raise RuntimeError(f"SetRealReg failed: {ret}")

        self.logger.info(
            "SetRealReg success screen_no=%s symbol=%s fids=%s",
            self.screen_no,
            symbol,
            REALTIME_FIDS,
        )

    def disconnect(self) -> None:
        self.dynamicCall("DisconnectRealData(QString)", self.screen_no)
        self.logger.info("DisconnectRealData screen_no=%s", self.screen_no)

    def _on_event_connect(self, err_code: int) -> None:
        self._connected = err_code == 0
        if self._login_loop is not None:
            self._login_loop.exit()

    def _on_receive_real_data(self, code: str, real_type: str, real_data: str) -> None:
        if self.symbol and code != self.symbol:
            return
        if real_type and real_type != REAL_TYPE_STOCK_TRADE:
            return

        try:
            tick = self._read_tick(code=code, real_type=real_type)
        except ValueError:
            self.logger.exception("failed to parse realtime tick code=%s real_type=%s", code, real_type)
            return

        self.logger.info("realtime tick received %s", tick)
        self.on_tick(tick)

    def _read_tick(self, *, code: str, real_type: str) -> Dict[str, object]:
        trade_time = self._get_real_data(code, FID_TRADE_TIME)
        price = clean_price(self._get_real_data(code, FID_PRICE), self.logger)
        acc_volume = clean_non_negative_int(self._get_real_data(code, FID_ACC_VOLUME), self.logger)
        previous_acc_volume = self._last_acc_volume_by_symbol.get(code)
        volume_delta = 0 if previous_acc_volume is None else max(0, acc_volume - previous_acc_volume)
        self._last_acc_volume_by_symbol[code] = acc_volume

        tick: Dict[str, object] = {
            "symbol": code,
            "real_type": real_type,
            "trade_time": trade_time,
            "minute_ts_kst": kst_minute_from_hhmmss(trade_time, self.logger),
            "price": price,
            "acc_volume": acc_volume,
            "volume_delta": volume_delta,
            "acc_value": self._clean_optional_int(self._get_real_data(code, FID_ACC_VALUE)),
            "open": self._clean_optional_price(self._get_real_data(code, FID_OPEN)),
            "high": self._clean_optional_price(self._get_real_data(code, FID_HIGH)),
            "low": self._clean_optional_price(self._get_real_data(code, FID_LOW)),
            "best_ask": self._clean_optional_price(self._get_real_data(code, FID_BEST_ASK)),
            "best_bid": self._clean_optional_price(self._get_real_data(code, FID_BEST_BID)),
        }
        return tick

    def _get_real_data(self, code: str, fid: int) -> str:
        return str(self.dynamicCall("GetCommRealData(QString, int)", code, fid)).strip()

    @staticmethod
    def _clean_optional_price(value: str) -> Optional[int]:
        text = str(value).strip()
        if not text:
            return None
        return clean_price(text)

    @staticmethod
    def _clean_optional_int(value: str) -> Optional[int]:
        text = str(value).strip()
        if not text:
            return None
        return clean_non_negative_int(text)
