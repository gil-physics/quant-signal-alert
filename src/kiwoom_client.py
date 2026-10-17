from __future__ import annotations

import logging
import platform
import time
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QEventLoop
from PyQt5.QAxContainer import QAxWidget


TR_CODE_OPT10080 = "opt10080"
RQ_NAME_OPT10080 = "\uc8fc\uc2dd\ubd84\ubd09\ucc28\ud2b8\uc870\ud68c\uc694\uccad"

FIELD_CODE = "\uc885\ubaa9\ucf54\ub4dc"
FIELD_TICK_RANGE = "\ud2f1\ubc94\uc704"
FIELD_ADJUSTED_PRICE = "\uc218\uc815\uc8fc\uac00\uad6c\ubd84"
FIELD_TRADE_TIME = "\uccb4\uacb0\uc2dc\uac04"
FIELD_OPEN = "\uc2dc\uac00"
FIELD_HIGH = "\uace0\uac00"
FIELD_LOW = "\uc800\uac00"
FIELD_CLOSE = "\ud604\uc7ac\uac00"
FIELD_VOLUME = "\uac70\ub798\ub7c9"


class KiwoomClient(QAxWidget):
    def __init__(self, logger: logging.Logger, screen_no: str = "1000") -> None:
        if platform.architecture()[0] != "32bit":
            raise RuntimeError("Kiwoom OpenAPI+ requires 32-bit Python.")

        super().__init__()
        self.logger = logger
        self.screen_no = screen_no

        ok = self.setControl("KHOPENAPI.KHOpenAPICtrl.1")
        if ok is False or self.isNull():
            raise RuntimeError("Failed to load KHOPENAPI.KHOpenAPICtrl.1")

        self.OnEventConnect.connect(self._on_event_connect)
        self.OnReceiveTrData.connect(self._on_receive_tr_data)
        self.OnReceiveMsg.connect(self._on_receive_msg)

        self._login_loop: Optional[QEventLoop] = None
        self._request_loop: Optional[QEventLoop] = None
        self._connected = False
        self._last_prev_next = "0"
        self._last_page_rows: List[Dict[str, str]] = []
        self._last_message = ""

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

    def request_minute_bars(
        self,
        *,
        symbol: str,
        tick: str,
        max_pages: int,
        request_delay_sec: float,
    ) -> List[Dict[str, str]]:
        all_rows: List[Dict[str, str]] = []
        prev_next = "0"

        for page in range(1, max_pages + 1):
            self.logger.info(
                "TR request start tr=%s rq=%s symbol=%s tick=%s page=%s prev_next=%s",
                TR_CODE_OPT10080,
                RQ_NAME_OPT10080,
                symbol,
                tick,
                page,
                prev_next,
            )
            self._request_opt10080_page(symbol=symbol, tick=tick, prev_next=prev_next)
            all_rows.extend(self._last_page_rows)

            self.logger.info(
                "TR response received page=%s rows=%s prev_next=%s message=%s",
                page,
                len(self._last_page_rows),
                self._last_prev_next,
                self._last_message,
            )

            if self._last_prev_next != "2":
                self.logger.info("continuous request finished page=%s", page)
                break

            if page >= max_pages:
                self.logger.info("continuous request stopped by max_pages=%s", max_pages)
                break

            self.logger.info("continuous request available, sleeping seconds=%s", request_delay_sec)
            prev_next = "2"
            time.sleep(request_delay_sec)

        self.logger.info("TR request complete total_raw_rows=%s", len(all_rows))
        return all_rows

    def _request_opt10080_page(self, *, symbol: str, tick: str, prev_next: str) -> None:
        self._last_page_rows = []
        self._last_prev_next = "0"
        self._last_message = ""

        self.dynamicCall("SetInputValue(QString, QString)", FIELD_CODE, symbol)
        self.dynamicCall("SetInputValue(QString, QString)", FIELD_TICK_RANGE, str(tick))
        self.dynamicCall("SetInputValue(QString, QString)", FIELD_ADJUSTED_PRICE, "1")

        self._request_loop = QEventLoop()
        ret = int(
            self.dynamicCall(
                "CommRqData(QString, QString, int, QString)",
                RQ_NAME_OPT10080,
                TR_CODE_OPT10080,
                int(prev_next),
                self.screen_no,
            )
        )
        if ret != 0:
            self.logger.error("CommRqData failed ret=%s message=%s", ret, self._last_message)
            raise RuntimeError(f"CommRqData failed: {ret}")

        self._request_loop.exec_()

    def _on_event_connect(self, err_code: int) -> None:
        self._connected = err_code == 0
        if self._login_loop is not None:
            self._login_loop.exit()

    def _on_receive_msg(self, screen_no: str, rq_name: str, tr_code: str, msg: str) -> None:
        self._last_message = str(msg).strip()
        if self._last_message:
            self.logger.info(
                "receive message screen=%s rq=%s tr=%s msg=%s",
                screen_no,
                rq_name,
                tr_code,
                self._last_message,
            )

    def _on_receive_tr_data(
        self,
        screen_no: str,
        rq_name: str,
        tr_code: str,
        record_name: str,
        prev_next: str,
        data_len: int,
        error_code: str,
        message: str,
        splm_msg: str,
    ) -> None:
        if rq_name != RQ_NAME_OPT10080:
            return

        self._last_prev_next = str(prev_next).strip()
        rows = self._get_rows_with_comm_data_ex(tr_code, rq_name)
        if rows is None:
            rows = self._get_rows_with_comm_data(tr_code, rq_name)

        self._last_page_rows = rows
        self.logger.info(
            "OnReceiveTrData tr=%s rq=%s rows=%s prev_next=%s",
            tr_code,
            rq_name,
            len(rows),
            self._last_prev_next,
        )
        if self._request_loop is not None:
            self._request_loop.exit()

    def _get_rows_with_comm_data_ex(
        self,
        tr_code: str,
        rq_name: str,
    ) -> Optional[List[Dict[str, str]]]:
        try:
            data = self.dynamicCall("GetCommDataEx(QString, QString)", tr_code, rq_name)
        except Exception:
            self.logger.exception("GetCommDataEx failed; falling back to GetCommData")
            return None

        if not data:
            return None

        rows: List[Dict[str, str]] = []
        for raw_row in data:
            if not isinstance(raw_row, (list, tuple)) or len(raw_row) < 6:
                return None

            values = [str(value).strip() for value in raw_row]
            mapped = self._map_comm_data_ex_row(values)
            if mapped is None:
                return None
            rows.append(mapped)

        self.logger.info("GetCommDataEx used rows=%s", len(rows))
        return rows

    def _map_comm_data_ex_row(self, values: List[str]) -> Optional[Dict[str, str]]:
        if len(values) >= 6 and self._looks_like_kiwoom_datetime(values[2]):
            return {
                "ts": values[2],
                "open": values[3],
                "high": values[4],
                "low": values[5],
                "close": values[0],
                "volume": values[1],
            }

        if len(values) >= 6 and self._looks_like_kiwoom_datetime(values[0]):
            return {
                "ts": values[0],
                "open": values[3],
                "high": values[4],
                "low": values[5],
                "close": values[1],
                "volume": values[2],
            }

        return None

    def _get_rows_with_comm_data(self, tr_code: str, rq_name: str) -> List[Dict[str, str]]:
        repeat_count = int(self.dynamicCall("GetRepeatCnt(QString, QString)", tr_code, rq_name))
        rows: List[Dict[str, str]] = []
        for index in range(repeat_count):
            rows.append(
                {
                    "ts": self._get_comm_data(tr_code, rq_name, index, FIELD_TRADE_TIME),
                    "open": self._get_comm_data(tr_code, rq_name, index, FIELD_OPEN),
                    "high": self._get_comm_data(tr_code, rq_name, index, FIELD_HIGH),
                    "low": self._get_comm_data(tr_code, rq_name, index, FIELD_LOW),
                    "close": self._get_comm_data(tr_code, rq_name, index, FIELD_CLOSE),
                    "volume": self._get_comm_data(tr_code, rq_name, index, FIELD_VOLUME),
                }
            )

        self.logger.info("GetCommData fallback used rows=%s", len(rows))
        return rows

    def _get_comm_data(self, tr_code: str, rq_name: str, index: int, item_name: str) -> str:
        return str(
            self.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                tr_code,
                rq_name,
                index,
                item_name,
            )
        ).strip()

    @staticmethod
    def _looks_like_kiwoom_datetime(value: Any) -> bool:
        text = str(value).strip()
        return len(text) in (12, 14) and text.isdigit()
