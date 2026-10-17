from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Tuple


KST = timezone(timedelta(hours=9))


def setup_logger(log_path: Path, name: str = "collector") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    existing_paths = {
        Path(handler.baseFilename)
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    }
    if log_path.resolve() not in {path.resolve() for path in existing_paths}:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def now_kst_text() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def now_kst() -> datetime:
    return datetime.now(KST)


def clean_price(value: Any, logger: logging.Logger | None = None) -> int:
    text = "" if value is None else str(value).strip().replace(",", "")
    if not text:
        message = "empty price value"
        if logger:
            logger.exception(message)
        raise ValueError(message)

    if text[0] in "+-":
        text = text[1:]

    try:
        return abs(int(text))
    except ValueError as exc:
        if logger:
            logger.exception("failed to clean price value=%r", value)
        raise ValueError(f"invalid price value: {value!r}") from exc


def clean_non_negative_int(value: Any, logger: logging.Logger | None = None) -> int:
    text = "" if value is None else str(value).strip().replace(",", "")
    if not text:
        return 0

    if text[0] == "+":
        text = text[1:]

    try:
        number = int(text)
    except ValueError as exc:
        if logger:
            logger.exception("failed to clean integer value=%r", value)
        raise ValueError(f"invalid integer value: {value!r}") from exc

    return abs(number)


def kst_minute_from_hhmmss(value: Any, logger: logging.Logger | None = None) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        message = f"invalid HHMMSS value: {value!r}"
        if logger:
            logger.error(message)
        raise ValueError(message)

    base = now_kst()
    hour = int(text[:2])
    minute = int(text[2:4])
    try:
        parsed = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except ValueError as exc:
        if logger:
            logger.exception("failed to parse realtime HHMMSS value=%r", value)
        raise ValueError(f"invalid HHMMSS value: {value!r}") from exc
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def normalize_ts_kst(value: Any, logger: logging.Logger | None = None) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) >= 14:
        text = text[:14]
    elif len(text) == 12:
        text = text + "00"
    else:
        message = f"invalid timestamp length: {value!r}"
        if logger:
            logger.exception(message)
        raise ValueError(message)

    try:
        parsed = datetime.strptime(text, "%Y%m%d%H%M%S")
    except ValueError as exc:
        if logger:
            logger.exception("failed to parse timestamp value=%r", value)
        raise ValueError(f"invalid timestamp value: {value!r}") from exc

    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def validate_minute_bars(
    df: Any,
    logger: logging.Logger,
) -> Tuple[Any, int]:
    import pandas as pd

    if df.empty:
        return df.copy(), 0

    work = df.copy()
    invalid = pd.Series(False, index=work.index)

    duplicated_ts = work.duplicated(subset=["symbol", "ts_kst"], keep="first")
    if duplicated_ts.any():
        logger.warning("duplicate timestamp rows skipped count=%s", int(duplicated_ts.sum()))
        invalid = invalid | duplicated_ts

    for idx, value in work["ts_kst"].items():
        try:
            datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            logger.warning("invalid timestamp row skipped index=%s ts_kst=%r", idx, value)
            invalid.loc[idx] = True

    ohlc_positive = (
        (work["open"] > 0)
        & (work["high"] > 0)
        & (work["low"] > 0)
        & (work["close"] > 0)
    )
    bad_ohlc_positive = ~ohlc_positive
    if bad_ohlc_positive.any():
        logger.warning("non-positive OHLC rows skipped count=%s", int(bad_ohlc_positive.sum()))
        invalid = invalid | bad_ohlc_positive

    high_valid = work["high"] >= work[["open", "close", "low"]].max(axis=1)
    bad_high = ~high_valid
    if bad_high.any():
        logger.warning("invalid high rows skipped count=%s", int(bad_high.sum()))
        invalid = invalid | bad_high

    low_valid = work["low"] <= work[["open", "close", "high"]].min(axis=1)
    bad_low = ~low_valid
    if bad_low.any():
        logger.warning("invalid low rows skipped count=%s", int(bad_low.sum()))
        invalid = invalid | bad_low

    volume_valid = work["volume"] >= 0
    bad_volume = ~volume_valid
    if bad_volume.any():
        logger.warning("negative volume rows skipped count=%s", int(bad_volume.sum()))
        invalid = invalid | bad_volume

    invalid_count = int(invalid.sum())
    valid = work.loc[~invalid].copy()
    valid = valid.sort_values(["symbol", "ts_kst"]).reset_index(drop=True)
    return valid, invalid_count
