from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from config import DEFAULT_SOURCE, MINUTE_BARS_DDL
from utils import (
    clean_non_negative_int,
    clean_price,
    normalize_ts_kst,
    now_kst_text,
    validate_minute_bars,
)


@dataclass
class SaveSummary:
    received_rows: int
    valid_rows: int
    invalid_rows: int
    sqlite_inserted: int
    sqlite_skipped_duplicates: int
    parquet_rows: int
    parquet_path: Path


class MinuteBarCollector:
    def __init__(self, *, db_path: Path, parquet_path: Path, logger: logging.Logger) -> None:
        self.db_path = db_path
        self.parquet_path = parquet_path
        self.logger = logger

    def initialize_storage(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(MINUTE_BARS_DDL)
            conn.commit()

    def collect_to_storage(self, *, symbol: str, raw_rows: Iterable[Dict[str, str]]) -> SaveSummary:
        raw_rows_list = list(raw_rows)
        received_rows = len(raw_rows_list)
        self.logger.info("received row count=%s", received_rows)

        df = self._raw_rows_to_dataframe(symbol=symbol, raw_rows=raw_rows_list)
        valid_df, invalid_rows = validate_minute_bars(df, self.logger)
        self.logger.info("valid rows=%s invalid rows=%s", len(valid_df), invalid_rows)

        inserted, skipped_duplicates = self._save_sqlite(valid_df)
        parquet_rows = self._save_parquet(valid_df)

        self.logger.info(
            "save summary received=%s valid=%s invalid=%s sqlite_inserted=%s "
            "sqlite_skipped_duplicates=%s parquet_rows=%s parquet_path=%s",
            received_rows,
            len(valid_df),
            invalid_rows,
            inserted,
            skipped_duplicates,
            parquet_rows,
            self.parquet_path,
        )

        return SaveSummary(
            received_rows=received_rows,
            valid_rows=len(valid_df),
            invalid_rows=invalid_rows,
            sqlite_inserted=inserted,
            sqlite_skipped_duplicates=skipped_duplicates,
            parquet_rows=parquet_rows,
            parquet_path=self.parquet_path,
        )

    def _raw_rows_to_dataframe(
        self,
        *,
        symbol: str,
        raw_rows: List[Dict[str, str]],
    ):
        import pandas as pd

        created_at = now_kst_text()
        cleaned_rows: List[Dict[str, object]] = []
        failed = 0

        for index, row in enumerate(raw_rows):
            try:
                cleaned_rows.append(
                    {
                        "symbol": symbol,
                        "ts_kst": normalize_ts_kst(row.get("ts"), self.logger),
                        "open": clean_price(row.get("open"), self.logger),
                        "high": clean_price(row.get("high"), self.logger),
                        "low": clean_price(row.get("low"), self.logger),
                        "close": clean_price(row.get("close"), self.logger),
                        "volume": clean_non_negative_int(row.get("volume"), self.logger),
                        "source": DEFAULT_SOURCE,
                        "created_at": created_at,
                    }
                )
            except ValueError:
                failed += 1
                self.logger.exception("row cleaning failed index=%s raw=%r", index, row)

        if failed:
            self.logger.warning("cleaning failed row count=%s", failed)

        columns = [
            "symbol",
            "ts_kst",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "created_at",
        ]
        df = pd.DataFrame(cleaned_rows, columns=columns)
        if not df.empty:
            df = df.sort_values(["symbol", "ts_kst"]).reset_index(drop=True)
        return df

    def _save_sqlite(self, df) -> tuple[int, int]:
        if df.empty:
            self.logger.info("SQLite save skipped: no valid rows")
            return 0, 0

        rows = [
            (
                row.symbol,
                row.ts_kst,
                int(row.open),
                int(row.high),
                int(row.low),
                int(row.close),
                int(row.volume),
                row.source,
                row.created_at,
            )
            for row in df.itertuples(index=False)
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(MINUTE_BARS_DDL)
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO minute_bars (
                    symbol, ts_kst, open, high, low, close, volume, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        inserted = int(cursor.rowcount if cursor.rowcount is not None else 0)
        skipped = len(rows) - inserted
        self.logger.info("SQLite inserted=%s duplicate_skipped=%s", inserted, skipped)
        return inserted, skipped

    def _save_parquet(self, df) -> int:
        import pandas as pd

        if df.empty:
            self.logger.info("Parquet save skipped: no valid rows")
            return self._existing_parquet_rows()

        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            merged = pd.concat([existing, df], ignore_index=True)
        else:
            merged = df.copy()

        merged = (
            merged.drop_duplicates(subset=["symbol", "ts_kst"], keep="last")
            .sort_values(["symbol", "ts_kst"])
            .reset_index(drop=True)
        )
        merged.to_parquet(self.parquet_path, index=False)
        self.logger.info("Parquet saved path=%s rows=%s", self.parquet_path, len(merged))
        return len(merged)

    def _existing_parquet_rows(self) -> int:
        import pandas as pd

        if not self.parquet_path.exists():
            return 0
        return len(pd.read_parquet(self.parquet_path))
