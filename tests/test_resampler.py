"""Tests for Step 5: 1m -> 3m/5m resampling."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    HAVE_PANDAS = False

from csv_importer import create_bars_tables  # noqa: E402


def _one_minute_df(symbol, day, n, start_hhmm="09:00"):
    """n consecutive 1-minute bars from day start_hhmm. minute i: open=100+i,
    high=110+i, low=90+i, close=100+i, volume=i+1."""
    import pandas as pd

    h, m = int(start_hhmm[:2]), int(start_hhmm[3:])
    rows = []
    for i in range(n):
        ts = pd.Timestamp(f"{day} {h:02d}:{m:02d}:00") + pd.Timedelta(minutes=i)
        rows.append(
            {
                "symbol": symbol,
                "bar_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": 100 + i,
                "high": 110 + i,
                "low": 90 + i,
                "close": 100 + i,
                "volume": i + 1,
            }
        )
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class ResampleTests(unittest.TestCase):
    def test_3m_counts_and_aggregation(self):
        from resampler import resample_ohlcv

        df = _one_minute_df("T", "2025-06-02", 10)  # 09:00..09:09
        out = resample_ohlcv(df, 3)
        # 3 complete buckets (09:00,09:03,09:06); the lone 09:09 bucket is incomplete -> dropped
        self.assertEqual(len(out), 3)
        first = out.iloc[0]
        self.assertEqual(first["bar_start"], "2025-06-02 09:00:00")
        self.assertEqual(first["bar_end"], "2025-06-02 09:03:00")
        self.assertEqual(int(first["open"]), 100)   # minute 0 open
        self.assertEqual(int(first["high"]), 112)   # max(110,111,112)
        self.assertEqual(int(first["low"]), 90)     # min(90,91,92)
        self.assertEqual(int(first["close"]), 102)  # minute 2 close
        self.assertEqual(int(first["volume"]), 6)   # 1+2+3

    def test_5m_counts(self):
        from resampler import resample_ohlcv

        df = _one_minute_df("T", "2025-06-02", 10)  # 09:00..09:09
        out = resample_ohlcv(df, 5)
        self.assertEqual(len(out), 2)  # 09:00-09:05, 09:05-09:10
        self.assertEqual(int(out.iloc[0]["volume"]), 15)  # 1+2+3+4+5

    def test_incomplete_last_bucket_dropped(self):
        from resampler import resample_ohlcv

        df = _one_minute_df("T", "2025-06-02", 7)  # 09:00..09:06
        out = resample_ohlcv(df, 3)
        self.assertEqual(len(out), 2)  # 09:00,09:03 complete; 09:06 alone dropped

    def test_pre_open_bars_ignored(self):
        from resampler import resample_ohlcv

        df = _one_minute_df("T", "2025-06-02", 3, start_hhmm="08:58")  # before open
        out = resample_ohlcv(df, 3)
        self.assertEqual(len(out), 0)

    def test_multi_symbol_and_day_boundary(self):
        from resampler import resample_ohlcv

        df = pd.concat(
            [
                _one_minute_df("AAA", "2025-06-02", 6),
                _one_minute_df("BBB", "2025-06-02", 6),
                _one_minute_df("AAA", "2025-06-03", 6),
            ],
            ignore_index=True,
        )
        out = resample_ohlcv(df, 3)
        # each (symbol, day) yields 2 complete 3m buckets -> 6 total
        self.assertEqual(len(out), 6)
        self.assertEqual(set(out["symbol"]), {"AAA", "BBB"})
        aaa_days = out[out["symbol"] == "AAA"]["bar_start"].str.slice(0, 10).nunique()
        self.assertEqual(aaa_days, 2)

    def test_empty_input(self):
        from resampler import resample_ohlcv

        out = resample_ohlcv(pd.DataFrame(columns=["symbol", "bar_start", "open", "high", "low", "close", "volume"]), 3)
        self.assertEqual(len(out), 0)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class StoreBarsTests(unittest.TestCase):
    def test_store_and_idempotent(self):
        from resampler import resample_ohlcv, store_bars

        conn = sqlite3.connect(":memory:")
        create_bars_tables(conn)
        out = resample_ohlcv(_one_minute_df("T", "2025-06-02", 10), 3)
        r1 = store_bars(conn, out, table="bars_3m", source="resampled_from_al_regular")
        self.assertEqual(r1["inserted"], 3)
        r2 = store_bars(conn, out, table="bars_3m", source="resampled_from_al_regular")
        self.assertEqual(r2["inserted"], 0)  # idempotent
        total = conn.execute("SELECT COUNT(*) FROM bars_3m WHERE symbol='T'").fetchone()[0]
        self.assertEqual(total, 3)
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
