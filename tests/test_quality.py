"""Tests for Step 6: data-quality checks."""

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

WINDOW = ("09:00:00", "15:30:00")


def _mk(bar_start, o, h, l, c, v, bar_end=None):
    return {"bar_start": bar_start, "open": o, "high": h, "low": l, "close": c, "volume": v, "bar_end": bar_end}


def _group(rows, interval=1):
    df = pd.DataFrame(rows)
    df["_ts"] = pd.to_datetime(df["bar_start"])
    df["_exp_end"] = (df["_ts"] + pd.to_timedelta(interval, unit="m")).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["bar_end"] = df["bar_end"].fillna(df["_exp_end"])
    return df.sort_values("bar_start").reset_index(drop=True)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class DayMetricsTests(unittest.TestCase):
    def _metrics(self, rows, *, mixing=0, lo=2, hi=10, interval=1):
        from quality import compute_day_metrics

        return compute_day_metrics(
            _group(rows, interval),
            session_window=WINDOW,
            spike_pct=0.10,
            range_pct=0.10,
            source_mixing_count=mixing,
            expected_lo=lo,
            expected_hi=hi,
        )

    def test_clean_ok(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10),
                _mk("2025-06-02 09:01:00", 100, 102, 100, 101, 12),
                _mk("2025-06-02 09:02:00", 101, 103, 100, 102, 8)]
        m = self._metrics(rows)
        self.assertEqual(m["status"], "OK")
        self.assertEqual(m["duplicate"], 0)
        self.assertEqual(m["actual"], 3)

    def test_duplicate_fail(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10),
                _mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10)]
        m = self._metrics(rows)
        self.assertEqual(m["duplicate"], 1)
        self.assertEqual(m["status"], "FAIL")

    def test_ohlc_error_fail(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 99, 101, 100, 10)]  # high<low
        self.assertEqual(self._metrics(rows)["status"], "FAIL")

    def test_price_zero_fail(self):
        rows = [_mk("2025-06-02 09:00:00", 0, 101, 99, 100, 10)]
        m = self._metrics(rows)
        self.assertEqual(m["price_zero"], 1)
        self.assertEqual(m["status"], "FAIL")

    def test_negative_volume_fail(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, -5)]
        self.assertEqual(self._metrics(rows)["status"], "FAIL")

    def test_bar_end_error_fail(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10, bar_end="2025-06-02 09:05:00")]
        m = self._metrics(rows, interval=1)
        self.assertEqual(m["bar_end_error"], 1)
        self.assertEqual(m["status"], "FAIL")

    def test_out_of_session_fail(self):
        rows = [_mk("2025-06-02 08:30:00", 100, 101, 99, 100, 10)]
        m = self._metrics(rows, lo=1, hi=10)
        self.assertEqual(m["out_of_session"], 1)
        self.assertEqual(m["status"], "FAIL")

    def test_source_mixing_fail(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10)]
        m = self._metrics(rows, mixing=2, lo=1, hi=10)
        self.assertEqual(m["source_mixing"], 2)
        self.assertEqual(m["status"], "FAIL")

    def test_spike_warn(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10),
                _mk("2025-06-02 09:01:00", 100, 140, 100, 130, 12)]  # +30% close
        m = self._metrics(rows, lo=1, hi=10)
        self.assertEqual(m["spike"], 1)
        self.assertEqual(m["status"], "WARN")

    def test_range_outlier_warn(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 130, 100, 110, 10)]  # 30% intrabar
        m = self._metrics(rows, lo=1, hi=10)
        self.assertEqual(m["range_outlier"], 1)
        self.assertEqual(m["status"], "WARN")

    def test_missing_below_band_warn(self):
        rows = [_mk("2025-06-02 09:00:00", 100, 101, 99, 100, 10)]
        m = self._metrics(rows, lo=5, hi=10)  # actual 1 < lo 5
        self.assertEqual(m["missing"], 4)
        self.assertEqual(m["status"], "WARN")


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class CheckTableTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        create_bars_tables(self.conn)

    def tearDown(self):
        self.conn.close()

    def _ins(self, bar_start, source="al_regular", o=100, h=101, l=99, c=100, v=10, bar_end=None):
        be = bar_end or (pd.to_datetime(bar_start) + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            "INSERT OR REPLACE INTO bars_1m "
            "(symbol,bar_start,bar_end,open,high,low,close,volume,source,created_at) "
            "VALUES ('T',?,?,?,?,?,?,?,?,'t')",
            (bar_start, be, o, h, l, c, v, source),
        )
        self.conn.commit()

    def test_source_mixing_detected(self):
        from quality import check_table

        for i in range(3):
            self._ins(f"2025-06-02 09:0{i}:00")
        self._ins("2025-06-02 09:05:00", source="junk_src")  # unexpected source on same date
        rows = check_table(
            self.conn, "T", table="bars_1m", timeframe="1m", interval_minutes=1,
            expected_source="al_regular",
            allowed_sources=["al_raw", "al_regular", "single_regular"],
            session_window=WINDOW,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_mixing"], 1)
        self.assertEqual(rows[0]["status"], "FAIL")

    def test_clean_table_ok(self):
        from quality import check_table

        for i in range(3):
            self._ins(f"2025-06-02 09:0{i}:00")
        rows = check_table(
            self.conn, "T", table="bars_1m", timeframe="1m", interval_minutes=1,
            expected_source="al_regular",
            allowed_sources=["al_raw", "al_regular", "single_regular"],
            session_window=WINDOW,
        )
        self.assertEqual(rows[0]["status"], "OK")
        self.assertEqual(rows[0]["source_mixing"], 0)

    def test_missing_symbol_is_warn(self):
        # No data at all = "missing" -> WARN per step6.md #9 (FAIL is for corrupt data).
        from quality import check_table

        rows = check_table(
            self.conn, "ZZZ", table="bars_1m", timeframe="1m", interval_minutes=1,
            expected_source="al_regular",
            allowed_sources=["al_raw", "al_regular", "single_regular"],
            session_window=WINDOW,
        )
        self.assertEqual(rows[0]["status"], "WARN")
        self.assertEqual(rows[0]["actual"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
