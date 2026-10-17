"""Tests for Step 4: al_regular validation / anomaly detection."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csv_importer import _bar_end, create_bars_tables  # noqa: E402
from data_validation import validate_source  # noqa: E402


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        create_bars_tables(self.conn)

    def tearDown(self):
        self.conn.close()

    def _ins(self, bar_start, o, h, l, c, v, *, source="al_regular", bar_end=None):
        self.conn.execute(
            "INSERT OR REPLACE INTO bars_1m "
            "(symbol,bar_start,bar_end,open,high,low,close,volume,source,created_at) "
            "VALUES ('T',?,?,?,?,?,?,?,?,'t')",
            (bar_start, bar_end or _bar_end(bar_start), o, h, l, c, v, source),
        )
        self.conn.commit()

    def _validate(self, **kw):
        return validate_source(self.conn, "T", source="al_regular", spike_pct=0.10, range_pct=0.10, **kw)

    def test_clean_is_ok(self):
        self._ins("2025-06-02 09:00:00", 100, 101, 99, 100, 10)
        self._ins("2025-06-02 09:01:00", 100, 102, 100, 101, 12)
        r = self._validate()
        self.assertEqual(r.severity, "OK")
        self.assertEqual(r.structural_total, 0)
        self.assertEqual(r.spike_count, 0)
        self.assertEqual(r.total_rows, 2)

    def test_missing_symbol(self):
        r = self._validate()
        self.assertEqual(r.severity, "MISSING")
        self.assertEqual(r.total_rows, 0)

    def test_ohlc_violation_is_severe(self):
        self._ins("2025-06-02 09:00:00", 100, 99, 101, 100, 10)  # high < low
        r = self._validate()
        self.assertGreaterEqual(r.ohlc_violations, 1)
        self.assertEqual(r.severity, "SEVERE")

    def test_out_of_window_is_severe(self):
        self._ins("2025-06-02 08:30:00", 100, 101, 99, 100, 10)  # before 09:00
        r = self._validate()
        self.assertGreaterEqual(r.out_of_window, 1)
        self.assertEqual(r.severity, "SEVERE")

    def test_bad_bar_end_is_severe(self):
        self._ins("2025-06-02 09:00:00", 100, 101, 99, 100, 10, bar_end="2025-06-02 09:05:00")
        r = self._validate()
        self.assertGreaterEqual(r.bad_bar_end, 1)
        self.assertEqual(r.severity, "SEVERE")

    def test_spike_detected_within_day(self):
        self._ins("2025-06-02 09:00:00", 100, 101, 99, 100, 10)
        self._ins("2025-06-02 09:01:00", 100, 140, 100, 130, 12)  # +30% close-to-close
        r = self._validate()
        self.assertGreaterEqual(r.spike_count, 1)
        self.assertEqual(r.severity, "REVIEW")
        self.assertGreaterEqual(r.max_abs_ret_pct, 29.9)
        self.assertTrue(r.spike_examples)

    def test_overnight_gap_is_not_a_spike(self):
        # big close change across a day boundary must NOT be flagged (LAG partitions by day)
        self._ins("2025-06-02 15:30:00", 100, 101, 99, 100, 10)
        self._ins("2025-06-03 09:00:00", 200, 201, 199, 200, 10)
        r = self._validate()
        self.assertEqual(r.spike_count, 0)

    def test_range_outlier_is_review(self):
        self._ins("2025-06-02 09:00:00", 100, 130, 100, 110, 10)  # (130-100)/100 = 30%
        r = self._validate()
        self.assertGreaterEqual(r.range_count, 1)
        self.assertEqual(r.severity, "REVIEW")


class BarsReaderTests(unittest.TestCase):
    def test_default_source_is_model_source(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not available in this runtime")
        from bars_reader import load_bars
        from settings import load_data_config

        conn = sqlite3.connect(":memory:")
        create_bars_tables(conn)
        model = load_data_config().model_source
        conn.execute(
            "INSERT INTO bars_1m VALUES ('T','2025-06-02 09:00:00','2025-06-02 09:01:00',1,1,1,1,1,?, 't')",
            (model,),
        )
        conn.execute(
            "INSERT INTO bars_1m VALUES ('T','2025-06-02 09:00:00','2025-06-02 09:01:00',9,9,9,9,9,'single_regular','t')"
        )
        conn.commit()
        df = load_bars("T", conn=conn)  # no source -> defaults to model_source
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["source"], model)
        self.assertEqual(int(df.iloc[0]["close"]), 1)
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
