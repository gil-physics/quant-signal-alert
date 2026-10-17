"""Tests for Step 3: multi-symbol historical 1-minute collector."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_collector import collect_1m, store_minute_records, symbol_coverage  # noqa: E402
from settings import load_active_symbols, load_data_config  # noqa: E402


def _rec(dt, o, h, l, c, v):
    return {"datetime": dt, "open": o, "high": h, "low": l, "close": c, "volume": v}

# Synthetic AL file: 08:30 (pre), 09:00 / 09:30 / 15:30 (regular), 16:00 (post).
_AL = """datetime,open,high,low,close,volume,value
20250602083000,100,101,99,100,10,
20250602090000,100,102,100,101,20,
20250602093000,101,103,101,102,30,
20250602153000,102,104,102,103,40,
20250602160000,103,105,103,104,50,
"""
# Single-exchange regular file: same 3 regular minutes, different volumes.
_SINGLE = """datetime,open,high,low,close,volume,value
20250602090000,100,102,100,101,15,
20250602093000,101,103,101,102,25,
20250602153000,102,104,102,103,35,
"""


class DataConfigTests(unittest.TestCase):
    def test_active_symbols(self):
        # The active set is user-controlled (started at 2, expanding as data
        # arrives); validate invariants instead of pinning a specific list.
        from settings import load_symbols

        all_codes = [s.code for s in load_symbols()]
        active = load_active_symbols()
        self.assertGreaterEqual(len(active), 1)
        self.assertTrue(set(active).issubset(set(all_codes)))
        self.assertEqual(len(active), len(set(active)), "active codes must be unique")
        self.assertEqual(active, [c for c in all_codes if c in set(active)])

    def test_data_classification(self):
        dc = load_data_config()
        self.assertEqual(dc.model_source, "al_regular")
        self.assertEqual(dc.raw_source, "al_raw")
        self.assertEqual(dc.comparison_source, "single_regular")


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "111111_AL_1min.csv").write_text(_AL, encoding="utf-8")
        (self.tmp / "111111_1min.csv").write_text(_SINGLE, encoding="utf-8")
        self.conn = sqlite3.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_source_split_and_window(self):
        s = collect_1m("111111", conn=self.conn, data_dir=self.tmp)
        self.assertEqual(s.per_source["al_raw"].inserted, 5)
        self.assertEqual(s.per_source["al_regular"].inserted, 3)  # 08:30 and 16:00 filtered out
        self.assertEqual(s.per_source["single_regular"].inserted, 3)
        self.assertTrue(s.ok)
        # every row carries the symbol; al_regular strictly inside the session
        bad = self.conn.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE source='al_regular' "
            "AND substr(bar_start,12) NOT BETWEEN '09:00:00' AND '15:30:00'"
        ).fetchone()[0]
        self.assertEqual(bad, 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(DISTINCT symbol) FROM bars_1m").fetchone()[0], 1
        )

    def test_idempotent(self):
        collect_1m("111111", conn=self.conn, data_dir=self.tmp)
        again = collect_1m("111111", conn=self.conn, data_dir=self.tmp)
        self.assertEqual(again.inserted_total, 0)
        # no duplicate primary keys
        dupes = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT symbol,bar_start,source,COUNT(*) c "
            "FROM bars_1m GROUP BY symbol,bar_start,source HAVING c>1)"
        ).fetchone()[0]
        self.assertEqual(dupes, 0)

    def test_missing_files_do_not_raise(self):
        s = collect_1m("999999", conn=self.conn, data_dir=self.tmp)  # no such CSVs
        self.assertFalse(s.ok)
        self.assertEqual(s.inserted_total, 0)
        self.assertEqual(sorted(s.missing_files), ["999999_1min.csv", "999999_AL_1min.csv"])

    def test_date_range_filter(self):
        s = collect_1m(
            "111111", start_date="2030-01-01", end_date="2030-12-31", conn=self.conn, data_dir=self.tmp
        )
        self.assertEqual(s.inserted_total, 0)  # all rows are in 2025

    def test_bar_end_is_plus_one_minute(self):
        collect_1m("111111", conn=self.conn, data_dir=self.tmp)
        bad = self.conn.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE bar_end != "
            "strftime('%Y-%m-%d %H:%M:%S', datetime(bar_start, '+1 minute'))"
        ).fetchone()[0]
        self.assertEqual(bad, 0)

    def test_coverage_report_row(self):
        collect_1m("111111", conn=self.conn, data_dir=self.tmp)
        row = symbol_coverage(
            self.conn, "111111", ["al_raw", "al_regular", "single_regular"], model_source="al_regular"
        )
        self.assertEqual(row["status"], "OK")
        self.assertEqual(row["model_rows"], 3)
        self.assertEqual(row["total_rows"], 11)


class StoreRecordsTests(unittest.TestCase):
    """The PyQt5-free half of the direct Kiwoom->bars_1m path."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.al = [
            _rec("20250602083000", 100, 101, 99, 100, 10),   # pre-market
            _rec("20250602090000", 100, 102, 100, 101, 20),  # regular
            _rec("20250602093000", 101, 103, 101, 102, 30),  # regular
            _rec("20250602153000", 102, 104, 102, 103, 40),  # regular boundary
            _rec("20250602160000", 103, 105, 103, 104, 50),  # post-market
        ]
        self.single = [
            _rec("20250602090000", 100, 102, 100, 101, 15),
            _rec("20250602093000", 101, 103, 101, 102, 25),
            _rec("20250602153000", 102, 104, 102, 103, 35),
        ]

    def tearDown(self):
        self.conn.close()

    def test_three_sources_from_records(self):
        s = store_minute_records(self.conn, "654321", al_records=self.al, single_records=self.single)
        self.assertEqual(s.per_source["al_raw"].inserted, 5)
        self.assertEqual(s.per_source["al_regular"].inserted, 3)  # pre/post filtered out
        self.assertEqual(s.per_source["single_regular"].inserted, 3)
        bad = self.conn.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE source='al_regular' "
            "AND substr(bar_start,12) NOT BETWEEN '09:00:00' AND '15:30:00'"
        ).fetchone()[0]
        self.assertEqual(bad, 0)

    def test_idempotent_records(self):
        store_minute_records(self.conn, "654321", al_records=self.al, single_records=self.single)
        again = store_minute_records(self.conn, "654321", al_records=self.al, single_records=self.single)
        self.assertEqual(again.inserted_total, 0)

    def test_no_single(self):
        s = store_minute_records(self.conn, "654321", al_records=self.al, single_records=None)
        self.assertNotIn("single_regular", s.per_source)
        self.assertEqual(s.inserted_total, 8)  # 5 al_raw + 3 al_regular


if __name__ == "__main__":
    unittest.main(verbosity=2)
