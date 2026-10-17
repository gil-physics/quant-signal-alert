"""Tests for Step 8: leakage-safe as-of feature merge."""

from __future__ import annotations

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

DAY = "2025-06-02"


def _ts(hhmm):
    return f"{DAY} {hhmm}:00"


def _f1(symbol, minutes):
    rows = [{
        "symbol": symbol, "source": "al_regular",
        "decision_time": _ts(m), "close": 100, "bar_end_1m": _ts(m), "ret_1m_1m": 0.0,
    } for m in minutes]
    return pd.DataFrame(rows)


def _f3(symbol, ends, close=300):
    rows = [{
        "symbol": symbol, "source": "resampled_from_al_regular",
        "bar_start_3m": _ts(s), "bar_end_3m": _ts(e), "close_3m": close, "ret_1bar_3m": 0.1,
    } for s, e in ends]
    return pd.DataFrame(rows)


def _f5(symbol, ends, close=500):
    rows = [{
        "symbol": symbol, "source": "resampled_from_al_regular",
        "bar_start_5m": _ts(s), "bar_end_5m": _ts(e), "close_5m": close, "ret_1bar_5m": 0.2,
    } for s, e in ends]
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class AsofMergeTests(unittest.TestCase):
    def setUp(self):
        from feature_merge import merge_asof_features

        f1 = _f1("T", ["09:0%d" % i for i in range(0, 10)] + ["09:10"])
        f3 = _f3("T", [("09:00", "09:03"), ("09:03", "09:06"), ("09:06", "09:09")])
        f5 = _f5("T", [("09:00", "09:05"), ("09:05", "09:10")])
        self.m = merge_asof_features(f1, f3, f5).set_index("decision_time")

    def test_step8_example_0907(self):
        row = self.m.loc[_ts("09:07")]
        self.assertEqual(row["bar_end_3m"], _ts("09:06"))   # 09:03-09:06
        self.assertEqual(row["bar_end_5m"], _ts("09:05"))   # 09:00-09:05

    def test_exact_match_allowed(self):
        # bar_end == decision_time is allowed (<=)
        self.assertEqual(self.m.loc[_ts("09:09")]["bar_end_3m"], _ts("09:09"))
        self.assertEqual(self.m.loc[_ts("09:10")]["bar_end_5m"], _ts("09:10"))

    def test_future_bar_not_used(self):
        # at 09:07 the 09:06-09:09 (bar_end 09:09) 3m bar must NOT be attached
        self.assertNotEqual(self.m.loc[_ts("09:07")]["bar_end_3m"], _ts("09:09"))
        self.assertEqual(self.m.loc[_ts("09:08")]["bar_end_5m"], _ts("09:05"))  # 09:10 excluded

    def test_before_first_bar_is_nan(self):
        self.assertTrue(pd.isna(self.m.loc[_ts("09:02")]["bar_end_3m"]))  # first 3m end 09:03
        self.assertTrue(pd.isna(self.m.loc[_ts("09:04")]["bar_end_5m"]))  # first 5m end 09:05

    def test_source_column_no_collision(self):
        # the 1m provenance source survives; no source_x/source_y
        self.assertIn("source", self.m.columns)
        self.assertNotIn("source_x", self.m.columns)
        self.assertEqual(set(self.m["source"].unique()), {"al_regular"})

    def test_no_leakage(self):
        from feature_merge import leakage_violations, merge_asof_features

        f1 = _f1("T", ["09:0%d" % i for i in range(0, 10)])
        f3 = _f3("T", [("09:00", "09:03"), ("09:03", "09:06"), ("09:06", "09:09")])
        f5 = _f5("T", [("09:00", "09:05"), ("09:05", "09:10")])
        v3, v5 = leakage_violations(merge_asof_features(f1, f3, f5))
        self.assertEqual((v3, v5), (0, 0))


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class CrossDayTests(unittest.TestCase):
    def test_no_borrowing_prior_day_feature(self):
        # day1 has a 5m feature; day2's early decision (before day2's first 5m) must be
        # NaN, NOT day1's stale feature.
        from feature_merge import merge_asof_features

        f1 = pd.concat([
            pd.DataFrame([{"symbol": "T", "source": "al_regular",
                           "decision_time": "2025-06-02 10:40:00", "close": 1, "bar_end_1m": "2025-06-02 10:40:00"}]),
            pd.DataFrame([{"symbol": "T", "source": "al_regular",
                           "decision_time": "2025-06-03 10:01:00", "close": 1, "bar_end_1m": "2025-06-03 10:01:00"}]),
        ], ignore_index=True)
        # only a day1 5m feature exists (bar_end 2025-06-02 10:40)
        f5 = pd.DataFrame([{"symbol": "T", "source": "resampled_from_al_regular",
                            "bar_start_5m": "2025-06-02 10:35:00", "bar_end_5m": "2025-06-02 10:40:00",
                            "close_5m": 9, "ret_1bar_5m": 0.0}])
        f3 = pd.DataFrame(columns=["symbol", "source", "bar_start_3m", "bar_end_3m", "close_3m", "ret_1bar_3m"])
        m = merge_asof_features(f1, f3, f5).set_index("decision_time")
        # day1 10:40 gets its own 5m (exact match)
        self.assertEqual(m.loc["2025-06-02 10:40:00"]["bar_end_5m"], "2025-06-02 10:40:00")
        # day2 10:01 must NOT borrow day1's 5m -> NaN
        self.assertTrue(pd.isna(m.loc["2025-06-03 10:01:00"]["bar_end_5m"]))


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class PerSymbolTests(unittest.TestCase):
    def test_symbols_do_not_cross(self):
        from feature_merge import merge_asof_features

        f1 = pd.concat([_f1("T", ["09:07"]), _f1("U", ["09:07"])], ignore_index=True)
        f3 = pd.concat([_f3("T", [("09:03", "09:06")], close=300),
                        _f3("U", [("09:03", "09:06")], close=999)], ignore_index=True)
        f5 = pd.concat([_f5("T", [("09:00", "09:05")], close=500),
                        _f5("U", [("09:00", "09:05")], close=777)], ignore_index=True)
        m = merge_asof_features(f1, f3, f5).set_index("symbol")
        self.assertEqual(m.loc["T"]["close_3m"], 300)
        self.assertEqual(m.loc["U"]["close_3m"], 999)
        self.assertEqual(m.loc["T"]["close_5m"], 500)
        self.assertEqual(m.loc["U"]["close_5m"], 777)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class LeakageDetectorTests(unittest.TestCase):
    def test_detector_flags_future_bar_end(self):
        from feature_merge import leakage_violations

        merged = pd.DataFrame([
            {"symbol": "T", "decision_time": _ts("09:07"), "bar_end_3m": _ts("09:09"), "bar_end_5m": _ts("09:05")},
            {"symbol": "T", "decision_time": _ts("09:08"), "bar_end_3m": _ts("09:06"), "bar_end_5m": _ts("09:10")},
        ])
        v3, v5 = leakage_violations(merged)
        self.assertEqual(v3, 1)  # 09:09 > 09:07
        self.assertEqual(v5, 1)  # 09:10 > 09:08

    def test_detector_ignores_nan(self):
        from feature_merge import leakage_violations

        merged = pd.DataFrame([
            {"symbol": "T", "decision_time": _ts("09:02"), "bar_end_3m": None, "bar_end_5m": None},
        ])
        self.assertEqual(leakage_violations(merged), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
