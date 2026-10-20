"""Economic/time-boundary tests for the frozen-signal short-horizon study."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_short_horizon_eval import (
    Day, bracket_path, entry_window, matched_baselines, measure_window,
    nonoverlap_keys, signals_from_bundle,
)


def signal(minute=600, direction=1, date="2026-01-05", **kw):
    return {"symbol": "AAA", "date": date, "decision_minute": minute,
            "decision_time": f"{date} {minute // 60:02d}:{minute % 60:02d}:00",
            "family": "trend", "direction": direction, "signal_close": 100.,
            "actionable": True, "types": "UP_ALERT", **kw}


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.day = Day("AAA", "2026-01-05", np.arange(600, 612),
                       np.array([[100 + i, 101 + i, 99 + i, 100.5 + i]
                                 for i in range(12)], float))

    def test_first_future_bar_is_included(self):
        (entry, window, start), error = entry_window(self.day, signal(), "next_open", 1)
        self.assertIsNone(error)
        self.assertEqual(entry, 100)
        self.assertEqual(start, 600)
        self.assertEqual(window[-1, 3], 100.5)

    def test_delay_and_legacy_comparisons_have_expected_times(self):
        corrected, _ = entry_window(self.day, signal(), "delay3_close", 3)
        legacy, _ = entry_window(self.day, signal(), "legacy_delay3_close", 3)
        self.assertEqual((corrected[0], corrected[2]), (102.5, 603))
        self.assertEqual((legacy[0], legacy[2]), (103.5, 604))
        no_delay, _ = entry_window(self.day, signal(), "decision_close", 1)
        skipped, _ = entry_window(self.day, signal(), "legacy_skip_first", 1)
        self.assertEqual(no_delay[1][0, 3], 100.5)
        self.assertEqual(skipped[1][0, 3], 101.5)

    def test_missing_or_incomplete_window_is_not_shortened(self):
        self.day.minutes[1:] += 1
        self.assertIsNotNone(entry_window(self.day, signal(), "next_open", 3)[1])
        self.assertIsNotNone(entry_window(self.day, signal(601), "next_open", 1)[1])
        self.assertIsNotNone(entry_window(self.day, signal(611), "next_open", 10)[1])

    def test_short_return_and_excursions_use_short_direction(self):
        result = measure_window(100, np.array([[100, 101, 97, 98.]]), -1)
        self.assertAlmostEqual(result["gross"], .02)
        self.assertAlmostEqual(result["mfe"], .03)
        self.assertAlmostEqual(result["mae"], -.01)


class PathTests(unittest.TestCase):
    def test_target_then_stop_remains_a_win(self):
        prices = np.array([[100, 100.6, 99.9, 100.3], [100.3, 100.4, 98, 99.]])
        r = bracket_path(100, prices, 1, .005, .005)
        self.assertEqual(r["outcome"], "target")
        self.assertEqual(r["exit_minute"], 1)

    def test_same_bar_ambiguity_and_optimistic_bound(self):
        r = bracket_path(100, np.array([[100, 101, 99, 100]]), 1, .005, .003)
        self.assertEqual(r["outcome"], "stop")
        self.assertTrue(r["ambiguous"])
        self.assertAlmostEqual(r["gross"], -.003)
        self.assertAlmostEqual(r["optimistic_gross"], .005)

    def test_gap_through_stop_executes_at_open(self):
        for direction, open_ in ((1, 98.), (-1, 102.)):
            r = bracket_path(100, np.array([[open_, open_ + .1, open_ - .1, open_]]),
                             direction, .005, .005)
            self.assertTrue(r["gap_stop"])
            self.assertAlmostEqual(r["gross"], -.02)

    def test_opening_target_precedes_later_intrabar_stop(self):
        r = bracket_path(100, np.array([[101, 102, 98, 99]]), 1, .005, .005)
        self.assertEqual(r["outcome"], "target")
        self.assertFalse(r["ambiguous"])
        self.assertEqual(r["gross"], .005)


class SignalTests(unittest.TestCase):
    def test_duplicate_directions_collapsed_and_outcomes_ignored(self):
        event = {"kind": "both", "decision_time": "2026-01-05 10:00:00", "close": 100,
                 "alerts": [{"dir": "up", "type": "UP_ALERT"},
                            {"dir": "up", "type": "SHORT_RISK_ALERT"},
                            {"dir": "neutral", "type": "VOLATILITY_ALERT"}]}
        bundle = {"meta": {"symbol": "AAA", "date": "2026-01-05"}, "events": [event]}
        clean = signals_from_bundle(bundle)
        event["outcome"] = {"future_return": -999}
        self.assertEqual(clean, signals_from_bundle(bundle))
        self.assertEqual(len(clean), 1)
        self.assertTrue(clean[0]["actionable"])

    def test_nonoverlap_is_prefix_causal_and_skips_opposite_ties(self):
        sig = [signal(600), signal(602), signal(610), signal(620), signal(620, -1)]
        accepted = nonoverlap_keys(sig, "next_open")
        self.assertEqual(len(accepted), 2)
        self.assertEqual(nonoverlap_keys(sig[:2], "next_open"),
                         {("AAA", sig[0]["decision_time"], 1)})
        # A three-minute delayed entry reserves capacity through 10:13.
        self.assertNotIn(("AAA", sig[2]["decision_time"], 1),
                         nonoverlap_keys(sig, "delay3_open"))

    def test_baseline_excludes_own_day_and_other_signal_times(self):
        days = {}
        for date, close in (("2026-01-05", 150), ("2026-01-06", 101),
                            ("2026-01-07", 120), ("2026-02-02", 160)):
            days[("AAA", date)] = Day("AAA", date, np.array([600]),
                                      np.array([[100, close, 100, close]], float))
        sig = [signal(), signal(date="2026-01-07")]
        frame = pd.DataFrame([{**sig[0], "entry_mode": "next_open",
                               "strategy": "hold_1m", "horizon": 1, "gross": .5}])
        result = matched_baselines(frame, sig, days).iloc[0]
        self.assertEqual(result["baseline_days"], 1)
        self.assertAlmostEqual(result["baseline_gross"], .01)
        self.assertAlmostEqual(result["baseline_edge"], .49)


if __name__ == "__main__":
    unittest.main()
