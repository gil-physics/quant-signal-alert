"""Tests for the path-dependent bracket primitive (src/bracket_eval.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bracket_eval as be  # noqa: E402


def _arr(highs, lows, closes, offsets):
    return (np.array(highs, float), np.array(lows, float),
            np.array(closes, float), np.array(offsets, float))


class SimulateBracketTests(unittest.TestCase):
    def test_target_first(self):
        h, l, c, off = _arr([101, 103, 104], [100, 101, 102], [100.5, 102, 103.5], [1, 2, 3])
        r = be.simulate_bracket(100.0, h, l, c, off, target_pct=0.03, stop_pct=0.015, time_exit_min=np.inf)
        self.assertEqual(r["outcome"], be.TARGET_FIRST)
        self.assertAlmostEqual(r["realized_return"], 0.03)
        self.assertAlmostEqual(r["mfe"], 0.03)        # max(101,103)/100-1
        self.assertAlmostEqual(r["mae"], 0.0)         # min(100,101)/100-1
        self.assertEqual(r["exit_offset_min"], 2.0)

    def test_stop_first(self):
        h, l, c, off = _arr([101, 102, 102], [99, 98, 97], [100, 98.5, 97.5], [1, 2, 3])
        r = be.simulate_bracket(100.0, h, l, c, off, target_pct=0.03, stop_pct=0.015, time_exit_min=np.inf)
        self.assertEqual(r["outcome"], be.STOP_FIRST)
        self.assertAlmostEqual(r["realized_return"], -0.015)
        self.assertAlmostEqual(r["mae"], -0.02)       # min(99,98)/100-1
        self.assertEqual(r["exit_offset_min"], 2.0)

    def test_same_bar_touch_counts_as_stop(self):
        h, l, c, off = _arr([104], [98], [101], [1])  # both target(103) and stop(98.5) in one bar
        r = be.simulate_bracket(100.0, h, l, c, off, target_pct=0.03, stop_pct=0.015, time_exit_min=np.inf)
        self.assertEqual(r["outcome"], be.STOP_FIRST)
        self.assertAlmostEqual(r["realized_return"], -0.015)

    def test_time_exit_ignores_touch_after_budget(self):
        h, l, c, off = _arr([101, 102, 200], [100, 100, 100], [100.5, 101, 150], [1, 2, 3])
        r = be.simulate_bracket(100.0, h, l, c, off, target_pct=0.03, stop_pct=0.015, time_exit_min=2.0)
        self.assertEqual(r["outcome"], be.TIME_EXIT)
        self.assertAlmostEqual(r["realized_return"], 0.01)   # close[1]/100-1
        self.assertEqual(r["exit_offset_min"], 2.0)
        # with no time budget the later bar reaches target
        r2 = be.simulate_bracket(100.0, h, l, c, off, target_pct=0.03, stop_pct=0.015, time_exit_min=np.inf)
        self.assertEqual(r2["outcome"], be.TARGET_FIRST)

    def test_no_eligible_bar_returns_none(self):
        h, l, c, off = _arr([101], [100], [100.5], [5])
        self.assertIsNone(be.simulate_bracket(100.0, h, l, c, off,
                                              target_pct=0.03, stop_pct=0.015, time_exit_min=2.0))


class ClassifyTests(unittest.TestCase):
    def test_groups(self):
        tgt = _arr([103], [100], [102], [1])
        stp = _arr([101], [98], [99], [1])
        none = _arr([101, 102], [99, 99], [100, 101.5], [1, 2])
        self.assertEqual(be.classify_outcome(100.0, *tgt, target_pct=0.03, stop_pct=0.015)[0], "A")
        self.assertEqual(be.classify_outcome(100.0, *stp, target_pct=0.03, stop_pct=0.015)[0], "B")
        g, res = be.classify_outcome(100.0, *none, target_pct=0.03, stop_pct=0.015)
        self.assertEqual(g, "C")
        self.assertAlmostEqual(res["realized_return"], 0.015)  # last close 101.5 -> +1.5%


class PrepareEntryTests(unittest.TestCase):
    def _bars(self):
        rows = []
        for i, t in enumerate(["09:30", "09:31", "09:32", "09:33", "09:34"]):
            rows.append({"symbol": "AAA", "bar_start": f"2026-01-05 {t}:00",
                         "high": 101 + i, "low": 99 + i, "close": 100 + i})
        return pd.DataFrame(rows)

    def test_delay0_and_delay3(self):
        by_day = be.day_arrays(self._bars())
        arr = by_day[("AAA", "2026-01-05")]
        # The 09:30 candle closes at 09:31, which is decision_time.
        dec = pd.Timestamp("2026-01-05 09:31:00")

        ep, h, l, c, off = be.prepare_entry(arr, dec, 100.0, delay_bars=0)
        self.assertEqual(ep, 100.0)
        self.assertEqual(len(h), 4)                       # 09:31..09:34
        self.assertEqual(list(off), [1.0, 2.0, 3.0, 4.0])

        ep3, h3, l3, c3, off3 = be.prepare_entry(arr, dec, 100.0, delay_bars=3)
        self.assertEqual(ep3, 103.0)                      # close of 09:33 (100+3)
        self.assertEqual(len(h3), 1)                       # only 09:34 left
        self.assertEqual(list(off3), [1.0])

    def test_delay_too_large_returns_none(self):
        by_day = be.day_arrays(self._bars())
        arr = by_day[("AAA", "2026-01-05")]
        dec = pd.Timestamp("2026-01-05 09:31:00")
        self.assertIsNone(be.prepare_entry(arr, dec, 100.0, delay_bars=4))  # entry=last, no post bar
        self.assertIsNone(be.prepare_entry(arr, dec, 100.0, delay_bars=5))  # no entry bar

    def test_first_future_minute_target_is_not_skipped(self):
        bars = self._bars()
        bars.loc[1, ["high", "low", "close"]] = [104, 100, 102]
        bars.loc[2:, ["high", "low", "close"]] = [101, 98, 99]
        arr = be.day_arrays(bars)[("AAA", "2026-01-05")]
        prepared = be.prepare_entry(arr, pd.Timestamp("2026-01-05 09:31"), 100)
        result = be.simulate_bracket(*prepared, target_pct=.03, stop_pct=.015,
                                     time_exit_min=1)
        self.assertEqual(result["outcome"], be.TARGET_FIRST)
        self.assertEqual(result["exit_offset_min"], 1)

    def test_missing_minutes_do_not_shift_delay_or_horizon(self):
        arr = be.day_arrays(self._bars().drop(index=1))[("AAA", "2026-01-05")]
        dec = pd.Timestamp("2026-01-05 09:31")
        self.assertIsNone(be.prepare_entry(arr, dec, 100, delay_bars=1))
        prepared = be.prepare_entry(arr, dec, 100)
        self.assertEqual(list(prepared[-1]), [2., 3., 4.])
        self.assertIsNone(be.simulate_bracket(*prepared, target_pct=.03,
                                             stop_pct=.015, time_exit_min=1))

    def test_last_future_bar_is_evaluable_at_zero_delay(self):
        arr = be.day_arrays(self._bars())[("AAA", "2026-01-05")]
        prepared = be.prepare_entry(arr, pd.Timestamp("2026-01-05 09:34"), 103)
        self.assertEqual(len(prepared[1]), 1)
        self.assertEqual(prepared[-1].tolist(), [1.])

    def test_invalid_delays_rejected(self):
        arr = be.day_arrays(self._bars())[("AAA", "2026-01-05")]
        for delay in (-1, .5, True):
            with self.assertRaises(ValueError):
                be.prepare_entry(arr, pd.Timestamp("2026-01-05 09:31"),
                                 100, delay_bars=delay)


class SummarizeTests(unittest.TestCase):
    def test_metrics(self):
        results = [
            {"outcome": be.TARGET_FIRST, "realized_return": 0.03, "mfe": 0.03, "mae": 0.0, "exit_offset_min": 2},
            {"outcome": be.STOP_FIRST, "realized_return": -0.015, "mfe": 0.01, "mae": -0.02, "exit_offset_min": 3},
            {"outcome": be.TIME_EXIT, "realized_return": 0.005, "mfe": 0.01, "mae": -0.005, "exit_offset_min": 30},
        ]
        s = be.summarize_bracket(results, stop_pct=0.015)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["target_first_rate"], 1 / 3)
        self.assertAlmostEqual(s["stop_first_rate"], 1 / 3)
        self.assertAlmostEqual(s["time_exit_rate"], 1 / 3)
        self.assertAlmostEqual(s["win_rate"], 2 / 3)
        self.assertAlmostEqual(s["avg_win"], (0.03 + 0.005) / 2)
        self.assertAlmostEqual(s["avg_loss"], -0.015)
        self.assertAlmostEqual(s["realized_return"], (0.03 - 0.015 + 0.005) / 3)
        self.assertAlmostEqual(s["expectancy_r"], ((0.03 - 0.015 + 0.005) / 3) / 0.015)

    def test_empty(self):
        s = be.summarize_bracket([], stop_pct=0.015)
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
