"""Tests for Step 10: forward-looking labels."""

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


def _bars(day, rows):
    """rows: list of (HH:MM, high, low)."""
    out = []
    for hhmm, hi, lo in rows:
        ts = pd.Timestamp(f"{day} {hhmm}:00")
        out.append({
            "bar_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "bar_end": (ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "high": hi, "low": lo,
        })
    return pd.DataFrame(out)


def _dec(day, hhmm, close=100):
    return pd.DataFrame([{"decision_time": f"{day} {hhmm}:00", "close": close}])


def _label(rows, dec_hhmm, close=100, day="2025-06-02"):
    from labeling import label_symbol
    out = label_symbol("T", _dec(day, dec_hhmm, close), _bars(day, rows))
    return out.iloc[0]


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class CurrentBarExclusionTests(unittest.TestCase):
    def test_current_bar_not_in_future(self):
        # current bar (09:00, ending at 09:01) has an extreme high; it must be excluded
        rows = [("09:00", 200, 50), ("09:01", 101, 99), ("09:02", 101, 99)]
        r = _label(rows, "09:01", close=100)
        self.assertEqual(r["up5_eod"], 0)   # would be 1 if the 09:00 high=200 leaked in
        self.assertEqual(r["up2_eod"], 0)   # future highs are 101 < 102
        self.assertEqual(r["n_future_bars_eod"], 2)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class HorizonTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(91):  # 09:00 .. 10:30
            ts = pd.Timestamp("2025-06-02 09:00:00") + pd.Timedelta(minutes=i)
            hhmm = ts.strftime("%H:%M")
            hi, lo = 101, 99
            if hhmm == "09:30":
                hi = 103
            if hhmm == "09:50":
                lo = 96
            rows.append((hhmm, hi, lo))
        self.r = _label(rows, "09:01", close=100)

    def test_up_down_thresholds_60m(self):
        self.assertEqual(self.r["up2_60m"], 1)   # 103 >= 102
        self.assertEqual(self.r["up3_60m"], 1)   # 103 >= 103
        self.assertEqual(self.r["up5_60m"], 0)   # 103 <  105
        self.assertEqual(self.r["down2_60m"], 1)  # 96 <= 98
        self.assertEqual(self.r["down3_60m"], 1)  # 96 <= 97
        self.assertEqual(self.r["down5_60m"], 0)  # 96 >  95

    def test_has_full_flags(self):
        self.assertEqual(self.r["has_full_60m"], 1)   # 60 min observable (data to 10:31)
        self.assertEqual(self.r["has_full_120m"], 0)
        self.assertEqual(self.r["has_full_180m"], 0)

    def test_event_fail(self):
        # +5% never reached, -1.5% stop (98.5) hit at 09:50 -> UP_FAIL
        self.assertEqual(self.r["up5_event"], "UP_FAIL")


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class PathEventTests(unittest.TestCase):
    def test_success_target_before_stop(self):
        rows = [("09:00", 100, 100), ("09:01", 106, 99), ("09:02", 101, 96)]  # +5% at 09:01, stop later
        r = _label(rows, "09:01", close=100)
        self.assertEqual(r["up5_event"], "UP5_SUCCESS")
        self.assertEqual(r["up5_eod"], 1)

    def test_fail_stop_before_target(self):
        rows = [("09:00", 100, 100), ("09:01", 101, 98), ("09:02", 106, 99)]  # stop(98.5) at 09:01, +5% later
        r = _label(rows, "09:01", close=100)
        self.assertEqual(r["up5_event"], "UP_FAIL")

    def test_same_bar_tie_is_fail(self):
        rows = [("09:00", 100, 100), ("09:01", 106, 98)]  # both target & stop in one bar -> pessimistic FAIL
        r = _label(rows, "09:01", close=100)
        self.assertEqual(r["up5_event"], "UP_FAIL")

    def test_no_event(self):
        rows = [("09:00", 100, 100), ("09:01", 101, 99), ("09:02", 101, 99)]
        r = _label(rows, "09:01", close=100)
        self.assertEqual(r["up5_event"], "NO_EVENT")
        self.assertEqual(r["up5_eod"], 0)
        self.assertEqual(r["down5_eod"], 0)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class DateIsolationTests(unittest.TestCase):
    def test_no_cross_day_lookahead(self):
        from labeling import label_symbol

        bars = pd.concat([
            _bars("2025-06-02", [("09:00", 100, 100), ("09:01", 101, 99)]),
            _bars("2025-06-03", [("09:00", 200, 50), ("09:01", 101, 99)]),  # day2 extreme
        ], ignore_index=True)
        dec = _dec("2025-06-02", "09:01", 100)  # decision on day1
        r = label_symbol("T", dec, bars).iloc[0]
        self.assertEqual(r["up5_eod"], 0)   # must NOT see day2's 200 high
        self.assertEqual(r["n_future_bars_eod"], 1)  # only 09:01 of day1


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class DistributionTests(unittest.TestCase):
    def test_distribution_shape(self):
        from labeling import EVENT_VALUES, LABEL_COLUMNS, label_distribution, label_symbol

        rows = [("09:00", 100, 100)] + [(f"09:{m:02d}", 101, 99) for m in range(1, 30)]
        out = label_symbol("T", _dec("2025-06-02", "09:01", 100), _bars("2025-06-02", rows))
        dist = label_distribution(out)
        labels = {d["label"] for d in dist}
        for col in LABEL_COLUMNS:
            self.assertIn(col, labels)
        for v in EVENT_VALUES:
            self.assertIn(f"up5_event={v}", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
