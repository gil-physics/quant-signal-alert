"""Tests for v3.0 premarket-context features (src/premarket_features.py).

Pure-function tests only: synthetic bar frames, no DB / no Kiwoom. Imports from src
(SRC_DIR on path); the scripts dir is NEVER added to the global sys.path (that would
re-trigger the threshold_sweep self-import break under discovery) -- the build script is
checked statically in test_v3_premarket_model_versions.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    HAVE_PANDAS = False


def _bar(sym, date, end_hms, o, h, l, c, v):
    """One 1-minute bar whose bar_end is `date end_hms` (bar_start = bar_end - 1m)."""
    end = f"{date} {end_hms}"
    start = str(pd.Timestamp(end) - pd.Timedelta(minutes=1))
    return {"symbol": sym, "bar_start": start, "bar_end": end,
            "open": o, "high": h, "low": l, "close": c, "volume": v}


def _regular_close_bar(sym, date, close):
    """The last regular bar of the day (bar_start 15:30 -> bar_end 15:31)."""
    return _bar(sym, date, "15:31:00", close, close, close, close, 10)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class PremarketWindowTests(unittest.TestCase):
    def _build(self, bars, **kw):
        from premarket_features import build_daily_premarket_features
        kw.setdefault("symbols", ["T"])
        kw.setdefault("target_symbol", "T")
        return build_daily_premarket_features(pd.DataFrame(bars), **kw)

    def test_window_selects_only_premarket_bar_ends(self):
        # case 1: only 08:00 <= bar_end < 09:00 bars enter the premarket aggregates.
        bars = [
            _bar("T", "2026-01-06", "08:05:00", 100, 110, 100, 105, 10),  # in
            _bar("T", "2026-01-06", "08:55:00", 105, 118, 104, 118, 30),  # in
            _bar("T", "2026-01-06", "09:00:00", 118, 999, 118, 900, 50),  # OUT (boundary, excluded)
            _bar("T", "2026-01-06", "09:30:00", 900, 999, 900, 950, 70),  # OUT (regular)
        ]
        out = self._build(bars).iloc[0]
        self.assertEqual(out["pm_T_bar_count"], 2)
        # range = premarket high/low - 1 (needs no prev close); high=118, low=100 from the
        # in-window bars only -- if the 999/900 excluded bars leaked it would blow up.
        self.assertAlmostEqual(out["pm_T_range"], 118 / 100 - 1)

    def test_regular_bars_not_used_for_premarket_aggregates(self):
        # case 2: a wild 10:00 regular bar must not leak into premarket high/low/close.
        bars = [
            _bar("T", "2026-01-06", "08:10:00", 100, 110, 100, 108, 10),
            _bar("T", "2026-01-06", "10:00:00", 108, 5000, 1, 4000, 99),  # regular, extreme
        ]
        out = self._build(bars).iloc[0]
        self.assertEqual(out["pm_T_bar_count"], 1)
        # high=110, low=100 from the single premarket bar; the 5000/1 regular bar must not leak.
        self.assertAlmostEqual(out["pm_T_range"], 110 / 100 - 1)

    def test_previous_regular_close_used_for_gap(self):
        # case 3: pm_ret = premarket_last_close / PREVIOUS regular close - 1, and the
        # SAME-day 15:30 close (which is AFTER the premarket window) must NOT be used.
        bars = [
            _bar("T", "2026-01-05", "08:10:00", 100, 120, 100, 118, 10),   # premarket day 1
            _regular_close_bar("T", "2026-01-05", 110),                    # 01-05 regular close = 110
            _bar("T", "2026-01-06", "08:10:00", 110, 130, 110, 120, 10),   # premarket day 2
            _regular_close_bar("T", "2026-01-06", 130),
        ]
        out = self._build(bars).set_index("trade_date")
        # 01-05 has premarket but NO prior regular close, and its own 15:30=110 is in the
        # future relative to 08:00-09:00 -> pm_ret is NaN (leakage guard).
        self.assertTrue(np.isnan(out.loc["2026-01-05"]["pm_T_ret"]))
        # 01-06 uses the genuine previous day's regular close (01-05 = 110).
        self.assertAlmostEqual(out.loc["2026-01-06"]["pm_T_ret"], 120 / 110 - 1)

    def test_target_ret_range_vwap(self):
        # case 4: ret / high_ret / low_ret / range / vwap / close_to_vwap.
        bars = [
            _regular_close_bar("T", "2026-01-05", 100),
            _bar("T", "2026-01-06", "08:01:00", 100, 110, 100, 105, 10),
            _bar("T", "2026-01-06", "08:02:00", 105, 120, 104, 118, 30),
        ]
        out = self._build(bars).set_index("trade_date").loc["2026-01-06"]
        self.assertAlmostEqual(out["pm_T_ret"], 118 / 100 - 1)
        self.assertAlmostEqual(out["pm_T_high_ret"], 120 / 100 - 1)
        self.assertAlmostEqual(out["pm_T_low_ret"], 100 / 100 - 1)
        self.assertAlmostEqual(out["pm_T_range"], 120 / 100 - 1)
        t1 = (110 + 100 + 105) / 3
        t2 = (120 + 104 + 118) / 3
        vwap = (t1 * 10 + t2 * 30) / 40
        self.assertAlmostEqual(out["pm_T_vwap"], vwap)
        self.assertAlmostEqual(out["pm_T_close_to_vwap"], 118 / vwap - 1)

    def test_last_5m_ret_anchored_5min_before_last_bar(self):
        bars = [_regular_close_bar("T", "2026-01-05", 100)]
        closes = [100, 101, 102, 103, 104, 110]   # 08:01 .. 08:06
        for i, c in enumerate(closes, start=1):
            bars.append(_bar("T", "2026-01-06", f"08:0{i}:00", c, c + 1, c - 1, c, 10))
        out = self._build(bars).set_index("trade_date").loc["2026-01-06"]
        # last close 110 (08:06); ref = close at bar_end <= 08:01 -> 100
        self.assertAlmostEqual(out["pm_T_last_5m_ret"], 110 / 100 - 1)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class AnchorFeatureTests(unittest.TestCase):
    """cases 5/6/8: anchor features + graceful handling of a missing anchor."""

    def _frame(self):
        bars = [
            _regular_close_bar("T", "2026-01-05", 100),
            _regular_close_bar("AN", "2026-01-05", 200),
            _bar("T", "2026-01-06", "08:10:00", 100, 115, 100, 110, 10),    # T +10%
            _bar("AN", "2026-01-06", "08:10:00", 200, 210, 198, 204, 20),   # AN +2%
        ]
        from premarket_features import build_daily_premarket_features
        return build_daily_premarket_features(
            pd.DataFrame(bars), symbols=["T", "AN"], target_symbol="T", anchors=["AN"],
        ).set_index("trade_date")

    def test_anchor_ret_and_spread(self):
        out = self._frame().loc["2026-01-06"]
        self.assertAlmostEqual(out["pm_AN_ret"], 204 / 200 - 1)
        self.assertAlmostEqual(out["pm_T_minus_AN_ret"], (110 / 100 - 1) - (204 / 200 - 1))
        self.assertEqual(out["pm_AN_has_data"], 1.0)

    def test_missing_anchor_does_not_crash(self):
        # case 8: anchor with no premarket bars -> NaN features + has_data flag 0.
        bars = [
            _regular_close_bar("T", "2026-01-05", 100),
            _bar("T", "2026-01-06", "08:10:00", 100, 115, 100, 110, 10),
        ]
        from premarket_features import build_daily_premarket_features
        out = build_daily_premarket_features(
            pd.DataFrame(bars), symbols=["T"], target_symbol="T", anchors=["MISSING"],
        ).set_index("trade_date").loc["2026-01-06"]
        self.assertEqual(out["pm_MISSING_has_data"], 0.0)
        self.assertTrue(np.isnan(out["pm_MISSING_ret"]))


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class UniverseFeatureTests(unittest.TestCase):
    def test_breadth_relative_and_rank(self):
        # case 7: 3 symbols, prev close 100 each. day2 returns A +10%, B +5%, C -10%.
        bars = []
        for s in ("A", "B", "C"):
            bars.append(_regular_close_bar(s, "2026-01-05", 100))
        for s, c in (("A", 110), ("B", 105), ("C", 90)):
            bars.append(_bar(s, "2026-01-06", "08:10:00", 100, c + 1, 89, c, 10))
        from premarket_features import build_daily_premarket_features
        out = build_daily_premarket_features(
            pd.DataFrame(bars), symbols=["A", "B", "C"], target_symbol="A",
            universe_symbols=["A", "B", "C"],
        ).set_index("trade_date").loc["2026-01-06"]
        self.assertAlmostEqual(out["pm_universe_breadth_pos"], 2 / 3)          # A,B up
        self.assertAlmostEqual(out["pm_universe_mean_ret"], (0.10 + 0.05 - 0.10) / 3)
        self.assertAlmostEqual(out["pm_A_relative_ret_to_universe"], 0.10 - (0.05) / 3)
        self.assertEqual(out["pm_A_rank_ret"], 1)                              # A strongest
        self.assertEqual(out["pm_universe_active_symbol_count"], 3)
        self.assertEqual(out["pm_universe_missing_symbol_count"], 0)

    def test_missing_premarket_day_flagged_not_crashed(self):
        # case 8: a day where the target has no premarket bars -> has_data 0, ret NaN.
        bars = [
            _regular_close_bar("A", "2026-01-05", 100),
            _bar("B", "2026-01-06", "08:10:00", 100, 110, 100, 105, 10),   # only B has premarket
        ]
        from premarket_features import build_daily_premarket_features
        out = build_daily_premarket_features(
            pd.DataFrame(bars), symbols=["A", "B"], target_symbol="A", universe_symbols=["A", "B"],
        ).set_index("trade_date").loc["2026-01-06"]
        self.assertEqual(out["pm_A_has_data"], 0.0)
        self.assertTrue(np.isnan(out["pm_A_ret"]))


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class LeakageTests(unittest.TestCase):
    def test_future_days_do_not_change_earlier_features(self):
        # case 10: an earlier day's features must depend only on that day + the prior
        # close, never on any LATER trading day.
        from premarket_features import build_daily_premarket_features
        base = [
            _regular_close_bar("T", "2026-01-05", 100),
            _bar("T", "2026-01-06", "08:10:00", 100, 120, 100, 118, 10),
            _regular_close_bar("T", "2026-01-06", 118),
        ]
        out1 = build_daily_premarket_features(pd.DataFrame(base), symbols=["T"], target_symbol="T") \
            .set_index("trade_date").loc["2026-01-06"]
        future = base + [
            _bar("T", "2026-01-07", "08:10:00", 999, 9999, 1, 5000, 999),
            _regular_close_bar("T", "2026-01-07", 5000),
        ]
        out2 = build_daily_premarket_features(pd.DataFrame(future), symbols=["T"], target_symbol="T") \
            .set_index("trade_date").loc["2026-01-06"]
        self.assertAlmostEqual(out1["pm_T_ret"], out2["pm_T_ret"])
        self.assertAlmostEqual(out1["pm_T_range"], out2["pm_T_range"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
