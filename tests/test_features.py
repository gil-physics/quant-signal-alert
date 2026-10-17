"""Tests for Step 7: per-timeframe feature builders (formulas + no leakage)."""

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


def _bars(symbol, day, n, *, start="09:00", open0=100):
    h, m = int(start[:2]), int(start[3:])
    rows = []
    for i in range(n):
        ts = pd.Timestamp(f"{day} {h:02d}:{m:02d}:00") + pd.Timedelta(minutes=i)
        be = ts + pd.Timedelta(minutes=1)
        rows.append({
            "symbol": symbol,
            "bar_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "bar_end": be.strftime("%Y-%m-%d %H:%M:%S"),
            "open": open0 + i, "high": 110 + i, "low": 90 + i, "close": 105 + i, "volume": 10 + i,
        })
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class MinuteFeatureTests(unittest.TestCase):
    def setUp(self):
        from feature_lib import build_minute_features, COLUMNS_1M

        self.COLUMNS_1M = COLUMNS_1M
        df = pd.concat([_bars("T", "2025-06-02", 70), _bars("T", "2025-06-03", 70)], ignore_index=True)
        self.out = build_minute_features(df, "al_regular")
        self.d1 = self.out[self.out["decision_time"].str.startswith("2025-06-02")].reset_index(drop=True)
        self.d2 = self.out[self.out["decision_time"].str.startswith("2025-06-03")].reset_index(drop=True)

    def test_columns_and_ids(self):
        self.assertEqual(list(self.out.columns), self.COLUMNS_1M)
        # decision_time == bar_end_1m for 1m (must be completely equal)
        self.assertTrue((self.out["decision_time"] == self.out["bar_end_1m"]).all())
        # provenance source tag
        self.assertEqual(set(self.out["source"].unique()), {"al_regular"})

    def test_candle_and_day_anchored_formulas(self):
        r0 = self.d1.iloc[0]
        self.assertAlmostEqual(r0["range_1m"], 110 / 90 - 1)
        self.assertAlmostEqual(r0["body_1m"], 105 / 100 - 1)
        self.assertAlmostEqual(r0["return_from_open_1m"], 105 / 100 - 1)
        self.assertAlmostEqual(r0["distance_from_day_high_1m"], 105 / 110 - 1)
        self.assertAlmostEqual(self.d1.iloc[1]["ret_1m_1m"], 106 / 105 - 1)
        self.assertAlmostEqual(self.d1.iloc[60]["ret_60m_1m"], 165 / 105 - 1)

    def test_first_bar_of_day_returns_are_nan(self):
        self.assertTrue(pd.isna(self.d1.iloc[0]["ret_1m_1m"]))
        self.assertTrue(pd.isna(self.d2.iloc[0]["ret_1m_1m"]))  # no cross-day reference

    def test_no_date_boundary_leak_in_cummax(self):
        # day2 first-bar distance_from_day_high uses day2's own high (110), not day1's max
        self.assertAlmostEqual(self.d2.iloc[0]["distance_from_day_high_1m"], 105 / 110 - 1)

    def test_volume_z_warmup(self):
        self.assertTrue(pd.isna(self.d1.iloc[5]["volume_z_20m_1m"]))
        self.assertFalse(pd.isna(self.d1.iloc[25]["volume_z_20m_1m"]))
        self.assertTrue(pd.isna(self.d1.iloc[40]["volume_z_60m_1m"]))

    def test_dropna_removes_warmup_only(self):
        from features_1m import FEATURE_COLUMNS

        clean = self.out.dropna(subset=FEATURE_COLUMNS)
        # longest window is 60 -> first 60 rows of each day drop, 10 remain per day, 2 days
        self.assertEqual(len(clean), 20)
        self.assertEqual(int(clean[FEATURE_COLUMNS].isna().sum().sum()), 0)

    def test_empty_input(self):
        from feature_lib import build_minute_features

        out = build_minute_features(
            pd.DataFrame(columns=["symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume"]),
            "al_regular",
        )
        self.assertEqual(list(out.columns), self.COLUMNS_1M)
        self.assertEqual(len(out), 0)


def _ma_bars(symbol, day, closes_by_time):
    """One symbol/day with explicit (time, close) bars; OHLC pinned to close ±1."""
    rows = []
    for t, c in closes_by_time:
        ts = f"{day} {t}:00"
        be = (pd.Timestamp(ts) + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({"symbol": symbol, "bar_start": ts, "bar_end": be,
                     "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 10})
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class MovingAverageFeatureTests(unittest.TestCase):
    # six ascending trading days; two bars each at 09:00 and 09:01.
    DAYS = ["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-09"]
    OPEN0900 = [100, 110, 120, 130, 140, 200]   # close at 09:00 per day
    OPEN0901 = [101, 111, 121, 131, 141, 202]   # close at 09:01 per day (= daily close)

    def setUp(self):
        from feature_lib import build_minute_features

        frames = [
            _ma_bars("T", d, [("09:00", a), ("09:01", b)])
            for d, a, b in zip(self.DAYS, self.OPEN0900, self.OPEN0901)
        ]
        self.out = build_minute_features(pd.concat(frames, ignore_index=True), "al_regular")

    def _row(self, day, tod):
        m = self.out[(self.out["decision_time"].str.startswith(day))
                     & (self.out["bar_end_1m"].str.startswith(day))]
        # decision_time == bar_end (one minute after bar_start); match the bar at `tod`
        target_end = (pd.Timestamp(f"{day} {tod}:00") + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        return self.out[self.out["decision_time"] == target_end].iloc[0]

    def test_ma_columns_kept_out_of_feature_columns(self):
        from features_1m import FEATURE_COLUMNS, COLUMNS

        for c in ("ma5_daily", "ma5_intraday", "dist_ma5_daily", "dist_ma5_intraday"):
            self.assertIn(c, COLUMNS)
            self.assertNotIn(c, FEATURE_COLUMNS)   # so warmup MA NaNs never drop a v0 row

    def test_daily_ma_is_prior_5_day_close_mean_and_constant(self):
        # daily close = last bar (09:01); D-5..D-1 closes = [101,111,121,131,141] -> mean 121
        d6_0900 = self._row("2025-06-09", "09:00")
        d6_0901 = self._row("2025-06-09", "09:01")
        self.assertAlmostEqual(d6_0900["ma5_daily"], 121.0)
        self.assertAlmostEqual(d6_0901["ma5_daily"], 121.0)   # constant across the day
        self.assertEqual(int(d6_0900["ma5_daily_n"]), 5)

    def test_intraday_ma_is_time_aligned(self):
        # 09:00 prior closes [100,110,120,130,140] -> 120 ; 09:01 -> [101..141] -> 121
        self.assertAlmostEqual(self._row("2025-06-09", "09:00")["ma5_intraday"], 120.0)
        self.assertAlmostEqual(self._row("2025-06-09", "09:01")["ma5_intraday"], 121.0)
        self.assertEqual(int(self._row("2025-06-09", "09:00")["ma5_intraday_n"]), 5)

    def test_distances_use_current_close(self):
        r = self._row("2025-06-09", "09:00")   # close 200
        self.assertAlmostEqual(r["dist_ma5_intraday"], 200 / 120 - 1)
        self.assertAlmostEqual(r["dist_ma5_daily"], 200 / 121 - 1)

    def test_warmup_is_nan_and_leakage_free(self):
        # day 5 has only 4 prior days -> MA undefined; the MA never sees the current day.
        d5 = self._row("2025-06-06", "09:00")
        self.assertTrue(pd.isna(d5["ma5_daily"]))
        self.assertTrue(pd.isna(d5["ma5_intraday"]))
        # day-6 MA equals the prior-day mean regardless of day-6's own (200/202) close
        self.assertAlmostEqual(self._row("2025-06-09", "09:00")["ma5_intraday"], 120.0)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class BarFeatureTests(unittest.TestCase):
    def test_3m_columns_and_formulas(self):
        import features_3m
        from feature_lib import build_bar_features

        df = pd.concat([_bars("T", "2025-06-02", 25), _bars("T", "2025-06-03", 25)], ignore_index=True)
        out = build_bar_features(df, "3m", "resampled_from_al_regular")
        self.assertEqual(list(out.columns), features_3m.COLUMNS)
        self.assertEqual(set(out["source"].unique()), {"resampled_from_al_regular"})
        d1 = out[out["bar_end_3m"].str.startswith("2025-06-02")].reset_index(drop=True)
        d2 = out[out["bar_end_3m"].str.startswith("2025-06-03")].reset_index(drop=True)
        self.assertAlmostEqual(d1.iloc[1]["ret_1bar_3m"], 106 / 105 - 1)
        self.assertAlmostEqual(d1.iloc[2]["ret_2bar_3m"], 107 / 105 - 1)
        self.assertAlmostEqual(d1.iloc[0]["range_1bar_3m"], 110 / 90 - 1)
        self.assertTrue(pd.isna(d2.iloc[0]["ret_1bar_3m"]))   # no cross-day leak
        self.assertTrue(pd.isna(d1.iloc[5]["volume_z_10bar_3m"]))
        self.assertFalse(pd.isna(d1.iloc[12]["volume_z_10bar_3m"]))

    def test_5m_suffix(self):
        import features_5m

        df = _bars("T", "2025-06-02", 25)
        out = features_5m.build(df, "resampled_from_al_regular")
        self.assertEqual(list(out.columns), features_5m.COLUMNS)
        self.assertIn("ret_1bar_5m", out.columns)
        self.assertIn("volume_z_20bar_5m", out.columns)
        self.assertEqual(set(out["source"].unique()), {"resampled_from_al_regular"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
