"""Tests for Step 15 short-direction forward metrics (src/short_entry_eval.py)."""

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

from directional_alert import select_online_alerts  # noqa: E402
from short_entry_eval import (  # noqa: E402
    BASELINE_COLUMNS,
    LONG_BASELINE_COLUMNS,
    LONG_EVAL_COLUMNS,
    LONG_METRIC_COLUMNS,
    SHORT_EVAL_COLUMNS,
    SHORT_METRIC_COLUMNS,
    baseline_same_time,
    baseline_same_time_long,
    evaluate_long_alerts,
    evaluate_short_alerts,
    paired_t_stat,
)


def _day(symbol, date, specs):
    """specs: list of (HH:MM, high, low, close)."""
    return pd.DataFrame([
        {"symbol": symbol, "bar_start": f"{date} {t}:00", "high": h, "low": l, "close": c}
        for t, h, l, c in specs
    ])


# decision 10:00 -> entry = close of the 10:03 bar (delay 3) = 100.0
FALLING_DAY = [
    ("10:00", 100.5, 99.8, 100.2),
    ("10:01", 100.4, 99.9, 100.1),
    ("10:02", 100.3, 99.8, 100.0),
    ("10:03", 100.2, 99.7, 100.0),
    ("10:04", 100.0, 99.0, 99.2),
    ("10:05", 99.4, 98.4, 98.6),
    ("10:06", 98.8, 97.9, 98.0),   # low 97.9 <= 98.0 -> down2_60m
    ("10:07", 98.2, 97.2, 97.5),
    ("10:08", 97.8, 96.5, 97.0),
]

RISING_DAY = [
    ("10:00", 100.3, 99.8, 100.1),
    ("10:01", 100.4, 99.9, 100.2),
    ("10:02", 100.5, 100.0, 100.3),
    ("10:03", 100.4, 99.9, 100.0),
    ("10:04", 101.0, 100.0, 100.8),
    ("10:05", 101.6, 100.6, 101.4),  # high 101.6 >= 101.5 -> adverse stop touch
    ("10:06", 102.0, 101.0, 101.8),
]


def _alert(symbol="AAA", dt="2026-01-05 10:00:00", close=100.2, **kw):
    row = {"symbol": symbol, "decision_time": dt, "close": close}
    row.update(kw)
    return row


class ShortMetricsTests(unittest.TestCase):
    def test_falling_path_is_favorable_for_a_short(self):
        bars = _day("AAA", "2026-01-05", FALLING_DAY)
        ev = evaluate_short_alerts(pd.DataFrame([_alert()]), bars)
        self.assertEqual(len(ev), 1)
        r = ev.iloc[0]
        self.assertEqual(float(r["entry_price"]), 100.0)
        # raw close returns stay raw: negative = short profit
        self.assertAlmostEqual(float(r["future_return_60m"]), 97.0 / 100.0 - 1)
        self.assertAlmostEqual(float(r["future_return_eod"]), 97.0 / 100.0 - 1)
        self.assertAlmostEqual(float(r["short_mfe"]), 1 - 96.5 / 100.0)   # positive good
        self.assertGreater(float(r["short_mfe"]), 0)
        self.assertAlmostEqual(float(r["short_mae"]), 0.0)                # no post-entry high above entry
        self.assertEqual(int(r["down2_60m"]), 1)
        self.assertEqual(int(r["reach_dn15_eod"]), 1)
        self.assertEqual(int(r["short_stop_first"]), 0)
        for m in (30, 60, 120):
            self.assertEqual(int(r[f"direction_hit_{m}m"]), 1)
        for col in SHORT_EVAL_COLUMNS:
            self.assertIn(col, ev.columns)

    def test_rising_path_stops_a_short_first(self):
        bars = _day("AAA", "2026-01-06", RISING_DAY)
        ev = evaluate_short_alerts(pd.DataFrame([_alert(dt="2026-01-06 10:00:00", close=100.1)]), bars)
        r = ev.iloc[0]
        self.assertEqual(int(r["short_stop_first"]), 1)
        self.assertAlmostEqual(float(r["short_mae"]), 0.02)   # max high 102 / entry 100 - 1
        self.assertEqual(int(r["down2_60m"]), 0)
        self.assertEqual(int(r["direction_hit_30m"]), 0)      # positive return = miss

    def test_same_bar_tie_counts_as_stop_first(self):
        # one post-entry bar touches BOTH +1.5% (102) and -1.5% (98.5)
        specs = [
            ("10:00", 100.2, 99.8, 100.0),
            ("10:01", 100.2, 99.8, 100.0),
            ("10:02", 100.2, 99.8, 100.0),
            ("10:03", 100.2, 99.8, 100.0),
            ("10:04", 102.0, 98.0, 100.0),
        ]
        bars = _day("AAA", "2026-01-08", specs)
        ev = evaluate_short_alerts(pd.DataFrame([_alert(dt="2026-01-08 10:00:00", close=100.0)]), bars)
        r = ev.iloc[0]
        self.assertEqual(int(r["down2_60m"]), 1)              # the favorable touch happened...
        self.assertEqual(int(r["short_stop_first"]), 1)       # ...but the tie is judged adverse-first

    def test_entry_is_decision_plus_3_bar_close(self):
        # closes 100..105; the alert's own close (100.0) must NOT be the entry
        specs = [(f"10:0{i}", 100 + i + 1, 100 + i - 0.5, 100 + i) for i in range(6)]
        bars = _day("AAA", "2026-01-07", specs)
        ev = evaluate_short_alerts(pd.DataFrame([_alert(dt="2026-01-07 10:00:00", close=100.0)]), bars)
        r = ev.iloc[0]
        self.assertEqual(int(r["entry_delay_minutes"]), 3)
        self.assertEqual(float(r["entry_price"]), 103.0)      # close of the 10:03 bar
        self.assertAlmostEqual(float(r["future_return_eod"]), 105.0 / 103.0 - 1)


class NoFuturePeekingTests(unittest.TestCase):
    def _alerts(self):
        return pd.DataFrame([
            _alert(dt="2026-01-05 10:00:00", close=100.2),
            _alert(dt="2026-01-05 10:04:00", close=99.2),
        ])

    def test_injected_future_column_changes_nothing(self):
        bars = _day("AAA", "2026-01-05", FALLING_DAY)

        def pipeline(alerts):
            selected = select_online_alerts(alerts, daily_alert_limit=5, cooldown_minutes=15)
            return selected, evaluate_short_alerts(selected, bars)

        clean_sel, clean_ev = pipeline(self._alerts())
        poisoned = self._alerts()
        poisoned["future_return_60m"] = [9.9, -9.9]   # garbage forward column
        dirty_sel, dirty_ev = pipeline(poisoned)

        # selection is untouched by the forward column
        self.assertEqual(clean_sel["decision_time"].tolist(), dirty_sel["decision_time"].tolist())
        # evaluation recomputes every forward metric from bars (injection overwritten)
        self.assertEqual(len(clean_ev), len(dirty_ev))
        for col in SHORT_METRIC_COLUMNS:
            np.testing.assert_allclose(
                clean_ev[col].to_numpy(dtype=float), dirty_ev[col].to_numpy(dtype=float)
            )
        self.assertFalse((dirty_ev["future_return_60m"] == 9.9).any())

    def test_cooldown_selection_is_causal(self):
        # 10:04 is inside the 15m cooldown of 10:00 -> suppressed
        full = select_online_alerts(self._alerts(), daily_alert_limit=5, cooldown_minutes=15)
        self.assertEqual(full["decision_time"].tolist(), ["2026-01-05 10:00:00"])
        # prefix property: selecting on only the earlier rows gives the same head
        prefix = select_online_alerts(self._alerts().iloc[:1], daily_alert_limit=5, cooldown_minutes=15)
        self.assertEqual(prefix["decision_time"].tolist(), full["decision_time"].tolist())
        # with a 3-minute cooldown the 4-minute gap fires both
        loose = select_online_alerts(self._alerts(), daily_alert_limit=5, cooldown_minutes=3)
        self.assertEqual(len(loose), 2)


class BaselineTests(unittest.TestCase):
    def test_same_time_baseline_uses_only_other_days(self):
        bars = pd.concat([
            _day("AAA", "2026-01-05", FALLING_DAY),
            _day("AAA", "2026-01-06", RISING_DAY),
        ], ignore_index=True)
        alerts = pd.DataFrame([_alert(dt="2026-01-05 10:00:00", close=100.2)])
        base = baseline_same_time(alerts, bars)
        self.assertEqual(list(base.columns), BASELINE_COLUMNS)
        self.assertEqual(len(base), 1)
        r = base.iloc[0]
        self.assertEqual(int(r["baseline_n_days"]), 1)        # own day excluded
        # rising-day 10:00 entry: close of its 10:03 bar = 100.0, EOD close 101.8
        self.assertAlmostEqual(float(r["baseline_future_return_60m"]), 101.8 / 100.0 - 1)
        self.assertEqual(int(r["baseline_short_stop_first"]), 1)
        # the alert's own falling day (-3%) never leaks into its baseline
        self.assertGreater(float(r["baseline_future_return_60m"]), 0)

    def test_paired_t_stat_on_future_return_60m(self):
        keys = ["2026-01-05 10:00:00", "2026-01-06 10:00:00", "2026-01-07 10:00:00"]
        ev = pd.DataFrame({
            "symbol": ["AAA"] * 3,
            "decision_time": keys,
            "future_return_60m": [-0.02, -0.01, -0.03],
        })
        base = pd.DataFrame({
            "symbol": ["AAA"] * 3,
            "decision_time": keys,
            "baseline_future_return_60m": [0.0, 0.0, 0.0],
        })
        res = paired_t_stat(ev, base)
        self.assertEqual(res["n"], 3)
        self.assertAlmostEqual(res["mean_diff"], -0.02)
        self.assertAlmostEqual(res["t_stat"], -0.02 / (0.01 / np.sqrt(3)), places=6)

        # NaN pairs are dropped; a single pair has no t-stat
        ev.loc[1, "future_return_60m"] = np.nan
        res2 = paired_t_stat(ev.iloc[:2], base.iloc[:2])
        self.assertEqual(res2["n"], 1)
        self.assertAlmostEqual(res2["mean_diff"], -0.02)
        self.assertTrue(np.isnan(res2["t_stat"]))

    def test_empty_inputs_are_graceful(self):
        empty = pd.DataFrame(columns=["symbol", "decision_time", "close"])
        bars = _day("AAA", "2026-01-05", FALLING_DAY)
        self.assertEqual(len(evaluate_short_alerts(empty, bars)), 0)
        self.assertEqual(len(baseline_same_time(empty, bars)), 0)
        res = paired_t_stat(empty.assign(future_return_60m=[]), pd.DataFrame())
        self.assertEqual(res["n"], 0)
        self.assertTrue(np.isnan(res["t_stat"]))


class LongMetricsTests(unittest.TestCase):
    """Bounce-long (dip-buy) mirror: LONG-direction metrics on the same fixtures."""

    def test_rising_path_is_favorable_for_a_long(self):
        bars = _day("AAA", "2026-01-06", RISING_DAY)
        ev = evaluate_long_alerts(pd.DataFrame([_alert(dt="2026-01-06 10:00:00", close=100.1)]), bars)
        self.assertEqual(len(ev), 1)
        r = ev.iloc[0]
        self.assertEqual(float(r["entry_price"]), 100.0)
        # raw close returns stay raw: positive = long profit
        self.assertAlmostEqual(float(r["future_return_60m"]), 101.8 / 100.0 - 1)
        self.assertAlmostEqual(float(r["future_return_eod"]), 101.8 / 100.0 - 1)
        self.assertAlmostEqual(float(r["long_mfe"]), 102.0 / 100.0 - 1)  # positive good
        self.assertGreater(float(r["long_mfe"]), 0)
        self.assertAlmostEqual(float(r["long_mae"]), 0.0)                # no post-entry low below entry
        self.assertEqual(int(r["up2_60m"]), 1)                           # high 102.0 >= entry*1.02
        self.assertEqual(int(r["reach_up15_eod"]), 1)
        self.assertEqual(int(r["long_stop_first"]), 0)
        for m in (30, 60, 120):
            self.assertEqual(int(r[f"direction_hit_{m}m"]), 1)
        for col in LONG_EVAL_COLUMNS:
            self.assertIn(col, ev.columns)

    def test_falling_path_stops_a_long_first(self):
        bars = _day("AAA", "2026-01-05", FALLING_DAY)
        ev = evaluate_long_alerts(pd.DataFrame([_alert()]), bars)
        r = ev.iloc[0]
        self.assertEqual(int(r["long_stop_first"]), 1)    # low 98.4 <= 98.5 before any +1.5% high
        self.assertAlmostEqual(float(r["long_mae"]), 1 - 96.5 / 100.0)   # positive bad
        self.assertEqual(int(r["up2_60m"]), 0)
        self.assertEqual(int(r["direction_hit_60m"]), 0)  # negative return = miss for a long

    def test_same_bar_tie_counts_as_stop_first_for_a_long_too(self):
        # one post-entry bar touches BOTH +1.5% (101.5) and -1.5% (98.5)
        specs = [
            ("10:00", 100.2, 99.8, 100.0),
            ("10:01", 100.2, 99.8, 100.0),
            ("10:02", 100.2, 99.8, 100.0),
            ("10:03", 100.2, 99.8, 100.0),
            ("10:04", 102.0, 98.0, 100.0),
        ]
        bars = _day("AAA", "2026-01-08", specs)
        ev = evaluate_long_alerts(pd.DataFrame([_alert(dt="2026-01-08 10:00:00", close=100.0)]), bars)
        r = ev.iloc[0]
        self.assertEqual(int(r["up2_60m"]), 1)            # the favorable touch happened...
        self.assertEqual(int(r["long_stop_first"]), 1)    # ...but the tie is judged adverse-first


class LongBaselineTests(unittest.TestCase):
    def test_long_same_time_baseline_uses_only_other_days(self):
        bars = pd.concat([
            _day("AAA", "2026-01-05", FALLING_DAY),
            _day("AAA", "2026-01-06", RISING_DAY),
        ], ignore_index=True)
        alerts = pd.DataFrame([_alert(dt="2026-01-06 10:00:00", close=100.1)])
        base = baseline_same_time_long(alerts, bars)
        self.assertEqual(list(base.columns), LONG_BASELINE_COLUMNS)
        self.assertEqual(len(base), 1)
        r = base.iloc[0]
        self.assertEqual(int(r["baseline_n_days"]), 1)    # own day excluded
        # falling-day 10:00 entry: close of its 10:03 bar = 100.0, EOD close 97.0
        self.assertAlmostEqual(float(r["baseline_future_return_60m"]), 97.0 / 100.0 - 1)
        self.assertEqual(int(r["baseline_long_stop_first"]), 1)
        # the alert's own rising day (+1.8%) never leaks into its baseline
        self.assertLess(float(r["baseline_future_return_60m"]), 0)

    def test_paired_t_stat_sign_is_positive_for_a_long_edge(self):
        keys = ["2026-01-05 10:00:00", "2026-01-06 10:00:00", "2026-01-07 10:00:00"]
        ev = pd.DataFrame({
            "symbol": ["AAA"] * 3,
            "decision_time": keys,
            "future_return_60m": [0.02, 0.01, 0.03],
        })
        base = pd.DataFrame({
            "symbol": ["AAA"] * 3,
            "decision_time": keys,
            "baseline_future_return_60m": [0.0, 0.0, 0.0],
        })
        res = paired_t_stat(ev, base)
        self.assertEqual(res["n"], 3)
        self.assertAlmostEqual(res["mean_diff"], 0.02)    # positive = long edge
        self.assertAlmostEqual(res["t_stat"], 0.02 / (0.01 / np.sqrt(3)), places=6)
        self.assertGreater(res["t_stat"], 0)


class LongNoFuturePeekingTests(unittest.TestCase):
    def _alerts(self):
        return pd.DataFrame([
            _alert(dt="2026-01-06 10:00:00", close=100.1),
            _alert(dt="2026-01-06 10:04:00", close=100.8),
        ])

    def test_injected_future_column_changes_nothing_for_the_long_path(self):
        bars = _day("AAA", "2026-01-06", RISING_DAY)

        def pipeline(alerts):
            selected = select_online_alerts(alerts, daily_alert_limit=5, cooldown_minutes=15)
            return selected, evaluate_long_alerts(selected, bars)

        clean_sel, clean_ev = pipeline(self._alerts())
        poisoned = self._alerts()
        poisoned["future_return_60m"] = [9.9, -9.9]   # garbage forward column
        poisoned["long_mfe"] = [7.7, -7.7]            # garbage forward column
        dirty_sel, dirty_ev = pipeline(poisoned)

        # selection is untouched by the forward columns
        self.assertEqual(clean_sel["decision_time"].tolist(), dirty_sel["decision_time"].tolist())
        # evaluation recomputes every forward metric from bars (injection overwritten)
        self.assertEqual(len(clean_ev), len(dirty_ev))
        for col in LONG_METRIC_COLUMNS:
            np.testing.assert_allclose(
                clean_ev[col].to_numpy(dtype=float), dirty_ev[col].to_numpy(dtype=float)
            )
        self.assertFalse((dirty_ev["future_return_60m"] == 9.9).any())
        self.assertFalse((dirty_ev["long_mfe"] == 7.7).any())


class ModuleGuardTests(unittest.TestCase):
    def test_no_order_api_and_no_hardcoded_symbol(self):
        text = (SRC_DIR / "short_entry_eval.py").read_text(encoding="utf-8")
        for token in ("SendOrder", "send_order", "place_order"):
            self.assertNotIn(token, text)
        self.assertNotIn("000660", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
