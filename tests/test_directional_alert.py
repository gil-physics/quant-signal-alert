"""Tests for the Step 13_5 directional alert system (src/directional_alert.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from directional_alert import (  # noqa: E402
    ALERT_COLUMNS,
    AlertThresholds,
    alert_fire_gate,
    compute_directional_alerts,
    select_online_alerts,
    select_topk_diagnostic,
    up_alert_gate,
)

_BASE = {
    "up_score": 1.0, "down_score": 1.0, "ret_5m_1m": 0.0, "ret_20m_1m": 0.0,
    "volume_z_20m_1m": 0.5, "rank_ret_20m_1m": 10, "active_symbol_count": 20,
    "relative_ret_20m_to_universe": 0.0, "universe_breadth": 0.5,
    "upper_shadow_1m": 0.0, "vwap_distance_1m": 0.0, "distance_from_day_high_1m": -0.02,
    "return_from_open_1m": 0.0, "range_1m": 0.002, "dist_ma5_intraday": 0.0,
}


def _row(**over):
    r = dict(_BASE)
    r.update(over)
    return r


def _frame(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "symbol", [f"S{i:02d}" for i in range(len(df))])
    df.insert(1, "decision_time", [f"2026-01-05 10:{i:02d}:00" for i in range(len(df))])
    return df


class AlertTypeGradeTests(unittest.TestCase):
    def _classify(self, **over):
        out = compute_directional_alerts(_frame([_row(**over)])).iloc[0]
        return out["alert_type"], out["alert_grade"]

    def test_grade_a_clean_strong_up(self):
        t, g = self._classify(up_score=5, down_score=0, ret_5m_1m=0.005, ret_20m_1m=0.012,
                              volume_z_20m_1m=1.5, rank_ret_20m_1m=1, relative_ret_20m_to_universe=0.01,
                              universe_breadth=0.7, dist_ma5_intraday=0.003,
                              distance_from_day_high_1m=-0.003, return_from_open_1m=0.015)
        self.assertEqual(t, "UP_CONTINUATION")
        self.assertEqual(g, "A")

    def test_grade_b_moderate_up(self):
        t, g = self._classify(up_score=3.5, down_score=1, ret_5m_1m=0.002, ret_20m_1m=0.005,
                              volume_z_20m_1m=1.2, rank_ret_20m_1m=3, relative_ret_20m_to_universe=0.002,
                              universe_breadth=0.6, dist_ma5_intraday=0.001,
                              distance_from_day_high_1m=-0.01)
        self.assertEqual(t, "UP_ALERT")
        self.assertEqual(g, "B")

    def test_grade_c_weak_up(self):
        t, g = self._classify(up_score=3, down_score=1.5, ret_5m_1m=0.001, ret_20m_1m=0.004,
                              volume_z_20m_1m=1.0, rank_ret_20m_1m=8, relative_ret_20m_to_universe=-0.001,
                              universe_breadth=0.4, dist_ma5_intraday=-0.001,
                              distance_from_day_high_1m=-0.02)
        self.assertEqual(t, "UP_ALERT")
        self.assertEqual(g, "C")

    def test_risk_alert_downside_dominant(self):
        t, g = self._classify(up_score=1, down_score=4, ret_5m_1m=-0.003, ret_20m_1m=-0.01,
                              volume_z_20m_1m=2.0, rank_ret_20m_1m=18,
                              relative_ret_20m_to_universe=-0.01, universe_breadth=0.3,
                              dist_ma5_intraday=-0.005)
        self.assertEqual(t, "RISK_ALERT")
        self.assertEqual(g, "RISK")

    def test_weakening_alert_momentum_fading(self):
        t, g = self._classify(up_score=3, down_score=1, ret_5m_1m=-0.002, ret_20m_1m=0.01,
                              volume_z_20m_1m=1.0, rank_ret_20m_1m=5, relative_ret_20m_to_universe=0.005,
                              universe_breadth=0.6, dist_ma5_intraday=0.001,
                              distance_from_day_high_1m=-0.004, return_from_open_1m=0.012)
        self.assertEqual(t, "WEAKENING_ALERT")

    def test_volatility_alert_no_direction(self):
        t, g = self._classify(up_score=1.5, down_score=1.25, ret_5m_1m=0.001, ret_20m_1m=0.001,
                              volume_z_20m_1m=2.2, rank_ret_20m_1m=10, range_1m=0.007)
        self.assertEqual(t, "VOLATILITY_ALERT")
        self.assertEqual(g, "C")

    def test_flat_row_is_no_alert(self):
        t, g = self._classify()
        self.assertEqual(t, "")
        self.assertEqual(g, "")

    def test_high_microstructure_risk_downgrades_bullish_to_risk(self):
        # strongly bullish but stretched + rejected + at-high -> graded RISK, reasons warn.
        out = compute_directional_alerts(_frame([_row(
            up_score=5, down_score=0, ret_5m_1m=0.004, ret_20m_1m=0.012, volume_z_20m_1m=3.0,
            rank_ret_20m_1m=1, relative_ret_20m_to_universe=0.01, universe_breadth=0.7,
            dist_ma5_intraday=0.004, upper_shadow_1m=0.006, vwap_distance_1m=0.015,
            distance_from_day_high_1m=0.0, return_from_open_1m=0.03, range_1m=0.008)]) ).iloc[0]
        self.assertGreaterEqual(out["risk_score"], 3.0)
        self.assertEqual(out["alert_grade"], "RISK")
        self.assertIn("upper_rejection", [out["reason_1"], out["reason_2"], out["reason_3"]])


class LeakageTests(unittest.TestCase):
    def test_label_columns_do_not_affect_alerts(self):
        rows = [_row(up_score=5, down_score=0, ret_20m_1m=0.012, dist_ma5_intraday=0.003,
                     distance_from_day_high_1m=-0.003, ret_5m_1m=0.004,
                     relative_ret_20m_to_universe=0.01, rank_ret_20m_1m=1)]
        base = compute_directional_alerts(_frame(rows))[ALERT_COLUMNS]
        leak = _frame(rows)
        leak["up2_60m"] = 999          # future label injected
        leak["up5_event"] = "UP5_SUCCESS"
        leaked = compute_directional_alerts(leak)[ALERT_COLUMNS]
        pd.testing.assert_frame_equal(base, leaked)

    def test_empty_frame_has_alert_columns(self):
        out = compute_directional_alerts(pd.DataFrame())
        for col in ALERT_COLUMNS:
            self.assertIn(col, out.columns)


class GateTests(unittest.TestCase):
    def test_up_gate_respects_each_knob(self):
        rows = [
            _row(up_score=5, down_score=0, ret_20m_1m=0.012, dist_ma5_intraday=0.003, ret_5m_1m=0.004,
                 distance_from_day_high_1m=-0.003, relative_ret_20m_to_universe=0.01, rank_ret_20m_1m=1),  # fires
            _row(up_score=3.9, down_score=0, ret_20m_1m=0.012, dist_ma5_intraday=0.003, ret_5m_1m=0.004,
                 distance_from_day_high_1m=-0.003, relative_ret_20m_to_universe=0.01, rank_ret_20m_1m=1),  # below up_threshold
            _row(up_score=5, down_score=3, ret_20m_1m=0.012, dist_ma5_intraday=0.003, ret_5m_1m=0.004,
                 distance_from_day_high_1m=-0.003, relative_ret_20m_to_universe=0.01, rank_ret_20m_1m=1),  # down>limit & margin<3
            _row(up_score=5, down_score=0, ret_20m_1m=0.012, dist_ma5_intraday=0.003, ret_5m_1m=0.004,
                 distance_from_day_high_1m=-0.003, relative_ret_20m_to_universe=0.01, rank_ret_20m_1m=9),  # rank>limit
        ]
        alerts = compute_directional_alerts(_frame(rows))
        thr = AlertThresholds(up_threshold=4.0, score_margin=3.0, down_risk_limit=2.0,
                              current_rank_limit=5)
        fired = up_alert_gate(alerts, thr).tolist()
        self.assertEqual(fired, [True, False, False, False])

    def test_fire_gate_emits_up_and_informational_types(self):
        rows = [
            _row(up_score=5, down_score=0, ret_20m_1m=0.012, dist_ma5_intraday=0.003, ret_5m_1m=0.004,
                 distance_from_day_high_1m=-0.003, relative_ret_20m_to_universe=0.01, rank_ret_20m_1m=1),  # UP fires
            _row(up_score=1, down_score=4, ret_5m_1m=-0.003, ret_20m_1m=-0.01, rank_ret_20m_1m=18,
                 relative_ret_20m_to_universe=-0.01, universe_breadth=0.3),                                # RISK fires
            _row(up_score=1.5, down_score=1.25, ret_5m_1m=0.001, ret_20m_1m=0.001, volume_z_20m_1m=2.6,
                 rank_ret_20m_1m=10, range_1m=0.007),                                                       # VOLATILITY fires
            _row(),                                                                                         # flat -> no alert
        ]
        alerts = compute_directional_alerts(_frame(rows))
        fired = alert_fire_gate(alerts, AlertThresholds())
        self.assertEqual(fired.tolist(), [True, True, True, False])
        self.assertEqual(alerts.loc[fired, "alert_type"].tolist(),
                         ["UP_CONTINUATION", "RISK_ALERT", "VOLATILITY_ALERT"])


class OnlineSelectionTests(unittest.TestCase):
    def _gated(self):
        df = pd.DataFrame([
            {"symbol": "AAA", "decision_time": "2026-01-05 10:00:00"},
            {"symbol": "BBB", "decision_time": "2026-01-05 10:05:00"},
            {"symbol": "AAA", "decision_time": "2026-01-05 10:20:00"},  # 20m < cooldown -> skip
            {"symbol": "AAA", "decision_time": "2026-01-05 10:40:00"},  # 40m >= cooldown -> fire
        ])
        return df

    def test_cooldown_and_causal_order(self):
        fired = select_online_alerts(self._gated(), daily_alert_limit=5, cooldown_minutes=30)
        self.assertEqual(
            list(zip(fired["symbol"], fired["decision_time"])),
            [("AAA", "2026-01-05 10:00:00"), ("BBB", "2026-01-05 10:05:00"),
             ("AAA", "2026-01-05 10:40:00")],
        )
        self.assertEqual(fired["alert_seq"].tolist(), [1, 2, 3])

    def test_daily_limit_truncates_in_time_order(self):
        fired = select_online_alerts(self._gated(), daily_alert_limit=2, cooldown_minutes=30)
        self.assertEqual(len(fired), 2)
        self.assertEqual(fired["symbol"].tolist(), ["AAA", "BBB"])

    def test_topk_diagnostic_can_pick_later_higher_score(self):
        # the strongest alert of the day comes late; the online stream would miss it,
        # but the (hindsight) top-k diagnostic surfaces it -> the two must differ.
        gated = pd.DataFrame([
            {"symbol": "AAA", "decision_time": "2026-01-05 10:00:00", "alert_score": 4.0,
             "direction_score": 3.0},
            {"symbol": "BBB", "decision_time": "2026-01-05 14:00:00", "alert_score": 7.0,
             "direction_score": 5.0},
        ])
        online = select_online_alerts(gated, daily_alert_limit=1, cooldown_minutes=30)
        topk = select_topk_diagnostic(gated, top_k=1)
        self.assertEqual(online["symbol"].tolist(), ["AAA"])      # earliest
        self.assertEqual(topk["symbol"].tolist(), ["BBB"])        # day's best (look-ahead)


if __name__ == "__main__":
    unittest.main(verbosity=2)
