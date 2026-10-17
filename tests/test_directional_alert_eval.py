"""Tests for Step 13_6 directional-alert evaluation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from directional_alert_eval import (  # noqa: E402
    ReportConfig,
    build_alert_catalog,
    evaluate_alerts,
    select_for_evaluation,
    summarize_alerts,
)


def _signal(symbol="AAA", dt="2026-01-05 10:00:00", **kw):
    row = {
        "symbol": symbol,
        "source": "al_regular",
        "decision_time": dt,
        "close": 100.0,
        "ret_5m_1m": 0.004,
        "ret_20m_1m": 0.012,
        "volume_z_20m_1m": 1.5,
        "active_symbol_count": 3,
        "rank_ret_5m_1m": 1,
        "rank_ret_20m_1m": 1,
        "rank_volume_z_20m_1m": 1,
        "universe_mean_ret_20m": 0.002,
        "relative_ret_20m_to_universe": 0.010,
        "relative_volume_z_to_universe": 0.5,
        "universe_breadth": 0.7,
        "up_score": 5.0,
        "down_score": 0.0,
        "is_up_candidate": 1,
        "is_down_candidate": 0,
        "primary_side": "UP",
        "reason_1": "up",
        "reason_2": "",
        "reason_3": "",
        "up_reasons": "up",
        "down_reasons": "",
    }
    row.update(kw)
    return row


def _micro(symbol="AAA", dt="2026-01-05 10:00:00", **kw):
    row = {
        "symbol": symbol,
        "decision_time": dt,
        "upper_shadow_1m": 0.0,
        "vwap_distance_1m": 0.002,
        "distance_from_day_high_1m": -0.003,
        "return_from_open_1m": 0.015,
        "range_1m": 0.003,
        "dist_ma5_intraday": 0.004,
    }
    row.update(kw)
    return row


def _bars(symbol="AAA"):
    rows = []
    prices = [100, 101, 102, 103, 104, 105]
    for i, close in enumerate(prices):
        ts = pd.Timestamp("2026-01-05 10:00:00") + pd.Timedelta(minutes=i)
        rows.append({
            "symbol": symbol,
            "bar_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "high": close + 1,
            "low": close - 0.5,
            "close": close,
        })
    return pd.DataFrame(rows)


class DirectionalAlertEvalTests(unittest.TestCase):
    def test_alert_catalog_does_not_take_labels_and_excludes_both(self):
        rs = pd.DataFrame([
            _signal(),
            _signal("BBB", "2026-01-05 10:01:00", is_down_candidate=1),
        ])
        micro = pd.DataFrame([
            _micro(),
            _micro("BBB", "2026-01-05 10:01:00"),
        ])
        out, stats = build_alert_catalog(rs, micro, created_at="now")
        self.assertEqual(out["symbol"].tolist(), ["AAA"])
        self.assertEqual(stats["both_excluded_count"], 1)
        self.assertIn("created_at", out.columns)

    def test_online_and_topk_modes_are_separated(self):
        rows = [
            _signal("AAA", "2026-01-05 10:00:00"),
            _signal("BBB", "2026-01-05 10:01:00", up_score=6.0),
        ]
        micro = [_micro("AAA", "2026-01-05 10:00:00"), _micro("BBB", "2026-01-05 10:01:00")]
        alerts, _ = build_alert_catalog(pd.DataFrame(rows), pd.DataFrame(micro))
        selected = select_for_evaluation(alerts, config=ReportConfig(daily_alert_limit=1, top_k=1))
        self.assertEqual(set(selected["selection_mode"]), {"online", "daily_topk"})
        online = selected[selected["selection_mode"] == "online"]
        topk = selected[selected["selection_mode"] == "daily_topk"]
        self.assertEqual(len(online), 1)
        self.assertEqual(len(topk), 1)

    def test_evaluate_alerts_uses_future_bars_and_labels_eval_only(self):
        alerts, _ = build_alert_catalog(pd.DataFrame([_signal()]), pd.DataFrame([_micro()]))
        selected = select_for_evaluation(alerts, config=ReportConfig(daily_alert_limit=1, top_k=0))
        labels = pd.DataFrame([{
            "symbol": "AAA",
            "decision_time": "2026-01-05 10:00:00",
            "up2_60m": 1,
            "up2_120m": 1,
            "up3_120m": 1,
            "down2_60m": 0,
            "up5_event": "UP_FAIL",
        }])
        ev = evaluate_alerts(selected, _bars(), labels, entry_delay_minutes=0)
        self.assertEqual(len(ev), 1)
        self.assertGreater(ev.iloc[0]["future_return_30m"], 0)
        self.assertEqual(int(ev.iloc[0]["direction_hit_30m"]), 1)
        self.assertEqual(int(ev.iloc[0]["UP_FAIL"]), 1)
        self.assertGreater(ev.iloc[0]["MFE"], 0.02)

    def test_summary_rollup(self):
        df = pd.DataFrame([
            {"selection_mode": "online", "alert_grade": "A", "symbol": "AAA",
             "future_return_30m": 0.01, "future_return_60m": 0.02, "future_return_120m": 0.03,
             "future_return_eod": 0.04, "direction_hit_30m": 1, "direction_hit_60m": 1,
             "direction_hit_120m": 1, "MFE": 0.05, "MAE": -0.01, "hit_up2_60m": 1,
             "hit_up2_120m": 1, "hit_up3_120m": 1, "down2_60m": 0, "UP_FAIL": 0,
             "stop_first": 0},
            {"selection_mode": "online", "alert_grade": "A", "symbol": "BBB",
             "future_return_30m": -0.01, "future_return_60m": -0.02, "future_return_120m": -0.03,
             "future_return_eod": -0.04, "direction_hit_30m": 0, "direction_hit_60m": 0,
             "direction_hit_120m": 0, "MFE": 0.01, "MAE": -0.03, "hit_up2_60m": 0,
             "hit_up2_120m": 0, "hit_up3_120m": 0, "down2_60m": 1, "UP_FAIL": 1,
             "stop_first": 1},
        ])
        out = summarize_alerts(df, by=["selection_mode", "alert_grade"])
        self.assertEqual(int(out.iloc[0]["n"]), 2)
        self.assertAlmostEqual(float(out.iloc[0]["direction_hit_60m"]), 0.5)
        self.assertAlmostEqual(float(out.iloc[0]["return_60m"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
