"""Tests for Step 12: daily top1/top2 backtest."""

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


def _signal(symbol="A", dt="2025-06-02 09:30:00", **kw):
    row = {
        "symbol": symbol,
        "source": "al_regular",
        "decision_time": dt,
        "close": 100,
        "ret_5m_1m": 0.01,
        "ret_20m_1m": 0.02,
        "volume_z_20m_1m": 3.0,
        "rank_ret_20m_1m": 1,
        "up_score": 5.0,
        "down_score": 0.5,
        "is_up_candidate": 1,
        "is_down_candidate": 0,
        "reason_1": "ret20>=0.8%",
        "reason_2": "volume_z>=2",
        "reason_3": "rank_ret20_top",
        "up_reasons": "ret20>=0.8%",
        "down_reasons": "",
    }
    row.update(kw)
    return row


def _label(symbol="A", dt="2025-06-02 09:30:00", **kw):
    row = {
        "symbol": symbol,
        "source": "al_regular",
        "decision_time": dt,
        "close": 100,
        "up2_eod": 1,
        "up3_eod": 0,
        "up5_eod": 0,
        "down2_eod": 0,
        "down3_eod": 0,
        "down5_eod": 0,
        "up5_event": "NO_EVENT",
    }
    row.update(kw)
    return row


def _bars():
    rows = []
    for i, close in enumerate([101, 102, 103, 104]):
        ts = pd.Timestamp("2025-06-02 09:30:00") + pd.Timedelta(minutes=i * 30)
        rows.append({
            "symbol": "A",
            "bar_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "high": close + 1,
            "low": close - 2,
            "close": close,
        })
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class DailyTopKBacktestTests(unittest.TestCase):
    def test_label_changes_do_not_affect_selection(self):
        from daily_topk_backtest import ThresholdConfig, apply_long_filter, join_signals_labels, select_daily_topk

        sig = pd.DataFrame([_signal()])
        labels_a = pd.DataFrame([_label(up2_eod=0, up5_event="UP_FAIL")])
        labels_b = pd.DataFrame([_label(up2_eod=1, up5_event="UP5_SUCCESS")])
        cfg = ThresholdConfig("x", 4, 1, 3)

        selected_a, _ = select_daily_topk(apply_long_filter(join_signals_labels(sig, labels_a), cfg)[0])
        selected_b, _ = select_daily_topk(apply_long_filter(join_signals_labels(sig, labels_b), cfg)[0])
        self.assertEqual(selected_a[["symbol", "decision_time"]].to_dict("records"),
                         selected_b[["symbol", "decision_time"]].to_dict("records"))

    def test_both_candidate_is_excluded(self):
        from daily_topk_backtest import ThresholdConfig, apply_long_filter, join_signals_labels

        sig = pd.DataFrame([_signal(is_down_candidate=1)])
        joined = join_signals_labels(sig, pd.DataFrame([_label()]))
        out, stats = apply_long_filter(joined, ThresholdConfig("x", 4, 1, 3))
        self.assertEqual(len(out), 0)
        self.assertEqual(stats["both_excluded_count"], 1)

    def test_topk_sort_and_cooldown(self):
        from daily_topk_backtest import select_daily_topk

        cands = pd.DataFrame([
            _signal("A", "2025-06-02 09:30:00", up_score=6, down_score=1, rank_ret_20m_1m=2),
            _signal("A", "2025-06-02 10:00:00", up_score=5.9, down_score=1, rank_ret_20m_1m=1),
            _signal("B", "2025-06-02 10:01:00", up_score=4, down_score=0, rank_ret_20m_1m=1),
        ])
        cands["date"] = cands["decision_time"].str.slice(0, 10)
        cands["time"] = cands["decision_time"].str.slice(11, 19)
        cands["score_margin"] = cands["up_score"] - cands["down_score"]
        cands["_decision_dt"] = pd.to_datetime(cands["decision_time"])
        selected, daily = select_daily_topk(cands, top_k=2, cooldown_minutes=60)
        self.assertEqual(selected["symbol"].tolist(), ["A", "B"])
        self.assertEqual(selected["daily_rank"].tolist(), [1, 2])
        self.assertEqual(int(daily.iloc[0]["cooldown_excluded_count"]), 1)
        self.assertEqual(int(daily.iloc[0]["after_cooldown_count"]), 2)

    def test_base_time_and_quality_filters(self):
        from daily_topk_backtest import apply_base_filters, join_signals_labels

        sig = pd.DataFrame([
            _signal("A", "2025-06-02 09:09:00"),
            _signal("A", "2025-06-02 15:00:00"),
            _signal("A", "2025-06-03 10:00:00"),
            _signal("A", "2025-06-04 10:00:00"),
            _signal("A", "2025-06-05 10:00:00"),
        ])
        lab = pd.DataFrame([_label("A", dt) for dt in sig["decision_time"]])
        joined = join_signals_labels(sig, lab)
        out, stats = apply_base_filters(
            joined,
            fail_quality_pairs={("A", "2025-06-03")},
            outlier_pairs={("A", "2025-06-04")},
        )
        self.assertEqual(out["decision_time"].tolist(), ["2025-06-05 10:00:00"])
        self.assertEqual(stats["before_min_time_excluded"], 1)
        self.assertEqual(stats["after_max_time_excluded"], 1)
        self.assertEqual(stats["after_time_filter"], 3)
        self.assertEqual(stats["fail_quality_excluded"], 1)
        self.assertEqual(stats["outlier_excluded"], 1)
        self.assertEqual(stats["after_quality_filter"], 1)

    def test_evaluate_selected_mfe_mae_and_returns(self):
        from daily_topk_backtest import evaluate_selected, join_signals_labels

        joined = join_signals_labels(pd.DataFrame([_signal()]), pd.DataFrame([_label(up2_eod=1, up5_event="UP_FAIL")]))
        joined["daily_rank"] = 1
        out = evaluate_selected(joined, _bars())
        r = out.iloc[0]
        self.assertAlmostEqual(r["mfe"], 0.05)
        self.assertAlmostEqual(r["mae"], -0.01)
        self.assertAlmostEqual(r["return_30m"], 0.01)
        self.assertAlmostEqual(r["return_60m"], 0.02)
        self.assertAlmostEqual(r["return_120m"], 0.04)
        self.assertEqual(r["hit_up2"], 1)
        self.assertEqual(r["stop_first"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
