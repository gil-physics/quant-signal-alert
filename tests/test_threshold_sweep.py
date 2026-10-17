"""Tests for Step 13 threshold sweep logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

import threshold_sweep as ts  # noqa: E402


def _signal(symbol="A", dt="2025-06-02 09:30:00", **kw):
    row = {
        "symbol": symbol,
        "source": "al_regular",
        "decision_time": dt,
        "close": 100.0,
        "ret_5m_1m": 0.01,
        "ret_20m_1m": 0.02,
        "volume_z_20m_1m": 3.0,
        "rank_ret_20m_1m": 1,
        "up_score": 5.0,
        "down_score": 0.5,
        "is_up_candidate": 1,
        "is_down_candidate": 0,
        "reason_1": "ret20",
        "reason_2": "volume",
        "reason_3": "rank",
        "up_reasons": "ret20",
        "down_reasons": "",
    }
    row.update(kw)
    return row


def _label(symbol="A", dt="2025-06-02 09:30:00", **kw):
    row = {
        "symbol": symbol,
        "source": "al_regular",
        "decision_time": dt,
        "up2_120m": 1,
        "up3_120m": 0,
        "up2_180m": 1,
        "up3_180m": 0,
        "up5_180m": 0,
        "up5_eod": 0,
        "down2_120m": 0,
        "down2_180m": 0,
        "has_full_120m": 1,
        "has_full_180m": 1,
        "up5_event": "NO_EVENT",
    }
    row.update(kw)
    return row


def _bars():
    rows = []
    closes = [100, 101, 102, 103, 106]
    for i, close in enumerate(closes):
        ts_ = pd.Timestamp("2025-06-02 09:30:00") + pd.Timedelta(minutes=i)
        rows.append(
            {
                "symbol": "A",
                "bar_start": ts_.strftime("%Y-%m-%d %H:%M:%S"),
                "open": close,
                "high": close + 1,
                "low": close,
                "close": close,
                "volume": 1000,
            }
        )
    return pd.DataFrame(rows)


class ThresholdSweepTests(unittest.TestCase):
    def test_labels_do_not_change_candidate_selection(self):
        signals = pd.DataFrame([_signal()])
        labels_a = pd.DataFrame([_label(up3_120m=0, up5_eod=0)])
        labels_b = pd.DataFrame([_label(up3_120m=1, up5_eod=1)])
        spec = ts.SweepSpec("x", up_threshold=4, down_risk_limit=1, score_margin=3, top_k=1, entry_delay=0)

        base_a, _ = ts.prepare_base_frame(signals, labels_a)
        base_b, _ = ts.prepare_base_frame(signals, labels_b)
        cand_a, _ = ts.build_candidate_pool(base_a, spec)
        cand_b, _ = ts.build_candidate_pool(base_b, spec)

        self.assertEqual(cand_a[["symbol", "decision_time"]].to_dict("records"),
                         cand_b[["symbol", "decision_time"]].to_dict("records"))

    def test_percentile_cutoff_is_fit_from_train(self):
        rows = [
            _signal("A", "2025-06-02 09:30:00", up_score=10, down_score=0, rank_ret_20m_1m=1),
            _signal("B", "2025-06-02 09:31:00", up_score=8, down_score=0, rank_ret_20m_1m=2),
            _signal("C", "2025-06-02 09:32:00", up_score=6, down_score=0, rank_ret_20m_1m=3),
        ]
        labels = [_label(r["symbol"], r["decision_time"]) for r in rows]
        base, _ = ts.prepare_base_frame(pd.DataFrame(rows), pd.DataFrame(labels))
        spec = ts.SweepSpec(
            "pct",
            up_threshold=4,
            down_risk_limit=1,
            score_margin=3,
            top_k=1,
            entry_delay=0,
            score_metric="up_score",
            top_percentile=0.5,
        )
        fitted = ts.fit_percentile_cutoffs(base, [spec])[0]
        self.assertAlmostEqual(fitted.percentile_cutoff, 8.0)

        cands, stats = ts.build_candidate_pool(base, fitted)
        self.assertEqual(cands["symbol"].tolist(), ["A", "B"])
        self.assertEqual(stats["after_percentile_filter"], 2)

    def test_run_threshold_sweep_evaluates_entry_delay(self):
        signals = pd.DataFrame([_signal(up_score=6, down_score=0)])
        labels = pd.DataFrame([_label(up3_120m=1, up3_180m=1)])
        base, base_stats = ts.prepare_base_frame(signals, labels)
        spec = ts.SweepSpec("delay3", up_threshold=4, down_risk_limit=1, score_margin=3, top_k=1, entry_delay=3)

        summary, evaluated = ts.run_threshold_sweep(
            base,
            _bars(),
            [spec],
            base_stats=base_stats,
            train_start="2025-06-02",
            train_end="2025-06-02",
            validation_start="2025-06-03",
            validation_end="2025-06-03",
        )

        train = summary[summary["train_or_validation"] == "train"].iloc[0]
        self.assertEqual(int(train["signal_count"]), 1)
        self.assertEqual(int(train["entry_delay"]), 3)
        self.assertEqual(float(train["hit_up2_rate"]), 1.0)
        self.assertEqual(float(train["up3_120m_rate"]), 1.0)
        self.assertAlmostEqual(float(evaluated.iloc[0]["entry_price"]), 103.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
