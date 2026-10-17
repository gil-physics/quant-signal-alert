"""Tests for Step 11: rule-based signal scoring."""

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


def _row(symbol="A", dt="2025-06-02 09:30:00", **overrides):
    row = {
        "symbol": symbol,
        "source": "al_regular",
        "decision_time": dt,
        "close": 100,
        "ret_5m_1m": 0.004,
        "ret_20m_1m": 0.010,
        "volume_z_20m_1m": 2.5,
        "active_symbol_count": 2,
        "rank_ret_5m_1m": 1,
        "rank_ret_20m_1m": 1,
        "rank_volume_z_20m_1m": 1,
        "universe_mean_ret_20m": 0.002,
        "relative_ret_20m_to_universe": 0.008,
        "relative_volume_z_to_universe": 1.0,
        "universe_breadth": 1.0,
    }
    row.update(overrides)
    return row


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class RuleSignalTests(unittest.TestCase):
    def test_strong_up_candidate(self):
        from rule_signal import build_rule_signals

        out = build_rule_signals(pd.DataFrame([_row()]))
        r = out.iloc[0]
        self.assertGreater(r["up_score"], r["down_score"])
        self.assertEqual(r["is_up_candidate"], 1)
        self.assertEqual(r["primary_side"], "UP")
        self.assertIn("ret20", r["reason_1"])

    def test_strong_down_candidate(self):
        from rule_signal import build_rule_signals

        out = build_rule_signals(pd.DataFrame([_row(
            ret_5m_1m=-0.004,
            ret_20m_1m=-0.011,
            rank_ret_5m_1m=2,
            rank_ret_20m_1m=2,
            universe_mean_ret_20m=-0.002,
            relative_ret_20m_to_universe=-0.009,
            universe_breadth=0.0,
        )]))
        r = out.iloc[0]
        self.assertGreater(r["down_score"], r["up_score"])
        self.assertEqual(r["is_down_candidate"], 1)
        self.assertEqual(r["primary_side"], "DOWN")
        self.assertIn("ret20", r["reason_1"])

    def test_labels_are_ignored(self):
        from rule_signal import build_rule_signals

        base = pd.DataFrame([_row()])
        with_label_zero = base.copy()
        with_label_one = base.copy()
        with_label_zero["up5_180m"] = 0
        with_label_one["up5_180m"] = 1

        a = build_rule_signals(with_label_zero)
        b = build_rule_signals(with_label_one)
        self.assertEqual(a.iloc[0]["up_score"], b.iloc[0]["up_score"])
        self.assertEqual(a.iloc[0]["down_score"], b.iloc[0]["down_score"])
        self.assertNotIn("up5_180m", a.columns)

    def test_missing_column_raises(self):
        from rule_signal import build_rule_signals

        df = pd.DataFrame([_row()]).drop(columns=["ret_20m_1m"])
        with self.assertRaises(ValueError):
            build_rule_signals(df)

    def test_nan_input_raises(self):
        from rule_signal import build_rule_signals

        df = pd.DataFrame([_row(ret_20m_1m=float("nan"))])
        with self.assertRaises(ValueError):
            build_rule_signals(df)

    def test_empty_input_has_output_columns(self):
        from rule_signal import OUTPUT_COLUMNS, build_rule_signals

        out = build_rule_signals(pd.DataFrame(columns=OUTPUT_COLUMNS))
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
