"""Tests for the MA-breakout rule variants (src/rule_signal_ma.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

import rule_signal_ma as rma  # noqa: E402
from rule_signal import build_rule_signals  # noqa: E402

DAY = "2026-01-05"
DTS = [f"{DAY} 09:5{i}:00" for i in range(3)]   # 09:50, 09:51, 09:52


def _feat(dt, close, **over):
    row = {
        "symbol": "AAA", "source": "al_regular", "decision_time": dt, "close": close,
        "ret_5m_1m": 0.01, "ret_20m_1m": 0.01, "volume_z_20m_1m": 2.5,
        "active_symbol_count": 1, "rank_ret_5m_1m": 1, "rank_ret_20m_1m": 1,
        "rank_volume_z_20m_1m": 1, "universe_mean_ret_20m": 0.0,
        "relative_ret_20m_to_universe": 0.005, "relative_volume_z_to_universe": 0.5,
        "universe_breadth": 1.0,
    }
    row.update(over)
    return row


def _fixture():
    # closes 100 -> 105 -> 98 ; intraday MA 100,102,100 ; daily MA constant 102.
    # bar0: first of day -> no cross; bar1: up cross; bar2: down cross.
    features = pd.DataFrame([_feat(DTS[0], 100), _feat(DTS[1], 105), _feat(DTS[2], 98)])
    ma = pd.DataFrame([
        {"symbol": "AAA", "decision_time": DTS[0], "ma5_daily": 102.0, "ma5_intraday": 100.0},
        {"symbol": "AAA", "decision_time": DTS[1], "ma5_daily": 102.0, "ma5_intraday": 102.0},
        {"symbol": "AAA", "decision_time": DTS[2], "ma5_daily": 102.0, "ma5_intraday": 100.0},
    ])
    return features, ma


class CrossDetectionTests(unittest.TestCase):
    def test_intraday_cross_flags(self):
        features, ma = _fixture()
        out = rma.build_ma_rule_signals(features, ma, mode="ma_only", ma_kind="intraday")
        out = out.sort_values("decision_time").reset_index(drop=True)
        self.assertEqual(list(out["ma_breakout"]), ["none", "up", "down"])
        self.assertEqual(list(out["up_cross"]), [0, 1, 0])
        self.assertEqual(list(out["down_cross"]), [0, 0, 1])

    def test_daily_cross_flags_same_pattern(self):
        features, ma = _fixture()
        out = rma.build_ma_rule_signals(features, ma, mode="ma_only", ma_kind="daily")
        out = out.sort_values("decision_time").reset_index(drop=True)
        self.assertEqual(list(out["ma_breakout"]), ["none", "up", "down"])

    def test_nan_ma_and_first_bar_no_cross(self):
        features, ma = _fixture()
        ma2 = ma.copy()
        ma2.loc[ma2["decision_time"] == DTS[1], "ma5_intraday"] = float("nan")
        out = rma.build_ma_rule_signals(features, ma2, mode="ma_only", ma_kind="intraday")
        out = out.sort_values("decision_time").reset_index(drop=True)
        # bar1's NaN MA kills its up cross; bar0 first-of-day already has none
        self.assertEqual(list(out["up_cross"]), [0, 0, 0])


class ModeScoringTests(unittest.TestCase):
    def setUp(self):
        self.features, self.ma = _fixture()
        self.v0 = build_rule_signals(self.features)
        self.v0i = self.v0.set_index("decision_time")

    def _indexed(self, mode, ma_kind):
        out = rma.build_ma_rule_signals(self.features, self.ma, mode=mode, ma_kind=ma_kind, base=self.v0)
        return out.set_index("decision_time")

    def test_ma_only_score_from_cross(self):
        out = self._indexed("ma_only", "intraday")
        self.assertEqual(out.loc[DTS[1], "up_score"], rma.MA_ONLY_SCORE)   # up cross
        self.assertEqual(out.loc[DTS[1], "down_score"], 0.0)
        self.assertEqual(out.loc[DTS[2], "down_score"], rma.MA_ONLY_SCORE) # down cross
        self.assertEqual(out.loc[DTS[0], "up_score"], 0.0)                 # no cross
        self.assertEqual(out.loc[DTS[1], "is_up_candidate"], 1)
        self.assertEqual(out.loc[DTS[0], "is_up_candidate"], 0)

    def test_v1_is_additive(self):
        out = self._indexed("v1", "intraday")
        w = rma.DEFAULT_BREAKOUT_WEIGHT
        # only the up-cross bar gains on up_score; only the down-cross bar on down_score
        self.assertAlmostEqual(out.loc[DTS[1], "up_score"], self.v0i.loc[DTS[1], "up_score"] + w)
        self.assertAlmostEqual(out.loc[DTS[0], "up_score"], self.v0i.loc[DTS[0], "up_score"])
        self.assertAlmostEqual(out.loc[DTS[2], "down_score"], self.v0i.loc[DTS[2], "down_score"] + w)

    def test_v2_filters_candidates_without_changing_score(self):
        out = self._indexed("v2", "intraday")
        # score is exactly v0's
        for dt in DTS:
            self.assertAlmostEqual(out.loc[dt, "up_score"], self.v0i.loc[dt, "up_score"])
        # is_up_candidate kept only where v0 said up AND an up cross occurred (bar1)
        for dt in DTS:
            expected = int(self.v0i.loc[dt, "is_up_candidate"] == 1 and dt == DTS[1])
            self.assertEqual(out.loc[dt, "is_up_candidate"], expected)

    def test_output_columns(self):
        out = rma.build_ma_rule_signals(self.features, self.ma, mode="v1", ma_kind="daily", base=self.v0)
        self.assertEqual(list(out.columns), rma.V_OUTPUT_COLUMNS)
        self.assertEqual(out["rule_version"].unique().tolist(), ["v1_daily"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
