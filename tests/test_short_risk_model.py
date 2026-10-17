"""Tests for Step 15 short-risk additions: up-risk label + v1_short registry.

All additions to src/risk_model.py must be strictly additive -- the v1 baseline
(feature set, filename, MAE label math) is frozen and asserted here again.
"""

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

import risk_model as rm  # noqa: E402


def _synthetic_bars(with_high=True):
    """One symbol, one day, five 1-minute bars (bar_start 09:00..09:04).

    The 09:02 bar carries the day-high (110) so both sides of the searchsorted
    side="left" convention are observable: a decision at 09:02 must INCLUDE it
    (first forward bar) while a decision at 09:03 must EXCLUDE it (it is then the
    decision bar itself).
    """
    bars = pd.DataFrame({
        "symbol": ["TESTSYM"] * 5,
        "bar_start": [f"2026-01-05 09:0{i}:00" for i in range(5)],
        "low": [99.0, 98.0, 97.0, 96.0, 95.0],
        "close": [100.0, 101.0, 102.0, 100.0, 99.0],
    })
    if with_high:
        bars["high"] = [105.0, 103.0, 110.0, 101.0, 102.0]
    return bars


def _decision_frame():
    # decision_time = the decision (feature) bar's bar_END; close = that bar's close.
    return pd.DataFrame({
        "symbol": ["TESTSYM"] * 3,
        "date": ["2026-01-05"] * 3,
        "decision_time": [
            "2026-01-05 09:02:00",   # decision bar = 09:01; forward = 09:02..09:04
            "2026-01-05 09:03:00",   # decision bar = 09:02 (day-high); forward = 09:03..09:04
            "2026-01-05 09:05:00",   # decision bar = 09:04 (last bar); no forward bars
        ],
        "close": [101.0, 102.0, 99.0],
    })


class ShortRiskLeakageGuardTests(unittest.TestCase):
    def test_new_label_columns_never_selected_as_features(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS + ["target_up_risk", "future_MFE"])
        cols = rm.available_feature_columns(frame)
        self.assertIn("up_score", cols)
        self.assertNotIn("target_up_risk", cols)
        self.assertNotIn("future_MFE", cols)

    def test_new_label_columns_rejected_when_passed_explicitly(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS + ["target_up_risk", "future_MFE"])
        with self.assertRaises(ValueError):
            rm.available_feature_columns(frame, rm.FEATURE_COLUMNS + ["target_up_risk"])
        with self.assertRaises(ValueError):
            rm.available_feature_columns(frame, rm.FEATURE_COLUMNS + ["future_MFE"])

    def test_new_columns_registered_in_label_schema(self):
        # build_model_frame keeps TARGET_COLUMNS/FUTURE_EVAL_COLUMNS -> the new label
        # columns flow into the modelling table (existence-filtered, never features).
        self.assertIn("target_up_risk", rm.TARGET_COLUMNS)
        self.assertIn("future_MFE", rm.FUTURE_EVAL_COLUMNS)


class UpRiskLabelTests(unittest.TestCase):
    def setUp(self):
        self.out = rm.add_future_risk_targets(_decision_frame(), _synthetic_bars())

    def test_suffix_max_includes_first_forward_bar(self):
        # 09:02 decision: forward window starts AT the bar whose bar_start == 09:02
        # (side="left"), so its 110 day-high counts. side="right" would skip it.
        self.assertAlmostEqual(self.out.loc[0, "future_MFE"], 110.0 / 101.0 - 1)
        self.assertEqual(self.out.loc[0, "target_up_risk"], 1)

    def test_suffix_max_excludes_decision_bar(self):
        # 09:03 decision: the 110 high belongs to the decision bar (09:02) and must
        # NOT count -- forward max = max(101, 102) = 102 -> MFE exactly 0.
        self.assertAlmostEqual(self.out.loc[1, "future_MFE"], 102.0 / 102.0 - 1)
        self.assertEqual(self.out.loc[1, "target_up_risk"], 0)

    def test_no_forward_bars_yields_nan_mfe_and_zero_target(self):
        self.assertTrue(np.isnan(self.out.loc[2, "future_MFE"]))
        self.assertEqual(self.out.loc[2, "target_up_risk"], 0)
        # identical convention on the existing MAE path
        self.assertTrue(np.isnan(self.out.loc[2, "future_MAE"]))
        self.assertEqual(self.out.loc[2, "target_mae_risk"], 0)

    def test_up_threshold_keyword_is_respected(self):
        out = rm.add_future_risk_targets(_decision_frame(), _synthetic_bars(),
                                         up_threshold=0.10)
        self.assertEqual(out.loc[0, "target_up_risk"], 0)   # +8.9% < 10%

    def test_mae_math_untouched(self):
        # suffix-min of low over 09:02..09:04 = 95 relative to close 101.
        self.assertAlmostEqual(self.out.loc[0, "future_MAE"], 95.0 / 101.0 - 1)
        self.assertEqual(self.out.loc[0, "target_mae_risk"], 1)


class BarsWithoutHighTests(unittest.TestCase):
    def test_missing_high_yields_nan_mfe_without_exception(self):
        out = rm.add_future_risk_targets(_decision_frame(), _synthetic_bars(with_high=False))
        self.assertIn("future_MFE", out.columns)
        self.assertIn("target_up_risk", out.columns)
        self.assertTrue(out["future_MFE"].isna().all())
        self.assertTrue((out["target_up_risk"] == 0).all())

    def test_mae_path_identical_with_and_without_high(self):
        with_high = rm.add_future_risk_targets(_decision_frame(), _synthetic_bars(with_high=True))
        without_high = rm.add_future_risk_targets(_decision_frame(), _synthetic_bars(with_high=False))
        pd.testing.assert_series_equal(with_high["future_MAE"], without_high["future_MAE"])
        pd.testing.assert_series_equal(with_high["target_mae_risk"], without_high["target_mae_risk"])
        pd.testing.assert_series_equal(with_high["future_return_60m"], without_high["future_return_60m"])

    def test_empty_frame_emits_stable_schema(self):
        out = rm.add_future_risk_targets(pd.DataFrame(), _synthetic_bars())
        for col in ("future_MAE", "future_return_60m", "target_mae_risk",
                    "future_MFE", "target_up_risk"):
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 0)


class V1FreezeAndShortRegistryTests(unittest.TestCase):
    def test_v1_baseline_frozen(self):
        self.assertEqual(len(rm.FEATURE_COLUMNS), 49)
        self.assertEqual(rm.MODEL_VERSIONS["v1"]["filename"], "risk_model_lgbm.txt")
        self.assertEqual(rm.MAE_RISK_THRESHOLD, -0.015)

    def test_up_risk_threshold_mirrors_mae_threshold(self):
        self.assertEqual(rm.UP_RISK_THRESHOLD, 0.015)
        self.assertEqual(rm.UP_RISK_THRESHOLD, -rm.MAE_RISK_THRESHOLD)

    def test_v1_short_registry_mirrors_v1_except_filename(self):
        self.assertTrue(str(rm.resolve_model_path("v1_short")).endswith("short_risk_model_lgbm.txt"))
        cfg = rm.model_version_config("v1_short")
        v1 = rm.model_version_config("v1")
        self.assertEqual(cfg["features"], v1["features"])
        self.assertEqual(cfg["relative"], v1["relative"])
        self.assertEqual(cfg["requires_cross_symbol"], v1["requires_cross_symbol"])
        self.assertEqual(cfg["premarket"], v1["premarket"])
        self.assertEqual(cfg["requires_premarket"], v1["requires_premarket"])
        self.assertNotEqual(cfg["filename"], v1["filename"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
