"""Tests for Step 14 risk model helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

import risk_model as rm  # noqa: E402


class RiskModelTests(unittest.TestCase):
    def test_feature_columns_exclude_future_and_target_like_names(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS + ["target_mae_risk", "future_MAE", "up5_event"])
        cols = rm.available_feature_columns(frame)
        self.assertIn("up_score", cols)
        self.assertNotIn("target_mae_risk", cols)
        self.assertNotIn("future_MAE", cols)
        self.assertNotIn("up5_event", cols)

    def test_date_split_has_no_overlap(self):
        frame = pd.DataFrame({
            "date": ["2025-06-02", "2026-02-28", "2026-03-01", "2026-06-19", "2026-06-20"],
            "x": range(5),
        })
        train, valid = rm.split_by_date(frame)
        self.assertEqual(train["date"].tolist(), ["2025-06-02", "2026-02-28"])
        self.assertEqual(valid["date"].tolist(), ["2026-03-01", "2026-06-19"])
        self.assertTrue(set(train["date"]).isdisjoint(set(valid["date"])))

    def test_auc_and_bucket_report(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(rm.roc_auc_score_np(y, p), 1.0)
        df = pd.DataFrame({
            "target_mae_risk": y,
            "target_up_fail": [0, 0, 1, 0],
            "target_down2_60m": [0, 0, 0, 1],
            "future_MAE": [-0.002, -0.003, -0.02, -0.03],
            "future_return_60m": [0.01, 0.0, -0.01, -0.02],
            "risk_prob": p,
        })
        report = rm.bucket_report(df)
        top = report[report["risk_bucket"] == "top_10%"].iloc[0]
        bottom = report[report["risk_bucket"] == "bottom_20%"].iloc[0]
        self.assertGreater(top["mae_risk_rate"], bottom["mae_risk_rate"])

    def test_numpy_logistic_learns_order(self):
        train = pd.DataFrame({
            "up_score": [0, 0, 1, 1],
            "down_score": [0, 0, 1, 1],
            "alert_score": [0, 0, 1, 1],
            "target_mae_risk": [0, 0, 1, 1],
        })
        features = ["up_score", "down_score", "alert_score"]
        pre = rm.FeaturePreprocessor(features).fit(train)
        model = rm.NumpyLogisticRiskModel(max_iter=60).fit(pre.transform(train), train["target_mae_risk"])
        prob = model.predict_proba(pre.transform(train))[:, 1]
        self.assertLess(prob[0], prob[-1])


class RiskModelVersioningTests(unittest.TestCase):
    """v2.0: feature-set versioning coexists with a frozen v1 baseline."""

    V2_NEW = ["universe_disp_ret_20m", "universe_breadth_5m", "universe_breadth_60m",
              "ret_20m_gap_to_best", "universe_disp_volume_z"]

    def test_v1_relative_columns_frozen(self):
        # v1 must stay exactly the original 8 cross-symbol features.
        self.assertEqual(rm.RELATIVE_FEATURE_COLUMNS, [
            "active_symbol_count", "rank_ret_5m_1m", "rank_ret_20m_1m", "rank_volume_z_20m_1m",
            "universe_mean_ret_20m", "relative_ret_20m_to_universe",
            "relative_volume_z_to_universe", "universe_breadth"])
        self.assertEqual(len(rm.FEATURE_COLUMNS), 49)

    def test_v2_is_strict_superset(self):
        self.assertTrue(set(rm.RELATIVE_FEATURE_COLUMNS).issubset(rm.RELATIVE_FEATURE_COLUMNS_V2))
        self.assertEqual(set(rm.RELATIVE_FEATURE_COLUMNS_V2) - set(rm.RELATIVE_FEATURE_COLUMNS),
                         set(self.V2_NEW))
        self.assertEqual(len(rm.FEATURE_COLUMNS_V2), 54)
        self.assertTrue(set(rm.FEATURE_COLUMNS).issubset(rm.FEATURE_COLUMNS_V2))

    def test_v2_new_columns_pass_leakage_filter(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS_V2 + ["target_mae_risk", "future_MAE"])
        cols = rm.available_feature_columns(frame, rm.FEATURE_COLUMNS_V2)
        for c in self.V2_NEW:
            self.assertIn(c, cols)
        self.assertNotIn("target_mae_risk", cols)
        self.assertNotIn("future_MAE", cols)

    def test_default_feature_columns_is_v1(self):
        # available_feature_columns / build_model_frame default to v1 -> v1 reproducible.
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS_V2)
        self.assertEqual(set(rm.available_feature_columns(frame)), set(rm.FEATURE_COLUMNS))

    def test_resolve_model_path(self):
        self.assertTrue(str(rm.resolve_model_path("v1")).endswith("risk_model_lgbm.txt"))
        self.assertTrue(str(rm.resolve_model_path("v2.0")).endswith("risk_model_v2_0_lgbm.txt"))
        self.assertFalse(rm.model_version_config("v1")["requires_cross_symbol"])
        self.assertTrue(rm.model_version_config("v2.0")["requires_cross_symbol"])
        with self.assertRaises(ValueError):
            rm.resolve_model_path("v9")


class RiskModelV21VersioningTests(unittest.TestCase):
    """v2.1a / v2.1b are SINGLE-leader lead-lag variants, each v1's 8 + one leader's 2
    columns. Neither may inherit v2.0's (rejected) universe dispersion/breadth set."""

    LEADER_A = ["leader_a_ret_5m_1m", "leader_a_ret_20m_1m"]
    LEADER_B = ["leader_b_ret_5m_1m", "leader_b_ret_20m_1m"]

    def test_v21_variants_are_v1_plus_one_leader_only(self):
        self.assertEqual(rm.RELATIVE_FEATURE_COLUMNS_V2_1A,
                         rm.RELATIVE_FEATURE_COLUMNS + self.LEADER_A)
        self.assertEqual(rm.RELATIVE_FEATURE_COLUMNS_V2_1B,
                         rm.RELATIVE_FEATURE_COLUMNS + self.LEADER_B)
        self.assertEqual(len(rm.FEATURE_COLUMNS_V2_1A), 51)  # 49 v1 + 2 lead-lag
        self.assertEqual(len(rm.FEATURE_COLUMNS_V2_1B), 51)
        # the two variants differ only by which leader's columns they carry.
        self.assertEqual(set(self.LEADER_A).isdisjoint(set(self.LEADER_B)), True)

    def test_v21_variants_exclude_v2_universe_features(self):
        v2_only = set(rm.RELATIVE_FEATURE_COLUMNS_V2) - set(rm.RELATIVE_FEATURE_COLUMNS)
        for cols in (rm.RELATIVE_FEATURE_COLUMNS_V2_1A, rm.RELATIVE_FEATURE_COLUMNS_V2_1B):
            self.assertTrue(v2_only.isdisjoint(set(cols)),
                            "v2.1x must not pull in v2.0's rejected universe features")

    def test_v21_columns_pass_leakage_filter(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS_V2_1A + self.LEADER_B
                             + ["target_mae_risk", "future_MAE"])
        cols = rm.available_feature_columns(frame, rm.FEATURE_COLUMNS_V2_1A + self.LEADER_B)
        for c in self.LEADER_A + self.LEADER_B:
            self.assertIn(c, cols)
        self.assertNotIn("target_mae_risk", cols)
        self.assertNotIn("future_MAE", cols)

    def test_v21_resolve_paths_and_guard(self):
        self.assertTrue(str(rm.resolve_model_path("v2.1a")).endswith("risk_model_v2_1a_lgbm.txt"))
        self.assertTrue(str(rm.resolve_model_path("v2.1b")).endswith("risk_model_v2_1b_lgbm.txt"))
        self.assertTrue(rm.model_version_config("v2.1a")["requires_cross_symbol"])
        self.assertTrue(rm.model_version_config("v2.1b")["requires_cross_symbol"])

    def test_v1_still_frozen_after_v21(self):
        # adding v2.1x must not have perturbed the v1 baseline set.
        self.assertEqual(len(rm.FEATURE_COLUMNS), 49)
        self.assertEqual(len(rm.RELATIVE_FEATURE_COLUMNS), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
