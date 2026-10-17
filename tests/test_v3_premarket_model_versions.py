"""Tests for v3.0 premarket model versioning, feature attachment, and no-order safety.

v3.0a/v3.0b coexist with a frozen v1 baseline. Imports from src only (SRC_DIR on path);
the build/gate scripts are checked by static file scan, never imported (the scripts dir
is kept off the global sys.path -- see test_premarket_features.py).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

import risk_model as rm  # noqa: E402


class V3VersionRegistryTests(unittest.TestCase):
    V3 = ["v3_0a_premarket_000660", "v3_0b_premarket_universe"]

    def test_resolver_recognizes_v3_variants(self):
        # case 11
        self.assertTrue(str(rm.resolve_model_path("v3_0a_premarket_000660"))
                        .endswith("risk_model_v3_0a_premarket_000660_lgbm.txt"))
        self.assertTrue(str(rm.resolve_model_path("v3_0b_premarket_universe"))
                        .endswith("risk_model_v3_0b_premarket_universe_lgbm.txt"))
        for v in self.V3:
            cfg = rm.model_version_config(v)
            self.assertTrue(cfg["requires_cross_symbol"], "v3.0 builds on v1.1 real cross-symbol")
            self.assertTrue(cfg["requires_premarket"])
            self.assertTrue(len(cfg["premarket"]) > 0)

    def test_v3a_is_target_only_v3b_is_superset(self):
        a = rm.PREMARKET_COLUMNS_V3_0A
        b = rm.PREMARKET_COLUMNS_V3_0B
        self.assertEqual(a, rm.PREMARKET_TARGET_COLUMNS)
        self.assertTrue(set(a).issubset(set(b)))
        self.assertEqual(set(b) - set(a), set(rm.PREMARKET_UNIVERSE_COLUMNS))
        # feature sets = v1 baseline + premarket subset
        self.assertEqual(rm.FEATURE_COLUMNS_V3_0A, rm.FEATURE_COLUMNS + a)
        self.assertEqual(rm.FEATURE_COLUMNS_V3_0B, rm.FEATURE_COLUMNS + b)

    def test_v3_relative_columns_are_v1_baseline(self):
        # v3.0 must NOT inherit v2.0's rejected universe set; it rides the v1 8.
        for v in self.V3:
            self.assertEqual(rm.model_version_config(v)["relative"], rm.RELATIVE_FEATURE_COLUMNS)

    def test_premarket_columns_pass_leakage_filter(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS_V3_0B + ["target_mae_risk", "future_MAE"])
        cols = rm.available_feature_columns(frame, rm.FEATURE_COLUMNS_V3_0B)
        for c in rm.PREMARKET_COLUMNS_V3_0B:
            self.assertIn(c, cols)
        self.assertNotIn("target_mae_risk", cols)
        self.assertNotIn("future_MAE", cols)

    def test_v1_and_v2_artifacts_untouched(self):
        # case 12: registering v3 must not perturb v1/v2.x.
        self.assertEqual(len(rm.RELATIVE_FEATURE_COLUMNS), 8)
        self.assertEqual(len(rm.FEATURE_COLUMNS), 49)
        self.assertTrue(str(rm.resolve_model_path("v1")).endswith("risk_model_lgbm.txt"))
        self.assertFalse(rm.model_version_config("v1")["requires_premarket"])
        self.assertFalse(rm.model_version_config("v2.0")["requires_premarket"])
        # v3 feature names never collide with the forbidden future/target token set
        for c in rm.PREMARKET_COLUMNS_V3_0B:
            self.assertFalse(any(tok in c.lower() for tok in rm.FORBIDDEN_FEATURE_TOKENS), c)


class MergePremarketTests(unittest.TestCase):
    """case 9: per-day premarket features join onto regular-session rows by date."""

    def _frame(self):
        return pd.DataFrame({
            "symbol": ["000660", "000660", "005930"],
            "decision_time": ["2026-01-06 09:31:00", "2026-01-06 10:15:00", "2026-01-07 09:31:00"],
            "date": ["2026-01-06", "2026-01-06", "2026-01-07"],
        })

    def _premarket(self):
        return pd.DataFrame({
            "trade_date": ["2026-01-06", "2026-01-07"],
            "pm_000660_ret": [0.01, -0.02],
            "pm_universe_breadth_pos": [0.8, 0.3],
        })

    def test_joins_same_day_broadcast(self):
        out = rm.merge_premarket_features(self._frame(), self._premarket(),
                                          ["pm_000660_ret", "pm_universe_breadth_pos"])
        self.assertEqual(len(out), 3)
        # both 2026-01-06 rows (different symbols/times) get the same day's premarket value
        d06 = out[out["date"] == "2026-01-06"]
        self.assertTrue((d06["pm_000660_ret"] == 0.01).all())
        self.assertAlmostEqual(out[out["date"] == "2026-01-07"]["pm_000660_ret"].iloc[0], -0.02)

    def test_noop_when_no_premarket(self):
        frame = self._frame()
        self.assertIs(rm.merge_premarket_features(frame, None, []), frame)
        self.assertIs(rm.merge_premarket_features(frame, self._premarket(), []), frame)
        out = rm.merge_premarket_features(frame, pd.DataFrame(), ["pm_000660_ret"])
        self.assertEqual(list(out.columns), list(frame.columns))

    def test_missing_premarket_day_is_nan(self):
        frame = pd.DataFrame({"symbol": ["000660"], "decision_time": ["2026-02-02 09:31:00"],
                              "date": ["2026-02-02"]})
        out = rm.merge_premarket_features(frame, self._premarket(), ["pm_000660_ret"])
        self.assertTrue(out["pm_000660_ret"].isna().all())


class NoOrderApiTests(unittest.TestCase):
    """case 13: the v3.0 feature/training/eval code never places orders."""

    FILES = [
        SRC_DIR / "premarket_features.py",
        SCRIPTS_DIR / "build_premarket_features.py",
        SCRIPTS_DIR / "evaluate_v3_premarket_gate.py",
    ]
    # call-form tokens (with "(") so that prose like "no SendOrder" in a docstring is fine
    # but an actual order-API invocation is caught.
    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(", ".SendOrder")

    def test_no_order_api_calls(self):
        for path in self.FILES:
            text = path.read_text(encoding="utf-8")
            for token in self.FORBIDDEN:
                self.assertNotIn(token, text, f"{path.name} must not invoke {token}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
