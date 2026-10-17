"""Tests for Step 17 fixed-horizon (60m) MAE/MFE labels.

The legacy rest-of-day label math is FROZEN and pinned by tests/test_short_risk_model.py
(which stays byte-unmodified); everything here covers the ADDITIVE horizon path:
window boundaries (half-open [decision, decision+60m)), truncation at the data end,
NaN PROPAGATION on the new targets (deliberately unlike the legacy NaN->0), the
default-path byte-identity regression, the minutes_since_open clock feature, and a
cross-module convention parity check against labeling._label_one (the labels table's
down2_60m uses the identical window).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import labeling  # noqa: E402
import risk_model as rm  # noqa: E402
import tcn_model as tm  # noqa: E402  (torch-free import; lazy-torch contract)


def _bars_71(with_high=True):
    """One symbol, one day, 71 one-minute bars (bar_start 09:00..10:10).

    Poison layout (base low/high = 100):
      i=4  (bar 09:04, the DECISION bar for a 09:05 decision) -> low 10 / high 500
      i=5  (bar 09:05, the FIRST forward bar)                 -> low 85 / high 115
      i=65 (bar 10:05, starts EXACTLY at decision+60m)        -> low 50 / high 200
    So a 09:05 decision's 60m window is bars 5..64: MAE_60m = 85/100-1 = -0.15
    (proves first-forward inclusion, decision-bar exclusion, and the half-open
    +60m boundary in one number) while rest-of-day MAE = 50/100-1 = -0.50.
    """
    n = 71
    lows = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows[4], highs[4] = 10.0, 500.0
    lows[5], highs[5] = 85.0, 115.0
    lows[65], highs[65] = 50.0, 200.0
    bars = pd.DataFrame({
        "symbol": ["TESTSYM"] * n,
        "bar_start": [f"2026-01-05 {9 + i // 60:02d}:{i % 60:02d}:00" for i in range(n)],
        "low": lows,
        "close": np.full(n, 100.0),
    })
    if with_high:
        bars["high"] = highs
    return bars


def _decisions(times, close=100.0):
    return pd.DataFrame({
        "symbol": ["TESTSYM"] * len(times),
        "date": ["2026-01-05"] * len(times),
        "decision_time": [f"2026-01-05 {t}" for t in times],
        "close": [close] * len(times),
    })


def _gap_bars():
    """09:00..09:05 then a >60-minute gap, then 11:00..11:05 (low 80 at 11:00)."""
    starts = [f"2026-01-05 09:0{i}:00" for i in range(6)] + \
             [f"2026-01-05 11:0{i}:00" for i in range(6)]
    lows = [100.0] * 6 + [80.0] + [100.0] * 5
    return pd.DataFrame({"symbol": ["TESTSYM"] * 12, "bar_start": starts,
                         "low": lows, "high": [100.0] * 12, "close": [100.0] * 12})


class HorizonWindowBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.out = rm.add_future_risk_targets(
            _decisions(["09:05:00", "10:00:00", "10:11:00"]), _bars_71(),
            horizon_minutes=60)

    def test_window_boundaries_in_one_number(self):
        # bars 5..64 only: includes first forward bar (85), excludes the decision bar
        # (10) and the bar starting exactly at decision+60m (50).
        self.assertAlmostEqual(self.out.loc[0, "future_MAE_60m"], 85.0 / 100.0 - 1)
        self.assertAlmostEqual(self.out.loc[0, "future_MFE_60m"], 115.0 / 100.0 - 1)
        # rest-of-day picks the 10:05 bar's 50/200 -- proves 60m != EOD
        self.assertAlmostEqual(self.out.loc[0, "future_MAE"], 50.0 / 100.0 - 1)
        self.assertAlmostEqual(self.out.loc[0, "future_MFE"], 200.0 / 100.0 - 1)

    def test_truncation_at_data_end_equals_rest_of_day(self):
        # 10:00 decision: [10:00, 11:00) truncated at the last bar -> identical windows.
        self.assertEqual(self.out.loc[1, "future_MAE_60m"], self.out.loc[1, "future_MAE"])
        self.assertEqual(self.out.loc[1, "future_MFE_60m"], self.out.loc[1, "future_MFE"])

    def test_subset_property_where_both_valid(self):
        both = self.out[self.out["future_MAE"].notna() & self.out["future_MAE_60m"].notna()]
        self.assertTrue((both["future_MAE_60m"] >= both["future_MAE"]).all())
        self.assertTrue((both["future_MFE_60m"] <= both["future_MFE"]).all())

    def test_validity_identity_with_future_return_60m(self):
        # same [start, end] mask by construction
        pd.testing.assert_series_equal(self.out["future_MAE_60m"].isna(),
                                       self.out["future_return_60m"].isna(),
                                       check_names=False)

    def test_no_forward_bars_nan_propagation_asymmetry(self):
        # 10:11 decision (past the last bar): legacy targets are 0, horizon targets NaN.
        self.assertTrue(np.isnan(self.out.loc[2, "future_MAE_60m"]))
        self.assertTrue(np.isnan(self.out.loc[2, "target_mae_risk_60m"]))
        self.assertTrue(np.isnan(self.out.loc[2, "target_up_risk_60m"]))
        self.assertEqual(self.out.loc[2, "target_mae_risk"], 0)
        self.assertEqual(self.out.loc[2, "target_up_risk"], 0)

    def test_targets_with_default_thresholds(self):
        # -0.15 <= -0.015 and +0.15 >= +0.015 -> 1.0 (float column)
        self.assertEqual(self.out.loc[0, "target_mae_risk_60m"], 1.0)
        self.assertEqual(self.out.loc[0, "target_up_risk_60m"], 1.0)
        self.assertEqual(str(self.out["target_mae_risk_60m"].dtype), "float64")


class HorizonGapAndNanTests(unittest.TestCase):
    def test_gap_over_60m_yields_nan_horizon_but_valid_eod(self):
        # 09:06 decision: no bar_start in [09:06, 10:06) (gap), next bar at 11:00.
        out = rm.add_future_risk_targets(_decisions(["09:06:00"]), _gap_bars(),
                                         horizon_minutes=60)
        self.assertTrue(np.isnan(out.loc[0, "future_MAE_60m"]))
        self.assertTrue(np.isnan(out.loc[0, "target_mae_risk_60m"]))
        self.assertAlmostEqual(out.loc[0, "future_MAE"], 80.0 / 100.0 - 1)   # EOD sees 11:00+
        self.assertEqual(out.loc[0, "target_mae_risk"], 1)                    # legacy convention
        self.assertTrue(np.isnan(out.loc[0, "future_return_60m"]))            # identity holds

    def test_horizon_thresholds_kwargs(self):
        out = rm.add_future_risk_targets(_decisions(["09:05:00"]), _bars_71(),
                                         horizon_minutes=60,
                                         horizon_mae_threshold=-0.10,
                                         horizon_up_threshold=0.20)
        self.assertEqual(out.loc[0, "target_mae_risk_60m"], 1.0)   # -0.15 <= -0.10
        self.assertEqual(out.loc[0, "target_up_risk_60m"], 0.0)    # +0.15 <  +0.20
        out2 = rm.add_future_risk_targets(_decisions(["09:05:00"]), _bars_71(),
                                          horizon_minutes=60,
                                          horizon_mae_threshold=-0.20)
        self.assertEqual(out2.loc[0, "target_mae_risk_60m"], 0.0)
        # legacy targets never react to the horizon kwargs
        self.assertEqual(out2.loc[0, "target_mae_risk"], 1)

    def test_bars_without_high(self):
        out = rm.add_future_risk_targets(_decisions(["09:05:00"]), _bars_71(with_high=False),
                                         horizon_minutes=60)
        self.assertTrue(np.isnan(out.loc[0, "future_MFE_60m"]))
        self.assertTrue(np.isnan(out.loc[0, "target_up_risk_60m"]))   # NaN, not 0
        self.assertEqual(out.loc[0, "target_up_risk"], 0)             # legacy stays 0
        self.assertAlmostEqual(out.loc[0, "future_MAE_60m"], 85.0 / 100.0 - 1)

    def test_empty_frame_schema(self):
        out = rm.add_future_risk_targets(pd.DataFrame(), _bars_71(), horizon_minutes=60)
        for col in ("future_MAE_60m", "future_MFE_60m",
                    "target_mae_risk_60m", "target_up_risk_60m"):
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 0)


class DefaultPathRegressionTests(unittest.TestCase):
    def test_default_path_is_byte_identical_and_emits_no_horizon_columns(self):
        decisions = _decisions(["09:05:00", "10:00:00", "10:11:00"])
        default = rm.add_future_risk_targets(decisions, _bars_71())
        with_h = rm.add_future_risk_targets(decisions, _bars_71(), horizon_minutes=60)
        for col in ("future_MAE_60m", "future_MFE_60m",
                    "target_mae_risk_60m", "target_up_risk_60m"):
            self.assertNotIn(col, default.columns)   # future_return_60m stays, of course
        pd.testing.assert_frame_equal(default, with_h[default.columns])

    def test_constant_and_registry_membership(self):
        self.assertEqual(rm.HORIZON_60M_MAIN_CUTOFF, "14:30:00")
        for c in ("target_mae_risk_60m", "target_up_risk_60m"):
            self.assertIn(c, rm.TARGET_COLUMNS)
        for c in ("future_MAE_60m", "future_MFE_60m"):
            self.assertIn(c, rm.FUTURE_EVAL_COLUMNS)

    def test_new_columns_rejected_as_features(self):
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS
                             + ["target_mae_risk_60m", "future_MAE_60m"])
        cols = rm.available_feature_columns(frame)
        self.assertNotIn("target_mae_risk_60m", cols)
        self.assertNotIn("future_MAE_60m", cols)
        with self.assertRaises(ValueError):
            rm.available_feature_columns(frame, rm.FEATURE_COLUMNS + ["future_MAE_60m"])


class BuildModelFrameHorizonTests(unittest.TestCase):
    def _inputs(self, times):
        alerts = pd.DataFrame({
            "symbol": ["TESTSYM"] * len(times),
            "decision_time": [f"2026-01-05 {t}" for t in times],
            "close": [100.0] * len(times),
            "source": ["al_regular"] * len(times),
        })
        empty = pd.DataFrame()
        labels = pd.DataFrame({
            "symbol": alerts["symbol"], "decision_time": alerts["decision_time"],
            "down2_60m": [0] * len(times), "up5_event": ["NO_EVENT"] * len(times),
        })
        return alerts, empty, empty, labels, _bars_71()

    def test_horizon_on_emits_labels_and_clock_feature(self):
        frame, _cols = rm.build_model_frame(*self._inputs(["09:05:00", "10:30:00"]),
                                            horizon_minutes=60)
        for col in ("future_MAE_60m", "target_mae_risk_60m", "minutes_since_open"):
            self.assertIn(col, frame.columns)
        # decision_time is the confirmed bar's bar_end -> 10:30 => 90 minutes after open
        row = frame[frame["decision_time"] == "2026-01-05 10:30:00"].iloc[0]
        self.assertEqual(row["minutes_since_open"], 90)
        self.assertEqual(frame[frame["decision_time"] == "2026-01-05 09:05:00"]
                         .iloc[0]["minutes_since_open"], 5)

    def test_horizon_off_is_unchanged(self):
        frame, _cols = rm.build_model_frame(*self._inputs(["09:05:00"]))
        for col in ("future_MAE_60m", "target_mae_risk_60m", "minutes_since_open"):
            self.assertNotIn(col, frame.columns)


class LabelingConventionParityTests(unittest.TestCase):
    """The labels table's down2_60m (labeling.py) uses the same half-open 60m window;
    P(future_MAE_60m <= -2%) must agree row-for-row on shared synthetic bars."""

    def _label_one_down2(self, bars, decision_time, close_t):
        minute = labeling._minute_of_day
        bs = np.asarray([minute(t) for t in bars["bar_start"]])
        hi = bars["high"].to_numpy(dtype=float)
        lo = bars["low"].to_numpy(dtype=float)
        be_last = bs[-1] + 1
        row = labeling._label_one("TESTSYM", decision_time, close_t,
                                  minute(decision_time), bs, hi, lo, be_last)
        return row["down2_60m"]

    def test_down2_parity_positive_and_negative(self):
        deep = _bars_71()                                  # window min -15% -> down2 = 1
        shallow = _bars_71()
        for col in ("low", "high"):
            shallow[col] = 100.0
        shallow.loc[10, "low"] = 99.5                      # -0.5% only -> down2 = 0
        for bars, expected in ((deep, 1), (shallow, 0)):
            out = rm.add_future_risk_targets(_decisions(["09:05:00"]), bars,
                                             horizon_minutes=60)
            ours = int(out.loc[0, "future_MAE_60m"] <= -0.02)
            theirs = self._label_one_down2(bars, "2026-01-05 09:05:00", 100.0)
            self.assertEqual(ours, expected)
            self.assertEqual(theirs, expected)


class SurveyScriptScanTests(unittest.TestCase):
    """Static scan (scripts stay off sys.path, house convention)."""

    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(", ".SendOrder")

    def test_survey_script_research_only_and_no_orders(self):
        path = SCRIPTS_DIR / "survey_60m_label_thresholds.py"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Research / evaluation only", text)
        for token in self.FORBIDDEN:
            self.assertNotIn(token, text)
        self.assertNotIn("import torch", text)


# --------------------------------------------------------------------------- #
# Phase B: user-decided constants, fold registry, v1_60m / v4_tcn_60m lineages
# --------------------------------------------------------------------------- #
class Step17ConstantsTests(unittest.TestCase):
    def test_60m_thresholds_pinned_and_mirrored(self):
        # USER-decided 2026-07-12 after the survey: keep +-1.5%.
        self.assertEqual(rm.MAE_RISK_THRESHOLD_60M, -0.015)
        self.assertEqual(rm.UP_RISK_THRESHOLD_60M, 0.015)
        self.assertEqual(rm.UP_RISK_THRESHOLD_60M, -rm.MAE_RISK_THRESHOLD_60M)
        # legacy constants untouched (frozen lineage)
        self.assertEqual(rm.MAE_RISK_THRESHOLD, -0.015)
        self.assertEqual(rm.UP_RISK_THRESHOLD, 0.015)

    def test_horizon_thresholds_decoupled_from_legacy_kwargs(self):
        # The 60m fallback is the pinned constant, NEVER the legacy kwarg: a caller
        # customizing the legacy threshold must not silently move the 60m label.
        out = rm.add_future_risk_targets(_decisions(["09:05:00"]), _bars_71(),
                                         horizon_minutes=60,
                                         mae_threshold=-0.6, up_threshold=1.5)
        self.assertEqual(out.loc[0, "target_mae_risk"], 0)       # legacy reacts (-0.5 > -0.6)
        self.assertEqual(out.loc[0, "target_up_risk"], 0)        # legacy reacts (1.0 < 1.5)
        self.assertEqual(out.loc[0, "target_mae_risk_60m"], 1.0)  # pinned -0.015, not -0.6
        self.assertEqual(out.loc[0, "target_up_risk_60m"], 1.0)   # pinned +0.015, not +1.5

    def test_cutoff_boundary_inclusive_string_compare(self):
        # zero-padded HH:MM:SS strings order lexicographically; 14:30:00 itself is MAIN.
        s = pd.Series(["14:29:59", "14:30:00", "14:30:01", "14:31:00"])
        self.assertEqual(list(s <= rm.HORIZON_60M_MAIN_CUTOFF), [True, True, False, False])


class WalkForwardFoldTests(unittest.TestCase):
    def test_folds_are_expanding_monthly_and_user_pinned(self):
        folds = rm.WALK_FORWARD_FOLDS_60M
        self.assertEqual(len(folds), 4)
        months = []
        for i, f in enumerate(folds):
            self.assertIsInstance(f, rm.SplitConfig)
            self.assertEqual(f.train_start, "2025-06-02")       # user: data starts here
            self.assertLess(f.train_end, f.validation_start)    # no train/val overlap
            self.assertLess(f.validation_start, f.validation_end)
            self.assertEqual(f.validation_start[:7], f.validation_end[:7])  # one month
            months.append(f.validation_start[:7])
            if i:
                prev = folds[i - 1]
                self.assertGreater(f.train_end, prev.train_end)             # expanding
                self.assertEqual(f.train_end[:7], prev.validation_end[:7])  # by the val month
        self.assertEqual(months, ["2026-03", "2026-04", "2026-05", "2026-06"])

    def test_walk_forward_fold_accessor(self):
        self.assertEqual(rm.walk_forward_fold(1), rm.WALK_FORWARD_FOLDS_60M[0])
        self.assertEqual(rm.walk_forward_fold(4), rm.WALK_FORWARD_FOLDS_60M[3])
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                rm.walk_forward_fold(bad)


class Step17RegistryTests(unittest.TestCase):
    def test_feature_columns_60m(self):
        self.assertEqual(list(rm.FEATURE_COLUMNS_60M),
                         list(rm.FEATURE_COLUMNS) + ["minutes_since_open"])
        self.assertEqual(len(rm.FEATURE_COLUMNS_60M), 50)
        frame = pd.DataFrame(columns=rm.FEATURE_COLUMNS_60M)
        cols = rm.available_feature_columns(frame, rm.FEATURE_COLUMNS_60M)
        self.assertIn("minutes_since_open", cols)   # passes the leakage-token guard

    def test_v1_60m_registered_and_baselines_frozen(self):
        cfg = rm.MODEL_VERSIONS["v1_60m"]
        self.assertEqual(cfg["features"], rm.FEATURE_COLUMNS_60M)
        self.assertEqual(cfg["relative"], rm.RELATIVE_FEATURE_COLUMNS)
        self.assertEqual(cfg["filename"], "risk_model_v1_60m_lgbm.txt")
        self.assertFalse(cfg["requires_cross_symbol"])
        self.assertFalse(cfg["requires_premarket"])
        self.assertNotIn("v1_short_60m", rm.MODEL_VERSIONS)   # one lineage at a time
        self.assertEqual(rm.MODEL_VERSIONS["v1"]["filename"], "risk_model_lgbm.txt")
        self.assertEqual(rm.MODEL_VERSIONS["v1_short"]["filename"], "short_risk_model_lgbm.txt")
        self.assertEqual(len(rm.FEATURE_COLUMNS), 49)

    def test_v4_tcn_60m_registry_and_paths(self):
        cfg = tm.tcn_version_config("v4_tcn_60m")
        self.assertEqual(cfg["target"], "target_mae_risk_60m")
        self.assertEqual(tuple(cfg["seeds"]), (42,))          # USER-decided single seed
        self.assertEqual(cfg["horizon_minutes"], 60)
        self.assertTrue(cfg["folds"])
        # same architecture inputs as v4_tcn (tod channel is a SequenceConfig default
        # the trainer keeps unless --no-tod)
        self.assertEqual(cfg["seq_features"], tm.TCN_VERSIONS["v4_tcn"]["seq_features"])
        self.assertEqual(cfg["static_features"], tm.TCN_VERSIONS["v4_tcn"]["static_features"])
        self.assertEqual(cfg["seq_len"], tm.SEQ_LEN)
        self.assertNotIn("v4_tcn_60m", rm.MODEL_VERSIONS)     # stays out of the LGBM registry
        # the Step 16 protocol is untouched
        self.assertEqual(tm.TCN_VERSIONS["v4_tcn"]["seeds"], tm.SUBMISSION_SEEDS)
        self.assertNotIn("folds", tm.TCN_VERSIONS["v4_tcn"])
        # fold-aware artifact paths
        p = tm.resolve_tcn_model_path("v4_tcn_60m", 42, fold=3)
        self.assertTrue(str(p).endswith("risk_model_v4_tcn_60m_fold3_seed42.pt"))
        with self.assertRaises(ValueError):
            tm.resolve_tcn_model_path("v4_tcn_60m", 42)       # fold is required
        self.assertTrue(str(tm.resolve_tcn_model_path("v4_tcn", 42))
                        .endswith("risk_model_v4_tcn_seed42.pt"))  # legacy path unchanged


class ReportMaeColKwargTests(unittest.TestCase):
    def _frame(self, n=20):
        return pd.DataFrame({
            "risk_prob": np.linspace(0.0, 1.0, n),
            "target_mae_risk": (np.arange(n) % 2),
            "target_mae_risk_60m": (np.arange(n) % 2).astype(float),
            "target_up_fail": np.zeros(n, dtype=int),
            "target_down2_60m": np.zeros(n, dtype=int),
            "future_MAE": np.linspace(-0.020, -0.001, n),
            "future_MAE_60m": np.linspace(-0.010, -0.0005, n),
            "future_return_60m": np.zeros(n),
            "alert_grade": ["A", "RISK"] * (n // 2),
        })

    def test_bucket_report_mae_col(self):
        f = self._frame()
        default = rm.bucket_report(f, target_col="target_mae_risk")
        explicit = rm.bucket_report(f, target_col="target_mae_risk", mae_col="future_MAE")
        pd.testing.assert_frame_equal(default, explicit)     # default path unchanged
        sixty = rm.bucket_report(f, target_col="target_mae_risk_60m",
                                 mae_col="future_MAE_60m")
        self.assertNotEqual(float(default.loc[0, "avg_MAE"]), float(sixty.loc[0, "avg_MAE"]))
        top10 = rm.top_fraction_mask(f["risk_prob"].to_numpy(), 0.10)
        exp = float(f.loc[top10, "future_MAE_60m"].mean())
        self.assertAlmostEqual(float(sixty.loc[0, "avg_MAE"]), exp, places=12)

    def test_existing_risk_grade_report_mae_col(self):
        f = self._frame()
        default = rm.existing_risk_grade_report(f, target_col="target_mae_risk")
        sixty = rm.existing_risk_grade_report(f, target_col="target_mae_risk_60m",
                                              mae_col="future_MAE_60m")
        risk_rows = f[f["alert_grade"] == "RISK"]
        self.assertAlmostEqual(float(default.loc[0, "avg_MAE"]),
                               float(risk_rows["future_MAE"].mean()), places=12)
        self.assertAlmostEqual(float(sixty.loc[0, "avg_MAE"]),
                               float(risk_rows["future_MAE_60m"].mean()), places=12)


class Step17ScriptScanTests(unittest.TestCase):
    """Static scans (scripts stay off sys.path, house convention)."""

    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(", ".SendOrder")

    def test_train_risk_model_pairing_fold_and_cutoff(self):
        text = (SCRIPTS_DIR / "train_risk_model.py").read_text(encoding="utf-8")
        # Step 20 extended the mutual-pairing map (v1_60m_prod); the seal itself stands.
        self.assertIn('REQUIRED_TARGET_BY_VERSION = {"v1_short": "target_up_risk", '
                      '"v1_60m": "target_mae_risk_60m",\n'
                      '                              "v1_60m_prod": "target_mae_risk_60m"}',
                      text)
        self.assertIn('train[train["time"] <= HORIZON_60M_MAIN_CUTOFF]', text)
        self.assertIn("WALK_FORWARD_FOLDS_60M", text)
        for token in self.FORBIDDEN:
            self.assertNotIn(token, text)

    def test_tcn_trainer_60m_main_population_and_reserved_suffixes(self):
        text = (SCRIPTS_DIR / "train_tcn_risk_model.py").read_text(encoding="utf-8")
        self.assertIn('train[train["time"] <= rm.HORIZON_60M_MAIN_CUTOFF]', text)
        self.assertIn("_tcn_reserved_suffixes", text)
        self.assertIn("WALK_FORWARD_FOLDS_60M", text)
        for token in self.FORBIDDEN:
            self.assertNotIn(token, text)

    def test_export_training_db_script_research_only(self):
        path = SCRIPTS_DIR / "export_tcn_training_db.py"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Research / evaluation only", text)
        self.assertNotIn("import torch", text)
        for token in self.FORBIDDEN:
            self.assertNotIn(token, text)
        # copies rows only -- must never write into the SOURCE db (attached read-only
        # by convention: every CREATE/INSERT targets the new file's main schema)
        self.assertNotIn("INSERT INTO src.", text)
        self.assertNotIn("UPDATE src.", text)
        self.assertNotIn("DROP TABLE src.", text)

    def test_gate_lineages_margins_and_seals(self):
        text = (SCRIPTS_DIR / "evaluate_tcn_gate.py").read_text(encoding="utf-8")
        self.assertIn('"mae_margin": 0.0005', text)   # legacy lineage = the v3 hardcode
        self.assertIn('"mae_margin": 0.0003', text)   # 60m scale-matched (USER-decided)
        self.assertIn('"target": "target_mae_risk_60m"', text)
        self.assertIn("cross-lineage parquet refused", text)   # hard target/fold seals
        self.assertIn("HORIZON_60M_MAIN_CUTOFF", text)
        for token in self.FORBIDDEN:
            self.assertNotIn(token, text)


class Step18CnnRegistryTests(unittest.TestCase):
    """v5_cnn_60m: performance-seeking 1D-CNN challenger on the 60m lineage."""

    def test_v5_registry_pins(self):
        cfg = tm.tcn_version_config("v5_cnn_60m")
        self.assertEqual(cfg["target"], "target_mae_risk_60m")
        self.assertEqual(tuple(cfg["seeds"]), (42,))          # pre-registered single seed
        self.assertEqual(cfg["arch"], "cnn1d")
        self.assertEqual((cfg["channels"], cfg["levels"], cfg["kernel_size"]), (64, 4, 5))
        self.assertTrue(cfg["holdout_by_day"])                # whole-trading-day holdout
        self.assertTrue(cfg["folds"])
        self.assertEqual(cfg["horizon_minutes"], 60)
        self.assertEqual(cfg["seq_features"], tm.TCN_VERSIONS["v4_tcn_60m"]["seq_features"])
        self.assertEqual(cfg["static_features"],
                         tm.TCN_VERSIONS["v4_tcn_60m"]["static_features"])
        self.assertNotIn("v5_cnn_60m", rm.MODEL_VERSIONS)
        p = tm.resolve_tcn_model_path("v5_cnn_60m", 42, fold=2)
        self.assertTrue(str(p).endswith("risk_model_v5_cnn_60m_fold2_seed42.pt"))
        with self.assertRaises(ValueError):
            tm.resolve_tcn_model_path("v5_cnn_60m", 42)       # fold-based

    def test_frame_cache_alias_is_label_compatible(self):
        cfg = tm.tcn_version_config("v5_cnn_60m")
        acfg = tm.tcn_version_config(cfg["frame_cache_alias"])
        self.assertEqual(acfg["target"], cfg["target"])
        self.assertEqual(acfg.get("horizon_minutes"), cfg.get("horizon_minutes"))
        self.assertEqual(acfg["seq_features"], cfg["seq_features"])

    def test_pre_step18_compat_defaults(self):
        # every pre-Step-18 artifact/protocol must load/behave unchanged
        self.assertEqual(tm.TCNConfig().arch, "tcn")
        self.assertFalse(tm.TrainConfig().holdout_by_day)
        for v in ("v4_tcn", "v4_tcn_60m"):
            self.assertNotIn("arch", tm.TCN_VERSIONS[v])          # -> family default tcn
            self.assertNotIn("channels", tm.TCN_VERSIONS[v])      # -> family default 32
            self.assertNotIn("holdout_by_day", tm.TCN_VERSIONS[v])

    def test_gate_and_trainer_static_scan(self):
        gate = (SCRIPTS_DIR / "evaluate_tcn_gate.py").read_text(encoding="utf-8")
        self.assertIn('"v5_cnn_60m"', gate)
        self.assertIn("risk_model_predictions_v5_cnn_60m_fold{fold}.parquet", gate)
        self.assertIn("cnn_vs_lgbm_60m_ab_report.md", gate)
        self.assertIn("risk_model_cnn_60m_gate.csv", gate)
        trainer = (SCRIPTS_DIR / "train_tcn_risk_model.py").read_text(encoding="utf-8")
        self.assertIn('cfg.get("arch", "tcn")', trainer)
        self.assertIn("frame_cache_alias", trainer)
        self.assertIn('cfg.get("holdout_by_day", False)', trainer)


class Step19MscnnRegistryTests(unittest.TestCase):
    """v6_mscnn_60m: multi-scale CNN tie-breaker on the 60m lineage."""

    def test_v6_registry_pins(self):
        cfg = tm.tcn_version_config("v6_mscnn_60m")
        self.assertEqual(cfg["target"], "target_mae_risk_60m")
        self.assertEqual(tuple(cfg["seeds"]), (42,))          # pre-registered single seed
        self.assertEqual(cfg["arch"], "mscnn")
        self.assertEqual((cfg["channels"], cfg["levels"]), (64, 4))
        self.assertEqual(tuple(cfg["kernel_sizes"]), (3, 7, 15))   # pre-registered ladder
        self.assertTrue(all(k % 2 == 1 for k in cfg["kernel_sizes"]))
        self.assertEqual(cfg["branch_channels"], 32)
        self.assertNotIn("kernel_size", cfg)                  # single-kernel knob refused
        self.assertTrue(cfg["holdout_by_day"])
        self.assertTrue(cfg["folds"])
        self.assertEqual(cfg["horizon_minutes"], 60)
        self.assertEqual(cfg["seq_features"], tm.TCN_VERSIONS["v4_tcn_60m"]["seq_features"])
        self.assertEqual(cfg["static_features"],
                         tm.TCN_VERSIONS["v4_tcn_60m"]["static_features"])
        self.assertNotIn("v6_mscnn_60m", rm.MODEL_VERSIONS)
        p = tm.resolve_tcn_model_path("v6_mscnn_60m", 42, fold=3)
        self.assertTrue(str(p).endswith("risk_model_v6_mscnn_60m_fold3_seed42.pt"))
        with self.assertRaises(ValueError):
            tm.resolve_tcn_model_path("v6_mscnn_60m", 42)     # fold-based

    def test_frame_cache_alias_is_label_compatible(self):
        cfg = tm.tcn_version_config("v6_mscnn_60m")
        acfg = tm.tcn_version_config(cfg["frame_cache_alias"])
        self.assertEqual(acfg["target"], cfg["target"])
        self.assertEqual(acfg.get("horizon_minutes"), cfg.get("horizon_minutes"))
        self.assertEqual(acfg["seq_features"], cfg["seq_features"])

    def test_pre_step19_compat_defaults(self):
        # every pre-Step-19 artifact/protocol must load/behave unchanged
        self.assertIsNone(tm.TCNConfig().kernel_sizes)
        self.assertIsNone(tm.TCNConfig().branch_channels)
        for v in ("v4_tcn", "v4_tcn_60m", "v5_cnn_60m"):
            self.assertNotIn("kernel_sizes", tm.TCN_VERSIONS[v])
            self.assertNotIn("branch_channels", tm.TCN_VERSIONS[v])

    def test_gate_and_trainer_static_scan(self):
        gate = (SCRIPTS_DIR / "evaluate_tcn_gate.py").read_text(encoding="utf-8")
        self.assertIn('"v6_mscnn_60m"', gate)
        self.assertIn("risk_model_predictions_v6_mscnn_60m_fold{fold}.parquet", gate)
        self.assertIn("mscnn_vs_lgbm_60m_ab_report.md", gate)
        self.assertIn("risk_model_mscnn_60m_gate.csv", gate)
        trainer = (SCRIPTS_DIR / "train_tcn_risk_model.py").read_text(encoding="utf-8")
        self.assertIn("--kernel-sizes", trainer)
        self.assertIn("--branch-channels", trainer)
        self.assertIn('cfg["kernel_sizes"]', trainer)
        self.assertIn("args.kernel_sizes is None", trainer)       # submission-knob guard
        self.assertIn("args.branch_channels is None", trainer)


class Step20ProductionTests(unittest.TestCase):
    """Step 20: production split + prod versions (champion & the two candidates)."""

    def test_production_split_pins(self):
        s = rm.PRODUCTION_SPLIT_60M
        self.assertEqual((s.train_start, s.train_end), ("2025-06-02", "2026-06-30"))
        self.assertEqual((s.validation_start, s.validation_end),
                         ("2026-07-01", "2026-07-10"))
        # the frozen research protocol is untouched: still exactly the 4 sealed folds
        self.assertEqual(len(rm.WALK_FORWARD_FOLDS_60M), 4)
        self.assertEqual(rm.WALK_FORWARD_FOLDS_60M[3].validation_end, "2026-06-30")

    def test_champion_prod_registry(self):
        cfg = rm.model_version_config("v1_60m_prod")
        self.assertEqual(cfg["features"], rm.FEATURE_COLUMNS_60M)
        self.assertEqual(cfg["filename"], "risk_model_v1_60m_prod_lgbm.txt")
        self.assertFalse(cfg["requires_cross_symbol"])
        trainer = (SCRIPTS_DIR / "train_risk_model.py").read_text(encoding="utf-8")
        self.assertIn('"v1_60m_prod": "target_mae_risk_60m"', trainer)
        self.assertIn("PRODUCTION_SPLIT_60M", trainer)
        self.assertIn('version in ("v1_60m", "v1_60m_prod")', trainer)   # 60m cutoff path

    def test_tcn_prod_registry_pins(self):
        for version, base in (("v4_tcn_60m_prod", "v4_tcn_60m"),
                              ("v5_cnn_60m_prod", "v5_cnn_60m")):
            cfg = tm.tcn_version_config(version)
            bcfg = tm.tcn_version_config(base)
            self.assertEqual(tuple(cfg["seeds"]), tm.SUBMISSION_SEEDS)   # 5-seed mean
            self.assertIs(cfg["split"], rm.PRODUCTION_SPLIT_60M)
            self.assertNotIn("folds", cfg)
            self.assertEqual(cfg["target"], "target_mae_risk_60m")
            self.assertEqual(cfg["horizon_minutes"], 60)
            self.assertEqual(cfg["frame_cache_alias"], "v4_tcn_60m")
            self.assertEqual(cfg["seq_features"], bcfg["seq_features"])
            self.assertEqual(cfg["static_features"], bcfg["static_features"])
            self.assertNotIn(version, rm.MODEL_VERSIONS)
            p = tm.resolve_tcn_model_path(version, 43)      # split versions: no fold
            self.assertTrue(str(p).endswith(f"risk_model_{version}_seed43.pt"))
        # the candidates must carry EXACTLY their research architectures
        v5p = tm.tcn_version_config("v5_cnn_60m_prod")
        self.assertEqual((v5p["arch"], v5p["channels"], v5p["levels"], v5p["kernel_size"]),
                         ("cnn1d", 64, 4, 5))
        self.assertTrue(v5p["holdout_by_day"])
        for key in ("arch", "channels", "kernel_size", "holdout_by_day"):
            self.assertNotIn(key, tm.tcn_version_config("v4_tcn_60m_prod"))

    def test_trainer_split_support_static_scan(self):
        trainer = (SCRIPTS_DIR / "train_tcn_risk_model.py").read_text(encoding="utf-8")
        self.assertIn('cfg.get("split")', trainer)
        self.assertIn("split.train_start, split.train_end", trainer)   # submission seal

    def test_selection_script_pre_registration(self):
        text = (SCRIPTS_DIR / "select_prod_60m.py").read_text(encoding="utf-8")
        self.assertIn("SELECTION_MARGIN = 0.005", text)
        self.assertIn('TIE_BREAK = "v5_cnn_60m_prod"', text)
        self.assertIn("assert_ensemble_consistency", text)             # mean-of-seeds seal
        self.assertIn("PRODUCTION_SPLIT_60M", text)
        for token in ("SendOrder(", "send_order(", "OrderRequest(", "place_order(",
                      ".SendOrder"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
