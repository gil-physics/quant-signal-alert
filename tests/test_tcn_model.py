"""Tests for Step 16 TCN challenger (v4_tcn): registry separation, sequence-window
causality, anchor-key integrity, padding-not-dropping, train-only standardization,
torch round-trips, and no-order safety.

Imports from src only (SRC_DIR on path); the train/gate scripts are checked by
static file scan, never imported (scripts dir stays off sys.path). torch-dependent
cases are skipped when torch is absent (same pattern as the lightgbm-dependent
tests) -- everything in sections A/B/D runs on the dependency-free runtime.
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import feature_lib  # noqa: E402
import risk_model as rm  # noqa: E402
import tcn_data  # noqa: E402
import tcn_model as tm  # noqa: E402  (must import WITHOUT torch: lazy-import contract)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# synthetic features_1m-shaped frames (tests pin invariants, not the real lists)
# --------------------------------------------------------------------------- #
def _bar_frame(rows):
    """rows: list of (symbol, decision_time, f1). f2 = f1 * 10 rides along."""
    return pd.DataFrame({
        "symbol": [r[0] for r in rows],
        "decision_time": [r[1] for r in rows],
        "f1": [float(r[2]) for r in rows],
        "f2": [float(r[2]) * 10.0 for r in rows],
    })


def _day_bars(symbol, date, start_minute, values):
    return [(symbol, f"{date} 09:{start_minute + i:02d}:00", v) for i, v in enumerate(values)]


_CFG = tcn_data.SequenceConfig(seq_len=4, seq_feature_columns=("f1", "f2"))


def _raw_windows(frame, decisions, *, cfg=_CFG, min_match_rate=1.0):
    """Windows over the RAW (unstandardized) bar matrix -- value containment stays
    directly observable."""
    idx = tcn_data.build_bar_index(frame, seq_config=cfg)
    j, day_start = tcn_data.attach_anchor_positions(decisions, idx,
                                                    min_match_rate=min_match_rate)
    return tcn_data.gather_windows(idx.bar_mat_raw, idx.bar_tod, j, day_start,
                                   seq_config=cfg), idx


def _decisions(symbol, times):
    return pd.DataFrame({"symbol": [symbol] * len(times), "decision_time": list(times)})


# --------------------------------------------------------------------------- #
# A. registry / freeze invariants (no torch)
# --------------------------------------------------------------------------- #
class RegistryAndFreezeTests(unittest.TestCase):
    def test_v4_registry_and_path(self):
        cfg = tm.tcn_version_config("v4_tcn")
        self.assertEqual(cfg["target"], "target_mae_risk")
        self.assertTrue(cfg["requires_cross_symbol"], "the challenger targets the v1.1 config")
        self.assertTrue(str(tm.resolve_tcn_model_path("v4_tcn", 42))
                        .endswith("risk_model_v4_tcn_seed42.pt"))
        with self.assertRaises(ValueError):
            tm.tcn_version_config("v9_unknown")

    def test_v4_lives_outside_the_lgbm_registry(self):
        # a MODEL_VERSIONS entry would pollute train_risk_model.py --risk-model-version
        # choices and misroute load_model's artifact dispatch onto a torch file.
        self.assertNotIn("v4_tcn", rm.MODEL_VERSIONS)

    def test_v1_baseline_still_frozen(self):
        self.assertEqual(len(rm.FEATURE_COLUMNS), 49)
        self.assertEqual(rm.MODEL_VERSIONS["v1"]["filename"], "risk_model_lgbm.txt")
        self.assertEqual(rm.MAE_RISK_THRESHOLD, -0.015)

    def test_channel_lists_reference_shared_constants(self):
        available = set(feature_lib.FEATURE_COLUMNS_1M) | set(feature_lib.MA_COLUMNS_1M)
        for c in tcn_data.SEQ_FEATURE_COLUMNS_V4:
            self.assertIn(c, available, f"sequence channel {c} is not a features_1m column")
        self.assertEqual(tcn_data.STATIC_FEATURE_COLUMNS_V4,
                         list(rm.ALERT_FEATURE_COLUMNS) + list(rm.RELATIVE_FEATURE_COLUMNS))
        self.assertEqual(len(tcn_data.STATIC_FEATURE_COLUMNS_V4), 13)

    def test_no_forbidden_tokens_in_channels(self):
        for c in tcn_data.SEQ_FEATURE_COLUMNS_V4 + tcn_data.STATIC_FEATURE_COLUMNS_V4:
            self.assertFalse(any(tok in c.lower() for tok in rm.FORBIDDEN_FEATURE_TOKENS), c)
        frame = pd.DataFrame(columns=tcn_data.STATIC_FEATURE_COLUMNS_V4 + ["target_mae_risk"])
        with self.assertRaises(ValueError):
            rm.available_feature_columns(frame,
                                         tcn_data.STATIC_FEATURE_COLUMNS_V4 + ["target_mae_risk"])

    def test_preregistered_seed_protocol(self):
        self.assertEqual(tm.SUBMISSION_SEEDS, (42, 43, 44, 45, 46))
        self.assertEqual(tm.TCN_VERSIONS["v4_tcn"]["seeds"], tm.SUBMISSION_SEEDS)
        a = np.array([0.2, 0.4, 0.6])
        b = np.array([0.4, 0.2, 1.0])
        np.testing.assert_allclose(tm.ensemble_probs([a, b]),
                                   np.vstack([a, b]).mean(axis=0))

    def test_ensemble_consistency_enforced(self):
        # the gate calls this; it must pass on the true mean and RAISE on a best-seed pick
        cols = ["risk_prob_seed42", "risk_prob_seed43"]
        good = pd.DataFrame({cols[0]: [0.2, 0.8], cols[1]: [0.4, 0.6]})
        good["risk_prob"] = good[cols].mean(axis=1)
        tm.assert_ensemble_consistency(good, cols)                # no raise
        bad = good.copy()
        bad["risk_prob"] = bad[cols[0]]                           # best-seed pick, not the mean
        with self.assertRaises(ValueError):
            tm.assert_ensemble_consistency(bad, cols)
        with self.assertRaises(ValueError):                      # no seed columns at all
            tm.assert_ensemble_consistency(good, [])

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_resolve_device(self):
        import torch

        self.assertEqual(tm.resolve_device(None).type, "cpu")
        self.assertEqual(tm.resolve_device("cpu").type, "cpu")
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):   # clear fail-fast, not a mid-epoch error
                tm.resolve_device("cuda:0")

    def test_receptive_field_and_required_levels(self):
        self.assertEqual(tm.receptive_field(4, 3), 61)           # default stack covers 60
        self.assertEqual(tm.required_levels(60, 3), 4)
        self.assertEqual(tm.required_levels(120, 3), 5)          # ablB must deepen to see 120
        self.assertGreaterEqual(tm.receptive_field(tm.required_levels(120, 3), 3), 120)
        with self.assertRaises(ValueError):                      # RF 61 < seq_len 120
            tm._check_receptive_field(tm.TCNConfig(levels=4, kernel_size=3),
                                      tcn_data.SequenceConfig(seq_len=120))

    def test_frame_fingerprint_is_value_sensitive(self):
        f1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        f2 = f1.copy()
        self.assertEqual(tm.frame_fingerprint(f1, ["a", "b"]), tm.frame_fingerprint(f2, ["a", "b"]))
        f2.loc[0, "a"] = 1.5                                     # a feature rebuild changes values
        self.assertNotEqual(tm.frame_fingerprint(f1, ["a", "b"]), tm.frame_fingerprint(f2, ["a", "b"]))


# --------------------------------------------------------------------------- #
# B. sequence construction causality (numpy/pandas only)
# --------------------------------------------------------------------------- #
class SequenceWindowTests(unittest.TestCase):
    def test_window_is_causal_and_poison_free(self):
        # bars 09:01..09:10 carry values 1..10; the 09:06 bar is poisoned. A decision
        # anchored at 09:05 must see exactly [2,3,4,5] (newest last = decision bar).
        values = list(range(1, 11))
        values[5] = 999.0                       # poison the 09:06 bar (index 5)
        frame = _bar_frame(_day_bars("SYMA", "2026-01-05", 1, values))
        win, _ = _raw_windows(frame, _decisions("SYMA", ["2026-01-05 09:05:00"]))
        np.testing.assert_array_equal(win[0, :, 0], [2.0, 3.0, 4.0, 5.0])
        self.assertNotIn(999.0, win[0, :, 0].tolist())
        self.assertTrue((win[0, :, -1] == 1.0).all(), "fully populated window has mask 1")

    def test_truncation_invariance(self):
        # adding LATER bars (any symbol) must leave earlier windows byte-identical.
        full_rows = (_day_bars("SYMA", "2026-01-05", 1, range(1, 11))
                     + _day_bars("SYMB", "2026-01-05", 1, range(101, 111)))
        trunc_rows = [r for r in full_rows if r[1] <= "2026-01-05 09:05:00"]
        decisions = pd.concat([_decisions("SYMA", ["2026-01-05 09:04:00", "2026-01-05 09:05:00"]),
                               _decisions("SYMB", ["2026-01-05 09:05:00"])], ignore_index=True)
        win_full, _ = _raw_windows(_bar_frame(full_rows), decisions)
        win_trunc, _ = _raw_windows(_bar_frame(trunc_rows), decisions)
        np.testing.assert_array_equal(win_full, win_trunc)

    def test_no_cross_day_bleed(self):
        # day-2 early windows must be padded, never filled from day-1 bars.
        rows = (_day_bars("SYMA", "2026-01-05", 1, [1, 2, 3, 4, 5])
                + _day_bars("SYMA", "2026-01-06", 1, [101, 102]))
        win, _ = _raw_windows(_bar_frame(rows), _decisions("SYMA", ["2026-01-06 09:02:00"]))
        mask = win[0, :, -1]
        np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(win[0, 2:, 0], [101.0, 102.0])
        self.assertTrue((win[0, :2, :] == 0.0).all(), "pre-day positions are zero pads")

    def test_anchor_requires_full_timestamp(self):
        frame = _bar_frame(_day_bars("SYMA", "2026-01-05", 1, [1, 2, 3]))
        idx = tcn_data.build_bar_index(frame, seq_config=_CFG)
        bad = pd.DataFrame({"symbol": ["SYMA"], "decision_time": ["09:02:00"]})
        with self.assertRaises(ValueError):
            tcn_data.attach_anchor_positions(bad, idx)
        # and the bar side too: a date-less features frame must be rejected at build
        bad_bars = _bar_frame([("SYMA", "09:01:00", 1.0)])
        with self.assertRaises(ValueError):
            tcn_data.build_bar_index(bad_bars, seq_config=_CFG)

    def test_same_clock_time_on_two_days_anchors_to_own_day(self):
        rows = (_day_bars("SYMA", "2026-01-05", 1, [1, 2, 3, 4, 5])
                + _day_bars("SYMA", "2026-01-06", 1, [101, 102, 103, 104, 105]))
        decisions = _decisions("SYMA", ["2026-01-05 09:05:00", "2026-01-06 09:05:00"])
        win, _ = _raw_windows(_bar_frame(rows), decisions)
        np.testing.assert_array_equal(win[0, :, 0], [2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(win[1, :, 0], [102.0, 103.0, 104.0, 105.0])

    def test_padding_never_dropping(self):
        # short day + one unanchorable decision: every input row still gets a window.
        frame = _bar_frame(_day_bars("SYMA", "2026-01-05", 1, [7, 8]))
        decisions = _decisions("SYMA", ["2026-01-05 09:01:00", "2026-01-05 09:59:00"])
        win, idx = _raw_windows(frame, decisions, min_match_rate=0.0)
        self.assertEqual(win.shape, (2, _CFG.seq_len, _CFG.n_channels))
        np.testing.assert_array_equal(win[0, :, -1], [0.0, 0.0, 0.0, 1.0])   # 3 pads + decision bar
        self.assertEqual(win[0, -1, 0], 7.0)                                  # newest = decision bar
        np.testing.assert_array_equal(win[1, :, -1], [0.0] * 4)               # unmatched -> all pad
        # ...but the DEFAULT match-rate guard refuses such a frame loudly
        with self.assertRaises(ValueError):
            tcn_data.attach_anchor_positions(decisions, idx)

    def test_window_is_time_ascending_with_decision_bar_last(self):
        frame = _bar_frame(_day_bars("SYMA", "2026-01-05", 1, [1, 2, 3, 4, 5, 6]))
        win, _ = _raw_windows(frame, _decisions("SYMA", ["2026-01-05 09:06:00"]))
        self.assertTrue((np.diff(win[0, :, 0]) > 0).all())
        self.assertEqual(win[0, -1, 0], 6.0)

    def test_duplicate_bars_rejected(self):
        rows = _day_bars("SYMA", "2026-01-05", 1, [1, 2]) * 2
        with self.assertRaises(ValueError):
            tcn_data.build_bar_index(_bar_frame(rows), seq_config=_CFG)


class StandardizationTests(unittest.TestCase):
    def _index(self):
        rows = (_day_bars("SYMA", "2026-01-05", 1, [1, 2, 3, 4])       # train day
                + _day_bars("SYMA", "2026-03-05", 1, [50, 60, 70, 80]))  # validation day
        return tcn_data.build_bar_index(_bar_frame(rows), seq_config=_CFG)

    def test_stats_use_train_range_only(self):
        idx = self._index()
        pre = tcn_data.SequencePreprocessor(["f1", "f2"]).fit(
            idx.bar_mat_raw, dates=idx.dates, train_end="2026-02-28")
        corrupted = idx.bar_mat_raw.copy()
        corrupted[idx.dates > "2026-02-28"] = 1e9   # poison the validation-range bars
        pre2 = tcn_data.SequencePreprocessor(["f1", "f2"]).fit(
            corrupted, dates=idx.dates, train_end="2026-02-28")
        np.testing.assert_array_equal(pre.mean_, pre2.mean_)
        np.testing.assert_array_equal(pre.scale_, pre2.scale_)
        np.testing.assert_array_equal(pre.median_, pre2.median_)
        self.assertEqual(pre.fit_train_end, "2026-02-28")

    def test_train_end_requires_dates(self):
        idx = self._index()
        with self.assertRaises(ValueError):
            tcn_data.SequencePreprocessor(["f1", "f2"]).fit(idx.bar_mat_raw,
                                                            train_end="2026-02-28")

    def test_roundtrip_and_nan_imputation(self):
        idx = self._index()
        mat = idx.bar_mat_raw.copy()
        mat[0, 0] = np.nan
        pre = tcn_data.SequencePreprocessor(["f1", "f2"]).fit(mat)
        out = pre.transform_matrix(mat)
        self.assertTrue(np.isfinite(out).all())
        pre2 = tcn_data.SequencePreprocessor.from_dict(pre.to_dict())
        np.testing.assert_allclose(pre2.transform_matrix(mat), out, rtol=0, atol=1e-12)

    def test_split_by_date_disjoint(self):
        frame = pd.DataFrame({"date": ["2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02"]})
        train, valid = rm.split_by_date(frame)
        self.assertFalse(set(train["date"]) & set(valid["date"]))


# --------------------------------------------------------------------------- #
# C. torch-dependent (skipped when torch is absent -- 64-bit env only)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_TORCH, "torch not installed (64-bit env only)")
class TorchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        rows = []
        for day in ("2026-01-05", "2026-01-06", "2026-01-07"):
            rows += _day_bars("SYMA", day, 1, rng.normal(size=30).round(3))
        cls.bar_index_frame = _bar_frame(rows)
        cls.bar_index = tcn_data.build_bar_index(cls.bar_index_frame, seq_config=_CFG)
        cls.seq_pre = tcn_data.SequencePreprocessor(["f1", "f2"]).fit(cls.bar_index.bar_mat_raw)

        decisions = []
        for day in ("2026-01-05", "2026-01-06", "2026-01-07"):
            decisions += [(f"{day} 09:{m:02d}:00") for m in range(5, 30, 2)]
        n = len(decisions)
        cls.train = pd.DataFrame({
            "symbol": ["SYMA"] * n,
            "decision_time": decisions,
            "s1": rng.normal(size=n),
            "s2": rng.normal(size=n),
            "target_mae_risk": (np.arange(n) % 3 == 0).astype(int),
        })
        cls.static_pre = rm.FeaturePreprocessor(["s1", "s2"]).fit(cls.train)
        cls.tcn_config = tm.TCNConfig(in_channels=_CFG.n_channels, channels=4, levels=2,
                                      kernel_size=2, static_dim=2, head_hidden=8)
        cls.train_config = tm.TrainConfig(batch_size=8, max_epochs=2)

    def _fit(self, seed=42):
        return tm.fit_tcn(self.train, self.bar_index, self.seq_pre, self.static_pre,
                          "target_mae_risk", seq_config=_CFG, tcn_config=self.tcn_config,
                          train_config=self.train_config, seed=seed)

    def test_predict_shape_range_and_order(self):
        state, history = self._fit()
        probs = tm.predict_tcn(state, self.train, self.bar_index, self.seq_pre,
                               self.static_pre, seq_config=_CFG, tcn_config=self.tcn_config)
        self.assertEqual(len(probs), len(self.train))
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        self.assertEqual(history["epochs_ran"], 2)
        perm = np.random.default_rng(1).permutation(len(self.train))
        shuffled = self.train.iloc[perm].reset_index(drop=True)
        probs_perm = tm.predict_tcn(state, shuffled, self.bar_index, self.seq_pre,
                                    self.static_pre, seq_config=_CFG, tcn_config=self.tcn_config)
        np.testing.assert_allclose(probs_perm, probs[perm], rtol=0, atol=1e-7)

    def test_seed_determinism(self):
        state1, _ = self._fit(seed=123)
        state2, _ = self._fit(seed=123)
        p1 = tm.predict_tcn(state1, self.train, self.bar_index, self.seq_pre,
                            self.static_pre, seq_config=_CFG, tcn_config=self.tcn_config)
        p2 = tm.predict_tcn(state2, self.train, self.bar_index, self.seq_pre,
                            self.static_pre, seq_config=_CFG, tcn_config=self.tcn_config)
        np.testing.assert_array_equal(p1, p2)

    def test_save_load_roundtrip(self):
        state, history = self._fit()
        probs = tm.predict_tcn(state, self.train, self.bar_index, self.seq_pre,
                               self.static_pre, seq_config=_CFG, tcn_config=self.tcn_config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk_model_v4_tcn_seed42.pt"
            tm.save_tcn(path, state, seq_pre=self.seq_pre, static_pre=self.static_pre,
                        seq_config=_CFG, tcn_config=self.tcn_config,
                        train_config=self.train_config, history=history,
                        metadata={"target": "target_mae_risk"})
            payload = tm.load_tcn(path)
        self.assertEqual(payload["seq_config"], _CFG)
        self.assertEqual(payload["tcn_config"], self.tcn_config)
        self.assertEqual(payload["metadata"]["target"], "target_mae_risk")
        np.testing.assert_allclose(
            payload["static_pre"].transform(self.train),
            self.static_pre.transform(self.train), rtol=0, atol=1e-12)
        probs2 = tm.predict_tcn(payload["state_dict"], self.train, self.bar_index,
                                payload["seq_pre"], payload["static_pre"],
                                seq_config=payload["seq_config"],
                                tcn_config=payload["tcn_config"])
        np.testing.assert_allclose(probs2, probs, rtol=0, atol=1e-7)

    def test_no_static_ablation_path(self):
        cfg = tm.TCNConfig(in_channels=_CFG.n_channels, channels=4, levels=1,
                           kernel_size=2, static_dim=0, head_hidden=8)
        static_pre = rm.FeaturePreprocessor([]).fit(self.train)
        state, _ = tm.fit_tcn(self.train, self.bar_index, self.seq_pre, static_pre,
                              "target_mae_risk", seq_config=_CFG, tcn_config=cfg,
                              train_config=self.train_config, seed=42)
        probs = tm.predict_tcn(state, self.train, self.bar_index, self.seq_pre,
                               static_pre, seq_config=_CFG, tcn_config=cfg)
        self.assertEqual(len(probs), len(self.train))

    def test_production_path_no_future_leak(self):
        # End-to-end guard through the ACTUAL predict plumbing (not the raw gather):
        # poisoning bars strictly AFTER every decision_time must not change any score.
        # Reuses the fitted seq_pre (stored stats) so the ONLY thing varying is the
        # poisoned future bar -- which must never enter any decision window. Catches a
        # causality leak in the transform-then-gather path even if gather_windows is fine.
        # (The train-only standardization guard itself is pinned by StandardizationTests.)
        state, _ = self._fit()
        base = tm.predict_tcn(state, self.train, self.bar_index, self.seq_pre,
                              self.static_pre, seq_config=_CFG, tcn_config=self.tcn_config)
        last_decision = self.train["decision_time"].max()
        poisoned = _bar_frame([
            (r.symbol, r.decision_time, (999.0 if r.decision_time > last_decision else r.f1))
            for r in self.bar_index_frame.itertuples()])
        idx2 = tcn_data.build_bar_index(poisoned, seq_config=_CFG)
        after = tm.predict_tcn(state, self.train, idx2, self.seq_pre, self.static_pre,
                               seq_config=_CFG, tcn_config=self.tcn_config)
        np.testing.assert_allclose(after, base, rtol=0, atol=1e-7)

    def test_standardized_pads_are_zero_in_production_path(self):
        # a short-window (early-day) decision must have standardized pad positions == 0
        # with mask == 0 in the path fit/predict run (transform THEN gather), not just
        # in the raw gather.
        bar_mat_std = self.seq_pre.transform_matrix(self.bar_index.bar_mat_raw)
        j, day_start = tcn_data.attach_anchor_positions(
            self.train.iloc[:1], self.bar_index, min_match_rate=0.0)
        win = tcn_data.gather_windows(bar_mat_std, self.bar_index.bar_tod, j, day_start,
                                      seq_config=_CFG)
        mask = win[0, :, -1]
        if (mask == 0).any():                          # first decision of the earliest day
            pad_positions = win[0, mask == 0, :-1]     # feature+tod channels of pad steps
            np.testing.assert_array_equal(pad_positions, np.zeros_like(pad_positions))


# --------------------------------------------------------------------------- #
# D. static file scans (scripts never imported)
# --------------------------------------------------------------------------- #
class DayBoundaryHoldoutTests(unittest.TestCase):
    """Step 18: the internal early-stopping holdout can split by whole trading days."""

    @staticmethod
    def _frame(days=5, rows_per_day=100):
        recs = []
        for d in range(days):
            date = f"2026-02-{d + 2:02d}"
            recs += [{"date": date, "decision_time": f"{date} 09:{i % 60:02d}:00"}
                     for i in range(rows_per_day)]
        return pd.DataFrame(recs)

    def test_day_granularity_and_fraction(self):
        f = self._frame()
        fit, ev = tm._day_boundary_holdout(f, 0.15)
        self.assertEqual(set(f.iloc[ev]["date"]), {"2026-02-06"})   # last day (100 >= 75)
        self.assertEqual(set(f.iloc[fit]["date"]) & set(f.iloc[ev]["date"]), set())
        self.assertEqual(len(fit) + len(ev), len(f))
        _fit2, ev2 = tm._day_boundary_holdout(f, 0.35)
        self.assertEqual(set(f.iloc[ev2]["date"]), {"2026-02-05", "2026-02-06"})

    def test_degenerate_cases(self):
        small = self._frame(days=2, rows_per_day=50)                # < 400 rows
        self.assertEqual(tm._day_boundary_holdout(small, 0.15), (None, None))
        f = self._frame()
        self.assertEqual(tm._day_boundary_holdout(f, 0), (None, None))
        self.assertEqual(tm._day_boundary_holdout(f, 1.0), (None, None))   # fit would be empty


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CNN1dNetTests(unittest.TestCase):
    """Step 18 CNN1dRiskNet: triple pad-guard (input zeroing, per-block re-mask,
    masked pooling) and the mask-driven last-valid-timestep extraction."""

    @staticmethod
    def _net(**kw):
        import torch as t
        from tcn_net import CNN1dRiskNet

        defaults = dict(in_channels=4, channels=8, levels=2, kernel_size=3,
                        dropout=0.0, static_dim=0, head_hidden=8)
        defaults.update(kw)
        t.manual_seed(0)
        net = CNN1dRiskNet(**defaults)
        net.eval()
        return net

    @staticmethod
    def _x(mask_rows):
        """(B, 4, T): 3 random feature channels + the pad mask as the LAST channel."""
        import torch as t

        masks = np.asarray(mask_rows, dtype=np.float32)
        b, tt = masks.shape
        feats = np.random.default_rng(3).normal(size=(b, 3, tt)).astype(np.float32)
        return t.from_numpy(np.concatenate([feats, masks[:, None, :]], axis=1))

    def test_padding_poison_invariance_and_all_pad_finite(self):
        import torch as t

        x = self._x([[1] * 10, [0] * 4 + [1] * 6, [0] * 10])
        net = self._net()
        with t.no_grad():
            base = net(x)
            pad = x[:, -1:, :] == 0
            poisoned = x.clone()
            poisoned[:, :-1, :] = t.where(pad.expand(-1, x.shape[1] - 1, -1),
                                          t.full_like(x[:, :-1, :], 1e6),
                                          x[:, :-1, :])
            out = net(poisoned)
        self.assertTrue(t.isfinite(base).all())                 # incl. the all-pad row
        t.testing.assert_close(out, base)                       # pads never leak in

    def test_last_valid_timestep_is_mask_driven(self):
        import torch as t

        # trailing-pad mask (never produced by gather_windows, but the net must honor
        # the MASK, not the left-padding convention): last valid position = index 5.
        x = self._x([[1] * 6 + [0] * 4])
        net = self._net()
        with t.no_grad():
            base = net(x)
            bump_valid = x.clone()
            bump_valid[0, 0, 5] += 1.0                          # the last VALID timestep
            bump_pad = x.clone()
            bump_pad[0, 0, 9] += 123.0                          # a padded timestep
            self.assertFalse(t.allclose(net(bump_valid), base))
            t.testing.assert_close(net(bump_pad), base)

    def test_even_kernel_rejected(self):
        with self.assertRaises(ValueError):
            self._net(kernel_size=4)

    def test_build_net_dispatch_and_rf_check_skip(self):
        from tcn_net import CNN1dRiskNet, TCNRiskNet

        cnn_cfg = tm.TCNConfig(in_channels=_CFG.n_channels, channels=8, levels=2,
                               kernel_size=3, static_dim=2, head_hidden=8, arch="cnn1d")
        self.assertIsInstance(tm._build_net(cnn_cfg), CNN1dRiskNet)
        self.assertIsInstance(tm._build_net(tm.TCNConfig()), TCNRiskNet)
        with self.assertRaises(ValueError):
            tm._build_net(tm.TCNConfig(arch="rnn"))
        # RF check is a tcn-family concept; global pooling sees the full window
        tm._check_receptive_field(cnn_cfg, tcn_data.SequenceConfig(seq_len=120))  # no raise
        with self.assertRaises(ValueError):
            tm._check_receptive_field(tm.TCNConfig(levels=2, kernel_size=3),
                                      tcn_data.SequenceConfig(seq_len=120))

    def test_fit_predict_cnn_end_to_end_with_day_holdout(self):
        rng = np.random.default_rng(5)
        days = ("2026-01-05", "2026-01-06", "2026-01-07")
        rows = []
        for day in days:
            rows += _day_bars("SYMA", day, 1, rng.normal(size=30).round(3))
        bar_index = tcn_data.build_bar_index(_bar_frame(rows), seq_config=_CFG)
        seq_pre = tcn_data.SequencePreprocessor(["f1", "f2"]).fit(bar_index.bar_mat_raw)
        decisions, dates = [], []
        for day in days:
            for m in range(5, 30, 2):
                decisions.append(f"{day} 09:{m:02d}:00")
                dates.append(day)
        n = len(decisions)
        train = pd.DataFrame({"symbol": ["SYMA"] * n, "decision_time": decisions,
                              "date": dates, "s1": rng.normal(size=n),
                              "s2": rng.normal(size=n),
                              "target_mae_risk": (np.arange(n) % 3 == 0).astype(int)})
        static_pre = rm.FeaturePreprocessor(["s1", "s2"]).fit(train)
        cfg = tm.TCNConfig(in_channels=_CFG.n_channels, channels=4, levels=2,
                           kernel_size=3, static_dim=2, head_hidden=8, arch="cnn1d")
        tc = tm.TrainConfig(batch_size=8, max_epochs=2, holdout_by_day=True)
        state, history = tm.fit_tcn(train, bar_index, seq_pre, static_pre,
                                    "target_mae_risk", seq_config=_CFG, tcn_config=cfg,
                                    train_config=tc, seed=7)
        probs = tm.predict_tcn(state, train, bar_index, seq_pre, static_pre,
                               seq_config=_CFG, tcn_config=cfg)
        self.assertEqual(len(probs), n)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        self.assertTrue(history["holdout_by_day"])
        for key in ("holdout_loss_best", "holdout_pr_auc_best", "final_train_loss"):
            self.assertIn(key, history)                          # overfit-watch fields


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class MultiScaleNetTests(unittest.TestCase):
    """Step 19 MultiScaleCNNRiskNet: the v5 pad triple-guard/pooling contract must
    hold verbatim for the multi-scale trunk (parallel branches + 1x1 fuse + residual
    stages are three NEW pad-leak surfaces)."""

    @staticmethod
    def _net(**kw):
        import torch as t
        from tcn_net import MultiScaleCNNRiskNet

        defaults = dict(in_channels=4, channels=8, levels=3, kernel_sizes=(3, 5),
                        branch_channels=4, dropout=0.0, static_dim=0, head_hidden=8)
        defaults.update(kw)
        t.manual_seed(0)
        net = MultiScaleCNNRiskNet(**defaults)
        net.eval()
        return net

    @staticmethod
    def _x(mask_rows):
        """(B, 4, T): 3 random feature channels + the pad mask as the LAST channel."""
        import torch as t

        masks = np.asarray(mask_rows, dtype=np.float32)
        b, tt = masks.shape
        feats = np.random.default_rng(3).normal(size=(b, 3, tt)).astype(np.float32)
        return t.from_numpy(np.concatenate([feats, masks[:, None, :]], axis=1))

    def test_padding_poison_invariance_and_all_pad_finite(self):
        import torch as t

        x = self._x([[1] * 10, [0] * 4 + [1] * 6, [0] * 10])
        net = self._net()
        with t.no_grad():
            base = net(x)
            pad = x[:, -1:, :] == 0
            poisoned = x.clone()
            poisoned[:, :-1, :] = t.where(pad.expand(-1, x.shape[1] - 1, -1),
                                          t.full_like(x[:, :-1, :], 1e6),
                                          x[:, :-1, :])
            out = net(poisoned)
        self.assertTrue(t.isfinite(base).all())                 # incl. the all-pad row
        t.testing.assert_close(out, base)                       # pads never leak in

    def test_last_valid_timestep_is_mask_driven(self):
        import torch as t

        x = self._x([[1] * 6 + [0] * 4])                        # trailing-pad mask
        net = self._net()
        with t.no_grad():
            base = net(x)
            bump_valid = x.clone()
            bump_valid[0, 0, 5] += 1.0                          # the last VALID timestep
            bump_pad = x.clone()
            bump_pad[0, 0, 9] += 123.0                          # a padded timestep
            self.assertFalse(t.allclose(net(bump_valid), base))
            t.testing.assert_close(net(bump_pad), base)

    def test_kernel_ladder_validation(self):
        with self.assertRaises(ValueError):
            self._net(kernel_sizes=(3, 4))                      # even kernel
        with self.assertRaises(ValueError):
            self._net(kernel_sizes=())                          # empty ladder

    def test_build_net_dispatch_registry_pins_and_fallback(self):
        from tcn_net import MultiScaleCNNRiskNet

        ms_cfg = tm.TCNConfig(in_channels=_CFG.n_channels, channels=8, levels=2,
                              kernel_size=None, static_dim=2, head_hidden=8,
                              arch="mscnn", kernel_sizes=(3, 5), branch_channels=4)
        net = tm._build_net(ms_cfg)
        self.assertIsInstance(net, MultiScaleCNNRiskNet)
        self.assertEqual(len(net.blocks[0].branches), 2)        # ladder honored
        # kernel_sizes/branch_channels unset -> the registry-pin defaults (3, 7, 15)x32
        fallback = tm._build_net(tm.TCNConfig(in_channels=4, channels=8, levels=2,
                                              static_dim=0, arch="mscnn"))
        self.assertEqual(len(fallback.blocks[0].branches), 3)
        cfg = tm.tcn_version_config("v6_mscnn_60m")
        self.assertEqual(tuple(cfg["kernel_sizes"]), (3, 7, 15))
        # RF check is a tcn-family concept; global pooling sees the full window
        tm._check_receptive_field(ms_cfg, tcn_data.SequenceConfig(seq_len=120))  # no raise

    def test_fit_predict_mscnn_end_to_end_with_day_holdout(self):
        rng = np.random.default_rng(5)
        days = ("2026-01-05", "2026-01-06", "2026-01-07")
        rows = []
        for day in days:
            rows += _day_bars("SYMA", day, 1, rng.normal(size=30).round(3))
        bar_index = tcn_data.build_bar_index(_bar_frame(rows), seq_config=_CFG)
        seq_pre = tcn_data.SequencePreprocessor(["f1", "f2"]).fit(bar_index.bar_mat_raw)
        decisions, dates = [], []
        for day in days:
            for m in range(5, 30, 2):
                decisions.append(f"{day} 09:{m:02d}:00")
                dates.append(day)
        n = len(decisions)
        train = pd.DataFrame({"symbol": ["SYMA"] * n, "decision_time": decisions,
                              "date": dates, "s1": rng.normal(size=n),
                              "s2": rng.normal(size=n),
                              "target_mae_risk": (np.arange(n) % 3 == 0).astype(int)})
        static_pre = rm.FeaturePreprocessor(["s1", "s2"]).fit(train)
        cfg = tm.TCNConfig(in_channels=_CFG.n_channels, channels=4, levels=2,
                           kernel_size=None, static_dim=2, head_hidden=8,
                           arch="mscnn", kernel_sizes=(3, 5), branch_channels=4)
        tc = tm.TrainConfig(batch_size=8, max_epochs=2, holdout_by_day=True)
        state, history = tm.fit_tcn(train, bar_index, seq_pre, static_pre,
                                    "target_mae_risk", seq_config=_CFG, tcn_config=cfg,
                                    train_config=tc, seed=7)
        probs = tm.predict_tcn(state, train, bar_index, seq_pre, static_pre,
                               seq_config=_CFG, tcn_config=cfg)
        self.assertEqual(len(probs), n)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        self.assertTrue(history["holdout_by_day"])
        for key in ("holdout_loss_best", "holdout_pr_auc_best", "final_train_loss"):
            self.assertIn(key, history)


class StaticScanTests(unittest.TestCase):
    NEW_SRC = [SRC_DIR / "tcn_data.py", SRC_DIR / "tcn_model.py", SRC_DIR / "tcn_net.py"]
    NEW_SCRIPTS = [SCRIPTS_DIR / "train_tcn_risk_model.py", SCRIPTS_DIR / "evaluate_tcn_gate.py"]
    # call-form tokens (with "(") so that prose like "no SendOrder" in a docstring is fine
    # but an actual order-API invocation is caught.
    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(", ".SendOrder")

    def test_files_exist(self):
        for path in self.NEW_SRC + self.NEW_SCRIPTS:
            self.assertTrue(path.exists(), path)

    def test_no_order_api_calls(self):
        for path in self.NEW_SRC + self.NEW_SCRIPTS:
            text = path.read_text(encoding="utf-8")
            for token in self.FORBIDDEN:
                self.assertNotIn(token, text, f"{path.name} must not invoke {token}")

    @staticmethod
    def _module_scope_imports(path):
        """AST names imported at MODULE scope (incl. top-level if/try/with blocks) --
        robust to indentation tricks a line-anchored regex would miss."""
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        names, stack = [], list(tree.body)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                # descend only into constructs that still execute at import time,
                # NOT into FunctionDef/ClassDef (those are the allowed lazy scope).
                for attr in ("body", "orelse", "finalbody", "handlers", "items"):
                    for child in getattr(node, attr, []) or []:
                        stack += getattr(child, "body", []) if isinstance(child, ast.ExceptHandler) else [child]
        return names

    def test_lazy_torch_contract(self):
        # tcn_data/tcn_model must import at load time WITHOUT torch: no module-scope
        # torch/tcn_net import (tcn_net.py is the ONLY module allowed torch at top
        # level, and it is only ever imported inside functions). AST-based so an
        # indented `if ...: import torch` at module scope is still caught.
        for path in (SRC_DIR / "tcn_data.py", SRC_DIR / "tcn_model.py"):
            imported = self._module_scope_imports(path)
            for banned in ("torch", "tcn_net"):
                self.assertNotIn(banned, imported,
                                 f"{path.name} must not import {banned} at module scope")
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("import_module", text,
                             f"{path.name}: importlib.import_module could smuggle an eager torch import")
        # functional proof: both modules imported at the top of this test module
        self.assertIn("tcn_model", sys.modules)
        self.assertIn("tcn_data", sys.modules)

    def test_gate_script_is_torch_free_and_research_only(self):
        gate = SCRIPTS_DIR / "evaluate_tcn_gate.py"
        self.assertNotIn("torch", self._module_scope_imports(gate),
                         "evaluate_tcn_gate.py must not import torch (torch-free gate contract)")
        text = gate.read_text(encoding="utf-8")
        self.assertNotIn("import torch", text)
        self.assertIn("Research / evaluation only", text)
        self.assertIn("Research / evaluation only",
                      (SCRIPTS_DIR / "train_tcn_risk_model.py").read_text(encoding="utf-8"))

    def test_no_hardcoded_symbol_codes(self):
        # belt (the repo-wide guard in tests/test_settings.py is the suspenders);
        # broadened to ANY 6-digit quoted code -- KOSPI codes are not all 0-leading
        # (e.g. 402340), which the repo-wide 0\d{5} guard misses.
        code_re = re.compile(r"['\"]\d{6}['\"]")
        for path in self.NEW_SRC + self.NEW_SCRIPTS:
            self.assertIsNone(code_re.search(path.read_text(encoding="utf-8")),
                              f"{path.name} must not hard-code a symbol code")

    def test_train_script_fits_standardization_on_train_only(self):
        # the leakage guard is opt-in on SequencePreprocessor.fit; pin that the script
        # actually passes train_end (a refactor dropping it would leak validation stats).
        text = re.sub(r"\s+", " ", (SCRIPTS_DIR / "train_tcn_risk_model.py").read_text(encoding="utf-8"))
        self.assertRegex(text, r"SequencePreprocessor\(.*?\)\.fit\(.*?train_end=args\.train_end")

    @staticmethod
    def _slice_fn(text, name):
        m = re.search(rf"\ndef {name}\(.*?\n(?=\ndef |\n# ---|\nif __name__)", text, re.DOTALL)
        return m.group(0) if m else None

    def test_gate_math_verbatim_vs_v3_template_modulo_mae_margin_kwarg(self):
        # Step 16 pinned decide_verdict BYTE-identical to evaluate_v3_premarket_gate.py.
        # Step 17 added exactly ONE additive kwarg (mae_margin, default 0.0005 = the v3
        # hardcode) so the 60m lineage can pass its user-decided scale-matched margin;
        # the pin is now "byte-identical after normalizing exactly that kwarg", which
        # still forbids ANY other drift of the GO/NO-GO semantics, and the default
        # margin keeps the v4_tcn lineage's behavior bit-for-bit.
        tcn = (SCRIPTS_DIR / "evaluate_tcn_gate.py").read_text(encoding="utf-8")
        v3 = (SCRIPTS_DIR / "evaluate_v3_premarket_gate.py").read_text(encoding="utf-8")
        tcn_fn = self._slice_fn(tcn, "decide_verdict")
        v3_fn = self._slice_fn(v3, "decide_verdict")
        self.assertIsNotNone(tcn_fn, "decide_verdict not found in evaluate_tcn_gate.py")
        self.assertIsNotNone(v3_fn, "decide_verdict not found in evaluate_v3_premarket_gate.py")
        normalized = tcn_fn.replace(
            "def decide_verdict(base, variant, *, mae_margin=0.0005):",
            "def decide_verdict(base, variant):")
        self.assertNotEqual(normalized, tcn_fn,
                            "the mae_margin kwarg with default 0.0005 must exist verbatim")
        normalized = normalized.replace("- mae_margin \\", "- 0.0005 \\")
        self.assertEqual(normalized, v3_fn,
                         "decide_verdict must stay byte-identical to the v3 gate template "
                         "apart from the additive mae_margin kwarg")

        def slice_const(text, name):
            m = re.search(rf"^{name} = .+$", text, re.MULTILINE)
            self.assertIsNotNone(m, f"{name} not found")
            return m.group(0).split("#")[0].strip()

        for const in ("RECENT_MONTHS", "GATE_RECENT_AUC_MIN", "GATE_WORST_MONTH_FLOOR"):
            self.assertEqual(slice_const(tcn, const), slice_const(v3, const),
                             f"{const} must match the v3 gate template")

    def test_gate_mae_margin_kwarg_behaviour(self):
        # exec the sliced decide_verdict (no script import; scripts stay off sys.path)
        # and pin the margin semantics: the default tolerates an A-grade MAE regression
        # of 0.0004, the 60m scale-matched margin 0.0003 rejects it.
        text = (SCRIPTS_DIR / "evaluate_tcn_gate.py").read_text(encoding="utf-8")
        src = self._slice_fn(text, "decide_verdict")
        ns = {"np": np, "RECENT_MONTHS": ("2026-05", "2026-06"),
              "GATE_RECENT_AUC_MIN": 0.010, "GATE_WORST_MONTH_FLOOR": -0.005}
        exec(src, ns)  # noqa: S102 (test-only, our own source)
        decide_verdict = ns["decide_verdict"]

        def metrics(recent, monthly, top10, bottom20, mae):
            return {"recent_auc": recent,
                    "monthly": pd.DataFrame({"month": list(monthly),
                                             "auc": list(monthly.values())}),
                    "top10_mae_risk_rate": top10, "bottom20_mae_risk_rate": bottom20,
                    "agrade_lowrisk": {"avg_MAE": mae}}

        base = metrics(0.60, {"2026-03": 0.60, "2026-04": 0.60, "2026-05": 0.60,
                              "2026-06": 0.60}, 0.30, 0.10, -0.0100)
        variant = metrics(0.62, {"2026-03": 0.62, "2026-04": 0.62, "2026-05": 0.62,
                                 "2026-06": 0.62}, 0.35, 0.10, -0.0104)  # MAE 0.0004 worse
        default = decide_verdict(base, variant)
        self.assertTrue(default["mae_ok"])
        self.assertEqual(default["verdict"], "GO")
        tight = decide_verdict(base, variant, mae_margin=0.0003)
        self.assertFalse(tight["mae_ok"])
        self.assertNotEqual(tight["verdict"], "GO")
        self.assertIn("A-grade low-risk MAE worse", "; ".join(tight["reasons"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
