"""Tests for serving-time feature/alert/risk scoring (src/realtime_features.py).

Uses only the numpy logistic model so it runs in the 32-bit runtime too (no
lightgbm/sklearn/matplotlib).
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import realtime_features as rf  # noqa: E402
from risk_model import (  # noqa: E402
    ALERT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FeaturePreprocessor,
    NumpyLogisticRiskModel,
)

SOURCE = "al_regular"


def _make_window(n_days=7, start="2025-06-02", minutes=180, seed=7):
    """Single symbol, n_days weekdays of 1m bars on a shared HH:MM grid."""
    rng = np.random.RandomState(seed)
    rows = []
    day = datetime.strptime(start, "%Y-%m-%d")
    made = 0
    price = 10000
    while made < n_days:
        if day.weekday() < 5:
            base = day.replace(hour=9, minute=0, second=0)
            for i in range(minutes):
                t0 = base + timedelta(minutes=i)
                t1 = t0 + timedelta(minutes=1)
                step = int(rng.randn() * 15)
                o = price
                c = max(1000, o + step)
                hi = max(o, c) + abs(int(rng.randn() * 8))
                lo = min(o, c) - abs(int(rng.randn() * 8))
                vol = int(abs(rng.randn()) * 1000) + 100
                rows.append({"symbol": "TST", "bar_start": t0.strftime("%Y-%m-%d %H:%M:%S"),
                             "bar_end": t1.strftime("%Y-%m-%d %H:%M:%S"),
                             "open": o, "high": hi, "low": lo, "close": c, "volume": vol})
                price = c
            made += 1
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def _make_symbol_window(symbol, seed, n_days=7, start="2025-06-02", minutes=180):
    """One symbol's n_days of 1m bars on the shared HH:MM grid (seed varies the path)."""
    rng = np.random.RandomState(seed)
    rows = []
    day = datetime.strptime(start, "%Y-%m-%d")
    made = 0
    price = 10000 + seed * 50
    while made < n_days:
        if day.weekday() < 5:
            base = day.replace(hour=9, minute=0, second=0)
            for i in range(minutes):
                t0 = base + timedelta(minutes=i)
                t1 = t0 + timedelta(minutes=1)
                o = price
                c = max(1000, o + int(rng.randn() * 15))
                hi = max(o, c) + abs(int(rng.randn() * 8))
                lo = min(o, c) - abs(int(rng.randn() * 8))
                rows.append({"symbol": symbol, "bar_start": t0.strftime("%Y-%m-%d %H:%M:%S"),
                             "bar_end": t1.strftime("%Y-%m-%d %H:%M:%S"),
                             "open": o, "high": hi, "low": lo, "close": c,
                             "volume": int(abs(rng.randn()) * 1000) + 100})
                price = c
            made += 1
        day += timedelta(days=1)
    return pd.DataFrame(rows)


class CrossSymbolServingTests(unittest.TestCase):
    """v1.1: build_serving_features_cross computes the REAL Step-9 cross-section."""

    def setUp(self):
        self.symbols = ["AAA", "BBB", "CCC"]
        self.window = pd.concat(
            [_make_symbol_window(s, seed=11 + i) for i, s in enumerate(self.symbols)],
            ignore_index=True)
        self.target = "AAA"

    def test_cross_matches_build_relative_features(self):
        from relative_features import RANK_INPUTS, build_relative_features
        serv = rf.build_serving_features_cross(self.window, target=self.target,
                                               source=SOURCE, active_count=len(self.symbols))
        rel = build_relative_features(rf._build_merged(self.window, SOURCE).dropna(subset=RANK_INPUTS))
        rel_t = rel[rel["symbol"] == self.target].set_index("decision_time")
        # compare only on REAL full-universe minutes (all symbols complete) -- those are the
        # rows where serv keeps the real cross-section rather than the neutral fallback.
        rel_full = rel_t[rel_t["active_symbol_count"] == len(self.symbols)]
        s = serv.set_index("decision_time")
        common = s.index.intersection(rel_full.index)
        self.assertGreater(len(common), 20)
        for c in rf.REL_COLUMNS:
            np.testing.assert_allclose(
                s.loc[common, c].astype(float).to_numpy(),
                rel_full.loc[common, c].astype(float).to_numpy(), rtol=0, atol=1e-9,
                err_msg=f"{c} differs from build_relative_features")

    def test_cross_features_are_not_neutral(self):
        # the whole point of v1.1: ranks/relative/breadth actually vary (not the
        # constant mid-rank/0/0.5 of the single-symbol neutral fill).
        serv = rf.build_serving_features_cross(self.window, target=self.target,
                                               source=SOURCE, active_count=len(self.symbols))
        full = serv[serv["active_symbol_count"] == len(self.symbols)]
        self.assertGreater(full["rank_ret_20m_1m"].nunique(), 1)
        self.assertGreater(full["relative_ret_20m_to_universe"].abs().max(), 0.0)

    def test_cross_causal_truncation_invariance(self):
        full = rf.build_serving_features_cross(self.window, target=self.target,
                                               source=SOURCE, active_count=len(self.symbols))
        last_day = self.window["bar_start"].str.slice(0, 10).max()
        cut = self.window[(self.window["bar_start"].str.slice(0, 10) < last_day) |
                          (self.window["bar_start"].str.slice(11, 16) <= "11:00")]
        cutserv = rf.build_serving_features_cross(cut, target=self.target,
                                                  source=SOURCE, active_count=len(self.symbols))
        dt = cutserv["decision_time"].iloc[-1]
        fr = full[full["decision_time"] == dt].iloc[0]
        cr = cutserv[cutserv["decision_time"] == dt].iloc[0]
        for c in rf.REL_COLUMNS:
            self.assertAlmostEqual(float(fr[c]), float(cr[c]), places=9,
                                   msg=f"{c} changed when later bars were added")

    def test_only_target_falls_back_to_neutral(self):
        only = self.window[self.window["symbol"] == self.target]
        serv = rf.build_serving_features_cross(only, target=self.target, source=SOURCE, active_count=20)
        single = rf.build_serving_features(only, source=SOURCE, active_count=20)
        s, g = serv.set_index("decision_time"), single.set_index("decision_time")
        common = s.index.intersection(g.index)
        self.assertGreater(len(common), 20)
        for c in rf.REL_COLUMNS:
            np.testing.assert_allclose(s.loc[common, c].astype(float).to_numpy(),
                                       g.loc[common, c].astype(float).to_numpy(), rtol=0, atol=1e-9,
                                       err_msg=f"{c} not neutral when only the target is present")

    def test_missing_peer_yields_partial_no_raise(self):
        last_day = self.window["bar_start"].str.slice(0, 10).max()
        win = self.window[~((self.window["symbol"] == "BBB") &
                            (self.window["bar_start"].str.slice(0, 10) == last_day))]
        serv = rf.build_serving_features_cross(win, target=self.target, source=SOURCE,
                                               active_count=len(self.symbols))
        last = serv[serv["decision_time"].str.slice(0, 10) == last_day]
        self.assertGreater(len(last), 0)
        # BBB absent that day -> real-cross minutes rank only AAA+CCC (count 2 appears).
        # (neutral-fallback warm-up rows keep count = active_count, so we assert presence
        # of the partial count rather than the max.)
        self.assertIn(2, set(last["active_symbol_count"].astype(int).tolist()))

    def test_empty_input_returns_empty(self):
        self.assertTrue(rf.build_serving_features_cross(
            pd.DataFrame(), target="AAA", source=SOURCE, active_count=3).empty)


class ServingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.window = _make_window()
        self.active_count = 20
        self.serv = rf.build_serving_features(self.window, source=SOURCE, active_count=self.active_count)

    def test_produces_all_non_alert_model_features(self):
        # build_serving_features produces every model feature EXCEPT the alert-score
        # features (up/down/alert/direction/risk_score), which score_serving adds.
        for col in FEATURE_COLUMNS:
            if col in ALERT_FEATURE_COLUMNS:
                self.assertNotIn(col, self.serv.columns)
            else:
                self.assertIn(col, self.serv.columns, f"serving frame missing {col}")

    def test_cross_sectional_features_are_neutral(self):
        self.assertTrue((self.serv["active_symbol_count"] == self.active_count).all())
        self.assertTrue((self.serv["universe_breadth"] == 0.5).all())
        self.assertTrue((self.serv["relative_ret_20m_to_universe"] == 0.0).all())
        # neutral rank = neither top-third nor bottom-third bucket
        mid = self.serv["rank_ret_20m_1m"].iloc[0]
        self.assertEqual(int(mid), max(1, (self.active_count + 1) // 2))

    def test_strictly_causal_truncation_invariance(self):
        """A past decision row is identical whether or not later bars exist."""
        last_day = self.window["bar_start"].str.slice(0, 10).max()
        day_bars = self.window[self.window["bar_start"].str.slice(0, 10) == last_day]
        prior = self.window[self.window["bar_start"].str.slice(0, 10) < last_day]
        # window truncated mid-day (first 90 of the last day)
        cut = pd.concat([prior, day_bars.iloc[:90]], ignore_index=True)
        serv_cut = rf.build_serving_features(cut, source=SOURCE, active_count=self.active_count)
        dt = serv_cut["decision_time"].iloc[-1]
        full_row = self.serv[self.serv["decision_time"] == dt].iloc[0]
        cut_row = serv_cut[serv_cut["decision_time"] == dt].iloc[0]
        for col in ("ret_20m_1m", "vwap_distance_1m", "dist_ma5_intraday", "volume_z_20bar_5m", "range_1m"):
            a, b = full_row[col], cut_row[col]
            if pd.isna(a) and pd.isna(b):
                continue
            self.assertAlmostEqual(float(a), float(b), places=9, msg=f"{col} changed with future bars")

    def test_score_serving_risk_prob_and_alert_columns(self):
        feats = [c for c in FEATURE_COLUMNS if c in self.serv.columns]
        train = self.serv.dropna(subset=feats)
        self.assertGreater(len(train), 50)
        pre = FeaturePreprocessor(feats).fit(train)
        y = (train["dist_ma5_intraday"].fillna(0) < 0).astype(int).to_numpy()  # synthetic target
        model = NumpyLogisticRiskModel(max_iter=40).fit(pre.transform(train), y)
        model.model_type = NumpyLogisticRiskModel.model_type

        res = rf.score_serving(self.serv, model, pre)
        self.assertGreater(len(res), 0)
        self.assertTrue(((res["risk_prob"] >= 0) & (res["risk_prob"] <= 1)).all())
        for col in ("alert_type", "alert_grade", "up_score", "down_score", "reason_1"):
            self.assertIn(col, res.columns)
        # warm-up rows (NaN rule inputs) are dropped, so every scored row has a usable score
        self.assertFalse(res["up_score"].isna().any())

    def test_empty_input_returns_empty(self):
        self.assertTrue(rf.build_serving_features(pd.DataFrame(), source=SOURCE, active_count=20).empty)
        self.assertEqual(len(rf.score_serving(pd.DataFrame(), None, None)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
