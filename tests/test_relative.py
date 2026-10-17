"""Tests for Step 9: cross-symbol relative-strength features."""

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


def _row(symbol, dt, ret5, ret20, volz, close=100):
    return {"symbol": symbol, "decision_time": dt, "close": close,
            "ret_5m_1m": ret5, "ret_20m_1m": ret20, "volume_z_20m_1m": volz}


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class RelativeFeatureTests(unittest.TestCase):
    def test_spec_example_three_symbols(self):
        from relative_features import build_relative_features

        # step9.md example: SK 1.2, Samsung 0.3, Hyundai -0.1 (ret_20m)
        df = pd.DataFrame([
            _row("005930", "09:10", 0.1, 0.3, 0.5),    # Samsung
            _row("000660", "09:10", 0.2, 1.2, 1.0),    # SK hynix
            _row("005380", "09:10", -0.05, -0.1, -0.2),  # Hyundai
        ])
        out = build_relative_features(df).set_index("symbol")
        # ranks: SK=1, Samsung=2, Hyundai=3
        self.assertEqual(out.loc["000660"]["rank_ret_20m_1m"], 1)
        self.assertEqual(out.loc["005930"]["rank_ret_20m_1m"], 2)
        self.assertEqual(out.loc["005380"]["rank_ret_20m_1m"], 3)
        # active_symbol_count
        self.assertTrue((out["active_symbol_count"] == 3).all())
        # universe mean and relative
        mean = (0.3 + 1.2 - 0.1) / 3
        self.assertAlmostEqual(out.loc["000660"]["universe_mean_ret_20m"], mean)
        self.assertAlmostEqual(out.loc["000660"]["relative_ret_20m_to_universe"], 1.2 - mean)
        # breadth: SK & Samsung > 0 -> 2/3
        self.assertAlmostEqual(out.loc["005930"]["universe_breadth"], 2 / 3)

    def test_rank_range_within_count(self):
        from relative_features import build_relative_features

        df = pd.DataFrame([
            _row("A", "09:10", 0.1, 0.3, 0.5),
            _row("B", "09:10", 0.2, 1.2, 1.0),
        ])
        out = build_relative_features(df)
        self.assertTrue(((out["rank_ret_20m_1m"] >= 1) & (out["rank_ret_20m_1m"] <= out["active_symbol_count"])).all())
        self.assertEqual(set(out["rank_ret_20m_1m"]), {1, 2})

    def test_two_symbols(self):
        from relative_features import build_relative_features

        df = pd.DataFrame([
            _row("A", "09:10", 0.1, 0.3, 0.5),
            _row("B", "09:10", 0.2, 1.2, 1.0),
        ])
        out = build_relative_features(df).set_index("symbol")
        self.assertEqual(out.loc["B"]["rank_ret_20m_1m"], 1)
        self.assertEqual(out.loc["A"]["rank_ret_20m_1m"], 2)
        self.assertAlmostEqual(out.loc["A"]["relative_ret_20m_to_universe"], 0.3 - (0.3 + 1.2) / 2)
        self.assertAlmostEqual(out.loc["A"]["universe_breadth"], 1.0)  # both up

    def test_per_decision_time_isolation(self):
        # ranks at one decision_time must not consider another decision_time's values
        from relative_features import build_relative_features

        df = pd.DataFrame([
            _row("A", "09:10", 0, 0.3, 0), _row("B", "09:10", 0, 0.1, 0),
            _row("A", "09:11", 0, 0.1, 0), _row("B", "09:11", 0, 0.9, 0),
        ])
        out = build_relative_features(df).set_index(["decision_time", "symbol"])
        self.assertEqual(out.loc[("09:10", "A")]["rank_ret_20m_1m"], 1)  # A higher at 09:10
        self.assertEqual(out.loc[("09:11", "B")]["rank_ret_20m_1m"], 1)  # B higher at 09:11

    def test_breadth_zero_when_none_up(self):
        from relative_features import build_relative_features

        df = pd.DataFrame([
            _row("A", "09:10", 0, -0.3, 0), _row("B", "09:10", 0, -0.1, 0),
        ])
        out = build_relative_features(df)
        self.assertTrue((out["universe_breadth"] == 0.0).all())

    def test_partial_decision_times(self):
        from relative_features import build_relative_features, partial_decision_times

        df = pd.DataFrame([
            _row("A", "09:10", 0, 0.3, 0), _row("B", "09:10", 0, 0.1, 0),  # full (2)
            _row("A", "09:11", 0, 0.2, 0),                                  # partial (1)
        ])
        out = build_relative_features(df)
        partial = partial_decision_times(out, active_total=2)
        self.assertEqual(list(partial.index), ["09:11"])
        # the partial row is still valid: count=1, rank=1
        only = out[out["decision_time"] == "09:11"].iloc[0]
        self.assertEqual(only["active_symbol_count"], 1)
        self.assertEqual(only["rank_ret_20m_1m"], 1)

    def test_nan_rank_input_raises_clearly(self):
        from relative_features import build_relative_features

        df = pd.DataFrame([
            _row("A", "09:10", 0.1, float("nan"), 0.5),
            _row("B", "09:10", 0.2, 1.2, 1.0),
        ])
        with self.assertRaises(ValueError):
            build_relative_features(df)

    def test_empty_input(self):
        from relative_features import OUTPUT_COLUMNS, build_relative_features

        out = build_relative_features(pd.DataFrame(columns=["symbol", "decision_time", "close", "ret_5m_1m", "ret_20m_1m", "volume_z_20m_1m"]))
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 0)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class V2RelativeFeatureTests(unittest.TestCase):
    """v2.0 richer universe features (dispersion, multi-horizon breadth, gap-to-leader)."""

    def test_dispersion_gap_and_breadth5m(self):
        import numpy as np
        from relative_features import build_relative_features
        df = pd.DataFrame([
            _row("A", "09:10", 0.2, 0.3, 0.5),
            _row("B", "09:10", -0.1, 1.2, 1.0),   # leader on ret_20m
            _row("C", "09:10", 0.05, -0.1, -0.2),
        ])
        out = build_relative_features(df).set_index("symbol")
        # cross-sectional dispersion = pandas (ddof=1) std of the 3 returns
        self.assertAlmostEqual(out.loc["A"]["universe_disp_ret_20m"],
                               float(np.std([0.3, 1.2, -0.1], ddof=1)))
        # gap-to-best: leader B = 0; all <= 0
        self.assertAlmostEqual(out.loc["B"]["ret_20m_gap_to_best"], 0.0)
        self.assertAlmostEqual(out.loc["A"]["ret_20m_gap_to_best"], 0.3 - 1.2)
        self.assertTrue((out["ret_20m_gap_to_best"] <= 1e-9).all())
        # breadth_5m: A & C up on ret_5m -> 2/3
        self.assertAlmostEqual(out.loc["A"]["universe_breadth_5m"], 2 / 3)

    def test_breadth_60m_optional_input(self):
        from relative_features import build_relative_features
        rows = []
        for sym, r60 in [("A", 0.01), ("B", -0.02), ("C", 0.03)]:
            d = _row(sym, "09:10", 0.1, 0.2, 0.0)
            d["ret_60m_1m"] = r60
            rows.append(d)
        out = build_relative_features(pd.DataFrame(rows)).set_index("symbol")
        self.assertAlmostEqual(out.loc["A"]["universe_breadth_60m"], 2 / 3)  # A,C up

    def test_breadth_60m_nan_symbol_skipped(self):
        from relative_features import build_relative_features
        rows = []
        for sym, r60 in [("A", 0.01), ("B", float("nan")), ("C", -0.03)]:
            d = _row(sym, "09:10", 0.1, 0.2, 0.0)
            d["ret_60m_1m"] = r60
            rows.append(d)
        out = build_relative_features(pd.DataFrame(rows)).set_index("symbol")
        self.assertAlmostEqual(out.loc["A"]["universe_breadth_60m"], 0.5)  # only A,C counted

    def test_breadth_60m_nan_when_input_absent(self):
        from relative_features import build_relative_features
        df = pd.DataFrame([_row("A", "09:10", 0.1, 0.3, 0.5), _row("B", "09:10", 0.2, 1.2, 1.0)])
        out = build_relative_features(df)
        self.assertTrue(out["universe_breadth_60m"].isna().all())

    def test_v2_isolation_across_decision_times(self):
        from relative_features import build_relative_features
        df = pd.DataFrame([
            _row("A", "09:10", 0, 0.3, 0), _row("B", "09:10", 0, 0.1, 0),
            _row("A", "09:11", 0, 0.1, 0), _row("B", "09:11", 0, 0.9, 0),
        ]).assign(ret_60m_1m=0.0)
        out = build_relative_features(df).set_index(["decision_time", "symbol"])
        # gap-to-best is computed within each decision_time, not across
        self.assertAlmostEqual(out.loc[("09:10", "A")]["ret_20m_gap_to_best"], 0.0)   # A leads at 09:10
        self.assertAlmostEqual(out.loc[("09:11", "B")]["ret_20m_gap_to_best"], 0.0)   # B leads at 09:11


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class V21LeaderLagFeatureTests(unittest.TestCase):
    """v2.1x causal leader lead-lag: each labeled leader's own return broadcast onto
    every symbol's row at the SAME decision_time (already-closed -> leakage-safe)."""

    def test_leader_columns_absent_by_default(self):
        from relative_features import OUTPUT_COLUMNS, build_relative_features
        df = pd.DataFrame([_row("L", "09:10", 0.1, 0.3, 0.5), _row("X", "09:10", 0.2, 1.2, 1.0)])
        out = build_relative_features(df)  # no leaders
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertNotIn("leader_a_ret_5m_1m", out.columns)

    def test_multiple_leaders_each_get_labeled_columns(self):
        from relative_features import build_relative_features, leader_columns
        df = pd.DataFrame([
            _row("LA", "09:10", 0.10, 0.30, 0.5),   # leader a
            _row("LB", "09:10", 0.40, 0.70, 0.5),   # leader b
            _row("X", "09:10", 0.20, 1.20, 1.0),
        ])
        out = build_relative_features(df, leaders=[("a", "LA"), ("b", "LB")]).set_index("symbol")
        for label in ("a", "b"):
            for c in leader_columns(label):
                self.assertIn(c, out.columns)
        # every symbol sees each leader's own returns at 09:10
        for sym in ("LA", "LB", "X"):
            self.assertAlmostEqual(out.loc[sym]["leader_a_ret_5m_1m"], 0.10)
            self.assertAlmostEqual(out.loc[sym]["leader_a_ret_20m_1m"], 0.30)
            self.assertAlmostEqual(out.loc[sym]["leader_b_ret_5m_1m"], 0.40)
            self.assertAlmostEqual(out.loc[sym]["leader_b_ret_20m_1m"], 0.70)

    def test_leader_is_causal_per_decision_time(self):
        # Leakage guard: the leader value at D must be the leader's return AS OF D,
        # never a later decision_time's value.
        from relative_features import build_relative_features
        df = pd.DataFrame([
            _row("L", "09:10", 0.1, 0.5, 0), _row("X", "09:10", 0.0, 0.2, 0),
            _row("L", "09:11", 0.9, 0.8, 0), _row("X", "09:11", 0.0, 0.3, 0),
        ])
        out = build_relative_features(df, leaders=[("a", "L")]).set_index(["decision_time", "symbol"])
        # X at 09:10 sees L@09:10 (0.1/0.5), NOT L@09:11 (0.9/0.8)
        self.assertAlmostEqual(out.loc[("09:10", "X")]["leader_a_ret_5m_1m"], 0.1)
        self.assertAlmostEqual(out.loc[("09:10", "X")]["leader_a_ret_20m_1m"], 0.5)
        self.assertAlmostEqual(out.loc[("09:11", "X")]["leader_a_ret_5m_1m"], 0.9)
        self.assertAlmostEqual(out.loc[("09:11", "X")]["leader_a_ret_20m_1m"], 0.8)

    def test_leader_absent_yields_nan_columns(self):
        from relative_features import build_relative_features
        df = pd.DataFrame([_row("X", "09:10", 0.2, 1.2, 1.0), _row("Y", "09:10", 0.1, 0.3, 0.5)])
        out = build_relative_features(df, leaders=[("b", "L")])  # leader not present
        self.assertIn("leader_b_ret_5m_1m", out.columns)
        self.assertTrue(out["leader_b_ret_5m_1m"].isna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
