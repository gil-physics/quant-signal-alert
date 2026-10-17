"""Tests for the Step 15 downtrend trend-alert module (src/short_entry.py).

Covers the 5-type taxonomy (SHORT_RISK_ALERT > DOWN_WEAKENING_ALERT >
DOWN_CONTINUATION > DOWN_ALERT > VOLATILITY_ALERT), the per-type fire gates,
the per-type cooldown of ShortFireGate, and the annotation-first model
integration (veto only when explicitly configured, DOWN types only).

Uses only NumpyLogisticRiskModel + FeaturePreprocessor so the suite stays
32-bit safe (no lightgbm/sklearn). Also statically checks that the module (and
any of its runner scripts that already exist) never references an order API,
and that the src module hardcodes no symbol literal.
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import short_entry as se  # noqa: E402
from realtime_features import build_serving_features  # noqa: E402
from risk_model import (  # noqa: E402
    FEATURE_COLUMNS,
    FeaturePreprocessor,
    NumpyLogisticRiskModel,
)

SOURCE = "al_regular"

# neutral base row: no trend, mid rank, away from the day low, at MA5/VWAP
_BASE = {
    "up_score": 1.0, "down_score": 1.0, "ret_5m_1m": 0.0, "ret_20m_1m": 0.0,
    "volume_z_20m_1m": 0.5, "rank_ret_20m_1m": 10, "active_symbol_count": 20,
    "relative_ret_20m_to_universe": 0.0, "universe_breadth": 0.5,
    "lower_shadow_1m": 0.0, "vwap_distance_1m": 0.0, "distance_from_day_low_1m": 0.02,
    "return_from_open_1m": 0.0, "range_1m": 0.002, "dist_ma5_intraday": 0.0,
}


def _row(**over):
    r = dict(_BASE)
    r.update(over)
    return r


def _short_row(**over):
    """A DOWN_ALERT row that passes every default rule-gate knob (down 4 / up 1 /
    margin 3 / bottom rank); tests override single fields to probe boundaries."""
    r = _row(down_score=4.0, up_score=1.0, ret_20m_1m=-0.01, ret_5m_1m=-0.003,
             relative_ret_20m_to_universe=-0.01, rank_ret_20m_1m=18,
             dist_ma5_intraday=-0.004, universe_breadth=0.3)
    r.update(over)
    return r


def _frame(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "symbol", [f"S{i:02d}" for i in range(len(df))])
    df.insert(1, "decision_time", [f"2026-01-05 10:{i:02d}:00" for i in range(len(df))])
    return df


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
                o = price
                c = max(1000, o + int(rng.randn() * 15))
                hi = max(o, c) + abs(int(rng.randn() * 8))
                lo = min(o, c) - abs(int(rng.randn() * 8))
                rows.append({"symbol": "TST", "bar_start": t0.strftime("%Y-%m-%d %H:%M:%S"),
                             "bar_end": t1.strftime("%Y-%m-%d %H:%M:%S"),
                             "open": o, "high": hi, "low": lo, "close": c,
                             "volume": int(abs(rng.randn()) * 1000) + 100})
                price = c
            made += 1
        day += timedelta(days=1)
    return pd.DataFrame(rows)


class ShortScoreTests(unittest.TestCase):
    def _one(self, **over):
        return se.compute_short_entry_columns(_frame([_row(**over)])).iloc[0]

    def test_short_score_below_ma5_bonus(self):
        self.assertAlmostEqual(self._one(down_score=2.0, dist_ma5_intraday=-0.001)["short_score"], 3.0)
        # at/above MA5 gets no bonus (strict < 0)
        self.assertAlmostEqual(self._one(down_score=2.0, dist_ma5_intraday=0.0)["short_score"], 2.0)

    def test_upside_risk_components_at_boundaries(self):
        # every mirrored microstructure trigger at its inclusive boundary
        out = self._one(up_score=1.0, lower_shadow_1m=0.003, vwap_distance_1m=-0.010,
                        distance_from_day_low_1m=0.001, return_from_open_1m=-0.020,
                        volume_z_20m_1m=2.5)
        self.assertAlmostEqual(out["upside_risk_score"], 1.0 + 1.5 + 1.0 + 1.0 + 0.5)
        # all triggers just outside their thresholds contribute nothing
        out = self._one(up_score=1.0, lower_shadow_1m=0.0029, vwap_distance_1m=-0.0099,
                        distance_from_day_low_1m=0.0011, return_from_open_1m=-0.021,
                        volume_z_20m_1m=2.49)
        self.assertAlmostEqual(out["upside_risk_score"], 1.0)

    def test_capitulation_needs_at_low_and_drop_from_open(self):
        # at the day low but not down from the open -> no capitulation point
        out = self._one(up_score=1.0, distance_from_day_low_1m=0.0005, return_from_open_1m=-0.005)
        self.assertAlmostEqual(out["upside_risk_score"], 1.0)


class AlertTypeTests(unittest.TestCase):
    """Taxonomy priority: SHORT_RISK_ALERT > DOWN_WEAKENING_ALERT >
    DOWN_CONTINUATION > DOWN_ALERT > VOLATILITY_ALERT (mirror of the long side)."""

    def _type(self, **over):
        return se.compute_short_entry_columns(_frame([_row(**over)])).iloc[0]["alert_type"]

    def test_neutral_row_has_no_alert(self):
        self.assertEqual(self._type(), "")

    def test_short_risk_boundary(self):
        # upside_risk 3.0 dominating short_score 1.0 -> SHORT_RISK_ALERT
        self.assertEqual(self._type(up_score=3.0, down_score=1.0), "SHORT_RISK_ALERT")
        # score just below the cut-off -> no alert
        self.assertEqual(self._type(up_score=2.99, down_score=1.0), "")

    def test_short_risk_requires_dominance(self):
        # upside_risk 3.0 == short_score 3.0 (down 2 + MA5 bonus): risk must be
        # strictly greater than the down case -> not SHORT_RISK, nothing else fits
        self.assertEqual(self._type(up_score=3.0, down_score=2.0, dist_ma5_intraday=-0.001), "")

    def test_short_risk_beats_weakening(self):
        # fading (ret20<0, ret5>0) AND dominant upside risk -> SHORT_RISK_ALERT wins
        self.assertEqual(self._type(up_score=3.5, down_score=1.0,
                                    ret_20m_1m=-0.005, ret_5m_1m=0.001), "SHORT_RISK_ALERT")

    def test_short_risk_beats_down_alert(self):
        # satisfies DOWN_ALERT (direction -1.5, down 3.0) and weakening (wick+down>=3)
        # but the wick+stretch make upside risk 4.0 > short 3.0 -> SHORT_RISK_ALERT
        self.assertEqual(self._type(up_score=1.5, down_score=3.0,
                                    lower_shadow_1m=0.004, vwap_distance_1m=-0.012),
                         "SHORT_RISK_ALERT")

    def test_weakening_fade_branch_boundaries(self):
        # down move fading: was falling over 20m, bouncing over 5m
        self.assertEqual(self._type(ret_20m_1m=-0.005, ret_5m_1m=0.001), "DOWN_WEAKENING_ALERT")
        self.assertEqual(self._type(ret_20m_1m=0.0, ret_5m_1m=0.001), "")    # needs ret20 < 0
        self.assertEqual(self._type(ret_20m_1m=-0.005, ret_5m_1m=0.0), "")   # needs ret5 > 0

    def test_weakening_wick_branch_boundaries(self):
        # meaningfully bearish (down>=3) + support/bounce wick
        self.assertEqual(self._type(down_score=3.0, lower_shadow_1m=0.003), "DOWN_WEAKENING_ALERT")
        self.assertEqual(self._type(down_score=2.99, lower_shadow_1m=0.003), "")

    def test_weakening_beats_continuation(self):
        # full continuation setup + a support wick -> DOWN_WEAKENING_ALERT wins
        self.assertEqual(self._type(down_score=4.0, dist_ma5_intraday=-0.001,
                                    distance_from_day_low_1m=0.001, ret_5m_1m=-0.001,
                                    lower_shadow_1m=0.003), "DOWN_WEAKENING_ALERT")

    def test_continuation_boundaries(self):
        cont = dict(down_score=3.0, dist_ma5_intraday=-0.001,
                    distance_from_day_low_1m=0.005, ret_5m_1m=-0.001)
        # short_score 4.0 / near-low 0.005 / ret5 < 0, below MA5 -> continuation
        self.assertEqual(self._type(**cont), "DOWN_CONTINUATION")
        # each leg broken -> falls through to DOWN_ALERT (dir -2.0, down 3.0)
        self.assertEqual(self._type(**{**cont, "distance_from_day_low_1m": 0.0051}), "DOWN_ALERT")
        self.assertEqual(self._type(**{**cont, "ret_5m_1m": 0.0}), "DOWN_ALERT")
        self.assertEqual(self._type(**{**cont, "dist_ma5_intraday": 0.0}), "DOWN_ALERT")

    def test_down_alert_boundaries(self):
        # direction exactly -1.5 and down_score exactly 3.0 -> DOWN_ALERT
        self.assertEqual(self._type(down_score=3.0, up_score=1.5), "DOWN_ALERT")
        self.assertEqual(self._type(down_score=3.0, up_score=1.51), "")   # dir -1.49
        self.assertEqual(self._type(down_score=2.99, up_score=1.49), "")  # down < 3

    def test_volatility_boundaries(self):
        self.assertEqual(self._type(volume_z_20m_1m=2.0), "VOLATILITY_ALERT")   # volz leg
        self.assertEqual(self._type(range_1m=0.006), "VOLATILITY_ALERT")        # range leg
        self.assertEqual(self._type(volume_z_20m_1m=1.99, range_1m=0.0059), "")
        # |direction| must be strictly below 1.5
        self.assertEqual(self._type(volume_z_20m_1m=2.0, up_score=2.5, down_score=1.0), "")

    def test_down_alert_beats_volatility(self):
        self.assertEqual(self._type(down_score=3.5, up_score=1.0,
                                    volume_z_20m_1m=2.5, range_1m=0.007), "DOWN_ALERT")

    def test_priority_order_matches_long_mirror(self):
        self.assertEqual(se.DOWN_ALERT_TYPES,
                         ["SHORT_RISK_ALERT", "DOWN_WEAKENING_ALERT", "DOWN_CONTINUATION",
                          "DOWN_ALERT", "VOLATILITY_ALERT"])
        self.assertEqual(se.DOWN_TYPES, {"DOWN_ALERT", "DOWN_CONTINUATION"})


class ShortGradeTests(unittest.TestCase):
    def _one(self, **over):
        return se.compute_short_entry_columns(_frame([_row(**over)])).iloc[0]

    def test_grade_a_clean_strong_down(self):
        out = self._one(down_score=4.0, up_score=1.0, dist_ma5_intraday=-0.004,
                        ret_20m_1m=-0.01, relative_ret_20m_to_universe=-0.01)
        self.assertAlmostEqual(out["short_score"], 5.0)
        self.assertAlmostEqual(out["direction_score"], -3.0)   # boundary: -3.0 <= -A_MARGIN
        self.assertEqual(out["short_grade"], "A")

    def test_grade_a_requires_rel_weakness(self):
        out = self._one(down_score=4.0, up_score=1.0, dist_ma5_intraday=-0.004,
                        ret_20m_1m=-0.01, relative_ret_20m_to_universe=0.001)
        self.assertEqual(out["short_grade"], "B")   # everything A-grade except rel20 < 0

    def test_grade_b_boundaries(self):
        out = self._one(down_score=2.5, up_score=1.0, dist_ma5_intraday=-0.002)
        self.assertAlmostEqual(out["short_score"], 3.5)         # boundary: >= B_SCORE
        self.assertAlmostEqual(out["direction_score"], -1.5)    # boundary: <= -B_MARGIN
        self.assertEqual(out["short_grade"], "B")
        out = self._one(down_score=2.49, up_score=1.0, dist_ma5_intraday=-0.002)
        self.assertEqual(out["short_grade"], "C")               # short_score 3.49 < 3.5

    def test_grade_c_flat_row(self):
        self.assertEqual(self._one()["short_grade"], "C")

    def test_short_risk_grade_boundary(self):
        # upside_risk_score exactly at the cut-off -> grade SHORT_RISK
        out = self._one(up_score=3.0, down_score=5.0, dist_ma5_intraday=-0.002)
        self.assertAlmostEqual(out["upside_risk_score"], 3.0)
        self.assertEqual(out["short_grade"], "SHORT_RISK")
        # just below -> falls through (upside 2.99 also blocks B's <= 2.5 -> C)
        out = self._one(up_score=2.99, down_score=5.0, dist_ma5_intraday=-0.002)
        self.assertEqual(out["short_grade"], "C")

    def test_short_risk_grade_overrides_grade_a(self):
        # a would-be A row with a support wick + over-extension below VWAP; the
        # grade flips to SHORT_RISK even though the TYPE stays a weakening alert
        out = self._one(down_score=4.0, up_score=1.0, dist_ma5_intraday=-0.004,
                        relative_ret_20m_to_universe=-0.01,
                        lower_shadow_1m=0.004, vwap_distance_1m=-0.012)
        self.assertAlmostEqual(out["upside_risk_score"], 3.5)
        self.assertEqual(out["short_grade"], "SHORT_RISK")
        self.assertEqual(out["alert_type"], "DOWN_WEAKENING_ALERT")

    def test_grade_vocabulary(self):
        self.assertEqual(se.SHORT_GRADES, ["A", "B", "C", "SHORT_RISK"])


class ReasonTests(unittest.TestCase):
    def _one(self, **over):
        return se.compute_short_entry_columns(_frame([_row(**over)])).iloc[0]

    def test_reason_priority_bounce_warnings_first(self):
        # a bounce warning must outrank a (stronger) trend confirmation
        out = self._one(ret_20m_1m=-0.01, ret_5m_1m=-0.001, lower_shadow_1m=0.004)
        self.assertEqual(out["reason_1"], "support_bounce_wick")
        self.assertEqual(out["reason_2"], "downtrend_20m")
        self.assertEqual(out["reason_3"], "momentum_falling_5m")

    def test_reason_order_within_bounce_block(self):
        out = self._one(ret_20m_1m=0.004, lower_shadow_1m=0.004, vwap_distance_1m=-0.012,
                        dist_ma5_intraday=-0.001)
        self.assertEqual([out["reason_1"], out["reason_2"], out["reason_3"]],
                         ["uptrend_20m", "support_bounce_wick", "stretched_below_vwap"])

    def test_confirmation_reason_order(self):
        out = self._one(ret_20m_1m=-0.009, ret_5m_1m=-0.002, dist_ma5_intraday=-0.003)
        self.assertEqual([out["reason_1"], out["reason_2"], out["reason_3"]],
                         ["downtrend_20m", "momentum_falling_5m", "below_5ma"])

    def test_empty_frame_has_short_columns(self):
        out = se.compute_short_entry_columns(pd.DataFrame())
        for col in se.SHORT_ENTRY_COLUMNS:
            self.assertIn(col, out.columns)
        self.assertIn("alert_type", se.SHORT_ENTRY_COLUMNS)

    def test_missing_required_column_raises(self):
        bad = _frame([_row()]).drop(columns=["down_score"])
        with self.assertRaises(ValueError):
            se.compute_short_entry_columns(bad)


class ShortEntryGateTests(unittest.TestCase):
    """short_entry_gate is the unchanged DOWN-type threshold gate."""

    def _gate(self, rows, thr):
        cols = se.compute_short_entry_columns(_frame(rows))
        return se.short_entry_gate(cols, thr).tolist()

    def test_gate_respects_each_knob(self):
        rows = [
            _short_row(),                                    # fires
            _short_row(down_score=3.99),                     # below down_threshold
            _short_row(up_score=1.01, down_score=4.51),      # up above up_risk_limit (margin still ok)
            _short_row(rank_ret_20m_1m=15),                  # rank not in the bottom-5 bucket
        ]
        fired = self._gate(rows, se.ShortThresholds())
        self.assertEqual(fired, [True, False, False, False])

    def test_margin_boundary(self):
        # direction -3.0 passes margin 3.0 but fails a tightened 3.5
        self.assertEqual(self._gate([_short_row()], se.ShortThresholds(score_margin=3.5)), [False])
        self.assertEqual(self._gate([_short_row(down_score=4.5)],
                                    se.ShortThresholds(score_margin=3.5)), [True])

    def test_rank_boundary_is_strict(self):
        # bottom bucket = rank > active - limit = 15; 16 passes, 15 does not
        self.assertEqual(self._gate([_short_row(rank_ret_20m_1m=16)], se.ShortThresholds()), [True])
        self.assertEqual(self._gate([_short_row(rank_ret_20m_1m=15)], se.ShortThresholds()), [False])

    def test_none_skips_up_risk_limit(self):
        rows = [_short_row(up_score=1.01, down_score=4.51)]
        self.assertEqual(self._gate(rows, se.ShortThresholds(up_risk_limit=None)), [True])

    def test_none_skips_bottom_rank_limit(self):
        rows = [_short_row(rank_ret_20m_1m=10)]
        self.assertEqual(self._gate(rows, se.ShortThresholds(bottom_rank_limit=None)), [True])

    def test_neutral_mid_rank_never_fires_default_gate(self):
        # documentation case: single-symbol serving fills the NEUTRAL mid rank
        # (ceil(20/2)=10), which never reaches the bottom bucket -- so the rank knob
        # blocks EVERY row in non-cross mode and must be disabled (None) there.
        rows = [_short_row(rank_ret_20m_1m=10)]
        self.assertEqual(self._gate(rows, se.ShortThresholds()), [False])

    def test_optional_confirms(self):
        below = se.ShortThresholds(require_below_ma5=True)
        self.assertEqual(self._gate([_short_row(dist_ma5_intraday=0.001)], below), [False])
        self.assertEqual(self._gate([_short_row(dist_ma5_intraday=-0.001)], below), [True])
        rel = se.ShortThresholds(require_rel_weakness=True)
        self.assertEqual(self._gate([_short_row(relative_ret_20m_to_universe=0.001)], rel), [False])
        self.assertEqual(self._gate([_short_row(relative_ret_20m_to_universe=-0.001)], rel), [True])

    def test_nan_inputs_never_fire(self):
        self.assertEqual(self._gate([_short_row(rank_ret_20m_1m=float("nan"))],
                                    se.ShortThresholds()), [False])


class ShortAlertFireGateTests(unittest.TestCase):
    """short_alert_fire_gate mirrors directional_alert.alert_fire_gate: DOWN
    types via the threshold gate, informational types via fixed conditions."""

    def _fire(self, rows, thr=None):
        cols = se.compute_short_entry_columns(_frame(rows))
        return se.short_alert_fire_gate(cols, thr or se.ShortThresholds()).tolist()

    def test_down_alert_fires_through_threshold_gate(self):
        row = _short_row()
        typ = se.compute_short_entry_columns(_frame([row])).iloc[0]["alert_type"]
        self.assertEqual(typ, "DOWN_ALERT")
        self.assertEqual(self._fire([row]), [True])

    def test_down_continuation_fires_through_threshold_gate(self):
        row = _short_row(distance_from_day_low_1m=0.004)
        typ = se.compute_short_entry_columns(_frame([row])).iloc[0]["alert_type"]
        self.assertEqual(typ, "DOWN_CONTINUATION")
        self.assertEqual(self._fire([row]), [True])

    def test_down_types_require_threshold_gate(self):
        # DOWN_ALERT typed but below down_threshold / outside the bottom-rank bucket
        self.assertEqual(self._fire([_short_row(down_score=3.99)]), [False])
        self.assertEqual(self._fire([_short_row(rank_ret_20m_1m=10)]), [False])

    def test_informational_types_bypass_threshold_gate(self):
        # weakening row failing the rank knob still fires (fixed gate, not swept)
        row = _short_row(lower_shadow_1m=0.003, rank_ret_20m_1m=10)
        typ = se.compute_short_entry_columns(_frame([row])).iloc[0]["alert_type"]
        self.assertEqual(typ, "DOWN_WEAKENING_ALERT")
        self.assertEqual(self._fire([row]), [True])

    def test_short_risk_fire_boundaries(self):
        # upside_risk >= 4.0 AND direction >= +3.0
        self.assertEqual(self._fire([_row(up_score=4.0, down_score=1.0)]), [True])
        self.assertEqual(self._fire([_row(up_score=3.99, down_score=0.99)]), [False])  # risk 3.99
        self.assertEqual(self._fire([_row(up_score=4.0, down_score=1.01)]), [False])   # dir 2.99

    def test_weakening_fire_boundary(self):
        # fade-typed rows fire only when the down case was meaningful (down >= 3)
        weak = dict(ret_20m_1m=-0.005, ret_5m_1m=0.001)
        self.assertEqual(self._fire([_row(down_score=3.0, **weak)]), [True])
        self.assertEqual(self._fire([_row(down_score=2.99, **weak)]), [False])

    def test_volatility_fire_boundaries(self):
        # needs BOTH volume_z >= 2.5 and range >= 0.006
        self.assertEqual(self._fire([_row(volume_z_20m_1m=2.5, range_1m=0.006)]), [True])
        self.assertEqual(self._fire([_row(volume_z_20m_1m=2.49, range_1m=0.007)]), [False])
        self.assertEqual(self._fire([_row(volume_z_20m_1m=2.6, range_1m=0.0059)]), [False])

    def test_volatility_without_range_column_never_fires(self):
        cols = se.compute_short_entry_columns(
            _frame([_row(volume_z_20m_1m=2.6)]).drop(columns=["range_1m"]))
        self.assertEqual(cols.iloc[0]["alert_type"], "VOLATILITY_ALERT")
        self.assertEqual(se.short_alert_fire_gate(cols, se.ShortThresholds()).tolist(), [False])

    def test_untyped_row_never_fires(self):
        self.assertEqual(self._fire([_row()]), [False])


def _fire_row(dt="2026-01-05 10:00:00", **over):
    r = {"decision_time": dt, "symbol": "TSTA", "close": 10000.0,
         "short_fires_rule": True, "alert_type": "DOWN_ALERT", "short_grade": "B",
         "up_score": 1.0, "down_score": 4.0, "direction_score": -3.0,
         "short_score": 5.0, "upside_risk_score": 1.0, "short_adverse_prob": 0.30,
         "reason_1": "downtrend_20m", "reason_2": "below_5ma", "reason_3": ""}
    r.update(over)
    return r


class ShortFireGateTests(unittest.TestCase):
    def _gate(self, **thr_over):
        return se.ShortFireGate("TSTA", se.ShortThresholds(**thr_over))

    def test_default_is_annotation_only(self):
        # short_adverse_prob_max defaults to None: no veto, NaN prob still fires
        self.assertIsNone(se.ShortThresholds().short_adverse_prob_max)
        alert = self._gate().maybe_fire(_fire_row(), short_adverse_prob=float("nan"))
        self.assertIsNotNone(alert)
        self.assertEqual(alert["short_risk_band"], "NA")

    def test_configured_veto_boundaries_on_down_types(self):
        # explicit veto 0.45: 0.44 fires, 0.45 is vetoed (>=), on BOTH down types
        for typ in sorted(se.DOWN_TYPES):
            g = self._gate(short_adverse_prob_max=0.45)
            self.assertIsNotNone(g.maybe_fire(_fire_row(alert_type=typ),
                                              short_adverse_prob=0.44))
            g = self._gate(short_adverse_prob_max=0.45)
            self.assertIsNone(g.maybe_fire(_fire_row(alert_type=typ),
                                           short_adverse_prob=0.45))

    def test_configured_veto_nan_is_conservative_on_down_types(self):
        # an unknown probability must NOT fire a DOWN row while the veto is set
        self.assertIsNone(self._gate(short_adverse_prob_max=0.45).maybe_fire(
            _fire_row(), short_adverse_prob=float("nan")))

    def test_informational_types_are_never_vetoed(self):
        for typ in ("SHORT_RISK_ALERT", "DOWN_WEAKENING_ALERT", "VOLATILITY_ALERT"):
            g = self._gate(short_adverse_prob_max=0.45)
            alert = g.maybe_fire(_fire_row(alert_type=typ), short_adverse_prob=0.90)
            self.assertIsNotNone(alert, typ)
            self.assertEqual(alert["alert_type"], typ)
            g = self._gate(short_adverse_prob_max=0.45)
            self.assertIsNotNone(g.maybe_fire(_fire_row(alert_type=typ),
                                              short_adverse_prob=float("nan")), typ)

    def test_veto_uses_argument_not_row_column(self):
        # the caller's score is authoritative even if the row carries a stale column
        alert = self._gate(short_adverse_prob_max=0.45).maybe_fire(
            _fire_row(short_adverse_prob=0.90), short_adverse_prob=0.30)
        self.assertIsNotNone(alert)
        self.assertAlmostEqual(alert["short_adverse_prob"], 0.30)

    def test_untyped_row_never_fires(self):
        self.assertIsNone(self._gate().maybe_fire(
            _fire_row(alert_type=""), short_adverse_prob=0.30))

    def test_session_window(self):
        self.assertIsNone(self._gate().maybe_fire(
            _fire_row("2026-01-05 09:09:00"), short_adverse_prob=0.30))
        self.assertIsNotNone(self._gate().maybe_fire(
            _fire_row("2026-01-05 09:10:00"), short_adverse_prob=0.30))
        self.assertIsNone(self._gate().maybe_fire(
            _fire_row("2026-01-05 15:00:00"), short_adverse_prob=0.30))
        self.assertIsNotNone(self._gate().maybe_fire(
            _fire_row("2026-01-05 14:59:00"), short_adverse_prob=0.30))

    def test_rule_flag_required(self):
        self.assertIsNone(self._gate().maybe_fire(
            _fire_row(short_fires_rule=False), short_adverse_prob=0.30))

    def test_per_type_cooldown(self):
        g = self._gate()
        # same type within cooldown suppressed, +15m fires again
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:00:00"), short_adverse_prob=0.3))
        self.assertIsNone(g.maybe_fire(_fire_row("2026-01-05 10:14:00"), short_adverse_prob=0.3))
        # a DIFFERENT type fires immediately (per-type state, mirror of the engine)
        self.assertIsNotNone(g.maybe_fire(
            _fire_row("2026-01-05 10:14:00", alert_type="DOWN_WEAKENING_ALERT"),
            short_adverse_prob=0.3))
        # ... and that other type now has its own cooldown
        self.assertIsNone(g.maybe_fire(
            _fire_row("2026-01-05 10:20:00", alert_type="DOWN_WEAKENING_ALERT"),
            short_adverse_prob=0.3))
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:15:00"), short_adverse_prob=0.3))

    def test_cooldown_scope_single_slot_vs_per_type(self):
        # sequence A@10:00, B@10:05, A@10:10 -- under single_slot B's fire REPLACES
        # the slot, so A re-fires at 10:10 even though its own 15 min have not elapsed
        g = se.ShortFireGate("TSTA", se.ShortThresholds(), cooldown_scope="single_slot")
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:00:00"),
                                          short_adverse_prob=0.3))
        self.assertIsNotNone(g.maybe_fire(
            _fire_row("2026-01-05 10:05:00", alert_type="DOWN_WEAKENING_ALERT"),
            short_adverse_prob=0.3))
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:10:00"),
                                          short_adverse_prob=0.3))
        # the same sequence under per_type (the default): A@10:10 is SUPPRESSED
        # because A's own per-type timer (10:00 + 15m) has not elapsed
        g = se.ShortFireGate("TSTA", se.ShortThresholds())
        self.assertEqual(g.cooldown_scope, "per_type")
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:00:00"),
                                          short_adverse_prob=0.3))
        self.assertIsNotNone(g.maybe_fire(
            _fire_row("2026-01-05 10:05:00", alert_type="DOWN_WEAKENING_ALERT"),
            short_adverse_prob=0.3))
        self.assertIsNone(g.maybe_fire(_fire_row("2026-01-05 10:10:00"),
                                       short_adverse_prob=0.3))

    def test_single_slot_same_type_still_cooled_down(self):
        # single_slot keeps the engine's same-type suppression: A@10:00 fires,
        # A@10:14 suppressed (within 15m), A@10:15 fires again
        g = se.ShortFireGate("TSTA", se.ShortThresholds(), cooldown_scope="single_slot")
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:00:00"),
                                          short_adverse_prob=0.3))
        self.assertIsNone(g.maybe_fire(_fire_row("2026-01-05 10:14:00"),
                                       short_adverse_prob=0.3))
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:15:00"),
                                          short_adverse_prob=0.3))

    def test_invalid_cooldown_scope_raises(self):
        with self.assertRaises(ValueError):
            se.ShortFireGate("TSTA", se.ShortThresholds(), cooldown_scope="both")

    def test_daily_limit_counts_all_types_and_day_reset(self):
        g = self._gate(daily_alert_limit=2)
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:00:00"), short_adverse_prob=0.3))
        self.assertIsNotNone(g.maybe_fire(
            _fire_row("2026-01-05 10:01:00", alert_type="DOWN_WEAKENING_ALERT"),
            short_adverse_prob=0.3))
        self.assertIsNone(g.maybe_fire(
            _fire_row("2026-01-05 10:02:00", alert_type="VOLATILITY_ALERT"),
            short_adverse_prob=0.3))
        # new date -> daily count and cooldowns auto-reset
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-06 09:30:00"), short_adverse_prob=0.3))

    def test_vetoed_row_does_not_consume_cooldown_or_limit(self):
        g = self._gate(short_adverse_prob_max=0.45, daily_alert_limit=1)
        self.assertIsNone(g.maybe_fire(_fire_row("2026-01-05 10:00:00"), short_adverse_prob=0.9))
        self.assertIsNotNone(g.maybe_fire(_fire_row("2026-01-05 10:01:00"), short_adverse_prob=0.3))

    def test_emitted_dict_keys_and_message(self):
        alert = self._gate().maybe_fire(_fire_row(), short_adverse_prob=0.30)
        self.assertEqual(set(alert.keys()), {
            "time", "symbol", "close", "alert_type", "short_grade",
            "up_score", "down_score", "direction_score", "short_score",
            "upside_risk_score", "short_adverse_prob", "short_risk_band",
            "reason_1", "reason_2", "reason_3", "thresholds_name", "message",
        })
        self.assertEqual(alert["alert_type"], "DOWN_ALERT")
        self.assertEqual(alert["thresholds_name"], "mirror_baseline")
        self.assertEqual(alert["short_risk_band"], "LOW")
        self.assertIn("하방 가능성 [DOWN_ALERT B]", alert["message"])
        self.assertIn("short_adverse_prob=0.30(LOW)", alert["message"])
        self.assertIn("✓ 저위험 하방", alert["message"])
        self.assertIn("reasons=downtrend_20m/below_5ma", alert["message"])


class BandMessageTests(unittest.TestCase):
    def test_short_risk_band(self):
        self.assertEqual(se.short_risk_band(float("nan")), "NA")
        self.assertEqual(se.short_risk_band(0.60), "HIGH")
        self.assertEqual(se.short_risk_band(0.59), "MED")
        self.assertEqual(se.short_risk_band(0.45), "MED")
        self.assertEqual(se.short_risk_band(0.44), "LOW")

    def test_message_heads_per_type(self):
        heads = {
            "DOWN_ALERT": "하방 가능성", "DOWN_CONTINUATION": "하방 유지 가능성",
            "SHORT_RISK_ALERT": "상승 위험", "DOWN_WEAKENING_ALERT": "하락 추세 약화",
            "VOLATILITY_ALERT": "변동성 확대",
        }
        for typ, head in heads.items():
            msg = se.short_message(_fire_row(alert_type=typ, short_adverse_prob=0.50))
            self.assertTrue(msg.startswith(f"{head} [{typ} B]"), msg)
            self.assertIn("short_adverse_prob=0.50(MED)", msg)

    def test_down_type_model_notes(self):
        for typ in sorted(se.DOWN_TYPES):
            low = se.short_message(_fire_row(alert_type=typ, short_adverse_prob=0.20))
            self.assertIn("✓ 저위험 하방", low)
            high = se.short_message(_fire_row(alert_type=typ, short_adverse_prob=0.70))
            self.assertIn("⚠ 모델 상승위험: 하방이지만 반등 주의", high)
            med = se.short_message(_fire_row(alert_type=typ, short_adverse_prob=0.50))
            self.assertNotIn("✓", med)
            self.assertNotIn("⚠", med)

    def test_informational_types_have_no_model_note(self):
        for typ in ("SHORT_RISK_ALERT", "DOWN_WEAKENING_ALERT", "VOLATILITY_ALERT"):
            for p in (0.20, 0.70):
                msg = se.short_message(_fire_row(alert_type=typ, short_adverse_prob=p))
                self.assertNotIn("✓", msg, typ)
                self.assertNotIn("⚠", msg, typ)


class ScoreServingShortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = _make_window()
        cls.serv = build_serving_features(cls.window, source=SOURCE, active_count=20)
        feats = [c for c in FEATURE_COLUMNS if c in cls.serv.columns]
        train = cls.serv.dropna(subset=feats)
        assert len(train) > 50
        cls.pre = FeaturePreprocessor(feats).fit(train)
        y = (train["ret_5m_1m"].fillna(0) < 0).astype(int).to_numpy()  # synthetic target
        model = NumpyLogisticRiskModel(max_iter=40).fit(cls.pre.transform(train), y)
        model.model_type = NumpyLogisticRiskModel.model_type
        cls.model = model
        # single-symbol serving has neutral ranks -> rank knob must be disabled
        cls.thr = se.ShortThresholds(bottom_rank_limit=None)
        cls.res = se.score_serving_short(cls.serv, cls.model, cls.pre, thresholds=cls.thr)

    def test_schema_and_time_order(self):
        self.assertGreater(len(self.res), 0)
        self.assertEqual(list(self.res.columns), se.SHORT_RESULT_COLUMNS)
        self.assertIn("alert_type", se.SHORT_RESULT_COLUMNS)
        self.assertTrue(self.res["decision_time"].is_monotonic_increasing)
        self.assertTrue(((self.res["short_adverse_prob"] >= 0) &
                         (self.res["short_adverse_prob"] <= 1)).all())
        # warm-up rows (NaN rule inputs) are dropped -> every row is scorable
        self.assertFalse(self.res["down_score"].isna().any())
        self.assertTrue(self.res["short_grade"].isin(se.SHORT_GRADES).all())
        self.assertTrue(self.res["alert_type"].isin([""] + se.DOWN_ALERT_TYPES).all())
        self.assertEqual(self.res["short_fires_rule"].dtype, bool)

    def test_fires_only_on_typed_rows(self):
        fired = self.res[self.res["short_fires_rule"]]
        if len(fired):
            self.assertTrue(fired["alert_type"].isin(se.DOWN_ALERT_TYPES).all())

    def test_none_model_yields_nan_prob_only(self):
        rule_only = se.score_serving_short(self.serv, None, None, thresholds=self.thr)
        self.assertTrue(rule_only["short_adverse_prob"].isna().all())
        pd.testing.assert_frame_equal(
            rule_only.drop(columns=["short_adverse_prob"]),
            self.res.drop(columns=["short_adverse_prob"]))

    def test_leakage_token_guard_rejects_future_features(self):
        serv = self.serv.copy()
        serv["future_MAE"] = 0.0
        bad_pre = FeaturePreprocessor(self.pre.feature_columns + ["future_MAE"])
        with self.assertRaises(ValueError):
            se.score_serving_short(serv, self.model, bad_pre, thresholds=self.thr)

    def test_empty_input_returns_empty_schema(self):
        out = se.score_serving_short(pd.DataFrame(), None, None, thresholds=self.thr)
        self.assertEqual(len(out), 0)
        self.assertEqual(list(out.columns), se.SHORT_RESULT_COLUMNS)

    def test_causal_truncation_invariance(self):
        # scoring bars truncated at time t must reproduce the full-day batch row at t
        # (decision_time = bar_end; any mismatch would be a 1-minute leak).
        last_day = self.window["bar_start"].str.slice(0, 10).max()
        day_bars = self.window[self.window["bar_start"].str.slice(0, 10) == last_day]
        prior = self.window[self.window["bar_start"].str.slice(0, 10) < last_day]
        cut = pd.concat([prior, day_bars.iloc[:90]], ignore_index=True)
        serv_cut = build_serving_features(cut, source=SOURCE, active_count=20)
        cut_res = se.score_serving_short(serv_cut, self.model, self.pre, thresholds=self.thr)
        self.assertGreater(len(cut_res), 0)
        dt = cut_res["decision_time"].iloc[-1]
        fr = self.res[self.res["decision_time"] == dt].iloc[0]
        cr = cut_res.iloc[-1]
        for col in se.SHORT_RESULT_COLUMNS:
            a, b = fr[col], cr[col]
            if isinstance(a, str):
                self.assertEqual(a, b, f"{col} changed when later bars were added")
            else:
                a, b = float(a), float(b)
                if math.isnan(a) and math.isnan(b):
                    continue
                self.assertAlmostEqual(a, b, places=9,
                                       msg=f"{col} changed when later bars were added")
        # the truncated feed is an exact PREFIX of the full batch (no future influence)
        prefix = self.res.iloc[:len(cut_res)].reset_index(drop=True)
        pd.testing.assert_frame_equal(cut_res, prefix, check_dtype=False)


class GuardTests(unittest.TestCase):
    """Static text guards: decision-support only, no hardcoded symbol, and the
    trend-alert reframe (no GO/NO-GO machinery left in the module)."""

    SCRIPT_NAMES = [
        "realtime_short_trend_alert.py",
        "realtime_long_short_trend_alert.py",
        "build_short_replay_alerts.py",
        "short_gate_sweep.py",
        "evaluate_short_entry_gate.py",
    ]

    def test_no_order_api_references(self):
        module_path = SRC_DIR / "short_entry.py"
        self.assertTrue(module_path.exists(), "src/short_entry.py must exist")
        paths = [module_path]
        for name in self.SCRIPT_NAMES:      # later build steps; skip the missing ones
            p = SCRIPTS_DIR / name
            if p.exists():
                paths.append(p)
        for p in paths:
            text = p.read_text(encoding="utf-8")
            for forbidden in ("SendOrder", "send_order", "sendorder", "place_order"):
                self.assertNotIn(forbidden, text, f"{p.name} must not reference {forbidden}")

    def test_no_hardcoded_symbol_literal(self):
        text = (SRC_DIR / "short_entry.py").read_text(encoding="utf-8")
        self.assertNotIn("000660", text, "src modules must not hardcode a symbol code")

    def test_mandated_naming_is_used(self):
        # user-mandated names; the shortened variants are forbidden everywhere
        text = (SRC_DIR / "short_entry.py").read_text(encoding="utf-8")
        self.assertIn("short_adverse_prob", text)
        self.assertIn("short_risk_band", text)
        self.assertNotIn("short_prob", text.replace("short_adverse_prob", ""))
        self.assertNotIn("short_band", text)

    def test_go_no_go_machinery_removed(self):
        # trend alerts need no GO/NO-GO: the validation flag and the single
        # entry-type constant are gone from the module
        self.assertFalse(hasattr(se, "DEFAULTS_VALIDATED"))
        self.assertFalse(hasattr(se, "SHORT_ENTRY_TYPE"))
        text = (SRC_DIR / "short_entry.py").read_text(encoding="utf-8")
        self.assertNotIn("DEFAULTS_VALIDATED", text)
        self.assertNotIn("미검증", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
