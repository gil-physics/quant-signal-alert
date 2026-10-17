"""Tests for the MA5x20 1-minute crossover study (src/ma_cross_study.py).

Hand-computed truths on tiny synthetic frames: regime-flip cross detection (incl.
the zero-touch ghost-recross guard and NaN regime breaks), warmup-variant timelines,
timestamp-bounded outcome windows (gap semantics, truncation), directional metric
math for long/short, whipsaw/resume timing boundaries, cluster de-duplication,
matched-baseline bounds, bands/ATR, and the static scan of the analysis script.
"""

from __future__ import annotations

import re
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

import ma_cross_study as mcs  # noqa: E402


def _bars(rows, symbol="TESTSYM"):
    """rows: list of (bar_start, open, high, low, close)."""
    return pd.DataFrame({
        "symbol": [symbol] * len(rows),
        "bar_start": [r[0] for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
    })


def _seq(date, start_hhmm, opens, closes=None, gap_pad=1.0):
    """Consecutive 1-minute bars from start_hhmm; high/low bracket open/close."""
    hh, mm = int(start_hhmm[:2]), int(start_hhmm[3:5])
    closes = list(opens) if closes is None else list(closes)
    rows = []
    for i, (o, c) in enumerate(zip(opens, closes)):
        t = hh * 60 + mm + i
        rows.append((f"{date} {t // 60:02d}:{t % 60:02d}:00",
                     o, max(o, c) + gap_pad, min(o, c) - gap_pad, c))
    return rows


def _d_frame(d_values, times=None, symbol="TESTSYM", date="2026-01-05"):
    """Minimal frame for detect_crossovers with an injected d column."""
    n = len(d_values)
    times = times or [f"{9 + i // 60:02d}:{i % 60:02d}:00" for i in range(n)]
    return pd.DataFrame({
        "symbol": [symbol] * n, "date": [date] * n, "time": times,
        "d": [float(v) if v is not None else np.nan for v in d_values],
    })


class RegimeCrossTests(unittest.TestCase):
    G = ["symbol", "date"]

    def _crosses(self, d_values, **kw):
        out = mcs.detect_crossovers(_d_frame(d_values, **kw), self.G)
        return out["cross"].tolist()

    def test_basic_up_and_down(self):
        self.assertEqual(self._crosses([-1, -1, 2]), [0, 0, 1])
        self.assertEqual(self._crosses([1, 1, -2]), [0, 0, -1])

    def test_zero_touch_same_side_is_not_a_recross(self):
        # -,0,0,+ -> ONE up cross at the +; -,0,- -> nothing (the ghost-recross guard)
        self.assertEqual(self._crosses([-1, 0, 0, 2]), [0, 0, 0, 1])
        self.assertEqual(self._crosses([-1, 0, -1]), [0, 0, 0])
        self.assertEqual(self._crosses([1, 0, 1]), [0, 0, 0])

    def test_no_cross_without_established_regime(self):
        # zeros (or NaN warmup) then + : no prior non-zero sign -> no cross
        self.assertEqual(self._crosses([0, 0, 2]), [0, 0, 0])
        self.assertEqual(self._crosses([None, None, 2]), [0, 0, 0])

    def test_nan_breaks_the_regime(self):
        self.assertEqual(self._crosses([-1, None, 2]), [0, 0, 0])       # break -> no cross
        self.assertEqual(self._crosses([-1, None, -1, 2]), [0, 0, 0, 1])  # re-established

    def test_eligibility_and_session(self):
        d = [-1, -1, 2, 2, 2]
        times = ["08:30:00", "08:31:00", "08:32:00", "15:29:00", "15:30:00"]
        reg = mcs.detect_crossovers(_d_frame(d, times=times), self.G)
        self.assertEqual(reg["eligible"].tolist(), [False, False, False, True, False])
        pre = mcs.detect_crossovers(_d_frame(d, times=times), self.G,
                                    signal_start=mcs.PREMARKET_START)
        self.assertEqual(pre["eligible"].tolist(), [True, True, True, True, False])
        self.assertEqual(pre["session"].tolist(),
                         ["premarket", "premarket", "premarket", "regular", "regular"])

    def test_variant_signal_start(self):
        self.assertEqual(mcs.variant_signal_start("regular"), mcs.REGULAR_START)
        self.assertEqual(mcs.variant_signal_start("preopen"), mcs.PREMARKET_START)
        self.assertEqual(mcs.variant_signal_start("carry"), mcs.PREMARKET_START)


class PipelineCrossTests(unittest.TestCase):
    def _reference_crosses(self, prices, fast=5, slow=20):
        """Independent slow-path implementation of the regime-cross convention."""
        crosses = [0] * len(prices)
        regime = None
        for i in range(len(prices)):
            if i + 1 < slow:
                continue
            ma_f = sum(prices[i - fast + 1:i + 1]) / fast
            ma_s = sum(prices[i - slow + 1:i + 1]) / slow
            d = ma_f - ma_s
            sign = 1 if d > 0 else (-1 if d < 0 else 0)
            if sign != 0:
                if regime is not None and regime != sign:
                    crosses[i] = sign
                regime = sign
        return crosses

    def test_pipeline_matches_reference(self):
        # 100-flat (d==0, NO regime yet), dip to 95 (regime -1 established, NOT a
        # cross), jump to 120 (up cross), fall to 90 (down cross).
        opens = [100.0] * 25 + [95.0] * 5 + [120.0] * 10 + [90.0] * 10
        bars = _bars(_seq("2026-01-05", "09:00", opens))
        vf, gcols = mcs.prepare_variant(bars, "regular")
        mf = mcs.add_mas_and_atr(vf, gcols, price_col="open")
        cf = mcs.detect_crossovers(mf, gcols)
        self.assertEqual(cf["cross"].tolist(), self._reference_crosses(opens))
        fired = cf.loc[cf["cross"] != 0, "cross"].tolist()
        self.assertEqual(fired, [1, -1])            # up then down; dip was regime setup
        self.assertTrue(cf["d"].iloc[:19].isna().all())             # MA20 warmup
        first_cross = int(np.nonzero(cf["cross"].to_numpy())[0][0])
        self.assertGreaterEqual(first_cross, 20)

    def test_p1_open_vs_p2_close_sources_differ(self):
        opens = [100.0] * 40
        closes = [100.0] * 25 + [95.0] * 5 + [120.0] * 10
        bars = _bars(_seq("2026-01-05", "09:00", opens, closes))
        vf, gcols = mcs.prepare_variant(bars, "regular")
        on_open = mcs.detect_crossovers(
            mcs.add_mas_and_atr(vf, gcols, price_col="open"), gcols)
        on_close = mcs.detect_crossovers(
            mcs.add_mas_and_atr(vf, gcols, price_col="close"), gcols)
        self.assertEqual(int((on_open["cross"] != 0).sum()), 0)     # flat opens: d == 0
        self.assertGreater(int((on_close["cross"] != 0).sum()), 0)

    def test_invalid_bar_suppresses_signals_in_its_window_span(self):
        opens = [100.0] * 25 + [95.0] * 5 + [120.0] * 10
        rows = _seq("2026-01-05", "09:00", opens)
        bars = _bars(rows)
        bars.loc[27, "high"] = bars.loc[27, "low"] - 1               # ohlc violation
        vf, gcols = mcs.prepare_variant(bars, "regular")
        mf = mcs.add_mas_and_atr(vf, gcols, price_col="open")
        self.assertTrue(bool(mf.loc[27, "bar_invalid"]))
        self.assertTrue(mf["d"].iloc[27:47].isna().all())           # slow window poisoned
        cf = mcs.detect_crossovers(mf, gcols)
        self.assertEqual(int((cf["cross"] != 0).sum()), 0)          # regime broken


class VariantWarmupTests(unittest.TestCase):
    def _two_day(self, premarket_bars):
        day1 = _seq("2026-01-05", "09:00", [100.0] * 30)
        pre = _seq("2026-01-06", "08:55", [100.0] * premarket_bars)[:premarket_bars]
        day2 = _seq("2026-01-06", "09:00", [100.0] * 30)
        return _bars(day1 + pre + day2)

    def _first_valid_time(self, bars, variant, day):
        vf, gcols = mcs.prepare_variant(bars, variant)
        mf = mcs.add_mas_and_atr(vf, gcols, price_col="open")
        day_rows = mf[mf["date"] == day]
        ok = day_rows[day_rows["ma_slow"].notna()]
        return ok["time"].iloc[0] if len(ok) else None

    def test_warmup_coverage_ordering(self):
        bars = self._two_day(premarket_bars=5)     # thin premarket: 08:55..08:59
        self.assertEqual(self._first_valid_time(bars, "regular", "2026-01-06"),
                         "09:19:00")               # 20th regular bar
        self.assertEqual(self._first_valid_time(bars, "preopen", "2026-01-06"),
                         "09:14:00")               # 5 premarket bars warm 5 minutes
        self.assertEqual(self._first_valid_time(bars, "carry", "2026-01-06"),
                         "08:55:00")               # previous day carries the window

    def test_carry_crosses_can_fire_in_premarket(self):
        # trend flip engineered around the day-2 premarket
        day1 = _seq("2026-01-05", "09:00", [100.0] * 20 + [90.0] * 10)
        pre = _seq("2026-01-06", "08:55", [130.0] * 5)
        day2 = _seq("2026-01-06", "09:00", [130.0] * 10)
        bars = _bars(day1 + pre + day2)
        vf, gcols = mcs.prepare_variant(bars, "carry")
        cf = mcs.detect_crossovers(mcs.add_mas_and_atr(vf, gcols, price_col="open"),
                                   gcols, signal_start=mcs.PREMARKET_START)
        pre_crosses = cf[(cf["cross"] != 0) & (cf["session"] == "premarket")
                         & cf["eligible"]]
        self.assertGreater(len(pre_crosses), 0)


class OutcomeWindowTests(unittest.TestCase):
    def _stream(self, rows):
        return mcs.per_bar_forward_stats(
            mcs.build_outcome_stream(_bars(rows)), windows=(3,))

    def test_hand_truth_long_and_short(self):
        # closes: 100, 101, 100.8, 99, ...; highs/lows bracket by +-1
        rows = _seq("2026-01-05", "09:00", [100.0, 101.0, 100.8, 99.0, 98.0],
                    closes=[100.0, 101.0, 100.8, 99.0, 98.0])
        pb = self._stream(rows)
        j0 = pb.iloc[[0]]
        long_m = mcs.directional_outcomes(j0, 1.0, (3,), threshold=0.007)
        # window bars 0..2: max close 101 -> +1.0% >= 0.7% touch
        self.assertEqual(float(long_m["w3_close_touch"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(long_m["w3_endpoint_ret"].iloc[0]),
                               100.8 / 100.0 - 1)
        self.assertAlmostEqual(float(long_m["w3_mae"].iloc[0]),
                               99.0 / 100.0 - 1)      # min low = 100-1
        self.assertAlmostEqual(float(long_m["w3_mfe"].iloc[0]),
                               102.0 / 100.0 - 1)     # max high = 101+1
        short_m = mcs.directional_outcomes(j0, -1.0, (3,), threshold=0.007)
        self.assertEqual(float(short_m["w3_close_touch"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(short_m["w3_mae"].iloc[0]),
                               -(102.0 / 100.0 - 1))  # adverse = max high

    def test_truncated_window_excluded(self):
        rows = _seq("2026-01-05", "09:00", [100.0] * 5)
        pb = self._stream(rows)
        self.assertFalse(bool(pb["complete_w3"].iloc[3]))   # 09:03 + 2min > 09:04
        self.assertTrue(bool(pb["complete_w3"].iloc[2]))
        m = mcs.directional_outcomes(pb.iloc[[3]], 1.0, (3,))
        self.assertTrue(np.isnan(float(m["w3_close_touch"].iloc[0])))

    def test_gap_windows_are_timestamp_bounded(self):
        # bars 09:00, 09:01, then 09:05 (thin premarket-like gap)
        rows = [("2026-01-05 09:00:00", 100.0, 101.0, 99.0, 100.0),
                ("2026-01-05 09:01:00", 100.0, 106.0, 99.0, 105.0),
                ("2026-01-05 09:05:00", 100.0, 200.0, 99.0, 190.0)]
        pb = self._stream(rows)
        # W=3 at j=0: [09:00, 09:03) holds bars 0,1 ONLY -- the 09:05 spike stays out
        self.assertAlmostEqual(float(pb["fwd_max_close_w3"].iloc[0]), 105.0)
        self.assertAlmostEqual(float(pb["fwd_max_high_w3"].iloc[0]), 106.0)
        self.assertAlmostEqual(float(pb["endpoint_close_w3"].iloc[0]), 105.0)
        # complete: 09:00 + 2min = 09:02 <= last bar 09:05 -> observable
        self.assertTrue(bool(pb["complete_w3"].iloc[0]))

    def test_auction_bar_in_window_but_incomplete_as_anchor(self):
        rows = _seq("2026-01-05", "15:27", [100.0] * 4)     # 15:27..15:30
        pb = self._stream(rows)
        self.assertEqual(pb["time"].iloc[-1], "15:30:00")
        self.assertTrue(bool(pb["complete_w3"].iloc[0]))    # 15:27+2 <= 15:30
        self.assertFalse(bool(pb["complete_w3"].iloc[-1]))  # auction anchor truncated


class WhipsawResumeTests(unittest.TestCase):
    def _events(self, specs):
        # specs: (bar_start, direction)
        return pd.DataFrame({
            "symbol": ["TESTSYM"] * len(specs),
            "date": [s[0][:10] for s in specs],
            "bar_start": [s[0] for s in specs],
            "direction": [s[1] for s in specs],
        })

    def test_whipsaw_boundary_and_resume(self):
        ev = self._events([("2026-01-05 10:00:00", 1),
                           ("2026-01-05 10:03:00", -1),
                           ("2026-01-05 10:10:00", 1)])
        out = mcs.attach_whipsaw_and_resume(ev, whipsaw_minutes=5, resume_minutes=15)
        first = out.iloc[0]
        self.assertTrue(bool(first["whipsaw"]))
        self.assertAlmostEqual(float(first["tt_whipsaw_min"]), 3.0)
        self.assertEqual(first["rev_bar_start"], "2026-01-05 10:03:00")
        self.assertTrue(bool(first["resume"]))                     # 10:10 within +15m
        self.assertAlmostEqual(float(first["tt_resume_min"]), 7.0)
        # the 10:03 down event: next up at 10:10 -> whipsaw for IT too (dt 7 > 5? no)
        self.assertFalse(bool(out.iloc[1]["whipsaw"]))             # 7min > 5min
        self.assertFalse(bool(out.iloc[2]["whipsaw"]))             # no later opposite

    def test_exactly_five_minutes_included(self):
        ev = self._events([("2026-01-05 10:00:00", 1),
                           ("2026-01-05 10:05:00", -1)])
        out = mcs.attach_whipsaw_and_resume(ev, whipsaw_minutes=5)
        self.assertTrue(bool(out.iloc[0]["whipsaw"]))
        ev2 = self._events([("2026-01-05 10:00:00", 1),
                            ("2026-01-05 10:06:00", -1)])
        out2 = mcs.attach_whipsaw_and_resume(ev2, whipsaw_minutes=5)
        self.assertFalse(bool(out2.iloc[0]["whipsaw"]))

    def test_cluster_first(self):
        ev = self._events([("2026-01-05 10:00:00", 1),
                           ("2026-01-05 10:05:00", 1),
                           ("2026-01-05 10:30:00", 1),
                           ("2026-01-05 10:31:00", -1)])
        out = mcs.mark_cluster_first(ev, cluster_minutes=15)
        self.assertEqual(out["cluster_first"].tolist(), [True, False, True, True])


class MatchedBaselineTests(unittest.TestCase):
    def test_bounds_and_exclusions(self):
        rows = _seq("2026-01-05", "09:00", [100.0] * 121)   # 09:00..11:00
        reg = mcs.build_outcome_stream(_bars(rows))
        ev = pd.DataFrame({"symbol": ["TESTSYM"], "date": ["2026-01-05"],
                           "bar_start": ["2026-01-05 10:00:00"], "direction": [1]})
        pos_event = 60
        ev_rows, bar_rows = mcs.matched_baseline_index(
            ev, reg, halfwidth_minutes=30, cross_positions=[pos_event])
        self.assertEqual(len(ev_rows), 60)                  # 61 bars in +-30m minus self
        times = reg["time"].to_numpy()[bar_rows]
        self.assertEqual(times.min(), "09:30:00")           # inclusive both ends
        self.assertEqual(times.max(), "10:30:00")
        self.assertNotIn("10:00:00", set(times))            # cross bar excluded


class BandAndAtrTests(unittest.TestCase):
    def test_band_boundaries_and_shares(self):
        bands = mcs.classify_bands([0.0005, 0.001, 0.0015, 0.003, 0.004])
        self.assertEqual(bands.tolist(), ["near", "near", "mid", "far", "far"])
        shares = bands.value_counts(normalize=True)
        self.assertAlmostEqual(float(shares.sum()), 1.0)

    def test_atr_constant_range(self):
        rows = [(f"2026-01-05 09:{i:02d}:00", 100.0, 101.0, 99.0, 100.0)
                for i in range(30)]
        vf, gcols = mcs.prepare_variant(_bars(rows), "regular")
        mf = mcs.add_mas_and_atr(vf, gcols, price_col="open")
        self.assertAlmostEqual(float(mf["atr"].iloc[-1]), 2.0)   # TR == high-low == 2
        self.assertTrue(mf["atr"].iloc[:19].isna().all())


class ScriptScanTests(unittest.TestCase):
    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(", ".SendOrder")

    def test_analysis_script_research_only(self):
        for name in ("analyze_ma_cross_1m.py", "export_ma_cross_phase_csv.py"):
            path = SCRIPTS_DIR / name
            self.assertTrue(path.exists(), name)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Research / evaluation only", text)
            for token in self.FORBIDDEN:
                self.assertNotIn(token, text, name)
            self.assertNotIn("import torch", text)
            self.assertIsNone(re.search(r"['\"]\d{6}['\"]", text),
                              f"{name}: no hard-coded symbol codes (CLI args only)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
