"""Tests for entry/selection variants (src/entry_variants.py).

These exercise the pure, DB-free logic that the conservative-baseline comparison
relies on: pre-open context, the soft selection penalty, daily top-k selection
with cooldown, delayed-entry evaluation, and the summary roll-up.

Run (cwd = quant):
    ..\\Python311-64\\python.exe -m unittest tests.test_entry_variants
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

import entry_variants as ev  # noqa: E402


def _day(rows):
    """Build a single-symbol day frame like bars_by_day() returns.

    rows: list of (time, high, low, close[, open]) -- date is fixed.
    """
    recs = []
    for r in rows:
        t = r[0]
        recs.append({
            "_dt": pd.Timestamp(f"2026-01-05 {t}:00"),
            "high": float(r[1]),
            "low": float(r[2]),
            "close": float(r[3]),
            "open": float(r[4]) if len(r) > 4 else float(r[3]),
        })
    return pd.DataFrame(recs)


def _candidate(symbol, time, score_margin, up_score, *, preopen_ok=True,
               rank=1.0, volume_z=0.0, upper_shadow=0.0):
    return {
        "symbol": symbol,
        "date": "2026-01-05",
        "_decision_dt": pd.Timestamp(f"2026-01-05 {time}:00"),
        "score_margin": float(score_margin),
        "up_score": float(up_score),
        "rank_ret_20m_1m": float(rank),
        "volume_z_20m_1m": float(volume_z),
        "upper_shadow_1m": float(upper_shadow),
        "preopen_ok": preopen_ok,
        "close": 100.0,
    }


class PreopenContextTests(unittest.TestCase):
    def _raw(self):
        # Symbol AAA: rising pre-open (ok). Symbol BBB: falling pre-open (not ok).
        # An out-of-window 09:05 bar must be ignored entirely.
        rows = [
            ("AAA", "2026-01-05 08:30:00", 100, 101, 99, 100, 1000),
            ("AAA", "2026-01-05 08:45:00", 101, 104, 101, 103, 1000),
            ("AAA", "2026-01-05 09:05:00", 103, 110, 90, 95, 9999),  # ignored
            ("BBB", "2026-01-05 08:30:00", 100, 101, 99, 100, 1000),
            ("BBB", "2026-01-05 08:45:00", 100, 100, 95, 96, 1000),
        ]
        return pd.DataFrame(rows, columns=[
            "symbol", "bar_start", "open", "high", "low", "close", "volume"])

    def test_window_and_ok_flag(self):
        ctx = ev.build_preopen_context(self._raw())
        self.assertEqual(set(ctx["symbol"]), {"AAA", "BBB"})

        aaa = ctx[ctx["symbol"] == "AAA"].iloc[0]
        bbb = ctx[ctx["symbol"] == "BBB"].iloc[0]

        # Only the two in-window bars (08:30, 08:45) are counted; 09:05 dropped.
        self.assertEqual(int(aaa["preopen_bar_count"]), 2)
        self.assertEqual(int(bbb["preopen_bar_count"]), 2)

        # AAA: last_close 103 / first_open 100 - 1 > 0.
        self.assertAlmostEqual(aaa["preopen_ret"], 103 / 100 - 1, places=6)
        self.assertTrue(bool(aaa["preopen_ok"]))

        # BBB falls: last_close 96 < first_open 100 -> ret < 0 -> not ok.
        self.assertLess(bbb["preopen_ret"], 0)
        self.assertFalse(bool(bbb["preopen_ok"]))

    def test_missing_preopen_is_not_ok(self):
        # No in-window bars at all -> frame is empty, but the ok column is bool.
        empty = pd.DataFrame(
            [("AAA", "2026-01-05 09:05:00", 100, 101, 99, 100, 1000)],
            columns=["symbol", "bar_start", "open", "high", "low", "close", "volume"],
        )
        ctx = ev.build_preopen_context(empty)
        self.assertEqual(len(ctx), 0)
        self.assertEqual(ctx["preopen_ok"].dtype, bool)


class EvaluateEntryTests(unittest.TestCase):
    def _ramp(self):
        # Monotonic ramp; close rises 100->104 over 5 minutes.
        return _day([
            ("09:30", 101, 99, 100),
            ("09:31", 102, 100, 101),
            ("09:32", 103, 101, 102),
            ("09:33", 104, 102, 103),
            ("09:34", 105, 103, 104),
        ])

    def test_delay0_enters_at_decision_close(self):
        day = self._ramp()
        out = ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=0)
        self.assertEqual(out["entry_price"], 100.0)
        self.assertEqual(out["future_bar_count"], 5)            # includes decision bar
        self.assertAlmostEqual(out["mfe"], 105 / 100 - 1, places=6)
        self.assertAlmostEqual(out["mae"], 99 / 100 - 1, places=6)

    def test_delay1_is_one_minute_after_decision_close(self):
        day = self._ramp()
        out = ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=1)
        # future.iloc[1] is 09:31 (close 101) = 1 minute after the decision close.
        self.assertEqual(out["entry_price"], 101.0)

    def test_delay3_is_three_minutes_after_decision_close(self):
        day = self._ramp()
        out = ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=3)
        # future.iloc[3] is 09:33 (close 103) = 3 minutes after the decision close;
        # eval window is strictly after the entry bar -> 09:34 only.
        self.assertEqual(out["entry_price"], 103.0)
        self.assertEqual(out["future_bar_count"], 1)
        self.assertAlmostEqual(out["mfe"], 105 / 103 - 1, places=6)
        self.assertAlmostEqual(out["mae"], 103 / 103 - 1, places=6)

    def test_delay_returns_none_when_not_enough_bars(self):
        day = self._ramp()  # 5 bars at iloc 0..4
        # delay 4 -> entry at iloc[4] (last bar); no bar left to evaluate -> None.
        self.assertIsNone(
            ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=4))
        # delay 5 -> not even an entry bar -> None.
        self.assertIsNone(
            ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=5))

    def test_stop_first_true_when_stop_precedes_target(self):
        day = _day([
            ("09:30", 100, 100, 100),
            ("09:31", 101, 98, 99),    # low 98 <= stop 98.5 -> stop hit first
            ("09:32", 106, 99, 105),   # high 106 >= target 105 -> target later
        ])
        out = ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=0)
        self.assertEqual(out["stop_first"], 1)

    def test_stop_first_false_when_target_precedes_stop(self):
        day = _day([
            ("09:30", 100, 100, 100),
            ("09:31", 106, 100, 105),  # target hit, never touched stop
        ])
        out = ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=0)
        self.assertEqual(out["stop_first"], 0)

    def test_hit_flags_track_mfe(self):
        day = self._ramp()  # mfe = 5%
        out = ev.evaluate_entry(day, pd.Timestamp("2026-01-05 09:30:00"), 100.0, delay_bars=0)
        self.assertEqual((out["hit_up2"], out["hit_up3"], out["hit_up5"]), (1, 1, 1))


class SelectionMarginTests(unittest.TestCase):
    def test_zero_penalty_is_plain_margin(self):
        cands = pd.DataFrame([
            _candidate("AAA", "09:30", 5, 6, preopen_ok=True),
            _candidate("BBB", "09:31", 4, 5, preopen_ok=False),
        ])
        out = ev.add_selection_margin(cands, preopen_penalty=0.0)
        pd.testing.assert_series_equal(
            out["_sel_margin"], out["score_margin"], check_names=False)

    def test_penalty_only_hits_weak_preopen(self):
        cands = pd.DataFrame([
            _candidate("AAA", "09:30", 5, 6, preopen_ok=True),
            _candidate("BBB", "09:31", 4, 5, preopen_ok=False),
        ])
        out = ev.add_selection_margin(cands, preopen_penalty=1.0)
        aaa = out[out["symbol"] == "AAA"].iloc[0]
        bbb = out[out["symbol"] == "BBB"].iloc[0]
        self.assertEqual(aaa["_sel_margin"], 5.0)         # ok -> unchanged
        self.assertEqual(bbb["_sel_margin"], 4.0 - 1.0)   # weak -> penalized


class SelectTopkTests(unittest.TestCase):
    def test_baseline_top2_by_margin(self):
        cands = ev.add_selection_margin(pd.DataFrame([
            _candidate("AAA", "09:30", 5, 6),
            _candidate("BBB", "09:40", 4, 5),
            _candidate("CCC", "09:50", 3, 5),
        ]), preopen_penalty=0.0)
        sel = ev.select_topk(cands, top_k=2, cooldown_minutes=60)
        self.assertEqual(sel["symbol"].tolist(), ["AAA", "BBB"])
        self.assertEqual(sel["daily_rank"].tolist(), [1, 2])

    def test_penalty_reranks_top2(self):
        # Without penalty: AAA(5), BBB(4), CCC(3) -> top2 AAA,BBB.
        # With penalty 1 on weak BBB: BBB 4->3 ties CCC 3; CCC wins on up_score.
        raw = pd.DataFrame([
            _candidate("AAA", "09:30", 5, 6, preopen_ok=True),
            _candidate("BBB", "09:40", 4, 4, preopen_ok=False),
            _candidate("CCC", "09:50", 3, 5, preopen_ok=True),
        ])
        base = ev.select_topk(ev.add_selection_margin(raw, preopen_penalty=0.0),
                              top_k=2, cooldown_minutes=60)
        self.assertEqual(base["symbol"].tolist(), ["AAA", "BBB"])

        penalized = ev.select_topk(ev.add_selection_margin(raw, preopen_penalty=1.0),
                                   top_k=2, cooldown_minutes=60)
        self.assertEqual(penalized["symbol"].tolist(), ["AAA", "CCC"])

    def test_cooldown_blocks_same_symbol_within_window(self):
        # Two AAA signals 20 min apart (< 60) plus one BBB. Cooldown keeps the
        # stronger AAA then fills with BBB.
        cands = ev.add_selection_margin(pd.DataFrame([
            _candidate("AAA", "09:30", 5, 6),
            _candidate("AAA", "09:50", 4, 6),
            _candidate("BBB", "09:31", 3, 6),
        ]), preopen_penalty=0.0)
        sel = ev.select_topk(cands, top_k=2, cooldown_minutes=60)
        self.assertEqual(sel["symbol"].tolist(), ["AAA", "BBB"])
        kept_aaa = sel[sel["symbol"] == "AAA"].iloc[0]
        self.assertEqual(kept_aaa["_decision_dt"], pd.Timestamp("2026-01-05 09:30:00"))

    def test_empty_candidates(self):
        self.assertEqual(len(ev.select_topk(pd.DataFrame())), 0)


class SummarizeTests(unittest.TestCase):
    def test_empty(self):
        s = ev.summarize(pd.DataFrame(), name="x")
        self.assertEqual(s["variant"], "x")
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["hit2"], 0.0)

    def test_means_and_concentration(self):
        df = pd.DataFrame([
            {"symbol": "AAA", "hit_up2": 1, "hit_up3": 0, "hit_up5": 0,
             "mfe": 0.02, "mae": -0.01, "stop_first": 0,
             "return_30m": 0.01, "return_60m": 0.0, "return_120m": 0.0, "return_eod": 0.0},
            {"symbol": "AAA", "hit_up2": 1, "hit_up3": 1, "hit_up5": 0,
             "mfe": 0.04, "mae": -0.02, "stop_first": 1,
             "return_30m": 0.03, "return_60m": 0.0, "return_120m": 0.0, "return_eod": 0.0},
        ])
        s = ev.summarize(df, name="v")
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["hit2"], 1.0)
        self.assertAlmostEqual(s["hit3"], 0.5)
        self.assertAlmostEqual(s["mfe"], 0.03)
        self.assertAlmostEqual(s["stop_first"], 0.5)
        # Both rows are AAA -> full concentration.
        self.assertAlmostEqual(s["symbol_concentration"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
