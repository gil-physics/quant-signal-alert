"""Tests for the morning-signal -> afternoon-extreme validation.

Pure metric/selection logic (src/morning_signal_validation.py) is unit-tested without
DB/Kiwoom; the CLI script is checked statically (no order API) and via a guarded
end-to-end replay-style subprocess smoke test on the real DB.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
# Only SRC_DIR on sys.path (pure logic lives there). The CLI script is referenced by
# file path / run as a subprocess — never imported in-process (keeps the global path
# clean so other test modules' imports are unaffected).
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    HAVE_PANDAS = False

if HAVE_PANDAS:
    from morning_signal_validation import (
        aggregate, classify_direction, compute_windows, daily_comparison, select_signals,
    )

AFT_START, AFT_END = "13:00:00", "15:30:00"


def _bar(tod, high, low, close):
    return {"tod": tod, "bar_end": f"2026-06-19 {tod}", "high": float(high),
            "low": float(low), "close": float(close)}


def _day_bars():
    rows = [
        _bar("09:30:00", 101, 99, 100),
        _bar("11:00:00", 105, 98, 104),    # pre-13 window (after a 10:00 signal)
        _bar("13:00:00", 102, 100, 101),   # price_13_00 = 101
        _bar("13:30:00", 103, 95, 96),     # afternoon low = 95
        _bar("14:00:00", 110, 104, 109),   # afternoon high = 110
        _bar("15:30:00", 108, 106, 107),   # afternoon close = 107
    ]
    return pd.DataFrame(rows)


def _alerts():
    # one trading day, several UP signals + one after cutoff
    base = dict(symbol="000660", date="2026-06-19", signal_direction="UP", risk_prob=0.3)
    rows = [
        {**base, "decision_time": "2026-06-19 09:30:00", "tod": "09:30:00", "alert_grade": "B", "alert_score": 3.0},
        {**base, "decision_time": "2026-06-19 10:00:00", "tod": "10:00:00", "alert_grade": "A", "alert_score": 5.0},
        {**base, "decision_time": "2026-06-19 10:30:00", "tod": "10:30:00", "alert_grade": "A", "alert_score": 5.0},
        {**base, "decision_time": "2026-06-19 11:30:00", "tod": "11:30:00", "alert_grade": "A", "alert_score": 9.0},  # after cutoff
    ]
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class DirectionAndWindowTests(unittest.TestCase):
    def test_classify_direction(self):
        self.assertEqual(classify_direction("UP_ALERT"), "UP")
        self.assertEqual(classify_direction("UP_CONTINUATION"), "UP")
        self.assertEqual(classify_direction("DOWN_ALERT"), "DOWN")
        self.assertEqual(classify_direction("SHORT_ALERT"), "DOWN")
        self.assertEqual(classify_direction("RISK_ALERT"), "OTHER")
        self.assertEqual(classify_direction(""), "OTHER")

    def test_classify_direction_risk_as_short(self):
        self.assertEqual(classify_direction("RISK_ALERT", risk_as_down=True), "DOWN")
        self.assertEqual(classify_direction("UP_ALERT", risk_as_down=True), "UP")
        self.assertEqual(classify_direction("WEAKENING_ALERT", risk_as_down=True), "OTHER")

    def test_up_signal_maps_to_afternoon_high(self):
        w = compute_windows(_day_bars(), alert_price=100.0, signal_tod="10:00:00",
                            direction="UP", afternoon_start=AFT_START, afternoon_end=AFT_END)
        self.assertAlmostEqual(w["afternoon_high"], 110.0)
        self.assertEqual(w["afternoon_high_time"], "2026-06-19 14:00:00")
        self.assertAlmostEqual(w["afternoon_high_return"], 0.10)
        self.assertAlmostEqual(w["afternoon_close_return"], 0.07)
        self.assertAlmostEqual(w["price_13_00"], 101.0)
        self.assertAlmostEqual(w["pre_13_mfe"], 0.05)    # 105/100-1
        self.assertAlmostEqual(w["pre_13_mae"], -0.02)   # 98/100-1
        self.assertAlmostEqual(w["full_day_mfe_after_signal"], 0.10)
        self.assertAlmostEqual(w["full_day_mae_after_signal"], -0.05)
        self.assertEqual(w["notes"], [])

    def test_down_signal_maps_to_afternoon_low_and_short_potential(self):
        w = compute_windows(_day_bars(), alert_price=100.0, signal_tod="10:00:00",
                            direction="DOWN", afternoon_start=AFT_START, afternoon_end=AFT_END)
        self.assertAlmostEqual(w["afternoon_low"], 95.0)
        self.assertEqual(w["afternoon_low_time"], "2026-06-19 13:30:00")
        self.assertAlmostEqual(w["short_potential"], 100.0 / 95.0 - 1.0)
        self.assertAlmostEqual(w["pre_13_mfe"], 100.0 / 98.0 - 1.0)   # favorable short
        self.assertAlmostEqual(w["pre_13_mae"], 0.05)                 # 105/100-1 adverse
        self.assertAlmostEqual(w["full_day_mfe_after_signal"], 100.0 / 95.0 - 1.0)
        self.assertAlmostEqual(w["full_day_mae_after_signal"], 0.10)

    def test_missing_afternoon_bars_notes_not_crash(self):
        morning_only = pd.DataFrame([_bar("09:30:00", 101, 99, 100), _bar("10:00:00", 102, 99, 101)])
        w = compute_windows(morning_only, alert_price=100.0, signal_tod="09:00:00",
                            direction="UP", afternoon_start=AFT_START, afternoon_end=AFT_END)
        self.assertIn("missing_afternoon_bars", w["notes"])
        self.assertNotEqual(w["afternoon_high"], w["afternoon_high"])  # NaN

    def test_missing_alert_price_notes(self):
        w = compute_windows(_day_bars(), alert_price=float("nan"), signal_tod="10:00:00",
                            direction="UP", afternoon_start=AFT_START, afternoon_end=AFT_END)
        self.assertIn("missing_alert_price", w["notes"])


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class SelectionTests(unittest.TestCase):
    def test_last_selection(self):
        out = select_signals(_alerts(), selection="last", cutoff_tod="11:00:00")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["tod"], "10:30:00")   # last before cutoff

    def test_first_selection(self):
        out = select_signals(_alerts(), selection="first", cutoff_tod="11:00:00")
        self.assertEqual(out.iloc[0]["tod"], "09:30:00")

    def test_strongest_prefers_A_then_latest(self):
        out = select_signals(_alerts(), selection="strongest", cutoff_tod="11:00:00")
        self.assertEqual(len(out), 1)
        # grades: 09:30 B, 10:00 A, 10:30 A -> A wins, tie -> latest = 10:30
        self.assertEqual(out.iloc[0]["tod"], "10:30:00")
        self.assertEqual(out.iloc[0]["alert_grade"], "A")

    def test_signals_after_cutoff_excluded(self):
        out = select_signals(_alerts(), selection="all", cutoff_tod="11:00:00")
        self.assertNotIn("11:30:00", set(out["tod"]))
        self.assertEqual(len(out), 3)

    def test_all_keeps_every_valid_morning_signal(self):
        out = select_signals(_alerts(), selection="all", cutoff_tod="11:00:00")
        self.assertEqual(set(out["tod"]), {"09:30:00", "10:00:00", "10:30:00"})


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class AggregateTests(unittest.TestCase):
    def test_hit_rates(self):
        df = pd.DataFrame({
            "signal_direction": ["UP"] * 5,
            "afternoon_high_return": [0.0, 0.006, 0.011, 0.016, 0.021],
            "short_potential": [0.0, 0.0, 0.006, 0.012, 0.025],
        })
        a = aggregate(df)
        self.assertEqual(a["n"], 5)
        self.assertAlmostEqual(a["hit_up_0_5"], 4 / 5)
        self.assertAlmostEqual(a["hit_up_1_0"], 3 / 5)
        self.assertAlmostEqual(a["hit_up_1_5"], 2 / 5)
        self.assertAlmostEqual(a["hit_up_2_0"], 1 / 5)
        self.assertAlmostEqual(a["avg_afternoon_high_return"], (0 + 0.006 + 0.011 + 0.016 + 0.021) / 5)
        self.assertAlmostEqual(a["hit_down_0_5"], 3 / 5)
        self.assertAlmostEqual(a["hit_down_1_0"], 2 / 5)


@unittest.skipUnless(HAVE_PANDAS, "pandas required")
class DailyComparisonTests(unittest.TestCase):
    def _trades(self):
        return pd.DataFrame([
            {"trade_date": "2026-06-18", "signal_direction": "UP", "signal_time": "2026-06-18 10:00:00",
             "alert_type": "UP_ALERT", "alert_grade": "A", "risk_prob": 0.3, "risk_bucket": "LOW",
             "alert_price": 100.0, "afternoon_high_return": 0.02, "afternoon_low_return": -0.01,
             "short_potential": 0.01, "afternoon_close_return": 0.005, "notes": ""},
            {"trade_date": "2026-06-19", "signal_direction": "DOWN", "signal_time": "2026-06-19 09:50:00",
             "alert_type": "RISK_ALERT", "alert_grade": "RISK", "risk_prob": 0.7, "risk_bucket": "HIGH",
             "alert_price": 200.0, "afternoon_high_return": 0.004, "afternoon_low_return": -0.03,
             "short_potential": 0.031, "afternoon_close_return": -0.02, "notes": ""},
        ])

    def _baseline_last(self):
        return pd.DataFrame([
            {"trade_date": "2026-06-18", "signal_time": "2026-06-18 10:59:00", "alert_price": 101.0,
             "afternoon_high_return": 0.012, "short_potential": 0.006},
            {"trade_date": "2026-06-19", "signal_time": "2026-06-19 10:59:00", "alert_price": 198.0,
             "afternoon_high_return": 0.009, "short_potential": 0.02},
        ])

    def test_one_row_per_day_single_direction(self):
        d = daily_comparison(self._trades(), self._baseline_last())
        self.assertEqual(len(d), 2)
        self.assertTrue(d["trade_date"].is_unique)
        longs = set(d[d.direction == "long"].trade_date)
        shorts = set(d[d.direction == "short"].trade_date)
        self.assertEqual(longs & shorts, set())          # no day is both

    def test_direction_appropriate_returns_and_edge(self):
        d = daily_comparison(self._trades(), self._baseline_last()).set_index("trade_date")
        # long day -> signal_return = afternoon_high_return; baseline = baseline high-return
        self.assertAlmostEqual(d.loc["2026-06-18", "signal_return"], 0.02)
        self.assertAlmostEqual(d.loc["2026-06-18", "baseline_return"], 0.012)
        self.assertAlmostEqual(d.loc["2026-06-18", "edge_vs_baseline"], 0.02 - 0.012)
        # short day -> signal_return = short_potential; baseline = baseline short_potential
        self.assertEqual(d.loc["2026-06-19", "direction"], "short")
        self.assertAlmostEqual(d.loc["2026-06-19", "signal_return"], 0.031)
        self.assertAlmostEqual(d.loc["2026-06-19", "baseline_return"], 0.02)
        self.assertAlmostEqual(d.loc["2026-06-19", "edge_vs_baseline"], 0.031 - 0.02)


class NoOrderApiTests(unittest.TestCase):
    SCRIPTS = ("validate_morning_signal_afternoon_extreme.py", "build_realtime_replay_alerts.py")

    def test_scripts_never_invoke_order_api(self):
        # call-form tokens (with "(") so prose like "no SendOrder" in a docstring is allowed.
        for name in self.SCRIPTS:
            src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            for forbidden in ("SendOrder(", "send_order(", ".SendOrder", "OrderRequest(", "place_order("):
                self.assertNotIn(forbidden, src, f"{name} must not invoke {forbidden}")


class RealtimeReplaySourceWiringTests(unittest.TestCase):
    """The validator must expose the 09:xx-capable realtime_replay feed (the fix for the
    offline directional_alerts feed that started at 10:01 and missed 09:00-10:00)."""

    def test_validator_wires_realtime_replay_source(self):
        src = (SCRIPTS_DIR / "validate_morning_signal_afternoon_extreme.py").read_text(encoding="utf-8")
        self.assertIn("realtime_replay", src)                 # --alerts-source choice
        self.assertIn("realtime_replay_alerts", src)          # the table it reads
        # auto must PREFER the faithful replay feed when present.
        self.assertIn("realtime_replay_alerts", src.split("def resolve_alerts_source", 1)[-1]
                      .split("def ", 1)[0])

    def test_capture_script_exists(self):
        self.assertTrue((SCRIPTS_DIR / "build_realtime_replay_alerts.py").exists())


def _smoke_skip_reason():
    try:
        import pandas  # noqa: F401
    except Exception:
        return "pandas unavailable"
    import sqlite3
    from config import DB_PATH
    if not Path(DB_PATH).exists():
        return "DB not present"
    conn = sqlite3.connect(DB_PATH)
    try:
        n = conn.execute("SELECT count(*) FROM bars_1m WHERE symbol='000660' "
                         "AND source='al_regular'").fetchone()[0]
    except Exception:
        return "bars_1m unavailable"
    finally:
        conn.close()
    return None if n else "no 000660 al_regular bars"


class EndToEndSmokeTests(unittest.TestCase):
    """Run the full CLI on the real DB (guarded). Covers baseline generation, file
    outputs, missing-data tolerance, and the no-order guarantee end to end."""

    def test_run_writes_all_outputs_and_baseline(self):
        reason = _smoke_skip_reason()
        if reason:
            self.skipTest(reason)
        import shutil
        import subprocess
        import tempfile

        out_dir = Path(tempfile.mkdtemp(prefix="morning_val_"))
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        cmd = [sys.executable, str(SCRIPTS_DIR / "validate_morning_signal_afternoon_extreme.py"),
               "--code", "000660", "--selection", "last", "--output-dir", str(out_dir)]
        out = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=600)
        text = out.stdout or ""
        self.assertEqual(out.returncode, 0, msg=text)
        self.assertIn("verdict(UP/long)=", text)
        self.assertNotIn("SendOrder", text)

        trades = out_dir / "morning_signal_afternoon_extreme_trades.csv"
        summary = out_dir / "morning_signal_afternoon_extreme_summary.csv"
        baseline = out_dir / "morning_signal_afternoon_extreme_baseline.csv"
        daily = out_dir / "morning_signal_afternoon_extreme_daily.csv"
        report = out_dir / "morning_signal_afternoon_extreme_report.md"
        for f in (trades, summary, baseline, daily, report):
            self.assertTrue(f.exists(), f"missing output {f.name}\n{text}")
        bl = baseline.read_text(encoding="utf-8")
        self.assertIn("last_bar_before_11", bl)
        self.assertIn("Verdict", report.read_text(encoding="utf-8"))
        # per-day file: one row per day, each day exactly one direction (no overlap)
        if HAVE_PANDAS:
            dd = pd.read_csv(daily)
            self.assertTrue(dd["trade_date"].is_unique, "a day appears more than once in daily CSV")
        # trades should have at least the header + some UP rows for 000660 over history
        tr = pd.read_csv(trades) if HAVE_PANDAS else None
        if tr is not None:
            self.assertGreater(len(tr), 0)
            self.assertIn("afternoon_high_return", tr.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
