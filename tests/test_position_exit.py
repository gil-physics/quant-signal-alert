"""Tests for the position exit-warning rules (src/position_exit.py).

Pure module: no Kiwoom, pandas, model, or DB needed. Also statically checks the
monitor script never references an order API (SendOrder).
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
# Only SRC_DIR goes on sys.path (position_exit lives there). The monitor script is
# referenced by file path / run as a subprocess, so we never import it in-process and
# never put SCRIPTS_DIR on the global path (that would perturb other test modules).
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_exit import (  # noqa: E402
    HOLD, SELL_WARNING, STRONG_SELL_WARNING, WEAK_SELL_WARNING,
    ExitConfig, PositionState, evaluate_exit,
)

ENTRY_T = datetime(2026, 6, 19, 10, 0, 0)


def make_pos(*, ret, mfe, mae=0.0, bars=5, entry=100.0):
    """A long PositionState built from desired return/MFE/MAE (drawdown derived)."""
    return PositionState.from_prices(
        symbol="000660", side="long", entry_time=ENTRY_T, entry_price=entry,
        current_time=datetime(2026, 6, 19, 10, 0, 0),
        current_price=entry * (1 + ret),
        highest_price_since_entry=entry * (1 + mfe),
        lowest_price_since_entry=entry * (1 + mae),
        bars_since_entry=bars,
    )


def snap(**kw):
    """A benign trend snapshot (above MA5/VWAP, low risk, UP); override per test."""
    base = dict(trend_state="UP", alert_type="", alert_grade="A", risk_prob=0.30,
                risk_bucket="LOW", close=101.0, ma5=100.0, vwap=99.0,
                above_ma5=True, above_vwap=True)
    base.update(kw)
    return base


class PositionMetricsTests(unittest.TestCase):
    def test_long_mfe_mae_drawdown(self):
        # 11) long-side excursion math
        pos = PositionState.from_prices(
            symbol="000660", side="long", entry_time=ENTRY_T, entry_price=100.0,
            current_time=ENTRY_T, current_price=103.0,
            highest_price_since_entry=105.0, lowest_price_since_entry=99.0,
            bars_since_entry=5)
        self.assertAlmostEqual(pos.current_return, 0.03)
        self.assertAlmostEqual(pos.mfe, 0.05)
        self.assertAlmostEqual(pos.mae, -0.01)
        self.assertAlmostEqual(pos.drawdown_from_peak, 103.0 / 105.0 - 1.0)

    def test_short_side_not_supported_yet(self):
        with self.assertRaises(NotImplementedError):
            PositionState.from_prices(
                symbol="000660", side="short", entry_time=ENTRY_T, entry_price=100.0,
                current_time=ENTRY_T, current_price=99.0,
                highest_price_since_entry=100.0, lowest_price_since_entry=98.0,
                bars_since_entry=5)


class ExitRuleTests(unittest.TestCase):
    def test_hold_when_nothing_triggers(self):
        # 1) healthy: modest unrealized gain, above structure, low risk, UP
        sig = evaluate_exit(make_pos(ret=0.006, mfe=0.006, mae=-0.001, bars=10), snap())
        self.assertEqual(sig.exit_state, HOLD)
        self.assertEqual(sig.exit_reason, "")
        self.assertEqual(sig.reason_details, [])

    def test_hold_with_empty_snapshot(self):
        # robustness: missing engine keys must not raise or false-fire
        sig = evaluate_exit(make_pos(ret=0.002, mfe=0.003, bars=5), {})
        self.assertEqual(sig.exit_state, HOLD)

    def test_hard_stop_strong(self):
        # 2) hard stop
        sig = evaluate_exit(make_pos(ret=-0.010, mfe=0.0, mae=-0.010, bars=5), snap())
        self.assertEqual(sig.exit_state, STRONG_SELL_WARNING)
        self.assertEqual(sig.exit_reason, "hard_stop")

    def test_profit_giveback_sell(self):
        # 3) gave a +0.9% MFE back to ~break-even
        sig = evaluate_exit(make_pos(ret=0.0005, mfe=0.009, mae=-0.002, bars=10), snap())
        self.assertEqual(sig.exit_state, SELL_WARNING)
        self.assertEqual(sig.exit_reason, "profit_giveback_to_breakeven")

    def test_trailing_drawdown_sell(self):
        # 4) ran to +1.5% then pulled back > 0.5% from the peak (still net +0.8%)
        sig = evaluate_exit(make_pos(ret=0.008, mfe=0.015, mae=-0.001, bars=10), snap())
        self.assertEqual(sig.exit_state, SELL_WARNING)
        self.assertEqual(sig.exit_reason, "trailing_drawdown")

    def test_ma5_vwap_break_sell(self):
        # 5) below BOTH MA5 and VWAP
        s = snap(close=98.0, ma5=100.0, vwap=99.0, above_ma5=False, above_vwap=False,
                 trend_state="CHOPPY")
        sig = evaluate_exit(make_pos(ret=0.002, mfe=0.004, bars=10), s)
        self.assertEqual(sig.exit_state, SELL_WARNING)
        self.assertEqual(sig.exit_reason, "ma5_vwap_break")

    def test_single_ma5_break_weak(self):
        # 6) below MA5 only (still above VWAP)
        s = snap(close=99.0, ma5=100.0, vwap=98.0, above_ma5=False, above_vwap=True,
                 trend_state="CHOPPY")
        sig = evaluate_exit(make_pos(ret=0.002, mfe=0.004, bars=10), s)
        self.assertEqual(sig.exit_state, WEAK_SELL_WARNING)
        self.assertEqual(sig.exit_reason, "ma5_break")

    def test_risk_high_trend_weakening_strong(self):
        # 7) model high-risk + weakening trend
        s = snap(risk_prob=0.85, trend_state="WEAKENING")
        sig = evaluate_exit(make_pos(ret=0.003, mfe=0.004, bars=10), s)
        self.assertEqual(sig.exit_state, STRONG_SELL_WARNING)
        self.assertEqual(sig.exit_reason, "risk_high_trend_weakening")

    def test_risk_medium_only_weak(self):
        # risk between med and high, trend not down -> weak only
        s = snap(risk_prob=0.65, trend_state="UP")
        sig = evaluate_exit(make_pos(ret=0.003, mfe=0.004, bars=10), s)
        self.assertEqual(sig.exit_state, WEAK_SELL_WARNING)
        self.assertEqual(sig.exit_reason, "risk_elevated")

    def test_opposite_signal_strong(self):
        # 8) trend flipped DOWN
        s = snap(trend_state="DOWN", risk_prob=0.30)
        sig = evaluate_exit(make_pos(ret=0.001, mfe=0.002, bars=10), s)
        self.assertEqual(sig.exit_state, STRONG_SELL_WARNING)
        self.assertEqual(sig.exit_reason, "opposite_signal")

    def test_time_stop_sell(self):
        # 9) 35 bars in, never made progress, sitting flat
        sig = evaluate_exit(make_pos(ret=-0.0005, mfe=0.003, mae=-0.003, bars=35), snap())
        self.assertEqual(sig.exit_state, SELL_WARNING)
        self.assertEqual(sig.exit_reason, "time_stop_no_progress")

    def test_multiple_rules_strongest_wins_and_all_listed(self):
        # 10) hard_stop (STRONG) + ma5_vwap_break (SELL) + time_stop (SELL)
        s = snap(close=98.0, ma5=100.0, vwap=99.0, above_ma5=False, above_vwap=False,
                 trend_state="CHOPPY")
        sig = evaluate_exit(make_pos(ret=-0.010, mfe=0.002, mae=-0.012, bars=35), s)
        self.assertEqual(sig.exit_state, STRONG_SELL_WARNING)
        self.assertEqual(sig.exit_reason, "hard_stop")            # highest priority wins
        self.assertEqual(set(sig.reason_details),
                         {"hard_stop", "ma5_vwap_break", "time_stop_no_progress"})
        self.assertEqual(sig.reason_details[0], "hard_stop")      # priority-ordered

    def test_config_overrides_thresholds(self):
        # tighter hard stop fires where the default would not
        pos = make_pos(ret=-0.004, mfe=0.0, mae=-0.004, bars=5)
        self.assertEqual(evaluate_exit(pos, snap()).exit_state, HOLD)
        cfg = ExitConfig(hard_stop=-0.003)
        self.assertEqual(evaluate_exit(pos, snap(), cfg).exit_state, STRONG_SELL_WARNING)


class NoOrderApiTests(unittest.TestCase):
    def test_monitor_never_references_send_order(self):
        # the monitor is decision-support only; it must never touch an order API.
        src = (SCRIPTS_DIR / "realtime_position_monitor.py").read_text(encoding="utf-8")
        for forbidden in ("SendOrder", "send_order", "sendorder"):
            self.assertNotIn(forbidden, src, f"monitor must not reference {forbidden}")


def _replay_smoke_skip_reason():
    """Skip the end-to-end replay smoke unless its data/deps are present."""
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return "lightgbm not available (64-bit only)"
    import sqlite3
    from config import DB_PATH
    if not Path(DB_PATH).exists():
        return "DB not present"
    if not (ROOT_DIR / "models" / "risk_model_lgbm.txt").exists():
        return "risk model not present"
    conn = sqlite3.connect(DB_PATH)
    try:
        n = conn.execute("SELECT count(*) FROM bars_1m WHERE symbol='000660' "
                         "AND substr(bar_start,1,10)='2026-06-19'").fetchone()[0]
    except Exception:
        return "bars_1m unavailable"
    finally:
        conn.close()
    return None if n else "no 000660 bars for 2026-06-19"


class ReplayMonitorSmokeTests(unittest.TestCase):
    """End-to-end replay of the monitor script (guarded; skips without data/deps)."""

    DATE = "2026-06-19"

    def test_replay_cross_symbol_prints_and_writes_ledger(self):
        reason = _replay_smoke_skip_reason()
        if reason:
            self.skipTest(reason)
        import subprocess

        from config import PROCESSED_DIR
        LEDGER_PATH = PROCESSED_DIR / "realtime_position_ledger.csv"
        if LEDGER_PATH.exists():
            LEDGER_PATH.unlink()
        # default (no --all-bars) exercises the on-change display; --cross-symbol
        # exercises the v1.1 universe path. Early entry guarantees state transitions.
        cmd = [sys.executable, str(SCRIPTS_DIR / "realtime_position_monitor.py"),
               "--mode", "replay", "--replay-date", self.DATE, "--code", "000660",
               "--side", "long", "--entry-price", "2856000", "--entry-time", "10:14",
               "--cross-symbol"]
        out = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                             timeout=600)
        text = out.stdout or ""
        self.assertEqual(out.returncode, 0, msg=text)
        # printed at least one position line after entry, and wrote the ledger
        self.assertIn("000660 LONG", text)
        self.assertIn("ledger ->", text)
        self.assertTrue(LEDGER_PATH.exists())
        # the monitor must run without ever touching an order API
        self.assertNotIn("SendOrder", text)

        # on-change suppression: far fewer printed state lines than monitored bars.
        import re
        printed = sum(text.count(f"LONG {s}") for s in
                      ("HOLD", "WEAK_SELL_WARNING", "SELL_WARNING", "STRONG_SELL_WARNING"))
        m = re.search(r"monitored (\d+) bars", text)
        self.assertIsNotNone(m, msg=text)
        monitored = int(m.group(1))
        self.assertGreaterEqual(printed, 1)
        self.assertLess(printed, monitored, "on-change should print fewer lines than bars")


if __name__ == "__main__":
    unittest.main(verbosity=2)
