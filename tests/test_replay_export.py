import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import replay_export as rx  # noqa: E402

EXPORTER = ROOT_DIR / "scripts" / "export_replay_bundle.py"


def _bars(start="2026-06-19 09:00:00", n=10, base=100.0):
    rows = []
    t0 = pd.Timestamp(start)
    for i in range(n):
        bs = t0 + pd.Timedelta(minutes=i)
        rows.append({"bar_start": bs.strftime("%Y-%m-%d %H:%M:%S"),
                     "bar_end": (bs + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                     "open": base + i, "high": base + i + 0.5, "low": base + i - 0.5,
                     "close": base + i})
    return pd.DataFrame(rows)


class LabelTests(unittest.TestCase):
    def test_reason_tokens_translate_and_unknown_passes_through(self):
        self.assertEqual(rx.reasons_ko(["uptrend_20m", "", None, "mystery"]),
                         ["20분 상승 추세", "mystery"])

    def test_every_reason_token_in_both_rule_modules_has_a_label(self):
        for name in ("directional_alert.py", "short_entry.py"):
            text = (SRC_DIR / name).read_text(encoding="utf-8")
            block = text.split("reason_specs = [", 1)[1].split("]", 1)[0]
            for token in re.findall(r'"([a-z0-9_]+)"\)', block):
                self.assertIn(token, rx.REASON_KO, f"{name}: {token}")

    def test_parse_message_reasons(self):
        msg = "상방 가능성 [UP_ALERT A] risk_prob=0.62(HIGH)  reasons=uptrend_20m/above_5ma"
        self.assertEqual(rx.parse_message_reasons(msg), ["uptrend_20m", "above_5ma"])
        self.assertEqual(rx.parse_message_reasons("no reasons here"), [])
        self.assertEqual(rx.parse_message_reasons("x  reasons="), [])

    def test_direction_follows_meaning_not_engine_side(self):
        # the runner marks any long-engine fire with ▲, including its downside alert
        self.assertEqual(rx.type_dir("long", "RISK_ALERT"), "down")
        self.assertEqual(rx.type_dir("long", "UP_ALERT"), "up")
        # the down system's SHORT_RISK_ALERT warns of an upside move
        self.assertEqual(rx.type_dir("down", "SHORT_RISK_ALERT"), "up")
        self.assertEqual(rx.type_dir("down", "DOWN_ALERT"), "down")
        self.assertEqual(rx.type_dir("down", "VOLATILITY_ALERT"), "neutral")
        self.assertEqual(rx.type_ko("down", "SHORT_RISK_ALERT"), "상승 위험")

    def test_finite_or_none(self):
        self.assertIsNone(rx.finite_or_none(float("nan")))
        self.assertIsNone(rx.finite_or_none(None))
        self.assertEqual(rx.finite_or_none(0.123456, 3), 0.123)

    def test_bar_label_for_decision(self):
        self.assertEqual(rx.bar_label_for_decision("2026-06-19 10:15:00"), "10:14")


class OutcomeTests(unittest.TestCase):
    def test_window_starts_at_decision_and_is_half_open(self):
        bars = _bars(n=10)
        # decision at the close of the 09:02 bar -> decision_time 09:03
        out = rx.forward_outcome(bars, "2026-06-19 09:03:00", 102.0, 3)
        # window = bars starting 09:03, 09:04, 09:05 (highs 103.5..105.5, lows 102.5..)
        self.assertAlmostEqual(out["hi"], 105.5 / 102.0 - 1.0, places=5)
        self.assertAlmostEqual(out["lo"], 102.5 / 102.0 - 1.0, places=5)
        self.assertEqual(out["reveal"], "09:05")
        self.assertTrue(out["full"])

    def test_truncated_window_is_flagged(self):
        bars = _bars(n=5)
        out = rx.forward_outcome(bars, "2026-06-19 09:03:00", 102.0, 60)
        self.assertEqual(out["reveal"], "09:04")
        self.assertFalse(out["full"])

    def test_end_of_day_and_empty_window(self):
        bars = _bars(n=5)
        eod = rx.forward_outcome(bars, "2026-06-19 09:01:00", 100.0, None)
        self.assertEqual(eod["reveal"], "09:04")
        self.assertIsNone(rx.forward_outcome(bars, "2026-06-19 09:05:00", 104.0, None))

    def test_touch_reveal_is_last_bar_of_ten_minute_window(self):
        self.assertEqual(rx.touch_reveal_label("2026-06-19 10:15:00"), "10:24")


class ValidateTests(unittest.TestCase):
    def _bundle(self, events):
        return {"bars": [["09:00", 1, 1, 1, 1, 1], ["09:01", 1, 1, 1, 1, 1],
                         ["09:02", 1, 1, 1, 1, 1]], "events": events}

    def test_clean_bundle(self):
        ev = {"id": 0, "bar": "09:00", "outcome": {"h60": {"reveal": "09:02"}}}
        self.assertEqual(rx.validate_bundle(self._bundle([ev])), [])

    def test_outcome_revealed_too_early_is_rejected(self):
        ev = {"id": 0, "bar": "09:01", "outcome": {"h60": {"reveal": "09:01"}}}
        self.assertTrue(rx.validate_bundle(self._bundle([ev])))

    def test_unknown_bar_is_rejected(self):
        self.assertTrue(rx.validate_bundle(self._bundle([{"id": 0, "bar": "11:00"}])))


class ExporterGuardTests(unittest.TestCase):
    def test_exporter_places_no_orders_and_hardcodes_no_symbol(self):
        text = EXPORTER.read_text(encoding="utf-8")
        for forbidden in ("SendOrder", "send_order", "sendorder", "place_order"):
            self.assertNotIn(forbidden, text)
        self.assertIsNone(re.search(r"""['"]0\d{5}['"]""", text))


if __name__ == "__main__":
    unittest.main()
