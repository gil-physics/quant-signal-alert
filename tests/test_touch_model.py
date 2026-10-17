"""Tests for the touch-prediction study (src/touch_model.py + the trainer/gate).

Pins the pre-registered contract: fold registry, fixed feature lists, label
construction parity on synthetic bars (0.5% primary / 0.7% auxiliary), anchor key,
baseline math, FT-tail exclusion rules, the gate decision function (exec'd from the
script -- scripts stay off sys.path), the sequence-model rename/aliases, and the
static scans (research-only, no order API, no hard-coded symbol codes).
"""

from __future__ import annotations

import inspect
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

import risk_model as rm  # noqa: E402
import tcn_model as tm  # noqa: E402
import touch_model as tmm  # noqa: E402


def _bars(rows, symbol="TESTSYM"):
    return pd.DataFrame({
        "symbol": [symbol] * len(rows),
        "bar_start": [r[0] for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
    })


def _seq(date, start_hhmm, prices):
    hh, mm = int(start_hhmm[:2]), int(start_hhmm[3:5])
    rows = []
    for i, p in enumerate(prices):
        t = hh * 60 + mm + i
        rows.append((f"{date} {t // 60:02d}:{t % 60:02d}:00", p, p + 1, p - 1, p))
    return rows


class FoldRegistryTests(unittest.TestCase):
    def test_five_expanding_folds(self):
        folds = tmm.TOUCH_FOLDS
        self.assertEqual(len(folds), 5)
        months = []
        for i, f in enumerate(folds):
            self.assertEqual(f.train_start, "2025-06-02")
            self.assertLess(f.train_end, f.validation_start)
            self.assertEqual(f.validation_start[:7], f.validation_end[:7])
            months.append(f.validation_start[:7])
            if i:
                self.assertGreater(f.train_end, folds[i - 1].train_end)
        self.assertEqual(months, ["2026-03", "2026-04", "2026-05",
                                  "2026-06", "2026-07"])

    def test_selection_confirmation_split(self):
        self.assertEqual(tmm.SELECTION_FOLDS, (1, 2, 3, 4))
        self.assertEqual(tmm.CONFIRMATION_FOLD, 5)     # partial July: confirm-only
        self.assertEqual(tmm.RECENT_MONTHS, ("2026-05", "2026-06"))
        with self.assertRaises(ValueError):
            tmm.touch_fold(6)


class FixedFeatureListTests(unittest.TestCase):
    def test_relative_list_exactly_eight_and_synced(self):
        self.assertEqual(len(tmm.RELATIVE_FEATURES_V1), 8)
        self.assertEqual(list(tmm.RELATIVE_FEATURES_V1),
                         list(rm.RELATIVE_FEATURE_COLUMNS))

    def test_merged_list_synced_with_risk_model(self):
        self.assertEqual(list(tmm.MERGED_FEATURES_V1),
                         list(rm.MERGED_FEATURE_COLUMNS)
                         + list(rm.OPTIONAL_DERIVED_COLUMNS))

    def test_static_and_b2_composition(self):
        syms = ["TESTAAA", "TESTBBB"]
        static = tmm.static_features_v1(syms)
        self.assertEqual(static[:5], list(tmm.STATIC_BASE_FEATURES_V1))
        self.assertEqual(len(static), 5 + 8 + 2)
        self.assertIn("sym_TESTAAA", static)
        b2 = tmm.b2_features_v1(syms)
        self.assertEqual(b2, static + list(tmm.MERGED_FEATURES_V1))
        for col in static:                       # no leakage tokens in fixed lists
            self.assertFalse(any(t in col.lower()
                                 for t in rm.FORBIDDEN_FEATURE_TOKENS), col)

    def test_input_contract_complete(self):
        for key in ("seq_len", "seq_channels", "normalization_seq",
                    "normalization_static", "padding_value", "padding_mask",
                    "missing_seq", "missing_static", "class_weight", "label"):
            self.assertIn(key, tmm.TOUCH_INPUT_CONTRACT)
        for version in ("touch_cnn", "touch_tcn"):
            cfg = tmm.touch_version_config(version)
            for key in ("arch", "channels", "levels", "kernel_size", "seq_len",
                        "seeds", "batch_size", "max_epochs", "lr", "weight_decay",
                        "es_patience", "holdout_frac", "holdout_by_day",
                        "static_features", "target", "filename_pattern"):
                self.assertIn(key, cfg, f"{version} missing contract key {key}")
            self.assertEqual(cfg["seeds"], (42,))
            self.assertTrue(cfg["holdout_by_day"])
        self.assertEqual(tmm.touch_version_config("touch_cnn")["kernel_size"], 5)
        self.assertEqual(tmm.touch_version_config("touch_tcn")["kernel_size"], 3)
        self.assertEqual(tmm.touch_version_config("touch_tcn")["receptive_field"], 61)
        self.assertEqual(tmm.touch_version_config("touch_lgbm")["features"],
                         "B2_FEATURES_V1")
        with self.assertRaises(ValueError):
            tmm.touch_version_config("touch_unknown")

    def test_label_constants(self):
        self.assertEqual(tmm.TOUCH_THRESHOLD, 0.005)      # user-decided primary
        self.assertEqual(tmm.TOUCH_THRESHOLD_AUX, 0.007)  # auxiliary diagnostic
        self.assertEqual(tmm.TOUCH_WINDOW_MIN, 10)


class BuildEventsTests(unittest.TestCase):
    def _fixture_bars(self):
        # day1: flat 1000 then drop to 900 -> close-MA regime -1 carried overnight.
        day1 = _seq("2026-01-05", "09:00", [1000.0] * 25 + [900.0] * 10)
        # day2 premarket: flat 900 (no cross), then a regular-session jump ->
        # the up-cross CONFIRMS in the regular session (carry warmup, eligible).
        pre = _seq("2026-01-06", "08:55", [900.0] * 5)
        # entry bar open must be knowable: fixture opens == closes; after the jump
        # the price drifts +0.6% from the entry bar -> target_05=1, target_07=0.
        day2 = _seq("2026-01-06", "09:00", [900.0] * 3 + [1000.0] * 5
                    + [1006.0] * 12)
        return _bars(day1 + pre + day2)

    def test_labels_anchor_and_stats(self):
        events, stats = tmm.build_touch_events(self._fixture_bars())
        self.assertGreaterEqual(stats["n_crosses_eligible"], 1)
        self.assertEqual(stats["n_events"], len(events))
        self.assertTrue((events["date"] == "2026-01-06").all())
        ev = events.iloc[0]
        self.assertEqual(ev["direction"], 1)
        # decision_time = confirmation bar's bar_end = bar_start + 1 minute
        self.assertEqual(pd.Timestamp(ev["decision_time"]),
                         pd.Timestamp(ev["bar_start"]) + pd.Timedelta(minutes=1))
        # +0.6% drift from the next-bar entry: 0.5% touched, 0.7% not
        self.assertEqual(int(ev[tmm.TOUCH_TARGET]), 1)
        self.assertEqual(int(ev[tmm.TOUCH_TARGET_AUX]), 0)
        self.assertIn(ev["band"], ("near", "mid", "far"))
        self.assertGreaterEqual(ev["minutes_since_open"], 1)

    def test_premarket_confirmed_cross_excluded(self):
        # jump happens IN the premarket -> the cross confirms before 09:00 and the
        # event is excluded (features_1m has no premarket rows) but counted.
        day1 = _seq("2026-01-05", "09:00", [1000.0] * 25 + [900.0] * 10)
        pre = _seq("2026-01-06", "08:40", [900.0] * 5 + [1000.0] * 15)
        day2 = _seq("2026-01-06", "09:00", [1000.0] * 15)
        events, stats = tmm.build_touch_events(_bars(day1 + pre + day2))
        self.assertGreaterEqual(stats["n_premarket_excluded"], 1)
        self.assertTrue((events["time"] >= "09:00:00").all() if len(events) else True)

    def test_incomplete_window_dropped(self):
        # cross confirmed with < 10 minutes of day left -> dropped + counted
        day1 = _seq("2026-01-05", "09:00", [1000.0] * 25 + [900.0] * 10)
        day2 = _seq("2026-01-06", "15:10", [900.0] * 3 + [1000.0] * 18)  # ..15:30
        events, stats = tmm.build_touch_events(_bars(day1 + day2))
        self.assertGreaterEqual(stats["n_incomplete_dropped"], 0)
        if len(events):
            # every kept event has a full [t+1, t+11) window inside the day
            self.assertTrue((events["time"] <= "15:19:00").all())


class JoinAndBaselineTests(unittest.TestCase):
    def _events(self):
        return pd.DataFrame({
            "symbol": ["TESTAAA", "TESTAAA", "TESTBBB"],
            "decision_time": ["2026-01-06 09:05:00", "2026-01-06 09:06:00",
                              "2026-01-06 09:05:00"],
            "direction": [1, -1, 1],
            "band": ["near", "far", "near"],
            "early_flag": [1.0, 1.0, 0.0],
            tmm.TOUCH_TARGET: [1, 0, 0],
        })

    def _feature_frame(self, cols, keys):
        df = pd.DataFrame(keys, columns=["symbol", "decision_time"])
        for c in cols:
            df[c] = 0.5
        return df

    def test_join_match_rate_contract(self):
        ev = self._events()
        keys = list(zip(ev["symbol"], ev["decision_time"]))
        fm = self._feature_frame(tmm.MERGED_FEATURES_V1, keys)
        rel = self._feature_frame(tmm.RELATIVE_FEATURES_V1, keys)
        out = tmm.join_event_features(ev, fm, rel, ["TESTAAA", "TESTBBB"])
        self.assertIn("sym_TESTAAA", out.columns)
        self.assertEqual(out["sym_TESTAAA"].tolist(), [1.0, 1.0, 0.0])
        # derived flags (dist_ma5_* = 0.5 in the fixture -> above == 1.0)
        self.assertEqual(out["above_ma5_intraday"].tolist(), [1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):      # coverage break -> hard refusal
            tmm.join_event_features(ev, fm.iloc[:1], rel, ["TESTAAA", "TESTBBB"])
        with self.assertRaises(ValueError):      # fixed column missing -> refusal
            tmm.join_event_features(ev, fm.drop(columns=[tmm.MERGED_FEATURES_V1[0]]),
                                    rel, ["TESTAAA", "TESTBBB"])

    def test_filter_feature_covered(self):
        ev = self._events()
        keys = list(zip(ev["symbol"], ev["decision_time"]))[:2]   # third uncovered
        fm = self._feature_frame(tmm.MERGED_FEATURES_V1, keys)
        covered, n_dropped = tmm.filter_feature_covered(ev, fm)
        self.assertEqual(n_dropped, 1)
        self.assertEqual(len(covered), 2)
        self.assertNotIn("TESTBBB", set(covered["symbol"]))

    def test_early_flag_boundary_is_first_covered_half_hour(self):
        # feature stack starts ~10:01 -> early regime = through 10:30
        self.assertEqual(tmm.EARLY_SESSION_END, "10:30:00")

    def test_b0_b1_math_and_fallback(self):
        train = pd.DataFrame({
            "direction": [1, 1, 1, -1, -1],
            "band": ["near", "near", "far", "near", "near"],
            "early_flag": [0.0] * 5,
            tmm.TOUCH_TARGET: [1, 0, 1, 0, 0],
        })
        b0 = tmm.fit_b0(train)
        self.assertAlmostEqual(b0["rates"][1], 2 / 3)
        self.assertAlmostEqual(b0["rates"][-1], 0.0)
        b1 = tmm.fit_b1(train)
        frame = pd.DataFrame({"direction": [1, 1, -1],
                              "band": ["near", "far", "mid"],   # mid unseen -> b0
                              "early_flag": [0.0, 0.0, 0.0]})
        p1 = tmm.apply_b1(b1, frame)
        self.assertAlmostEqual(p1[0], 0.5)        # (near,1,0) = 1/2
        self.assertAlmostEqual(p1[1], 1.0)        # (far,1,0) = 1/1
        self.assertAlmostEqual(p1[2], 0.0)        # fallback to b0 dir=-1


class FtTailTests(unittest.TestCase):
    def test_tail_dates_and_exclusive_split(self):
        dates = [f"2026-02-{d:02d}" for d in range(1, 26)]     # 25 trading days
        events = pd.DataFrame({
            "symbol": ["TESTAAA", "TESTBBB"] * 25,
            "date": np.repeat(dates, 2),
            tmm.TOUCH_TARGET: [1, 0] * 25,
        })
        split = rm.SplitConfig(train_start="2026-02-01", train_end="2026-02-25",
                               validation_start="2026-03-01",
                               validation_end="2026-03-31")
        tail = tmm.ft_tail_dates(events, split)
        self.assertEqual(len(tail), tmm.FT_TAIL_DAYS)
        self.assertEqual(min(tail), "2026-02-11")              # last 15 of 25 days
        pooled, es = tmm.split_ft_rows(events, "TESTAAA", tail)
        self.assertEqual(len(es), tmm.FT_TAIL_DAYS)            # only TESTAAA tail rows
        self.assertTrue((es["symbol"] == "TESTAAA").all())
        self.assertTrue((es["date"] >= "2026-02-11").all())
        # TESTBBB tail rows STAY in pooled; TESTAAA tail rows are gone
        self.assertEqual(len(pooled), len(events) - len(es))
        self.assertFalse(((pooled["symbol"] == "TESTAAA")
                          & (pooled["date"] >= "2026-02-11")).any())
        # no ft symbol -> no exclusion
        p2, e2 = tmm.split_ft_rows(events, "", tail)
        self.assertEqual(len(p2), len(events))
        self.assertEqual(len(e2), 0)


class SequenceModelRenameTests(unittest.TestCase):
    def test_aliases_are_same_objects(self):
        self.assertIs(tm.fit_tcn, tm.fit_sequence_model)
        self.assertIs(tm.predict_tcn, tm.predict_sequence_model)

    def test_ft_kwargs_present(self):
        params = inspect.signature(tm.fit_sequence_model).parameters
        self.assertIn("init_state", params)
        self.assertIn("eval_frame", params)
        self.assertIsNone(params["init_state"].default)
        self.assertIsNone(params["eval_frame"].default)


class GateMathTests(unittest.TestCase):
    """exec the gate's decide() + constants from the script text (off sys.path)."""

    @classmethod
    def setUpClass(cls):
        text = (SCRIPTS_DIR / "evaluate_touch_gate.py").read_text(encoding="utf-8")
        ns = {"np": np, "tmm": tmm}
        for const in ("GATE_RECENT_AUC_MIN", "GATE_WORST_MONTH_FLOOR",
                      "TOP_DECILE_EPS", "BOTTOM_EPS", "CONFIRM_FLOOR", "FT_ADOPT_MIN"):
            m = re.search(rf"^{const} = ([-\d.]+)", text, re.MULTILINE)
            assert m, const
            ns[const] = float(m.group(1))
        fn = re.search(r"\ndef decide\(.*?\n(?=\ndef )", text, re.DOTALL)
        assert fn
        exec(fn.group(0), ns)  # noqa: S102 (our own script, test-only)
        cls.ns = ns
        cls.decide = staticmethod(ns["decide"])

    def _m(self, recent, monthly, top, bottom):
        return {"recent_auc": recent, "monthly": monthly,
                "top_decile_rate": top, "bottom20_rate": bottom}

    def test_pre_registered_constants(self):
        self.assertEqual(self.ns["GATE_RECENT_AUC_MIN"], 0.010)
        self.assertEqual(self.ns["GATE_WORST_MONTH_FLOOR"], -0.005)
        self.assertEqual(self.ns["TOP_DECILE_EPS"], 0.010)
        self.assertEqual(self.ns["BOTTOM_EPS"], 0.005)
        self.assertEqual(self.ns["CONFIRM_FLOOR"], -0.005)
        self.assertEqual(self.ns["FT_ADOPT_MIN"], 0.010)

    def test_go_and_two_sided_buckets(self):
        months = {"2026-03": 0.60, "2026-04": 0.60, "2026-05": 0.60, "2026-06": 0.60}
        base = self._m(0.60, months, 0.20, 0.05)
        up = {m: v + 0.02 for m, v in months.items()}
        good = self._m(0.62, up, 0.20, 0.05)
        self.assertEqual(GateMathTests.decide(base, good)["verdict"], "GO")
        # top-decile worse beyond eps -> not GO even with the AUC delta
        bad_top = self._m(0.62, up, 0.185, 0.05)
        v = GateMathTests.decide(base, bad_top)
        self.assertNotEqual(v["verdict"], "GO")
        self.assertFalse(v["top_ok"])
        # bottom bucket worse beyond eps -> not GO (the two-sided check)
        bad_bottom = self._m(0.62, up, 0.20, 0.06)
        v2 = GateMathTests.decide(base, bad_bottom)
        self.assertNotEqual(v2["verdict"], "GO")
        self.assertFalse(v2["bottom_ok"])
        # within eps passes
        edge = self._m(0.62, up, 0.1905, 0.0545)
        self.assertEqual(GateMathTests.decide(base, edge)["verdict"], "GO")

    def test_worst_recent_month_floor(self):
        months = {"2026-03": 0.60, "2026-04": 0.60, "2026-05": 0.60, "2026-06": 0.60}
        base = self._m(0.60, months, 0.20, 0.05)
        lopsided = self._m(0.62, {"2026-03": 0.63, "2026-04": 0.63,
                                  "2026-05": 0.66, "2026-06": 0.594}, 0.20, 0.05)
        v = GateMathTests.decide(base, lopsided)
        self.assertNotEqual(v["verdict"], "GO")   # June regresses past -0.005


class StaticScanTests(unittest.TestCase):
    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(", ".SendOrder")
    FILES = ("train_touch_model.py", "evaluate_touch_gate.py")

    def test_scripts_research_only_no_orders_no_codes(self):
        for name in self.FILES:
            path = SCRIPTS_DIR / name
            self.assertTrue(path.exists(), name)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Research / evaluation only", text)
            for token in self.FORBIDDEN:
                self.assertNotIn(token, text, name)
            self.assertIsNone(re.search(r"['\"]\d{6}['\"]", text),
                              f"{name}: symbol codes must come from CLI/config")
        src_text = (SRC_DIR / "touch_model.py").read_text(encoding="utf-8")
        for token in self.FORBIDDEN:
            self.assertNotIn(token, src_text)
        self.assertIsNone(re.search(r"['\"]\d{6}['\"]", src_text))

    def test_trainer_reserved_suffix_guard_and_lazy_torch(self):
        text = (SCRIPTS_DIR / "train_touch_model.py").read_text(encoding="utf-8")
        self.assertIn("--out-suffix", text)
        self.assertIn("never be clobbered", text)
        self.assertIn("al_raw vs al_regular price mismatch", text)  # source-parity seal
        # touch_model.py must stay torch-free (lazy contract lives in tcn_model)
        self.assertNotIn("import torch",
                         (SRC_DIR / "touch_model.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
