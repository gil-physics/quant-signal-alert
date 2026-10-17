"""Tests for the Step-22 touch serving stack (prod registry, grade map, adapter
glue, runner/parity/grade-map scripts).

Pins: the touch_cnn_prod registry entry (frozen knobs == touch_cnn, 5 seeds,
two-stage refit, registry split), grade-map math (percentile thresholds, per-grade
rates/N, monotonicity hard-fail, catch-all application, JSON round-trip, no 0.7%
surface), build_touch_setups <-> build_touch_events consistency, the shared
serving glue (coverage rule replicating the features_1m ETL dropna, serving-frame
build on synthetic bars), trainer prod/multi-seed guard functions (exec'd from the
script text -- scripts stay off sys.path), and the static scans for the new files
(no order API, docstring conventions, lazy torch, no hard-coded symbol codes, no
hard-coded session-clock literals in the runner).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import touch_model as tmm  # noqa: E402
import touch_serving as tsv  # noqa: E402
from feature_lib import FEATURE_COLUMNS_1M  # noqa: E402


class ProdRegistryTests(unittest.TestCase):
    def test_submission_seeds_and_split(self):
        self.assertEqual(tmm.TOUCH_SUBMISSION_SEEDS, (42, 43, 44, 45, 46))
        split = tmm.TOUCH_PRODUCTION_SPLIT
        self.assertEqual(split.train_start, "2025-06-02")
        self.assertIsNone(split.train_end)        # resolved at run time (latest day)
        self.assertIsNone(split.validation_start)  # user decision: no forward val
        self.assertIsNone(split.validation_end)

    def test_prod_entry_freezes_touch_cnn_knobs(self):
        prod = tmm.touch_version_config("touch_cnn_prod")
        cnn = tmm.touch_version_config("touch_cnn")
        for knob in ("kind", "target", "seq_len", "batch_size", "max_epochs", "lr",
                     "weight_decay", "es_patience", "holdout_frac", "holdout_by_day",
                     "static_features", "arch", "channels", "levels", "kernel_size",
                     "pooling"):
            self.assertEqual(prod[knob], cnn[knob], f"prod knob {knob} drifted")
        self.assertEqual(prod["seeds"], tmm.TOUCH_SUBMISSION_SEEDS)
        self.assertFalse(prod["folds"])
        self.assertIs(prod["split"], tmm.TOUCH_PRODUCTION_SPLIT)
        self.assertTrue(prod["two_stage_refit"])
        self.assertEqual(prod["filename_pattern"],
                         "touch_model_cnn_prod_seed{seed}.pt")


def _oof(n=1000, invert=False):
    scores = np.arange(n, dtype=float) / (n - 1)
    rank = np.arange(n)
    y = np.zeros(n, dtype=int)
    y[rank >= 900] = 1                                     # A: rate 1.0
    y[(rank >= 700) & (rank < 900) & (rank % 3 == 0)] = 1  # B: ~1/3
    y[(rank >= 200) & (rank < 700) & (rank % 10 == 0)] = 1  # C: ~1/10
    if invert:
        y = y[::-1].copy()
    return pd.DataFrame({
        "prob": scores,
        tmm.TOUCH_TARGET: y,
        "month": np.where(rank % 2 == 0, "2026-05", "2026-06"),
    })


class GradeMapTests(unittest.TestCase):
    def _oof(self, n=1000, invert=False):
        return _oof(n, invert)

    def test_boundaries_pre_registered(self):
        self.assertEqual(tmm.GRADE_BOUNDARIES,
                         (("A", 0.90), ("B", 0.70), ("C", 0.20), ("D", 0.0)))

    def test_build_thresholds_counts_and_monthly(self):
        oof = self._oof()
        gm = tmm.build_grade_map(oof)
        names = [g for g, _ in tmm.GRADE_BOUNDARIES]
        self.assertEqual(sum(gm["grades"][g]["n"] for g in names), len(oof))
        self.assertEqual(gm["n_rows"], len(oof))
        for g, pct in tmm.GRADE_BOUNDARIES:
            self.assertAlmostEqual(gm["thresholds"][g],
                                   float(np.quantile(oof["prob"], pct)))
        # ~10/20/50/20 partition (quantile edge rounding tolerated)
        self.assertAlmostEqual(gm["grades"]["A"]["n"], 100, delta=2)
        self.assertAlmostEqual(gm["grades"]["B"]["n"], 200, delta=2)
        self.assertAlmostEqual(gm["grades"]["C"]["n"], 500, delta=2)
        self.assertAlmostEqual(gm["grades"]["D"]["n"], 200, delta=2)
        # rates monotone by construction; A = 1.0, D = 0.0
        self.assertAlmostEqual(gm["grades"]["A"]["touch_rate"], 1.0)
        self.assertAlmostEqual(gm["grades"]["D"]["touch_rate"], 0.0)
        for g in names:
            monthly = gm["grades"][g]["monthly"]
            self.assertEqual(sum(v["n"] for v in monthly.values()),
                             gm["grades"][g]["n"])
        # the map never carries the 0.7% auxiliary label (user modification #2)
        self.assertNotIn("07", json.dumps(gm))
        self.assertEqual(gm["target"], tmm.TOUCH_TARGET)

    def test_monotonicity_hard_fail(self):
        with self.assertRaises(ValueError):
            tmm.build_grade_map(self._oof(invert=True))

    def test_degenerate_scores_fail(self):
        oof = self._oof()
        oof["prob"] = 0.5                                    # all thresholds equal
        with self.assertRaises(ValueError):
            tmm.build_grade_map(oof)
        oof2 = self._oof()
        oof2.loc[0, "prob"] = np.nan
        with self.assertRaises(ValueError):
            tmm.build_grade_map(oof2)

    def test_apply_grade_catch_all_and_nan(self):
        gm = tmm.build_grade_map(self._oof())
        grades = tmm.apply_grade(
            np.array([2.0, gm["thresholds"]["B"], -1.0, np.nan]), gm)
        self.assertEqual(grades[0], "A")       # above every threshold
        self.assertEqual(grades[1], "B")       # boundary is inclusive
        self.assertEqual(grades[2], "D")       # below the OOF min -> catch-all
        self.assertIsNone(grades[3])           # scoring failure is never graded

    def test_json_roundtrip(self):
        gm = tmm.build_grade_map(self._oof())
        gm2 = json.loads(json.dumps(gm))
        scores = np.array([0.05, 0.5, 0.85, 0.99])
        self.assertEqual(list(tmm.apply_grade(scores, gm)),
                         list(tmm.apply_grade(scores, gm2)))


class BuildTouchSetupsTests(unittest.TestCase):
    def _bars(self):
        def seq(date, start_hhmm, prices, symbol="TESTSYM"):
            hh, mm = int(start_hhmm[:2]), int(start_hhmm[3:5])
            rows = []
            for i, p in enumerate(prices):
                t = hh * 60 + mm + i
                rows.append({"symbol": symbol,
                             "bar_start": f"{date} {t // 60:02d}:{t % 60:02d}:00",
                             "open": p, "high": p + 1, "low": p - 1, "close": p})
            return rows
        day1 = seq("2026-01-05", "09:00", [1000.0] * 25 + [900.0] * 10)
        day2 = seq("2026-01-06", "09:00", [900.0] * 3 + [1000.0] * 5 + [1006.0] * 12)
        return pd.DataFrame(day1 + day2)

    def test_setups_match_events_setup_columns(self):
        bars = self._bars()
        setups, sstats = tmm.build_touch_setups(bars)
        events, estats = tmm.build_touch_events(bars)
        self.assertEqual(sstats["n_crosses_eligible"], estats["n_crosses_eligible"])
        self.assertEqual(sstats["n_premarket_excluded"],
                         estats["n_premarket_excluded"])
        # every labelled event exists in the setups with identical setup features
        m = events.merge(setups, on=["symbol", "decision_time"],
                         suffixes=("_e", "_s"), validate="one_to_one")
        self.assertEqual(len(m), len(events))
        for col in ("direction", "band", "early_flag", "minutes_since_open"):
            self.assertTrue((m[f"{col}_e"] == m[f"{col}_s"]).all(), col)
        np.testing.assert_allclose(m["dist_pct_e"], m["dist_pct_s"])
        # the setups carry what the live runner needs beyond the event columns
        for col in ("close", "ma_fast", "ma_slow", "atr", "open"):
            self.assertIn(col, setups.columns)


class ServingGlueTests(unittest.TestCase):
    def test_feature_covered_rows_replicates_etl_rule(self):
        frame = pd.DataFrame({c: [1.0, 2.0] for c in FEATURE_COLUMNS_1M})
        frame["symbol"] = ["TESTAAA", "TESTAAA"]
        frame.loc[1, "ret_60m_1m"] = np.nan          # warmup row -> dropped
        out = tsv.feature_covered_rows(frame)
        self.assertEqual(len(out), 1)
        with self.assertRaises(ValueError):
            tsv.feature_covered_rows(frame.drop(columns=["ret_60m_1m"]))

    def test_validate_serving_schema(self):
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        tsv.validate_serving_schema(df, ["a"], tag="t")
        with self.assertRaises(ValueError):
            tsv.validate_serving_schema(df, ["a", "missing"], tag="t")
        with self.assertRaises(ValueError):
            tsv.validate_serving_schema(df, ["b"], tag="t")

    def test_resolve_touch_model_path(self):
        p = tsv.resolve_touch_model_path(seed=43)
        self.assertEqual(p.name, "touch_model_cnn_prod_seed43.pt")
        p2 = tsv.resolve_touch_model_path(seed=43, artifact_suffix="_smoke")
        self.assertEqual(p2.name, "touch_model_cnn_prod_smoke_seed43.pt")

    def test_load_grade_map_missing_and_bad(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                tsv.load_grade_map(Path(td) / "nope.json")
            bad = Path(td) / "bad.json"
            bad.write_text(json.dumps({"boundaries": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                tsv.load_grade_map(bad)
            gm = tmm.build_grade_map(_oof())
            good = Path(td) / "ok.json"
            good.write_text(json.dumps(gm), encoding="utf-8")
            loaded = tsv.load_grade_map(good)
            rate, n = tsv.grade_display(loaded, "A")
            self.assertAlmostEqual(rate, 1.0)
            self.assertGreater(n, 0)

    def test_adapter_missing_artifacts_raise(self):
        # the runner catches this and degrades to breakout-only (modification #7)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                tsv.TouchModelAdapter(models_dir=td)

    def test_build_serving_frames_synthetic(self):
        rows = []
        for symbol, base in (("TESTAAA", 1000.0), ("TESTBBB", 500.0)):
            for i in range(71):                     # 09:00..10:10 regular bars
                t = 9 * 60 + i
                ts_ = f"2026-01-06 {t // 60:02d}:{t % 60:02d}:00"
                te = t + 1
                rows.append({"symbol": symbol, "bar_start": ts_,
                             "bar_end": f"2026-01-06 {te // 60:02d}:{te % 60:02d}:00",
                             "open": base + i, "high": base + i + 1,
                             "low": base + i - 1, "close": base + i,
                             "volume": 100 + i})
        covered, rel = tsv.build_serving_frames(pd.DataFrame(rows), source="al_regular")
        self.assertGreater(len(covered), 0)
        # coverage rule == ETL dropna: ret_60m needs 60 prior bars, so the first
        # covered decision_time is ~10:01 -- nothing earlier survives
        self.assertTrue((covered["decision_time"].str.slice(11, 19) >= "10:00:00").all())
        for c in FEATURE_COLUMNS_1M:
            self.assertIn(c, covered.columns)
            self.assertFalse(covered[c].isna().any(), c)
        # relatives: one row per covered (symbol, decision_time), fixed 8 columns
        self.assertEqual(len(rel), len(covered))
        for c in tmm.RELATIVE_FEATURES_V1:
            self.assertIn(c, rel.columns)


class TrainerGuardTests(unittest.TestCase):
    """exec resolve_seeds/_default_knobs/artifact_names from the trainer text."""

    @classmethod
    def setUpClass(cls):
        text = (SCRIPTS_DIR / "train_touch_model.py").read_text(encoding="utf-8")
        ns = {"tmm": tmm, "Path": Path, "MODELS_DIR": Path("models"),
              "PROCESSED_DIR": Path("processed"), "DB_PATH": Path("db.sqlite")}
        for name in ("resolve_seeds", "_default_knobs", "artifact_names"):
            # a top-level def runs until the next column-0 statement (bodies are
            # indented, so this is robust to blank-line style)
            m = re.search(rf"^def {name}\(.*?(?=^[A-Za-z@])", text,
                          re.MULTILINE | re.DOTALL)
            assert m, name
            exec(m.group(0), ns)  # noqa: S102 (our own script, test-only)
        cls.ns = ns
        cls.text = text

    def _args(self, **kw):
        base = dict(seeds="", seed=42, mode="pretrain", max_epochs=None,
                    db=str(Path("db.sqlite")))
        base.update(kw)
        return SimpleNamespace(**base)

    def test_resolve_seeds(self):
        fn = self.ns["resolve_seeds"]
        self.assertEqual(fn(self._args(seeds="42,43"), {}), (42, 43))
        self.assertEqual(fn(self._args(), {}), (42,))
        self.assertEqual(fn(self._args(mode="prod"),
                            {"seeds": tmm.TOUCH_SUBMISSION_SEEDS}),
                         tmm.TOUCH_SUBMISSION_SEEDS)
        with self.assertRaises(SystemExit):
            fn(self._args(seeds="42,42"), {})
        with self.assertRaises(SystemExit):
            fn(self._args(seeds="a,b"), {})

    def test_default_knobs_seed_sets(self):
        fn = self.ns["_default_knobs"]
        cfg = {"max_epochs": 40}
        self.assertTrue(fn(self._args(), cfg, (42,)))
        self.assertTrue(fn(self._args(), cfg, tmm.TOUCH_SUBMISSION_SEEDS))
        self.assertFalse(fn(self._args(), cfg, (42, 43)))       # smoke set
        self.assertFalse(fn(self._args(max_epochs=2), cfg, (42,)))

    def test_artifact_names_namespaces(self):
        fn = self.ns["artifact_names"]
        model, pred = fn("cnn", "pretrain", "", "", 1, 42)
        self.assertEqual(model.name, "touch_model_cnn_fold1_seed42.pt")   # frozen
        self.assertEqual(pred.name, "touch_predictions_cnn_fold1.parquet")
        model5, pred5 = fn("cnn", "pretrain", "", "", 2, 43, n_seeds=5)
        self.assertEqual(model5.name, "touch_model_cnn_5seed_fold2_seed43.pt")
        self.assertEqual(pred5.name, "touch_predictions_cnn_5seed_fold2.parquet")
        prod, none = fn("cnn", "prod", "", "", None, 44)
        self.assertEqual(prod.name, "touch_model_cnn_prod_seed44.pt")
        self.assertIsNone(none)
        prod_s, _ = fn("cnn", "prod", "", "_smoke", None, 44)
        self.assertEqual(prod_s.name, "touch_model_cnn_prod_smoke_seed44.pt")

    def test_prod_mode_declared(self):
        self.assertIn('"prod"', self.text)
        self.assertIn("two_stage_refit", self.text)
        self.assertIn("lr_schedule=used_lr", self.text)      # stage-1 lr replayed
        self.assertIn("holdout_frac=0.0", self.text)          # stage-2: no holdout/ES


class ServingStaticScanTests(unittest.TestCase):
    FORBIDDEN = ("SendOrder(", "send_order(", "OrderRequest(", "place_order(",
                 ".SendOrder")
    SCRIPT_FILES = ("realtime_touch_alert.py", "check_touch_alert_parity.py",
                    "build_touch_grade_map.py")

    def test_no_order_api_and_no_symbol_literals(self):
        paths = [SCRIPTS_DIR / n for n in self.SCRIPT_FILES] \
            + [SRC_DIR / "touch_serving.py"]
        for path in paths:
            self.assertTrue(path.exists(), path.name)
            text = path.read_text(encoding="utf-8")
            for token in self.FORBIDDEN:
                self.assertNotIn(token, text, path.name)
            self.assertIsNone(re.search(r"['\"]\d{6}['\"]", text), path.name)

    def test_docstring_conventions(self):
        runner = (SCRIPTS_DIR / "realtime_touch_alert.py").read_text(encoding="utf-8")
        self.assertIn("decision-support ONLY", runner)
        self.assertIn("NEVER places", runner)
        serving = (SRC_DIR / "touch_serving.py").read_text(encoding="utf-8")
        self.assertIn("decision-support ONLY", serving)
        for name in ("check_touch_alert_parity.py", "build_touch_grade_map.py"):
            text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            self.assertIn("Research / evaluation only", text)

    def test_lazy_torch(self):
        for path in (SRC_DIR / "touch_serving.py",
                     SCRIPTS_DIR / "realtime_touch_alert.py",
                     SCRIPTS_DIR / "check_touch_alert_parity.py",
                     SCRIPTS_DIR / "build_touch_grade_map.py"):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"^import torch|^from torch", text,
                                        re.MULTILINE), path.name)

    def test_runner_no_session_clock_literals(self):
        # user modification #4: the tier-2 cutoff is DERIVED from REGULAR_END and
        # TOUCH_WINDOW_MIN -- no 15:xx literal may appear in the runner.
        text = (SCRIPTS_DIR / "realtime_touch_alert.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"15:\d{2}", text))
        self.assertIn("GRADE_LAST_DECISION", text)
        self.assertIn("REGULAR_END", text)

    def test_runner_two_tier_and_failure_isolation(self):
        text = (SCRIPTS_DIR / "realtime_touch_alert.py").read_text(encoding="utf-8")
        # tier-1 survives scoring failures; no cooldown suppression; grade line
        # cites the OOF rate, never a probability word
        self.assertIn("grade_disabled_reason", text)
        self.assertIn("점수 실패", text)
        self.assertIn("워밍업", text)
        self.assertIn("OOF", text)
        self.assertNotIn("확률=", text)
        for field in ("cross_confirmed_time", "alert_emitted_time",
                      "alert_latency_ms", "price_at_alert",
                      "offline_reference_entry", "entry_slippage_vs_open_t1",
                      "realized_touch_05", "realized_mae"):
            self.assertIn(field, text)
        # the 0.7% auxiliary label never reaches the serving surface (mod #2)
        self.assertNotIn("realized_touch_07", text)
        self.assertNotIn("TOUCH_TARGET_AUX", text)


if __name__ == "__main__":
    unittest.main()
