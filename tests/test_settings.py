"""Tests for Step 0 (baseline) and Step 1 (symbol list) config loading.

Run from anywhere:
    python -m unittest discover -s quant/tests
or:
    ..\\Python311-64\\python.exe -m unittest tests.test_settings   (cwd = quant)
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

import settings  # noqa: E402
from settings import (  # noqa: E402
    active_symbol_codes,
    default_symbol,
    load_app_config,
    load_symbols,
    normalize_symbol_code,
)


class NormalizeSymbolTests(unittest.TestCase):
    def test_zero_pads_short_numeric_codes(self):
        self.assertEqual(normalize_symbol_code("28260"), "028260")
        self.assertEqual(normalize_symbol_code("12450"), "012450")

    def test_leaves_full_codes_unchanged(self):
        self.assertEqual(normalize_symbol_code("373220"), "373220")
        self.assertEqual(normalize_symbol_code("005930"), "005930")

    def test_strips_whitespace(self):
        self.assertEqual(normalize_symbol_code("402340 "), "402340")


class SymbolListTests(unittest.TestCase):
    def test_symbols_load_and_are_valid(self):
        symbols = load_symbols()
        self.assertGreaterEqual(len(symbols), 3)
        codes = [s.code for s in symbols]
        self.assertEqual(len(codes), len(set(codes)), "symbol codes must be unique")
        by_code = {s.code: s for s in symbols}
        self.assertIn("005930", by_code)
        self.assertEqual(by_code["005930"].name, "삼성전자")
        self.assertEqual(by_code["005930"].sector, "semiconductor")

    def test_codes_are_stripped(self):
        # The fixture intentionally contains "402340 " with a trailing space.
        codes = [s.code for s in load_symbols()]
        self.assertIn("402340", codes)
        for c in codes:
            self.assertEqual(c, c.strip(), "codes must be stripped of whitespace")

    def test_active_symbols_are_valid_subset(self):
        # The active set is user-controlled (started at 2, expanded as data arrives);
        # validate invariants instead of pinning a specific count.
        all_codes = [s.code for s in load_symbols()]
        active = active_symbol_codes()
        self.assertGreaterEqual(len(active), 1)
        self.assertTrue(set(active).issubset(set(all_codes)))
        self.assertEqual(len(active), len(set(active)), "active codes must be unique")
        # active codes preserve file order
        self.assertEqual(active, [c for c in all_codes if c in set(active)])

    def test_default_symbol_is_first_active(self):
        self.assertEqual(default_symbol(), "005930")

    def test_codes_are_strings_not_ints(self):
        # Leading-zero codes must survive as strings.
        for s in load_symbols():
            self.assertIsInstance(s.code, str)

    def test_unquoted_code_is_rejected(self):
        # An unquoted YAML code (000660 -> int 660) would silently lose leading
        # zeros; load_symbols must reject it with a clear error instead.
        import tempfile

        bad = "symbols:\n  - code: 000660\n    name: x\n    sector: y\n    active: true\n"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(bad)
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                load_symbols(path)
        finally:
            Path(path).unlink()


class BaselineConfigTests(unittest.TestCase):
    def test_session_matches_spec(self):
        session = load_app_config().session
        self.assertEqual(session.train_start, "09:00:00")
        self.assertEqual(session.train_end, "15:30:00")
        self.assertEqual(session.raw_start, "08:00:00")
        self.assertEqual(session.raw_end, "20:00:00")

    def test_signals_match_spec(self):
        signals = load_app_config().signals
        self.assertEqual(signals.max_signals_per_day, 2)
        self.assertEqual(signals.cooldown_minutes_per_symbol, 60)
        self.assertEqual(signals.min_minutes_after_open, 10)
        self.assertEqual(signals.no_new_signal_minutes_before_close, 30)


class FallbackParserTests(unittest.TestCase):
    """The built-in parser must agree with PyYAML on the real config files."""

    def _assert_fallback_matches(self, path: Path):
        text = path.read_text(encoding="utf-8")
        fallback = settings._parse_simple_yaml(text)
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed; only fallback path exercised")
            return
        self.assertEqual(fallback, yaml.safe_load(text))

    def test_symbols_file(self):
        self._assert_fallback_matches(ROOT_DIR / "config" / "symbols_test.yaml")

    def test_config_file(self):
        self._assert_fallback_matches(ROOT_DIR / "config" / "config.yaml")


class PerSymbolPathTests(unittest.TestCase):
    def test_parquet_paths_embed_symbol(self):
        import config

        self.assertIn("000660", str(config.raw_parquet_path("000660")))
        self.assertIn("123456", str(config.feature_parquet_path("123456")))


class NoHardcodedSymbolTests(unittest.TestCase):
    """Step 1 completion criterion: no symbol code is hard-coded in source logic."""

    QUOTED_CODE = re.compile(r"""['"]0\d{5}['"]""")

    def test_no_quoted_six_digit_code_in_sources(self):
        offenders = []
        for folder in (SRC_DIR, SCRIPTS_DIR):
            for py in folder.glob("*.py"):
                for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                    if self.QUOTED_CODE.search(line):
                        offenders.append(f"{py.name}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "hard-coded symbol codes found:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
