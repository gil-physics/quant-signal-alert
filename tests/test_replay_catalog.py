import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from publish_validation_replays import publish


class ReplayCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.output = root / "source", root / "output"
        self.source.mkdir()
        self.output.mkdir()
        self.symbol = "123456"
        self.dates = ["2026-03-03", "2026-03-04"]
        self.entries = []
        for date in self.dates:
            meta = {
                "date": date, "symbol": self.symbol, "name": self.symbol,
                "counts": {"long": 0, "down": 0, "touch": 0, "touch_graded": 0},
                "open": 100, "close": 101, "prev_close": 100, "change_pct": 0.01,
            }
            bundle = {"version": 1, "meta": meta,
                      "bars": [["09:00", 100, 101, 100, 101, 20]],
                      "ma": {"ma5": [100], "ma20": [100]}, "events": []}
            target = self.source / date / f"{self.symbol}.json"
            target.parent.mkdir()
            target.write_text(json.dumps(bundle), encoding="utf-8")
            self.entries.append({"date": date, "bars": 1, "counts": meta["counts"],
                                 "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        (self.source / "manifest.json").write_text(json.dumps({
            "symbol": self.symbol, "dates": self.entries, "errors": [],
        }))
        self.featured = root / "featured.json"
        self.featured.write_text(json.dumps({"dates": [{"date": self.dates[0], "label": "sample"}]}))

    def run_publish(self):
        return publish(self.source, self.output, self.featured, "Sample")

    def test_all_dates_available_but_only_featured_get_home_cards(self):
        index = self.run_publish()
        self.assertEqual([d["date"] for d in index["dates"]], self.dates[:1])
        self.assertEqual([d["date"] for d in index["availableDates"]], self.dates)
        for entry in index["availableDates"]:
            relative = Path(entry["file"]).relative_to("data")
            original = json.loads((self.source / relative).read_text())
            published = json.loads((self.output / relative).read_text())
            for key in ("bars", "ma", "events"):
                self.assertEqual(original[key], published[key])
            self.assertEqual(published["meta"]["name"], "Sample")
        before = {p: p.read_bytes() for p in self.output.rglob("*.json")}
        self.run_publish()
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob("*.json")})

    def test_source_tamper_rejected_before_any_files_are_written(self):
        target = self.source / self.dates[-1] / f"{self.symbol}.json"
        target.write_text(target.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_publish()
        self.assertEqual(list(self.output.rglob("*.json")), [])

    def test_existing_different_signals_are_not_overwritten(self):
        self.run_publish()
        target = self.output / self.dates[-1] / f"{self.symbol}.json"
        bundle = json.loads(target.read_text())
        bundle["events"] = [{"id": 1}]
        target.write_text(json.dumps(bundle))
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.run_publish()
        self.assertEqual(before, target.read_bytes())


if __name__ == "__main__":
    unittest.main()
