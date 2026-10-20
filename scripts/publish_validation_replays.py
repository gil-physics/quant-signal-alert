"""Prepare validated replay JSON for Hosting; does not compute signals or deploy."""

import argparse
import hashlib
import json
import sys
from datetime import date as calendar_date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_catalog import write_catalog


def publish(source: Path, destination: Path, featured: Path, name: str | None = None):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["errors"] or not manifest["dates"]:
        raise ValueError("Only a complete, successful validation manifest can be published")
    symbol = manifest["symbol"]
    if not symbol.isdigit():
        raise ValueError("Invalid symbol")
    if name is None:
        current = json.loads((destination / "index.json").read_text(encoding="utf-8"))
        if current["symbol"] != symbol:
            raise ValueError("Catalog symbol differs from validation manifest")
        name = current["name"]
    labels = {d["date"]: d.get("label", "") for d in
              json.loads(featured.read_text(encoding="utf-8"))["dates"]}
    pending, seen = [], set()
    # Check every source and existing file before making any changes.
    for entry in manifest["dates"]:
        date = entry["date"]
        if calendar_date.fromisoformat(date).isoformat() != date or date in seen:
            raise ValueError(f"Invalid or repeated date: {date}")
        seen.add(date)
        raw = (source / date / f"{symbol}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"{date}: source differs from validated SHA-256")
        bundle = json.loads(raw)
        meta = bundle["meta"]
        if (meta["date"], meta["symbol"], len(bundle["bars"]), meta["counts"]) != (
                date, symbol, entry["bars"], entry["counts"]):
            raise ValueError(f"{date}: source metadata differs from manifest")
        target = destination / date / f"{symbol}.json"
        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            if any(old[key] != bundle[key] for key in ("bars", "ma", "events")):
                raise ValueError(f"{date}: existing chart/signals differ; refusing overwrite")
            # Preserve original featured exports, including generation timestamps.
            if old["meta"]["name"] == name:
                continue
        meta["name"] = name  # Validation uses the symbol code as its display name.
        pending.append((target, json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))))
    if not set(labels).issubset(seen):
        raise ValueError("Featured dates must be present in the validation manifest")
    extras = {p.parent.name for p in destination.glob(f"*/{symbol}.json")} - seen
    if extras:
        raise ValueError(f"Destination contains dates outside the validation manifest: {sorted(extras)}")
    for target, text in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return write_catalog(destination, symbol, name, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/short_horizon_replay")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "app/public/data")
    parser.add_argument("--featured", type=Path, default=ROOT / "app/replay_dates.json")
    parser.add_argument("--name", help="Display name; defaults to the current app catalog")
    args = parser.parse_args()
    catalog = publish(args.source, args.out_dir, args.featured, args.name)
    print(f"Prepared {len(catalog['availableDates'])} dates, {len(catalog['dates'])} featured; no inference or deployment.")


if __name__ == "__main__":
    main()
