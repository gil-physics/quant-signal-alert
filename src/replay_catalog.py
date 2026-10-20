"""Build the small replay catalog without loading models or market databases."""

import json
from pathlib import Path


def write_catalog(data_dir: Path, symbol: str, name: str, date_labels: dict) -> dict:
    featured, available = [], []
    for path in sorted(data_dir.glob(f"*/{symbol}.json")):
        date = path.parent.name
        file = f"data/{date}/{symbol}.json"
        available.append({"date": date, "file": file})
        if date not in date_labels:
            continue
        bundle = json.loads(path.read_text(encoding="utf-8"))
        meta = bundle["meta"]
        closes = [bar[4] for bar in bundle["bars"]]
        step = max(1, len(closes) // 48)
        featured.append({
            "date": date, "label": date_labels[date],
            "change_pct": meta["change_pct"], "counts": meta["counts"],
            "open": meta["open"], "close": meta["close"],
            "prev_close": meta["prev_close"],
            "spark": closes[::step] + ([closes[-1]] if (len(closes) - 1) % step else []),
            "file": file,
        })
    catalog = {
        "version": 2, "symbol": symbol, "name": name,
        "dates": featured, "availableDates": available,
    }
    (data_dir / "index.json").write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return catalog
