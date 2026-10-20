"""Generate a separate, reproducible set of the app's exact replay bundles.

Uses the existing exporter and frozen training universe, with read-only DB
connections. Run with the Windows 64-bit model runtime. Existing app data is
never overwritten. Parallel workers share no mutable engine state.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
# Avoid oversubscribing model inference while generating independent days.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def build_one(job):
    date, symbol, outdir = job
    import export_replay_bundle as export
    from risk_model import load_model, resolve_model_path
    if not hasattr(build_one, "models"):
        lm, lp, _ = load_model(str(resolve_model_path("v1")))
        sm, sp, _ = load_model(export.lsta.DEFAULT_SHORT_MODEL_PATH)
        build_one.models = (lm, lp, sm, sp)
    with sqlite3.connect((ROOT / "db" / "market_data.sqlite").as_uri() + "?mode=ro",
                         uri=True) as conn:
        bundle = export.build_bundle(conn, date, symbol, symbol,
                                     export.trained_universe(), build_one.models)
    path = Path(outdir) / date / f"{symbol}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    return {"date": date, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bars": len(bundle["bars"]), "counts": bundle["meta"]["counts"],
            "grade_misses": bundle["meta"]["touch_grade_misses"]}


def main():
    from settings import realtime_default_symbol, load_data_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-07-16")
    parser.add_argument("--code", default=realtime_default_symbol())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default=str(ROOT / "data/processed/short_horizon_replay"))
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect((ROOT / "db/market_data.sqlite").as_uri() + "?mode=ro",
                         uri=True) as conn:
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT substr(bar_start,1,10) FROM bars_1m "
            "WHERE source=? AND symbol=? AND substr(bar_start,1,10) BETWEEN ? AND ? "
            "ORDER BY 1", (load_data_config().model_source, args.code, args.start, args.end))]
    print(f"Generate {len(dates)} dates, {args.workers} workers -> {out}", flush=True)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(build_one, (d, args.code, str(out))): d for d in dates}
        for future in as_completed(jobs):
            date = jobs[future]
            try:
                results.append(future.result())
                print(f"Completed {len(results)}/{len(dates)}: {date}", flush=True)
            except Exception as exc:
                errors.append({"date": date, "error": repr(exc)})
                print(f"FAILED {date}: {exc!r}", flush=True)
    manifest = {"requested_start": args.start, "requested_end": args.end,
                "symbol": args.code, "dates": sorted(results, key=lambda r: r["date"]),
                "errors": errors, "generation": "existing app exporter; frozen trained universe",
                "command": sys.argv}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
