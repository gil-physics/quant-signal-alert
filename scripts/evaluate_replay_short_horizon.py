"""Evaluate frozen replay alerts without fitting or changing signal thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from replay_short_horizon_eval import (
    Day, cluster_interval, evaluate_signals, matched_baselines,
    signals_from_bundle, summarize,
)


def timing_audit(frame, curated):
    """Paired comparison: identical events in both modes, no sample mismatch."""
    rows = []
    keys = ["symbol", "date", "decision_time", "family", "direction", "strategy"]
    for old_mode, new_mode in (("legacy_skip_first", "decision_close"),
                               ("legacy_delay3_close", "delay3_close"),
                               ("decision_close", "next_open")):
        old = frame[frame["entry_mode"] == old_mode]
        new = frame[frame["entry_mode"] == new_mode]
        paired = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
        paired["delta"] = paired["gross_new"] - paired["gross_old"]
        for family, part in paired.groupby("family"):
            for scope in ("all_dates", "curated_dates", "other_dates"):
                mask = part["date"].isin(curated)
                selected = part if scope == "all_dates" else part[
                    mask if scope == "curated_dates" else ~mask]
                for strategy, group in selected.groupby("strategy"):
                    low, high = cluster_interval(group, "delta")
                    rows.append({
                        "old_mode": old_mode, "new_mode": new_mode, "family": family,
                        "scope": scope, "strategy": strategy, "paired_n": len(group),
                        "old_gross": group["gross_old"].mean(),
                        "new_gross": group["gross_new"].mean(), "delta": group["delta"].mean(),
                        "delta_ci_low": low, "delta_ci_high": high,
                        "outcome_changed": (group["outcome_old"] != group["outcome_new"]).sum(),
                        "loss_to_win": ((group["gross_old"] <= 0) & (group["gross_new"] > 0)).sum(),
                        "win_to_loss": ((group["gross_old"] > 0) & (group["gross_new"] <= 0)).sum(),
                    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "app/public/data")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.bundle_dir.glob("*/*.json"))
    if not paths:
        raise SystemExit("No replay bundles found")
    days, signals, fingerprints = {}, [], []
    for path in paths:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        day = Day.from_bundle(bundle)
        key = (day.symbol, day.date)
        if key in days:
            raise ValueError(f"duplicate symbol/date: {key}")
        days[key] = day
        signals.extend(signals_from_bundle(bundle))
        fingerprints.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    curated = {r["date"] for r in json.loads((ROOT / "app/replay_dates.json").read_text())["dates"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(days)} symbol-days, {len(signals)} directional events", flush=True)
    frame, excluded = evaluate_signals(signals, days)
    print(f"{len(frame)} evaluations, {len(excluded)} excluded event/mode/horizon windows", flush=True)
    frame = matched_baselines(frame, signals, days)
    summary = summarize(frame, signals, curated)
    frame.to_csv(args.out_dir / "events.csv", index=False)
    excluded.to_csv(args.out_dir / "exclusions.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    timing_audit(frame, curated).to_csv(args.out_dir / "timing_audit.csv", index=False)
    monthly = frame.groupby(["family", "entry_mode", "strategy", frame["date"].str[:7]]).agg(
        n=("gross", "size"), gross=("gross", "mean"), baseline_edge=("baseline_edge", "mean"))
    monthly["net_cost_0.002"] = monthly["gross"] - .002
    monthly.to_csv(args.out_dir / "monthly.csv")
    manifest = {
        "command": sys.argv, "dates": sorted({d.date for d in days.values()}),
        "signals": len(signals), "evaluations": len(frame),
        "excluded_windows": len(excluded), "inputs": fingerprints,
        "bootstrap": {"unit": "whole trading date", "repetitions": 2000, "seed": 20261020},
        "baseline": "same symbol/month/clock/direction, other non-signal dates",
        "notes": ["Event returns, not cumulative portfolio returns.",
                  "All configurations reported; no parameter selection.",
                  "Down returns assume a hypothetical short, no borrow/availability model.",
                  "MFE/MAE describe the full horizon even for early bracket exits.",
                  "Legacy entry modes reproduce the old indexing for diagnostics only."],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    primary = summary[(summary["scope"] == "all_dates") &
                      (summary["selection"] == "all_alerts") &
                      (summary["direction"] == 0) &
                      (summary["entry_mode"] == "next_open")]
    print(primary[["cohort", "strategy", "n", "gross", "net_cost_0.002",
                   "baseline_edge"]].to_string(index=False), flush=True)
    print(f"Results -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
