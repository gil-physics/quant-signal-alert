"""Step 22: freeze the touch-alert grade map from the 5-seed OOF fold parquets.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Builds
models/touch_grade_map.json, the ONLY score->grade mapping the live runner may use
(user directive: the mapping comes from the walk-forward OOF ensemble predictions,
NEVER from the prod model's own training predictions).

Inputs (must exist first -- the 5-seed OOF rerun on the GPU server):
  data/processed/touch_predictions_cnn_5seed_fold{1..5}.parquet (+ .meta.json)

Seals before pooling:
  * all five folds present, sidecar target/threshold/window identical and equal to
    the pre-registered label constants;
  * sidecar seeds == touch_model.TOUCH_SUBMISSION_SEEDS (the prod ensemble's seeds,
    so the frozen thresholds live in the serving score space);
  * validation ranges pairwise non-overlapping (each OOF row scored exactly once);
  * prob == mean(prob_seed{s}) re-verified per fold (ensemble consistency).

The grade scheme is pre-registered in touch_model.GRADE_BOUNDARIES (A >= p90,
B p70-90, C p20-70, D < p20); touch_model.build_grade_map HARD-FAILS unless the
per-grade OOF touch rates are monotone non-increasing A->D (user modification #6).
Probabilities are class-weighted and UNCALIBRATED -- the map reports each grade's
HISTORICAL +0.5% touch rate, which is what the runner displays (never "확률 X%").

Run (cwd = quant, 64-bit):
  ..\\Python311-64\\python.exe scripts\\build_touch_grade_map.py
  ..\\Python311-64\\python.exe scripts\\build_touch_grade_map.py --force   (replace)
  smoke: ... --pred-tag _2seed_smoke --seeds "42,43" --out-suffix _smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

import touch_model as tmm
from config import PROCESSED_DIR
from utils import now_kst_text

MODELS_DIR = ROOT_DIR / "models"
GRADE_MAP_NAME = "touch_grade_map{suffix}.json"
ENSEMBLE_ATOL = 1e-9


def parse_args():
    p = argparse.ArgumentParser(
        description="Freeze the touch grade map from the 5-seed OOF parquets.")
    p.add_argument("--pred-tag", default="_5seed",
                   help="prediction-file tag between the model name and _fold "
                        "(default _5seed = the canonical 5-seed OOF rerun).")
    p.add_argument("--seeds", default="",
                   help="expected seed list (comma); default = the pre-registered "
                        "TOUCH_SUBMISSION_SEEDS. Smoke maps must also set --out-suffix.")
    p.add_argument("--out-suffix", default="",
                   help="grade-map namespace for smoke/diagnostic builds; the "
                        "canonical (empty) suffix is RESERVED for the pre-registered "
                        "5-seed inputs.")
    p.add_argument("--force", action="store_true",
                   help="replace an existing grade map (default: refuse -- the map "
                        "is a FROZEN artifact).")
    args = p.parse_args()
    if args.out_suffix and not args.out_suffix.startswith("_"):
        p.error("--out-suffix must start with '_'.")
    return args


def expected_seeds(args):
    if args.seeds:
        try:
            seeds = tuple(int(s) for s in args.seeds.split(","))
        except ValueError:
            raise SystemExit(f"--seeds must be a comma list of ints, got {args.seeds!r}")
    else:
        seeds = tuple(tmm.TOUCH_SUBMISSION_SEEDS)
    canonical_inputs = (seeds == tuple(tmm.TOUCH_SUBMISSION_SEEDS)
                        and args.pred_tag == "_5seed")
    if not canonical_inputs and not args.out_suffix:
        raise SystemExit(
            "non-canonical inputs (seeds/pred-tag) must use a distinct --out-suffix "
            "so the frozen canonical grade map can never be clobbered.")
    return seeds


def load_fold(pred_tag, fold, seeds):
    path = PROCESSED_DIR / f"touch_predictions_cnn{pred_tag}_fold{fold}.parquet"
    meta_path = Path(str(path) + ".meta.json")
    if not path.exists() or not meta_path.exists():
        raise SystemExit(f"missing OOF artifact for fold {fold}: {path.name} "
                         "(+ .meta.json) -- run the multi-seed pretrain first "
                         "(train_touch_model.py --model cnn --seeds ...).")
    frame = pd.read_parquet(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # -- seals ------------------------------------------------------------------
    if meta.get("target") != tmm.TOUCH_TARGET or \
            meta.get("threshold") != tmm.TOUCH_THRESHOLD or \
            meta.get("window_min") != tmm.TOUCH_WINDOW_MIN:
        raise SystemExit(f"fold {fold}: sidecar label contract mismatch "
                         f"(target={meta.get('target')!r} threshold="
                         f"{meta.get('threshold')!r} window={meta.get('window_min')!r})")
    if tuple(meta.get("seeds") or ()) != tuple(seeds):
        raise SystemExit(f"fold {fold}: sidecar seeds {meta.get('seeds')} != expected "
                         f"{list(seeds)} -- the grade map must share the prod "
                         "ensemble's seed set.")
    seed_cols = [f"prob_seed{s}" for s in seeds]
    missing = [c for c in seed_cols if c not in frame.columns]
    if missing:
        raise SystemExit(f"fold {fold}: parquet lacks per-seed columns {missing}")
    mean = frame[seed_cols].to_numpy(dtype=float).mean(axis=1)
    if not np.allclose(frame["prob"].to_numpy(dtype=float), mean, atol=ENSEMBLE_ATOL):
        raise SystemExit(f"fold {fold}: prob != mean(prob_seed*) -- ensemble "
                         "consistency seal failed.")
    return frame, meta


def main() -> int:
    args = parse_args()
    seeds = expected_seeds(args)
    out_path = MODELS_DIR / GRADE_MAP_NAME.format(suffix=args.out_suffix)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists -- the grade map is a FROZEN "
                         "artifact; pass --force only to intentionally replace it.")

    n_folds = len(tmm.TOUCH_FOLDS)
    frames, sources, ranges = [], [], []
    for fold in range(1, n_folds + 1):
        frame, meta = load_fold(args.pred_tag, fold, seeds)
        frames.append(frame)
        ranges.append((meta["validation_start"], meta["validation_end"]))
        sources.append({"fold": fold,
                        "file": f"touch_predictions_cnn{args.pred_tag}_fold{fold}.parquet",
                        "rows": int(len(frame)),
                        "validation_start": meta["validation_start"],
                        "validation_end": meta["validation_end"],
                        "validation_auc": meta.get("validation_auc")})
    for i in range(len(ranges)):
        for k in range(i + 1, len(ranges)):
            lo = max(ranges[i][0], ranges[k][0])
            hi = min(ranges[i][1], ranges[k][1])
            if lo <= hi:
                raise SystemExit(f"folds {i + 1} and {k + 1} have overlapping "
                                 f"validation ranges ({lo}..{hi}) -- rows would be "
                                 "double-counted in the pooled OOF.")

    oof = pd.concat(frames, ignore_index=True)
    print(f"[{now_kst_text()}] grade map from {n_folds} folds, "
          f"{len(oof):,} pooled OOF rows, seeds={list(seeds)}")
    grade_map = tmm.build_grade_map(oof)          # hard-fails on non-monotone rates
    grade_map["sources"] = sources
    grade_map["seeds"] = list(seeds)
    grade_map["built_at"] = now_kst_text()

    names = [g for g, _ in tmm.GRADE_BOUNDARIES]
    print(f"  monotonicity A>=B>=C>=D: PASS   base_rate={grade_map['base_rate']:.4f}")
    print(f"  {'grade':<6} {'score range':<24} {'touch rate':>10} {'N':>8}   monthly")
    for idx, g in enumerate(names):
        thr = grade_map["thresholds"][g]
        upper = grade_map["thresholds"][names[idx - 1]] if idx else None
        rng = (f">= {thr:.4f}" if idx == 0
               else f"[{thr:.4f}, {upper:.4f})" if idx < len(names) - 1
               else f"< {grade_map['thresholds'][names[-2]]:.4f} (catch-all)")
        st = grade_map["grades"][g]
        monthly = " ".join(f"{mo[-2:]}:{v['touch_rate']:.2f}"
                           for mo, v in sorted(st["monthly"].items()))
        print(f"  {g:<6} {rng:<24} {st['touch_rate']:>10.4f} {st['n']:>8,}   {monthly}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(grade_map, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\ngrade map -> {out_path}")
    print("note: " + tmm.GRADE_MAP_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
