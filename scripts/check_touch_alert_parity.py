"""Step 22: touch-alert serving parity check (runner path == offline study path).

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Proves, over
sampled historical days, that the live runner's code paths reproduce the offline
touch study exactly:

  (a) DETECTION: the runner recomputes the carry MA5x20 crosses over only
      TAIL_DAYS prior al_raw days + today (scripts/realtime_touch_alert.py). For
      every sampled day the truncated cross set must EQUAL the full-history
      build_touch_setups cross set (symbol/decision_time/direction/band and
      dist_pct/dist_atr to float tolerance) -- this is also the tail-length
      sufficiency proof (regime truncation would show up here).
  (b) SCORING: adapter probabilities computed from the runner's serving frames
      (bars -> realtime_features -> coverage rule) must match the same adapter fed
      the OFFLINE DB tables (features_merged / features_relative) to <= --tol,
      and the mapped grades must be identical. Requires the prod (or --artifact-
      suffix smoke) weights; without them the check downgrades to detection-only
      unless --require-scoring.

Run (cwd = quant, 64-bit):
  ..\\Python311-64\\python.exe scripts\\check_touch_alert_parity.py
  ..\\Python311-64\\python.exe scripts\\check_touch_alert_parity.py --artifact-suffix _smoke --grade-map models\\touch_grade_map_smoke.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (str(SRC_DIR), str(ROOT_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

import touch_model as tmm
import touch_serving as ts
from config import DB_PATH, connect
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text

TAIL_DAYS = 2                    # MUST mirror realtime_touch_alert.TAIL_DAYS
FEATURE_HISTORY_DAYS = 8         # MUST mirror realtime_touch_alert.FEATURE_HISTORY_DAYS
DIST_ATOL = 1e-12
KEY_COLS = ["symbol", "decision_time", "direction"]


def parse_args():
    p = argparse.ArgumentParser(description="Touch-alert serving parity check.")
    p.add_argument("--recent-days", type=int, default=20)
    p.add_argument("--old-days", type=int, default=10,
                   help="additional older days sampled with a fixed rng seed.")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="max |prob_runner - prob_offline| per event.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--models-dir", default=None)
    p.add_argument("--artifact-suffix", default="",
                   help="prod-weight namespace (e.g. _smoke).")
    p.add_argument("--grade-map", default=None,
                   help="grade-map JSON (default canonical); grades compared only "
                        "when it loads.")
    p.add_argument("--require-scoring", action="store_true",
                   help="fail (exit 1) when the scoring leg cannot run.")
    return p.parse_args()


def sample_days(all_days, recent_n, old_n):
    recent = all_days[-recent_n:] if recent_n else []
    older_pool = all_days[TAIL_DAYS:-recent_n] if recent_n else all_days[TAIL_DAYS:]
    older_pool = [d for d in older_pool if d not in set(recent)]
    rng = np.random.default_rng(42)
    older = sorted(rng.choice(older_pool, size=min(old_n, len(older_pool)),
                              replace=False).tolist()) if older_pool and old_n else []
    return older + list(recent)


def day_setups_key(setups, day):
    rows = setups[setups["date"] == day]
    return rows.sort_values(["symbol", "decision_time"]).reset_index(drop=True)


def compare_detection(full_day, trunc_day, day):
    problems = []
    k_full = set(map(tuple, full_day[KEY_COLS].to_numpy()))
    k_trunc = set(map(tuple, trunc_day[KEY_COLS].to_numpy()))
    if k_full != k_trunc:
        problems.append(f"cross-set mismatch: only-full={len(k_full - k_trunc)} "
                        f"only-trunc={len(k_trunc - k_full)}")
        return problems
    if len(full_day):
        m = full_day.merge(trunc_day, on=KEY_COLS, suffixes=("_f", "_t"),
                           validate="one_to_one")
        if not (m["band_f"] == m["band_t"]).all():
            problems.append("band mismatch")
        for col in ("dist_pct", "dist_atr", "early_flag", "minutes_since_open"):
            if not np.allclose(m[f"{col}_f"].astype(float),
                               m[f"{col}_t"].astype(float),
                               atol=DIST_ATOL, equal_nan=True):
                problems.append(f"{col} mismatch")
        if not (m["cluster_first_f"].astype(bool)
                == m["cluster_first_t"].astype(bool)).all():
            problems.append("cluster_first mismatch")
    return problems


def load_day_bars(conn, symbols, source, lo, hi):
    ph = ",".join("?" * len(symbols))
    return pd.read_sql_query(
        "SELECT symbol, bar_start, bar_end, open, high, low, close, volume "
        f"FROM bars_1m WHERE source=? AND symbol IN ({ph}) "
        "AND substr(bar_start,1,10) BETWEEN ? AND ? ORDER BY symbol, bar_start",
        conn, params=(source, *symbols, lo, hi))


def offline_frames(conn, source, day):
    merged_cols = [c for c in tmm.MERGED_FEATURES_V1
                   if c not in tmm.DERIVED_MERGED_FLAGS]
    fm = pd.read_sql_query(
        "SELECT symbol, decision_time, "
        + ", ".join(f'"{c}"' for c in merged_cols)
        + " FROM features_merged WHERE source=? AND substr(decision_time,1,10)=?",
        conn, params=(source, day))
    rel = pd.read_sql_query(
        "SELECT symbol, decision_time, "
        + ", ".join(f'"{c}"' for c in tmm.RELATIVE_FEATURES_V1)
        + " FROM features_relative WHERE source=? AND substr(decision_time,1,10)=?",
        conn, params=(source, day))
    return fm, rel


def main() -> int:
    args = parse_args()
    symbols = [normalize_symbol_code(s) for s in load_active_symbols()]
    model_source = load_data_config().model_source

    adapter = None
    grade_map = None
    scoring_note = ""
    try:
        adapter = ts.TouchModelAdapter(models_dir=args.models_dir,
                                       artifact_suffix=args.artifact_suffix,
                                       device=args.device,
                                       expected_symbols=symbols)
    except Exception as exc:                        # noqa: BLE001
        scoring_note = f"{type(exc).__name__}: {exc}"
        if args.require_scoring:
            print(f"scoring leg unavailable: {scoring_note}")
            return 1
    try:
        grade_map = ts.load_grade_map(args.grade_map)
    except Exception as exc:                        # noqa: BLE001
        print(f"  (grade map unavailable -- grades not compared: {exc})")

    conn = connect(DB_PATH)
    try:
        bars_all = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close FROM bars_1m "
            "WHERE source='al_raw'", conn)
        bars_all = bars_all[bars_all["symbol"].isin(symbols)].reset_index(drop=True)
        all_days = sorted(bars_all["bar_start"].str.slice(0, 10).unique())
        days = sample_days(all_days, args.recent_days, args.old_days)
        print(f"[{now_kst_text()}] parity over {len(days)} sampled days "
              f"({days[0]}..{days[-1]}), tail={TAIL_DAYS}d, "
              f"scoring={'ON' if adapter else 'OFF (' + scoring_note + ')'}")

        full_setups, _ = tmm.build_touch_setups(bars_all)
        det_fail = score_fail = 0
        checked_events = 0
        max_dprob = 0.0
        bar_dates = bars_all["bar_start"].str.slice(0, 10)
        for day in days:
            di = all_days.index(day)
            if di < TAIL_DAYS:
                print(f"  {day}: skipped (not enough prior days for the tail)")
                continue
            lo = all_days[di - TAIL_DAYS]
            trunc_bars = bars_all[(bar_dates >= lo) & (bar_dates <= day)]
            trunc_setups, _ = tmm.build_touch_setups(trunc_bars)
            full_day = day_setups_key(full_setups, day)
            trunc_day = day_setups_key(trunc_setups, day)
            problems = compare_detection(full_day, trunc_day, day)
            if problems:
                det_fail += 1
                print(f"  {day}: DETECTION FAIL -- {'; '.join(problems)}")
                continue
            line = f"  {day}: crosses={len(full_day)} detection OK"

            if adapter is not None and len(full_day):
                fm_db, rel_db = offline_frames(conn, model_source, day)
                events, _n_drop = tmm.filter_feature_covered(trunc_day, fm_db)
                if len(events):
                    # runner-path frames: prior al_regular days + the day's regular
                    # bars through the SHARED serving builder (coverage rule incl.)
                    feat_days = [d for d in all_days if d < day][-FEATURE_HISTORY_DAYS:]
                    bars_srv = load_day_bars(conn, symbols, model_source,
                                             feat_days[0] if feat_days else day, day)
                    covered_srv, rel_srv = ts.build_serving_frames(
                        bars_srv, source=model_source)
                    cov_keys = set(zip(covered_srv["symbol"],
                                       covered_srv["decision_time"]))
                    ev_keys = list(zip(events["symbol"], events["decision_time"]))
                    if not all(k in cov_keys for k in ev_keys):
                        score_fail += 1
                        print(line + " | SCORING FAIL -- serving coverage does not "
                                     "include every offline-covered event")
                        continue
                    p_off = adapter.score_events(events, fm_db, rel_db)
                    p_run = adapter.score_events(events, covered_srv, rel_srv)
                    d = float(np.max(np.abs(p_off - p_run)))
                    max_dprob = max(max_dprob, d)
                    checked_events += len(events)
                    ok = d <= args.tol
                    if ok and grade_map is not None:
                        g_off = tmm.apply_grade(p_off, grade_map)
                        g_run = tmm.apply_grade(p_run, grade_map)
                        ok = bool((g_off == g_run).all())
                        if not ok:
                            line += " | grade mismatch"
                    if not ok:
                        score_fail += 1
                        print(line + f" | SCORING FAIL max|Δprob|={d:.2e}")
                        continue
                    line += f" | scoring OK n={len(events)} max|Δprob|={d:.2e}"
                else:
                    line += " | scoring skipped (no covered events)"
            print(line)
    finally:
        conn.close()

    print(f"\ndetection failures: {det_fail}  scoring failures: {score_fail}  "
          f"events scored: {checked_events:,}  max|Δprob|={max_dprob:.2e}")
    if det_fail or score_fail:
        print("PARITY: FAIL")
        return 1
    if adapter is None and args.require_scoring:
        return 1
    print("PARITY: PASS" + ("" if adapter else " (detection only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
