"""Per-phase CSV exports of the MA5x20 crossover events for ONE symbol.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Reads the events
parquet written by scripts/analyze_ma_cross_1m.py and writes, per phase (P1/P2i/P2r):

  1. <prefix>_<symbol>_<phase>.csv          -- every cross of that phase for the
     symbol (all variants side by side, one row per cross: timestamp, direction,
     proximity band, +0.7% close/extreme touch flags for each window, endpoint/MAE,
     whipsaw fields).
  2. <prefix>_<symbol>_summary_<phase>.csv  -- the aggregate view, computed WITHIN
     each (variant, direction): band rows (near/mid/far) plus the band="ALL" row =
     the USER-defined baseline (touched crosses / all crosses of that variant,
     direction and phase). Rates use complete windows only; truncated counts kept.
  3. <prefix>_<symbol>_monthly_<phase>.csv  -- close-touch rates per (variant,
     direction, month), every available month (the recent-regime comparison).

Run:
  ..\\Python311-64\\python.exe scripts\\export_ma_cross_phase_csv.py --symbol <code>
"""

from __future__ import annotations

import argparse
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

import pandas as pd

from config import PROCESSED_DIR
from ma_cross_study import PHASES
from utils import now_kst_text

EVENT_COLUMNS = [
    "variant", "symbol", "date", "time", "bar_start", "session", "direction",
    "band", "atr_band", "dist_pct", "dist_atr", "entry",
    "w10_close_touch", "w15_close_touch", "w10_extreme_touch", "w15_extreme_touch",
    "w10_endpoint_ret", "w15_endpoint_ret", "w10_mae", "w15_mae",
    "w10_mfe", "w15_mfe", "w10_complete", "w15_complete",
    "whipsaw", "tt_whipsaw_min", "resume", "tt_resume_min", "exit_ret",
    "cluster_first", "body_ret",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-phase CSV export of MA5x20 cross events for one symbol.")
    p.add_argument("--symbol", required=True,
                   help="the symbol code to export (e.g. the main symbol).")
    p.add_argument("--events", default=str(PROCESSED_DIR / "ma_cross_1m_events.parquet"),
                   help="events parquet written by analyze_ma_cross_1m.py.")
    p.add_argument("--out-prefix", default="ma_cross_1m")
    return p.parse_args()


def summarize_phase(df):
    """(variant, direction) 안에서 band별 + band='ALL'(사용자 정의 베이스라인 =
    터치한 교차 수 / 전체 교차 수) 행을 만든다."""
    rows = []
    for (variant, direction), g in df.groupby(["variant", "direction"], sort=True):
        subsets = [("ALL", g)] + [(b, g[g["band"] == b]) for b in ("near", "mid", "far")]
        for band, s in subsets:
            row = {"variant": variant, "direction": direction, "band": band,
                   "n": int(len(s))}
            for w in (10, 15):
                comp = s[s[f"w{w}_complete"] == True]  # noqa: E712
                row[f"w{w}_n"] = int(len(comp))
                row[f"w{w}_n_truncated"] = int(len(s) - len(comp))
                if not len(comp):
                    continue
                row[f"w{w}_close_touch"] = float(comp[f"w{w}_close_touch"].mean())
                row[f"w{w}_extreme_touch"] = float(comp[f"w{w}_extreme_touch"].mean())
                row[f"w{w}_endpoint_mean"] = float(comp[f"w{w}_endpoint_ret"].mean())
                row[f"w{w}_mae_med"] = float(comp[f"w{w}_mae"].median())
                row[f"w{w}_p_mae_le_7"] = float((comp[f"w{w}_mae"] <= -0.007).mean())
            row["whipsaw_rate"] = float(s["whipsaw"].mean()) if len(s) else None
            rows.append(row)
    out = pd.DataFrame(rows)
    out["is_baseline"] = out["band"] == "ALL"
    return out


def monthly_phase(df):
    """(variant, direction, month)별 종가터치율 -- 레짐 비교용."""
    rows = []
    for (variant, direction, month), g in df.groupby(
            ["variant", "direction", "month"], sort=True):
        row = {"variant": variant, "direction": direction, "month": month,
               "n": int(len(g))}
        for w in (10, 15):
            comp = g[g[f"w{w}_complete"] == True]  # noqa: E712
            row[f"w{w}_n"] = int(len(comp))
            if len(comp):
                row[f"w{w}_close_touch"] = float(comp[f"w{w}_close_touch"].mean())
                row[f"w{w}_extreme_touch"] = float(comp[f"w{w}_extreme_touch"].mean())
                row[f"w{w}_mae_med"] = float(comp[f"w{w}_mae"].median())
        row["whipsaw_rate"] = float(g["whipsaw"].mean()) if len(g) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    events_path = Path(args.events)
    if not events_path.exists():
        raise SystemExit(f"events parquet not found: {events_path} - run "
                         "scripts/analyze_ma_cross_1m.py first.")
    ev = pd.read_parquet(events_path)
    ev = ev[ev["symbol"] == args.symbol].copy()
    if ev.empty:
        raise SystemExit(f"no events for symbol {args.symbol!r} in {events_path.name}")
    print(f"[{now_kst_text()}] export symbol={args.symbol} rows={len(ev):,} "
          f"phases={sorted(ev['phase'].unique())} variants={sorted(ev['variant'].unique())}")

    for phase in PHASES:
        pf = ev[ev["phase"] == phase]
        if pf.empty:
            continue
        cols = [c for c in EVENT_COLUMNS if c in pf.columns]
        detail = pf.sort_values(["variant", "bar_start"])[cols]
        detail_path = PROCESSED_DIR / f"{args.out_prefix}_{args.symbol}_{phase}.csv"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

        summary = summarize_phase(pf)
        summary_path = PROCESSED_DIR / f"{args.out_prefix}_{args.symbol}_summary_{phase}.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        monthly = monthly_phase(pf)
        monthly_path = PROCESSED_DIR / f"{args.out_prefix}_{args.symbol}_monthly_{phase}.csv"
        monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        base = summary[summary["is_baseline"]]
        print(f"\n[{phase}] events -> {detail_path.name} ({len(detail):,} rows)")
        print(f"[{phase}] summary -> {summary_path.name}")
        print(f"[{phase}] monthly -> {monthly_path.name}")
        show = [c for c in ("variant", "direction", "n", "w10_close_touch",
                            "w15_close_touch", "whipsaw_rate") if c in base.columns]
        print(base[show].to_string(index=False,
                                   float_format=lambda v: f"{v:,.4f}"))
        recent = monthly[monthly["month"] >= "2026-03"]
        if len(recent):
            piv = recent.pivot_table(index=["variant", "direction"], columns="month",
                                     values="w10_close_touch")
            print(f"[{phase}] W10 close-touch by month (2026-03+):")
            print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
