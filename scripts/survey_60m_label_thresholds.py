"""Step 17: threshold/dispersion survey for the fixed-horizon (60m) MAE/MFE labels.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. This script trains
nothing and adopts nothing: it reports the statistics the user needs to PICK the
Step 17 label thresholds (MAE/MFE event thresholds) and the 60m gate's A-grade MAE
margin, BEFORE any 60m-lineage model is trained.

Reported (main population = decisions with time-of-day <= --cutoff, i.e. always a
full 60-minute window; later decisions are the late-session diagnostic):
  1. population decomposition (total / main / late / horizon-NaN rows)
  2. event counts + positive rates per split & month, rest-of-day vs 60m, at the
     candidate thresholds (MAE: --mae-thresholds, MFE: --mfe-thresholds)
  3. dispersion of future_MAE / future_MAE_60m (and MFE): std + quantiles -- the
     basis for the 60m gate mae_ok margin decision
  4. hour-of-day positive-rate profile, old vs new label -- success check: the
     MONOTONE residual-time decay should be greatly reduced under the fixed horizon
     (intraday seasonality remaining is fine)
  5. late-session (> cutoff) diagnostic: truncated-window == rest-of-day equality
  6. cross-check vs the labels table: mean(future_MAE_60m <= -2%) ~= target_down2_60m
     rate (labeling.py uses the identical half-open 60m window convention)

Run:
  ..\\Python311-64\\python.exe scripts\\survey_60m_label_thresholds.py
"""

from __future__ import annotations

import argparse
import sqlite3
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

import risk_model as rm
from config import DB_PATH, PROCESSED_DIR
from settings import load_active_symbols, load_data_config
from utils import now_kst_text

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def parse_args():
    p = argparse.ArgumentParser(description="Step 17: 60m-label threshold/dispersion survey.")
    p.add_argument("--train-start", default="2025-06-02")
    p.add_argument("--train-end", default="2026-02-28")
    p.add_argument("--validation-start", default="2026-03-01")
    p.add_argument("--validation-end", default="2026-06-26",
                   help="matches the 60m-lineage gate frame.")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--mae-thresholds", default="-0.010,-0.0125,-0.015")
    p.add_argument("--mfe-thresholds", default="0.010,0.0125,0.015")
    p.add_argument("--cutoff", default=rm.HORIZON_60M_MAIN_CUTOFF,
                   help="main-population boundary on the decision time-of-day (inclusive).")
    p.add_argument("--out", default=str(PROCESSED_DIR / "label_60m_threshold_survey.csv"))
    return p.parse_args()


def _load_inputs(conn, model_source, symbols):
    alerts = pd.read_sql_query("SELECT * FROM directional_alerts WHERE source = ?", conn, params=(model_source,))
    fm = pd.read_sql_query("SELECT * FROM features_merged WHERE source = ?", conn, params=(model_source,))
    rel = pd.read_sql_query("SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,))
    labels = pd.read_sql_query("SELECT * FROM labels WHERE source = ?", conn, params=(model_source,))
    bars = pd.read_sql_query("SELECT symbol, bar_start, low, high, close FROM bars_1m WHERE source = ?",
                             conn, params=(model_source,))
    for frame in (alerts, fm, rel, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return alerts, fm, rel, labels, bars


def event_rows(frame, split_name, *, group_col, mae_thresholds, mfe_thresholds):
    """Event counts at each candidate threshold, old vs new label, per group."""
    rows = []
    specs = [("MAE", "future_MAE", "rest_of_day", "le"), ("MAE", "future_MAE_60m", "60m", "le"),
             ("MFE", "future_MFE", "rest_of_day", "ge"), ("MFE", "future_MFE_60m", "60m", "ge")]
    for side, col, kind, op in specs:
        thresholds = mae_thresholds if side == "MAE" else mfe_thresholds
        for thr in thresholds:
            for group, g in frame.groupby(group_col, sort=True):
                x = g[col]
                n = int(x.notna().sum())
                pos = int((x <= thr).sum() if op == "le" else (x >= thr).sum())
                rows.append({"section": "events", "split": split_name, "group": str(group),
                             "label_kind": kind, "side": side, "threshold": thr,
                             "n": n, "n_pos": pos,
                             "pos_rate": round(pos / n, 4) if n else float("nan")})
    return rows


def dispersion_rows(frame, split_name):
    rows = []
    for side, col, kind in [("MAE", "future_MAE", "rest_of_day"), ("MAE", "future_MAE_60m", "60m"),
                            ("MFE", "future_MFE", "rest_of_day"), ("MFE", "future_MFE_60m", "60m")]:
        x = frame[col].dropna()
        row = {"section": "dispersion", "split": split_name, "group": "all",
               "label_kind": kind, "side": side, "n": int(len(x)),
               "mean": round(float(x.mean()), 5), "std": round(float(x.std()), 5)}
        for q in QUANTILES:
            row[f"q{int(q * 100):02d}"] = round(float(x.quantile(q)), 5)
        rows.append(row)
    return rows


def fmt_table(df):
    return df.to_string(index=False)


def main():
    args = parse_args()
    mae_thresholds = [float(t) for t in args.mae_thresholds.split(",") if t.strip()]
    mfe_thresholds = [float(t) for t in args.mfe_thresholds.split(",") if t.strip()]
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    split = rm.SplitConfig(train_start=args.train_start, train_end=args.train_end,
                           validation_start=args.validation_start, validation_end=args.validation_end)

    print(f"[{now_kst_text()}] 60m-label threshold survey source={model_source} symbols={len(symbols)} "
          f"cutoff={args.cutoff} mae_thr={mae_thresholds} mfe_thr={mfe_thresholds}")
    conn = sqlite3.connect(args.db)
    try:
        alerts, fm, rel, labels, bars = _load_inputs(conn, model_source, symbols)
    finally:
        conn.close()

    data, _cols = rm.build_model_frame(alerts, fm, rel, labels, bars, horizon_minutes=60)
    train, valid = rm.split_by_date(data, split=split)
    all_rows = []

    for split_name, frame in (("train", train), ("validation", valid)):
        frame = frame.copy()
        frame["month"] = frame["date"].str.slice(0, 7)
        frame["hour"] = frame["time"].str.slice(0, 2)
        is_main = frame["time"] <= args.cutoff
        main = frame[is_main]
        late = frame[~is_main]

        # 1. population decomposition
        pop = {
            "section": "population", "split": split_name, "group": "all",
            "n_total": len(frame), "n_main": len(main), "n_late": len(late),
            "n_main_horizon_nan": int(main["future_MAE_60m"].isna().sum()),
            "n_old_valid_new_nan": int((main["future_MAE"].notna()
                                        & main["future_MAE_60m"].isna()).sum()),
        }
        all_rows.append(pop)
        print(f"\n== population [{split_name}] total={pop['n_total']:,} main(<= {args.cutoff})="
              f"{pop['n_main']:,} late={pop['n_late']:,} "
              f"main_horizon_nan={pop['n_main_horizon_nan']} "
              f"old_valid_new_nan={pop['n_old_valid_new_nan']}")

        # 2. events per month (MAIN population) + pooled
        all_rows += event_rows(main.assign(_pool="pooled"), split_name, group_col="_pool",
                               mae_thresholds=mae_thresholds, mfe_thresholds=mfe_thresholds)
        all_rows += event_rows(main, split_name, group_col="month",
                               mae_thresholds=mae_thresholds, mfe_thresholds=mfe_thresholds)

        # 3. dispersion (MAIN population)
        all_rows += dispersion_rows(main, split_name)

        # 4. hour-of-day profile (MAIN population) at every candidate threshold
        hour_rows = event_rows(main, split_name, group_col="hour",
                               mae_thresholds=mae_thresholds, mfe_thresholds=mfe_thresholds)
        for r in hour_rows:
            r["section"] = "hour_profile"
        all_rows += hour_rows

        # 5. late-session diagnostic: truncated 60m window == rest-of-day, by construction
        if len(late):
            both = late[late["future_MAE"].notna() & late["future_MAE_60m"].notna()]
            eq = float((both["future_MAE"] == both["future_MAE_60m"]).mean()) if len(both) else float("nan")
            late_rows = event_rows(late.assign(_pool="late"), split_name, group_col="_pool",
                                   mae_thresholds=mae_thresholds, mfe_thresholds=mfe_thresholds)
            for r in late_rows:
                r["section"] = "late_session"
            all_rows += late_rows
            all_rows.append({"section": "late_session", "split": split_name, "group": "late",
                             "label_kind": "equality", "side": "MAE",
                             "n": int(len(both)), "pos_rate": round(eq, 6)})
            print(f"   late-session: n={len(late):,} truncated==rest_of_day equality={eq:.4%}")

        # 6. cross-check vs labels table (60m window convention parity)
        chk = frame[frame["future_MAE_60m"].notna()]
        mae_2pct = float((chk["future_MAE_60m"] <= -0.02).mean()) if len(chk) else float("nan")
        down2 = float(chk["target_down2_60m"].mean()) if len(chk) else float("nan")
        all_rows.append({"section": "down2_crosscheck", "split": split_name, "group": "all",
                         "n": int(len(chk)), "mae60_le_2pct_rate": round(mae_2pct, 5),
                         "target_down2_60m_rate": round(down2, 5),
                         "abs_gap": round(abs(mae_2pct - down2), 5)})
        print(f"   down2 cross-check: P(MAE_60m<=-2%)={mae_2pct:.4f} vs down2_60m rate={down2:.4f} "
              f"(gap {abs(mae_2pct - down2):.4f})")

    out = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # console pivots (validation, main population) -- the decision tables
    ev = out[(out["section"] == "events") & (out["split"] == "validation") & (out["group"] == "pooled")]
    if len(ev):
        print("\n== validation pooled event rates (MAIN population)")
        print(fmt_table(ev[["side", "label_kind", "threshold", "n", "n_pos", "pos_rate"]]))
    disp = out[(out["section"] == "dispersion") & (out["split"] == "validation")]
    if len(disp):
        qcols = [f"q{int(q * 100):02d}" for q in QUANTILES]
        print("\n== validation dispersion (MAIN population; gate mae_ok-margin basis)")
        print(fmt_table(disp[["side", "label_kind", "n", "mean", "std"] + qcols]))
    hp = out[(out["section"] == "hour_profile") & (out["split"] == "validation")
             & (out["side"] == "MAE") & (out["label_kind"].isin(["rest_of_day", "60m"]))]
    if len(hp):
        for thr in mae_thresholds:
            sub = hp[hp["threshold"] == thr]
            piv = sub.pivot_table(index="group", columns="label_kind", values="pos_rate")
            print(f"\n== validation hour-of-day MAE pos_rate @ {thr:+.4f} "
                  "(monotone residual-time decay should be greatly reduced under 60m)")
            print(piv.to_string())

    print(f"\nsurvey -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
