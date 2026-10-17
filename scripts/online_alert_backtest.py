"""Step 13_5: online (time-ordered) directional-alert backtest.

Reframes rule_signals as a DIRECTIONAL ALERT system (see src/directional_alert.py)
and evaluates it the way a live alert feed would behave -- NOT as a fixed daily
top1/top2 buy list:

  1. Build alert attributes (type / grade / scores / reasons) from features known
     at decision_time. No label or future price enters alert generation.
  2. Apply base filters (09:10-15:00, FAIL/outlier quality dates).
  3. ONLINE firing: walk each day in time order, emit alerts as their conditions
     are met, capped by a daily limit with a per-symbol cooldown. No future row
     influences an earlier firing decision.
  4. Score each fired alert FORWARD (30/60/120m + eod returns, MFE, MAE,
     up2/up3 reach, downside-first risk) at entry_delay. Labels are eval-only.
  5. Compare grade-level performance (A vs B vs C, RISK downside) and sweep the
     firing thresholds. Daily top-k is kept ONLY as a ranking diagnostic.

Persists: ``directional_alerts`` (alert catalogue), ``online_alert_results``
(fired alerts x configs + forward performance) tables, and an
``alert_grade_summary`` CSV. Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\online_alert_backtest.py
"""

from __future__ import annotations

import dataclasses
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

from bracket_eval import day_arrays, prepare_entry
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from daily_topk_backtest import apply_base_filters, quality_exclusion_pairs
from directional_alert import (
    ALERT_GRADES,
    ALERT_TYPES,
    AlertThresholds,
    alert_fire_gate,
    compute_directional_alerts,
    select_online_alerts,
    select_topk_diagnostic,
)
from settings import load_active_symbols, load_data_config
from utils import now_kst_text

MICRO_COLUMNS = [
    "upper_shadow_1m", "vwap_distance_1m", "distance_from_day_high_1m",
    "return_from_open_1m", "range_1m", "dist_ma5_intraday",
]
LABEL_COLUMNS = ["up2_60m", "up2_120m", "up3_120m", "up2_eod", "up3_eod", "up5_eod", "up5_event"]
CATALOGUE_COLUMNS = [
    "symbol", "source", "decision_time", "date", "close",
    "up_score", "down_score", "alert_score", "risk_score", "direction_score",
    "alert_type", "alert_grade", "reason_1", "reason_2", "reason_3",
    "ret_20m_1m", "volume_z_20m_1m", "rank_ret_20m_1m", "dist_ma5_intraday",
]
DOWN_STOP, UP_TARGET = 0.015, 0.015   # balanced downside-first risk band

BASELINE = AlertThresholds(name="baseline")
# one-knob-at-a-time sweep around the baseline (instruction 10)
SWEEP = [
    BASELINE,
    dataclasses.replace(BASELINE, name="up_thr_3.0", up_threshold=3.0),
    dataclasses.replace(BASELINE, name="up_thr_5.0", up_threshold=5.0),
    dataclasses.replace(BASELINE, name="margin_2.0", score_margin=2.0),
    dataclasses.replace(BASELINE, name="margin_4.0", score_margin=4.0),
    dataclasses.replace(BASELINE, name="down_lim_1.0", down_risk_limit=1.0),
    dataclasses.replace(BASELINE, name="down_lim_none", down_risk_limit=None),
    dataclasses.replace(BASELINE, name="rank_3", current_rank_limit=3),
    dataclasses.replace(BASELINE, name="rank_10", current_rank_limit=10),
    dataclasses.replace(BASELINE, name="daily_3", daily_alert_limit=3),
    dataclasses.replace(BASELINE, name="daily_10", daily_alert_limit=10),
    dataclasses.replace(BASELINE, name="cooldown_15", cooldown_minutes=15),
    dataclasses.replace(BASELINE, name="cooldown_60", cooldown_minutes=60),
    dataclasses.replace(BASELINE, name="delay_0", entry_delay_minutes=0),
    dataclasses.replace(BASELINE, name="delay_5", entry_delay_minutes=5),
]


# --------------------------------------------------------------------------- #
# Forward performance of a fired alert (eval-only; future bars from the entry)
# --------------------------------------------------------------------------- #
def forward_alert_metrics(ep, highs, lows, closes, offsets):
    out = {}
    for h in (30, 60, 120):
        w = closes[offsets <= h]
        out[f"return_{h}m"] = float(w[-1] / ep - 1) if len(w) else np.nan
    out["return_eod"] = float(closes[-1] / ep - 1) if len(closes) else np.nan
    out["mfe"] = float(highs.max() / ep - 1) if len(highs) else np.nan
    out["mae"] = float(lows.min() / ep - 1) if len(lows) else np.nan

    def reach(level, horizon):
        hw = highs[offsets <= horizon]
        return int((hw >= ep * (1 + level)).any()) if len(hw) else 0

    out["hit_up2_60m"] = reach(0.02, 60)
    out["hit_up2_120m"] = reach(0.02, 120)
    out["hit_up3_120m"] = reach(0.03, 120)

    up_hits = np.nonzero(highs >= ep * (1 + UP_TARGET))[0]
    dn_hits = np.nonzero(lows <= ep * (1 - DOWN_STOP))[0]
    u = int(up_hits[0]) if up_hits.size else None
    s = int(dn_hits[0]) if dn_hits.size else None
    out["stop_first"] = int(s is not None and (u is None or s <= u))   # down 1.5% before up 1.5% (tie->down)
    return out


def evaluate_fired(online, by_day, delay, labels_map, cache):
    """Attach forward performance + eval-only label hits to each fired alert."""
    if online is None or len(online) == 0:
        return pd.DataFrame()
    keep = ["symbol", "source", "date", "decision_time", "close", "alert_seq",
            "alert_type", "alert_grade", "up_score", "down_score",
            "alert_score", "risk_score", "direction_score",
            "reason_1", "reason_2", "reason_3"]
    rows = []
    for r in online[[c for c in keep if c in online.columns]].to_dict("records"):
        key = (r["symbol"], r["decision_time"], delay)
        m = cache.get(key, "MISS")
        if m == "MISS":
            arr = by_day.get((r["symbol"], r["date"]))
            m = None
            if arr is not None:
                prep = prepare_entry(arr, pd.Timestamp(r["decision_time"]), float(r["close"]),
                                     delay_bars=delay)
                if prep is not None:
                    m = forward_alert_metrics(*prep)
            cache[key] = m
        if m is None:
            continue
        rows.append({**r, **m, **labels_map.get((r["symbol"], r["decision_time"]), {})})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _group_summary(ev, dimension, key, value):
    g = ev[ev[key] == value]
    n = len(g)
    if n == 0:
        return None
    return {
        "dimension": dimension, "bucket": value, "count": n, "share": round(n / len(ev), 4),
        "return_30m_mean": round(float(g["return_30m"].mean()), 6),
        "return_60m_mean": round(float(g["return_60m"].mean()), 6),
        "return_120m_mean": round(float(g["return_120m"].mean()), 6),
        "return_eod_mean": round(float(g["return_eod"].mean()), 6),
        "dir_hit_30m": round(float((g["return_30m"] > 0).mean()), 4),
        "dir_hit_60m": round(float((g["return_60m"] > 0).mean()), 4),
        "dir_hit_120m": round(float((g["return_120m"] > 0).mean()), 4),
        "avg_mfe": round(float(g["mfe"].mean()), 6),
        "avg_mae": round(float(g["mae"].mean()), 6),
        "stop_first_rate": round(float(g["stop_first"].mean()), 4),
        "hit_up2_60m": round(float(g["hit_up2_60m"].mean()), 4),
        "hit_up2_120m": round(float(g["hit_up2_120m"].mean()), 4),
        "hit_up3_120m": round(float(g["hit_up3_120m"].mean()), 4),
        "label_up2_eod": round(float(g["up2_eod"].mean()), 4) if "up2_eod" in g else float("nan"),
        "label_up5_eod": round(float(g["up5_eod"].mean()), 4) if "up5_eod" in g else float("nan"),
        "symbol_concentration": round(float(g["symbol"].value_counts().max() / n), 4),
    }


def grade_summary(ev, n_days):
    if ev is None or len(ev) == 0:
        return pd.DataFrame()
    rows = [_group_summary(ev, "grade", "alert_grade", grd) for grd in ALERT_GRADES]
    rows += [_group_summary(ev, "type", "alert_type", typ) for typ in ALERT_TYPES]
    return pd.DataFrame([r for r in rows if r is not None])


def config_summary(name, ev, gated_n, n_days):
    n = len(ev)
    grade_counts = ev["alert_grade"].value_counts().to_dict() if n else {}
    return {
        "config": name,
        "gated_candidates": int(gated_n),
        "alert_count": int(n),
        "days_with_alert": int(ev["date"].nunique()) if n else 0,
        "avg_alerts_per_day": round(n / n_days, 3) if n_days else 0.0,
        "A": int(grade_counts.get("A", 0)), "B": int(grade_counts.get("B", 0)),
        "C": int(grade_counts.get("C", 0)), "RISK": int(grade_counts.get("RISK", 0)),
        "return_60m_mean": round(float(ev["return_60m"].mean()), 6) if n else 0.0,
        "dir_hit_60m": round(float((ev["return_60m"] > 0).mean()), 4) if n else 0.0,
        "avg_mfe": round(float(ev["mfe"].mean()), 6) if n else 0.0,
        "avg_mae": round(float(ev["mae"].mean()), 6) if n else 0.0,
        "stop_first_rate": round(float(ev["stop_first"].mean()), 4) if n else 0.0,
        "symbol_concentration": round(float(ev["symbol"].value_counts().max() / n), 4) if n else 0.0,
    }


def _latest_quality_report():
    reports = sorted(PROCESSED_DIR.glob("quality_report_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _fmt(x, pct=True):
    if x != x:
        return "  n/a"
    return f"{x*100:+.3f}%" if pct else f"{x:.3f}"


def main():
    ensure_directories()
    symbols = load_active_symbols()
    model_source = load_data_config().model_source

    conn = sqlite3.connect(DB_PATH)
    try:
        rs = pd.read_sql_query("SELECT * FROM rule_signals WHERE source = ?", conn, params=(model_source,))
        mg = pd.read_sql_query(
            f"SELECT symbol, decision_time, {', '.join(MICRO_COLUMNS)} FROM features_merged WHERE source = ?",
            conn, params=(model_source,))
        lab = pd.read_sql_query(
            f"SELECT symbol, decision_time, {', '.join(LABEL_COLUMNS)} FROM labels WHERE source = ?",
            conn, params=(model_source,))
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn, params=(model_source,))
    finally:
        conn.close()

    for f in (rs, mg, lab, bars):
        f.drop(f.index[~f["symbol"].isin(symbols)], inplace=True)
        f.reset_index(drop=True, inplace=True)
    if rs.empty:
        raise SystemExit("rule_signals empty for active symbols - run build_rule_signals.py first.")

    merged = rs.merge(mg, on=["symbol", "decision_time"], how="left", validate="one_to_one")
    alerts = compute_directional_alerts(merged)
    alerts["date"] = alerts["decision_time"].str.slice(0, 10)
    alerts["time"] = alerts["decision_time"].str.slice(11, 19)

    quality_path = _latest_quality_report()
    fail_pairs, outlier_pairs = quality_exclusion_pairs(
        pd.read_csv(quality_path) if quality_path else None)
    base, base_stats = apply_base_filters(alerts, fail_quality_pairs=fail_pairs, outlier_pairs=outlier_pairs)
    n_days = int(base["date"].nunique())

    print(f"[{now_kst_text()}] online directional-alert backtest")
    print(f"  symbols={len(symbols)} source={model_source} quality_report={quality_path.name if quality_path else 'none'}")
    print(f"  rows raw={base_stats['raw_rows']} -> time {base_stats['after_time_filter']} "
          f"-> quality {base_stats['after_quality_filter']}   trading_days={n_days}")
    print(f"  period {base['date'].min()} ~ {base['date'].max()}")

    # ---- alert catalogue (baseline fire-gate candidates; config-independent reference) ----
    catalogue = base[alert_fire_gate(base, BASELINE)].copy()
    cat_out = catalogue[[c for c in CATALOGUE_COLUMNS if c in catalogue.columns]].reset_index(drop=True)

    labels_map = {
        (r["symbol"], r["decision_time"]): {c: r[c] for c in LABEL_COLUMNS}
        for r in lab.to_dict("records")
    }
    by_day = day_arrays(bars)
    cache = {}

    # ---- run every threshold config (online) ----
    config_rows, eval_frames = [], []
    baseline_eval = None
    for cfg in SWEEP:
        gated = base[alert_fire_gate(base, cfg)]
        online = select_online_alerts(
            gated, daily_alert_limit=cfg.daily_alert_limit, cooldown_minutes=cfg.cooldown_minutes)
        ev = evaluate_fired(online, by_day, cfg.entry_delay_minutes, labels_map, cache)
        config_rows.append(config_summary(cfg.name, ev, len(gated), n_days))
        if len(ev):
            ev.insert(0, "config", cfg.name)
            eval_frames.append(ev)
        if cfg.name == "baseline":
            baseline_eval = ev

    config_df = pd.DataFrame(config_rows)
    online_results = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    grades = grade_summary(baseline_eval, n_days)

    # ---- daily top-k DIAGNOSTIC (ranking only; sees the whole day, never traded) ----
    base_gated = base[alert_fire_gate(base, BASELINE)]
    topk = select_topk_diagnostic(base_gated, top_k=2)
    topk_eval = evaluate_fired(topk.assign(alert_seq=topk["daily_rank"]),
                               by_day, BASELINE.entry_delay_minutes, labels_map, cache)

    # ---- persist ----
    conn = sqlite3.connect(DB_PATH)
    try:
        cat_out.to_sql("directional_alerts", conn, if_exists="replace", index=False)
        if len(online_results):
            online_results.to_sql("online_alert_results", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    stamp = now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
    grade_path = PROCESSED_DIR / f"alert_grade_summary_{stamp}.csv"
    config_path = PROCESSED_DIR / f"alert_threshold_compare_{stamp}.csv"
    grades.to_csv(grade_path, index=False, encoding="utf-8-sig")
    config_df.to_csv(config_path, index=False, encoding="utf-8-sig")

    # ---- report ----
    print(f"\n=== ALERT CATALOGUE (baseline gate; {len(cat_out)} candidate alerts) ===")
    print("  type distribution:")
    for typ, c in catalogue["alert_type"].value_counts().items():
        print(f"    {typ:<18}{c:>8}")

    bn = len(baseline_eval)
    print(f"\n=== ONLINE STREAM (baseline; {bn} fired, {bn/n_days:.2f}/day) ===")
    hdr = "  bucket          count   ret30m   ret60m  ret120m  dir60%      MFE      MAE  stop1st  up3_120"
    for dim in ("grade", "type"):
        print(f"  [{dim}]")
        print(hdr)
        for _, r in grades[grades["dimension"] == dim].iterrows():
            print(f"  {str(r['bucket']):<14}{int(r['count']):>6}{_fmt(r['return_30m_mean']):>9}"
                  f"{_fmt(r['return_60m_mean']):>9}{_fmt(r['return_120m_mean']):>9}{r['dir_hit_60m']*100:>7.1f}"
                  f"{_fmt(r['avg_mfe']):>9}{_fmt(r['avg_mae']):>9}{r['stop_first_rate']*100:>8.1f}{r['hit_up3_120m']*100:>8.1f}")

    print("\n  --- good-alert-system checks ---")
    by_g = {r["bucket"]: r for _, r in grades[grades["dimension"] == "grade"].iterrows()}
    by_t = {r["bucket"]: r for _, r in grades[grades["dimension"] == "type"].iterrows()}

    def g60(b):
        return by_g[b]["return_60m_mean"] if b in by_g else float("nan")

    print(f"    A ret60m {_fmt(g60('A'))}  vs B {_fmt(g60('B'))}  vs C {_fmt(g60('C'))}   (want A > B > C)")
    if "RISK_ALERT" in by_t:
        rt = by_t["RISK_ALERT"]
        print(f"    RISK_ALERT type: ret60m {_fmt(rt['return_60m_mean'])}  stop_first {rt['stop_first_rate']*100:.1f}%  "
              f"MAE {_fmt(rt['avg_mae'])}   (want downside)")
    overall_dir = float((baseline_eval["return_60m"] > 0).mean())
    print(f"    overall dir_hit_60m {overall_dir*100:.1f}%   (>50% = better than coin flip)")

    print(f"\n=== THRESHOLD SWEEP (online; alerts/day & A/B/C/RISK mix & 60m perf) ===")
    print("  config           gated  fired  /day    A    B    C RISK  ret60m   dir60%  stop1st")
    for r in config_rows:
        print(f"  {r['config']:<15}{r['gated_candidates']:>6}{r['alert_count']:>6}{r['avg_alerts_per_day']:>6.2f}"
              f"{r['A']:>5}{r['B']:>5}{r['C']:>5}{r['RISK']:>5}{_fmt(r['return_60m_mean']):>9}"
              f"{r['dir_hit_60m']*100:>7.1f}{r['stop_first_rate']*100:>8.1f}")

    print(f"\n=== DAILY TOP-K DIAGNOSTIC (ranking only, look-ahead; NOT the online stream) ===")
    if len(topk_eval):
        for rank in (1, 2):
            g = topk_eval[topk_eval["alert_seq"] == rank]
            if len(g):
                print(f"  top{rank}: n={len(g)} ret60m {_fmt(g['return_60m'].mean())} "
                      f"dir60% {(g['return_60m']>0).mean()*100:.1f} stop1st {g['stop_first'].mean()*100:.1f}%")
        print(f"  online baseline ret60m {_fmt(baseline_eval['return_60m'].mean())} "
              f"dir60% {(baseline_eval['return_60m']>0).mean()*100:.1f}%  "
              f"(top-k look-ahead premium is the gap above)")

    print(f"\ndirectional_alerts -> db ({len(cat_out)} rows)")
    print(f"online_alert_results -> db ({len(online_results)} rows, {len(SWEEP)} configs)")
    print(f"alert_grade_summary -> {grade_path}")
    print(f"threshold compare -> {config_path}")


if __name__ == "__main__":
    main()
