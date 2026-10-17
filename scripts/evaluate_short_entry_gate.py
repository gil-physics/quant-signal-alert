"""Step 15: forward-edge study of the DOWN entry types -- holdout only.

The trend-alert system (src/short_entry.py) always fires its five typed alerts;
they carry no GO/NO-GO gate. This script answers a SEPARATE, optional question:
do the two entry-flavoured types (DOWN_ALERT / DOWN_CONTINUATION) additionally
carry short-side forward edge, i.e. could they double as conservative short-entry
timing? A GO means yes for the tested threshold combination; it does NOT change
whether the trend alerts fire (they always do). The verdict is recorded as
research history, not wired into the runner.

Takes the top train-ranked qualifying combinations from the threshold sweep
(scripts/short_gate_sweep.py), re-derives their fired sets from the pre-gate
``short_replay_candidates`` through the SAME offline gating/selection path, and
judges them ON THE HOLDOUT WINDOW ONLY against two baselines:

  1. Time-of-day-matched: unconditional entry at the same HH:MM:SS on every
     OTHER holdout day (paired per fire; short_entry_eval.baseline_same_time +
     paired_t_stat; baseline days are restricted to the holdout window so no
     train-regime day leaks into the judgement).
  2. Frozen RISK_ALERT-as-short numbers (Step 13_5, alert_type_summary.csv):
     fwd60 = +0.13%, dir-hit = 39.6% -- down signals historically mean-reverted,
     and the mirror gate must beat that history.

GO requires ALL holdout criteria (plan section 5); PARTIAL = beats BOTH
baselines while exactly one of {t_stat, n, stop_first} fails (every other
criterion passes). Writes data/processed/short_entry_gate_report.md +
short_entry_gate_summary.csv and exits with a clear verdict message.

Research only -- no orders, no Kiwoom, no order API. A GO verdict is recorded
here as evidence that the DOWN entry types also time short entries well; the
trend-alert runner fires those types regardless of this outcome.

    ..\\Python311-64\\python.exe scripts\\evaluate_short_entry_gate.py
    ..\\Python311-64\\python.exe scripts\\evaluate_short_entry_gate.py --summary data\\processed\\short_gate_sweep_summary_<stamp>.csv
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
for _p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

import short_gate_sweep as sgs
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from settings import load_data_config
from short_entry import ShortThresholds
from short_entry_eval import baseline_same_time, evaluate_short_alerts, paired_t_stat
from utils import now_kst_text

# --------------------------------------------------------------------------- #
# Holdout GO criteria (plan section 5; ALL must hold)
# --------------------------------------------------------------------------- #
GO_MIN_FIRES = 40             # evaluable holdout fires
GO_MIN_DAYS = 20              # distinct holdout fire days
GO_AVG_FWD60_MAX = -0.0010    # avg future_return_60m <= -0.10% (negative = short profit)
GO_PAIRED_EDGE_MAX = -0.0015  # paired (alert - same-time baseline) <= -0.15pp
GO_T_MAX = -2.0               # paired t-stat <= -2.0
GO_DIR_HIT60_MIN = 0.55       # direction_hit_60m rate >= 55%
GO_STOP_FIRST_MAX = 0.35      # +1.5% adverse-first rate <= 35%
GO_SHORT_MAE_MAX = 0.015      # avg short_mae <= +1.5%
GO_DOWN2_60M_MIN = 0.08       # -2% within 60m reached on >= 8% of fires
GO_MAX_POSITIVE_MONTHS = 1    # holdout months with positive avg fwd60 <= 1
PARTIAL_RELAXABLE = ("t_stat", "n", "stop_first")   # PARTIAL may fail exactly one of these

# frozen Step 13_5 evidence: RISK_ALERT rows traded as shorts mean-reverted
# (data/processed/alert_type_summary.csv) -- baseline 2 the gate must beat.
RISK_ALERT_SHORT_FWD60 = 0.0013
RISK_ALERT_SHORT_HIT60 = 0.396

SUMMARY_COLUMNS = [
    "combo_name", "train_rank",
    "down_threshold", "up_risk_limit", "score_margin", "bottom_rank_limit",
    "short_adverse_prob_max",
    "train_n", "train_avg_future_return_60m",
    "holdout_n", "holdout_days", "holdout_avg_future_return_60m",
    "baseline_avg_future_return_60m", "paired_mean_diff", "paired_t_stat", "paired_n",
    "holdout_dir_hit_60m_rate", "holdout_stop_first_rate",
    "holdout_avg_short_mfe", "holdout_avg_short_mae",
    "holdout_down2_60m_rate", "holdout_reach_dn15_eod_rate", "holdout_n_positive_months",
    "beats_tod_baseline", "beats_risk_alert_baseline", "criteria_failed", "verdict",
]

REPORT_NAME = "short_entry_gate_report.md"
SUMMARY_NAME = "short_entry_gate_summary.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 15: judge the top train-ranked SHORT_ENTRY combos GO/NO-GO "
                    "on the holdout window (research only).")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--candidates", default=None,
                   help="explicit candidates parquet (default: DB table, then the parquet dump).")
    p.add_argument("--summary", default=None,
                   help="sweep summary CSV (default: latest short_gate_sweep_summary_*.csv, "
                        f"then DB table {sgs.SUMMARY_TABLE}).")
    p.add_argument("--top", type=int, default=5,
                   help="how many train-ranked combos to judge (default 5).")
    p.add_argument("--train-end", default="2026-02-28")
    p.add_argument("--validation-start", default="2026-03-01")
    p.add_argument("--validation-end", default="2026-06-26")
    p.add_argument("--out-dir", default=str(PROCESSED_DIR),
                   help="directory for the report + summary CSV (default data/processed).")
    return p.parse_args()


def _latest_sweep_summary():
    files = sorted(PROCESSED_DIR.glob("short_gate_sweep_summary_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_summary(conn, path_arg):
    if path_arg:
        return pd.read_csv(path_arg), str(path_arg)
    latest = _latest_sweep_summary()
    if latest is not None:
        return pd.read_csv(latest), str(latest)
    if sgs._table_exists(conn, sgs.SUMMARY_TABLE):
        return pd.read_sql_query(f"SELECT * FROM {sgs.SUMMARY_TABLE}", conn), f"db:{sgs.SUMMARY_TABLE}"
    raise SystemExit("no sweep summary found - run scripts/short_gate_sweep.py first.")


def thresholds_from_row(row):
    """Rebuild the exact ShortThresholds a sweep summary row was produced with."""
    def _f(v):
        return None if pd.isna(v) else float(v)

    return replace(
        ShortThresholds(),
        name=str(row["combo_name"]),
        down_threshold=float(row["down_threshold"]),
        up_risk_limit=_f(row["up_risk_limit"]),
        score_margin=float(row["score_margin"]),
        bottom_rank_limit=None if pd.isna(row["bottom_rank_limit"]) else int(row["bottom_rank_limit"]),
        short_adverse_prob_max=_f(row["short_adverse_prob_max"]),
        daily_alert_limit=None if pd.isna(row["daily_alert_limit"]) else int(row["daily_alert_limit"]),
        cooldown_minutes=int(row["cooldown_minutes"]),
        min_time=str(row["min_time"]),
        max_time=str(row["max_time"]),
    )


# NaN-safe comparators: a missing metric FAILS its criterion (conservative)
def _le(x, lim):
    return x == x and x <= lim


def _ge(x, lim):
    return x == x and x >= lim


def _lt(x, lim):
    return x == x and x < lim


def _gt(x, lim):
    return x == x and x > lim


def _n_positive_months(evaluated):
    if len(evaluated) == 0:
        return 0
    m = evaluated.dropna(subset=["future_return_60m"])
    if m.empty:
        return 0
    months = m.groupby(m["date"].str.slice(0, 7))["future_return_60m"].mean()
    return int((months > 0).sum())


def _metrics_block(evaluated):
    if len(evaluated) == 0:
        return {"n": 0, "days": 0, "avg_fwd60": float("nan"), "dir_hit60": float("nan"),
                "stop_first": float("nan"), "avg_short_mfe": float("nan"),
                "avg_short_mae": float("nan"), "down2_60m": float("nan"),
                "reach_dn15_eod": float("nan"), "n_positive_months": 0}
    return {
        "n": int(len(evaluated)),
        "days": int(evaluated["date"].nunique()),
        "avg_fwd60": float(evaluated["future_return_60m"].mean()),
        "dir_hit60": float(evaluated["direction_hit_60m"].mean()),
        "stop_first": float(evaluated["short_stop_first"].mean()),
        "avg_short_mfe": float(evaluated["short_mfe"].mean()),
        "avg_short_mae": float(evaluated["short_mae"].mean()),
        "down2_60m": float(evaluated["down2_60m"].mean()),
        "reach_dn15_eod": float(evaluated["reach_dn15_eod"].mean()),
        "n_positive_months": _n_positive_months(evaluated),
    }


def evaluate_combo(row, candidates, bars, args):
    """Re-derive one combo's fires (same offline path as the sweep) and collect
    every holdout gate metric + the paired time-of-day baseline."""
    thr = thresholds_from_row(row)
    delay = int(row["entry_delay_minutes"])
    fires = sgs.select_fires(sgs.apply_combo(candidates, thr), thr)
    train_sel = fires[fires["date"] <= args.train_end]
    hold_sel = fires[(fires["date"] >= args.validation_start)
                     & (fires["date"] <= args.validation_end)]

    train_eval = evaluate_short_alerts(train_sel, bars, entry_delay_minutes=delay)
    hold_eval = evaluate_short_alerts(hold_sel, bars, entry_delay_minutes=delay)

    # baseline days restricted to the holdout window: the paired comparison must
    # not borrow train-regime days
    bar_dates = bars["bar_start"].str.slice(0, 10)
    hold_bars = bars[(bar_dates >= args.validation_start) & (bar_dates <= args.validation_end)]
    baseline = baseline_same_time(hold_eval, hold_bars, entry_delay_minutes=delay)
    paired = paired_t_stat(hold_eval, baseline, column="future_return_60m")

    train_m = _metrics_block(train_eval)
    hold_m = _metrics_block(hold_eval)
    return {
        "combo_name": thr.name,
        "train_rank": int(row["train_rank"]),
        "down_threshold": thr.down_threshold,
        "up_risk_limit": thr.up_risk_limit,
        "score_margin": thr.score_margin,
        "bottom_rank_limit": thr.bottom_rank_limit,
        "short_adverse_prob_max": thr.short_adverse_prob_max,
        "train_n": train_m["n"],
        "train_avg_future_return_60m": train_m["avg_fwd60"],
        "train_dir_hit_60m_rate": train_m["dir_hit60"],
        "train_stop_first_rate": train_m["stop_first"],
        "holdout_n": hold_m["n"],
        "holdout_days": hold_m["days"],
        "holdout_avg_future_return_60m": hold_m["avg_fwd60"],
        "baseline_avg_future_return_60m": paired["mean_baseline"],
        "paired_mean_diff": paired["mean_diff"],
        "paired_t_stat": paired["t_stat"],
        "paired_n": paired["n"],
        "holdout_dir_hit_60m_rate": hold_m["dir_hit60"],
        "holdout_stop_first_rate": hold_m["stop_first"],
        "holdout_avg_short_mfe": hold_m["avg_short_mfe"],
        "holdout_avg_short_mae": hold_m["avg_short_mae"],
        "holdout_down2_60m_rate": hold_m["down2_60m"],
        "holdout_reach_dn15_eod_rate": hold_m["reach_dn15_eod"],
        "holdout_n_positive_months": hold_m["n_positive_months"],
    }


def decide_verdict(res):
    """GO / PARTIAL / NO-GO on the holdout metrics (see module constants)."""
    checks = {
        "n": res["holdout_n"] >= GO_MIN_FIRES and res["holdout_days"] >= GO_MIN_DAYS,
        "avg_fwd60": _le(res["holdout_avg_future_return_60m"], GO_AVG_FWD60_MAX),
        "paired_edge": _le(res["paired_mean_diff"], GO_PAIRED_EDGE_MAX),
        "t_stat": _le(res["paired_t_stat"], GO_T_MAX),
        "dir_hit60": _ge(res["holdout_dir_hit_60m_rate"], GO_DIR_HIT60_MIN),
        "stop_first": _le(res["holdout_stop_first_rate"], GO_STOP_FIRST_MAX),
        "short_mae": _le(res["holdout_avg_short_mae"], GO_SHORT_MAE_MAX),
        "down2_60m": _ge(res["holdout_down2_60m_rate"], GO_DOWN2_60M_MIN),
        "train_sign": _lt(res["train_avg_future_return_60m"], 0.0),
        "monthly": res["holdout_n_positive_months"] <= GO_MAX_POSITIVE_MONTHS,
    }
    failed = [k for k, v in checks.items() if not v]
    beats_tod = _lt(res["paired_mean_diff"], 0.0)
    beats_risk = (_lt(res["holdout_avg_future_return_60m"], RISK_ALERT_SHORT_FWD60)
                  and _gt(res["holdout_dir_hit_60m_rate"], RISK_ALERT_SHORT_HIT60))
    if not failed:
        verdict = "GO"
    elif (beats_tod and beats_risk and len(failed) == 1
          and failed[0] in PARTIAL_RELAXABLE):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    return verdict, failed, beats_tod, beats_risk


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{x * 100:+.2f}%" if isinstance(x, float) and x == x else "n/a"


def _rate(x):
    return f"{x * 100:.1f}%" if isinstance(x, float) and x == x else "n/a"


def _num(x, fmt="{:+.2f}"):
    return fmt.format(x) if isinstance(x, float) and x == x else "n/a"


def _headline(overall, results):
    if overall == "GO":
        names = ", ".join(f"`{r['combo_name']}`" for r in results if r["verdict"] == "GO")
        return (f"**Verdict: GO — {names} passed every holdout criterion.** The DOWN entry types "
                "carry short-side forward edge at this threshold combination and could double as "
                "conservative short-entry timing (the trend alerts fire regardless).")
    if overall == "PARTIAL":
        names = ", ".join(f"`{r['combo_name']}`" for r in results if r["verdict"] == "PARTIAL")
        return (f"**Verdict: PARTIAL — {names} beat both baselines but failed one of "
                "{t_stat, n, stop_first}.** Weak short-entry edge; the DOWN types remain trend "
                "alerts, not a validated entry timer.")
    return ("**Verdict: NO-GO — no combination passed the holdout gate.** The DOWN types carry no "
            "short-entry edge (mean-reversion history, Step 13_5, stands); they remain informational "
            "trend alerts, which is their intended role.")


def write_report(results, overall, args, sources, out_dir):
    lines = []
    lines.append("# SHORT_ENTRY gate — holdout GO/NO-GO report (Step 15)\n")
    lines.append(_headline(overall, results) + "\n")
    lines.append(
        f"The top {len(results)} train-ranked combinations from the threshold sweep "
        f"(`{sources['summary']}`) were re-gated on the pre-gate replay candidates "
        f"(`{sources['candidates']}`, {sources['n_candidates']:,} rows) through the same offline "
        "path the sweep uses (rule gate → model veto → session window → causal cooldown/"
        "daily-limit selection) and judged **on the holdout window only** "
        f"({args.validation_start} → {args.validation_end}; train ranking window ends "
        f"{args.train_end}). Forward metrics follow the project execution model: entry = close "
        "of the decision + delay bar; raw close returns, so **negative = short profit**.\n")
    lines.append("Baselines the gate must beat:\n")
    lines.append(
        "1. **Time-of-day-matched** — unconditional entry at the same HH:MM:SS on every other "
        "holdout day (paired per fire; edge = alert − baseline, negative = short edge).")
    lines.append(
        f"2. **Frozen RISK_ALERT-as-short** (Step 13_5, alert_type_summary.csv) — fwd60 "
        f"{_pct(RISK_ALERT_SHORT_FWD60)}, dir-hit {_rate(RISK_ALERT_SHORT_HIT60)}: down alerts "
        "historically mean-reverted.\n")

    if not results:
        lines.append("## No qualifying combinations\n")
        lines.append("The sweep produced no combination meeting the minimum samples "
                     "(train ≥ 30 fires; holdout ≥ 20 fires on ≥ 15 days), so nothing could be "
                     "judged. Verdict defaults to NO-GO.\n")
    else:
        lines.append(f"## 1. Top train-ranked combinations (train ≤ {args.train_end})\n")
        lines.append("| rank | combo | train n | avg fwd60 | dir-hit60 | stop-first |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            lines.append(f"| {r['train_rank']} | `{r['combo_name']}` | {r['train_n']} | "
                         f"{_pct(r['train_avg_future_return_60m'])} | "
                         f"{_rate(r['train_dir_hit_60m_rate'])} | "
                         f"{_rate(r['train_stop_first_rate'])} |")
        lines.append("")

        lines.append(f"## 2. Holdout metrics ({args.validation_start} → {args.validation_end})\n")
        lines.append("| combo | n | days | avg fwd60 | baseline fwd60 | paired edge | t | "
                     "dir-hit60 | stop-first | avg MAE | down2_60m | +months |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in results:
            lines.append(
                f"| `{r['combo_name']}` | {r['holdout_n']} | {r['holdout_days']} | "
                f"{_pct(r['holdout_avg_future_return_60m'])} | "
                f"{_pct(r['baseline_avg_future_return_60m'])} | "
                f"{_pct(r['paired_mean_diff'])} | {_num(r['paired_t_stat'])} | "
                f"{_rate(r['holdout_dir_hit_60m_rate'])} | {_rate(r['holdout_stop_first_rate'])} | "
                f"{_pct(r['holdout_avg_short_mae'])} | {_rate(r['holdout_down2_60m_rate'])} | "
                f"{r['holdout_n_positive_months']} |")
        lines.append("")

    lines.append("## 3. Gate criteria (ALL must hold on holdout for GO)\n")
    lines.append(f"- n ≥ {GO_MIN_FIRES} evaluable fires on ≥ {GO_MIN_DAYS} distinct days")
    lines.append(f"- avg fwd60 ≤ {_pct(GO_AVG_FWD60_MAX)} AND paired edge ≤ "
                 f"{_pct(GO_PAIRED_EDGE_MAX)} AND paired t ≤ {GO_T_MAX:g}")
    lines.append(f"- dir-hit60 ≥ {_rate(GO_DIR_HIT60_MIN)}")
    lines.append(f"- stop-first ≤ {_rate(GO_STOP_FIRST_MAX)} AND avg short_mae ≤ "
                 f"{_pct(GO_SHORT_MAE_MAX)}")
    lines.append(f"- down2_60m ≥ {_rate(GO_DOWN2_60M_MIN)}")
    lines.append(f"- train avg fwd60 < 0 (sign consistency) AND ≤ {GO_MAX_POSITIVE_MONTHS} "
                 "positive holdout month(s)\n")
    lines.append("PARTIAL = beats BOTH baselines while exactly one of {t_stat, n, stop_first} "
                 "fails (every other criterion passes).\n")

    if results:
        lines.append("## 4. Verdicts\n")
        lines.append("| combo | verdict | beats ToD baseline | beats RISK_ALERT baseline | failed criteria |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            failed = ", ".join(r["criteria_failed"]) if r["criteria_failed"] else "—"
            lines.append(f"| `{r['combo_name']}` | **{r['verdict']}** | {r['beats_tod_baseline']} | "
                         f"{r['beats_risk_alert_baseline']} | {failed} |")
        lines.append("")

    lines.append("## Status\n")
    lines.append("- The trend-alert runner fires the five typed alerts unconditionally; this "
                 "study never gates it.")
    if overall == "GO":
        lines.append("- The GO combination additionally shows short-entry edge — record it in "
                     "README Step 15 as an optional entry-timing overlay for the DOWN types.")
    else:
        lines.append("- No short-entry edge; the DOWN types stay purely informational trend "
                     "alerts. Documented here per project convention (failed edge is recorded).")
    lines.append(f"- Generated {now_kst_text()} by scripts/evaluate_short_entry_gate.py "
                 "(research only, no orders).\n")

    out_path = Path(out_dir) / REPORT_NAME
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    args = parse_args()
    ensure_directories()
    source = load_data_config().model_source

    conn = sqlite3.connect(args.db)
    try:
        summary, summary_src = load_summary(conn, args.summary)
        candidates, cand_src = sgs.load_candidates(conn, args.candidates)
        bars = sgs.load_bars(conn, source, candidates["symbol"].unique())
    finally:
        conn.close()

    needed = {"combo_name", "split", "min_sample_ok", "train_rank", "entry_delay_minutes"}
    if not needed.issubset(summary.columns):
        raise SystemExit(f"sweep summary {summary_src} is missing columns "
                         f"{sorted(needed - set(summary.columns))} - rerun short_gate_sweep.py.")

    train_rows = summary[(summary["split"] == "train")
                         & (summary["min_sample_ok"] == 1)
                         & summary["train_rank"].notna()]
    top = train_rows.sort_values("train_rank").head(args.top)

    print(f"[{now_kst_text()}] short-entry gate judgement summary={summary_src} "
          f"candidates={cand_src} top={len(top)}/{args.top}")
    print(f"  holdout: {args.validation_start} ~ {args.validation_end} "
          f"(train ranking window <= {args.train_end})")

    results = []
    for row in top.to_dict("records"):
        res = evaluate_combo(row, candidates, bars, args)
        verdict, failed, beats_tod, beats_risk = decide_verdict(res)
        res.update({"verdict": verdict, "criteria_failed": failed,
                    "beats_tod_baseline": beats_tod, "beats_risk_alert_baseline": beats_risk})
        results.append(res)
        print(f"  #{res['train_rank']} {res['combo_name']:<40} {verdict:<8} "
              f"holdout n={res['holdout_n']} fwd60={_pct(res['holdout_avg_future_return_60m'])} "
              f"edge={_pct(res['paired_mean_diff'])} t={_num(res['paired_t_stat'])} "
              f"hit={_rate(res['holdout_dir_hit_60m_rate'])} "
              f"failed={','.join(failed) or 'none'}")

    order = {"GO": 0, "PARTIAL": 1, "NO-GO": 2}
    overall = min((r["verdict"] for r in results), key=order.get) if results else "NO-GO"

    csv_rows = []
    for r in results:
        row = {k: r[k] for k in SUMMARY_COLUMNS if k != "criteria_failed"}
        row["criteria_failed"] = ";".join(r["criteria_failed"])
        csv_rows.append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / SUMMARY_NAME
    pd.DataFrame(csv_rows, columns=SUMMARY_COLUMNS).to_csv(csv_path, index=False,
                                                           encoding="utf-8-sig")
    report_path = write_report(
        results, overall, args,
        {"summary": summary_src, "candidates": cand_src, "n_candidates": len(candidates)},
        args.out_dir)

    print(f"\nreport -> {report_path}")
    print(f"summary -> {csv_path}")
    if overall == "GO":
        winners = ", ".join(r["combo_name"] for r in results if r["verdict"] == "GO")
        print(f"\nVERDICT: GO — {winners} show short-entry edge on the holdout; the DOWN types "
              f"could double as entry timing (trend alerts fire regardless).")
    elif overall == "PARTIAL":
        partials = ", ".join(r["combo_name"] for r in results if r["verdict"] == "PARTIAL")
        print(f"\nVERDICT: PARTIAL — {partials} beat both baselines but failed one of "
              f"t_stat/n/stop_first; no validated short-entry edge.")
    else:
        print("\nVERDICT: NO-GO — no combination shows short-entry edge; the DOWN types remain "
              "informational trend alerts (their intended role).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
