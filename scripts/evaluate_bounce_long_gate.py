"""Bounce-long (dip-buy) gate: GO/NO-GO judgement -- holdout only.

Takes the top train-ranked qualifying combinations from the bounce-long sweep
(scripts/bounce_long_sweep.py), re-derives their fired sets from the pre-gate
``short_replay_candidates`` through the SAME offline gating/selection path, and
judges them ON THE HOLDOUT WINDOW ONLY against the time-of-day-matched
baseline: unconditional LONG entry at the same HH:MM:SS on every OTHER holdout
day (paired per fire; short_entry_eval.baseline_same_time_long + paired_t_stat;
baseline days are restricted to the holdout window so no train-regime day leaks
into the judgement).

Reuse framing: the gate scores the v1_short model output ``short_adverse_prob``
-- P(rest-of-day upside MFE >= +1.5%), the short's adverse-move probability --
AS a bounce probability for LONG entries on dip minutes. The train window is
IN-SAMPLE for that model (it was trained there), so train numbers only rank
combinations and only the holdout window judges them.

GO requires ALL holdout criteria; PARTIAL = beats the time-of-day baseline
(positive paired edge) while exactly one of {t_stat, n, long_stop_first} fails
(every other criterion passes). Writes data/processed/bounce_long_gate_report.md
+ bounce_long_gate_summary.csv and exits with a clear verdict message.

Research only -- no orders, no Kiwoom, no order API. A GO verdict is recorded
here; wiring the winning combination into any live decision-support layer is a
separate, deliberate step.

    ..\\Python311-64\\python.exe scripts\\evaluate_bounce_long_gate.py
    ..\\Python311-64\\python.exe scripts\\evaluate_bounce_long_gate.py --summary data\\processed\\bounce_long_sweep_summary_<stamp>.csv
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
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

import bounce_long_sweep as bls
import short_gate_sweep as sgs
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from settings import load_data_config
from short_entry_eval import baseline_same_time_long, evaluate_long_alerts, paired_t_stat
from utils import now_kst_text

# --------------------------------------------------------------------------- #
# Holdout GO criteria (ALL must hold; LONG sign: positive = profit)
# --------------------------------------------------------------------------- #
GO_MIN_FIRES = 40             # evaluable holdout fires
GO_MIN_DAYS = 20              # distinct holdout fire days
GO_AVG_FWD60_MIN = 0.0010     # avg future_return_60m >= +0.10%
GO_PAIRED_EDGE_MIN = 0.0015   # paired (alert - same-time baseline) >= +0.15pp
GO_T_MIN = 2.0                # paired t-stat >= +2.0
GO_DIR_HIT60_MIN = 0.55       # direction_hit_60m rate >= 55%
GO_STOP_FIRST_MAX = 0.35      # -1.5% adverse-first rate <= 35%
GO_LONG_MAE_MAX = 0.015       # avg long_mae <= +1.5%
GO_UP2_60M_MIN = 0.08         # +2% within 60m reached on >= 8% of fires
GO_MAX_NEGATIVE_MONTHS = 1    # holdout months with negative avg fwd60 <= 1
PARTIAL_RELAXABLE = ("t_stat", "n", "long_stop_first")  # PARTIAL may fail exactly one of these

SUMMARY_COLUMNS = [
    "combo_name", "train_rank",
    "bounce_prob_min", "down_floor", "upside_risk_min", "long_risk_max",
    "train_n", "train_avg_future_return_60m",
    "holdout_n", "holdout_days", "holdout_avg_future_return_60m",
    "baseline_avg_future_return_60m", "paired_mean_diff", "paired_t_stat", "paired_n",
    "holdout_dir_hit_60m_rate", "holdout_long_stop_first_rate",
    "holdout_avg_long_mfe", "holdout_avg_long_mae",
    "holdout_up2_60m_rate", "holdout_reach_up15_eod_rate", "holdout_n_negative_months",
    "beats_tod_baseline", "criteria_failed", "verdict",
]

REPORT_NAME = "bounce_long_gate_report.md"
SUMMARY_NAME = "bounce_long_gate_summary.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Judge the top train-ranked bounce-long combos GO/NO-GO on the "
                    "holdout window (v1_short reused as bounce probability; research only).")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--candidates", default=None,
                   help="explicit candidates parquet (default: DB table, then the parquet dump).")
    p.add_argument("--summary", default=None,
                   help="sweep summary CSV (default: latest bounce_long_sweep_summary_*.csv, "
                        f"then DB table {bls.SUMMARY_TABLE}).")
    p.add_argument("--top", type=int, default=5,
                   help="how many train-ranked combos to judge (default 5).")
    p.add_argument("--train-end", default="2026-02-28")
    p.add_argument("--validation-start", default="2026-03-01")
    p.add_argument("--validation-end", default="2026-06-26")
    p.add_argument("--out-dir", default=str(PROCESSED_DIR),
                   help="directory for the report + summary CSV (default data/processed).")
    return p.parse_args()


def _latest_sweep_summary():
    files = sorted(PROCESSED_DIR.glob("bounce_long_sweep_summary_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_summary(conn, path_arg):
    if path_arg:
        return pd.read_csv(path_arg), str(path_arg)
    latest = _latest_sweep_summary()
    if latest is not None:
        return pd.read_csv(latest), str(latest)
    if sgs._table_exists(conn, bls.SUMMARY_TABLE):
        return pd.read_sql_query(f"SELECT * FROM {bls.SUMMARY_TABLE}", conn), f"db:{bls.SUMMARY_TABLE}"
    raise SystemExit("no sweep summary found - run scripts/bounce_long_sweep.py first.")


def combo_from_row(row):
    """Rebuild the exact BounceCombo a sweep summary row was produced with."""
    def _f(v):
        return None if pd.isna(v) else float(v)

    return bls.BounceCombo(
        name=str(row["combo_name"]),
        bounce_prob_min=_f(row["bounce_prob_min"]),
        down_floor=float(row["down_floor"]),
        upside_risk_min=_f(row["upside_risk_min"]),
        long_risk_max=_f(row["long_risk_max"]),
        daily_alert_limit=int(row["daily_alert_limit"]),
        cooldown_minutes=int(row["cooldown_minutes"]),
        min_time=str(row["min_time"]),
        max_time=str(row["max_time"]),
    )


# NaN-safe comparators: a missing metric FAILS its criterion (conservative)
def _le(x, lim):
    return x == x and x <= lim


def _ge(x, lim):
    return x == x and x >= lim


def _gt(x, lim):
    return x == x and x > lim


def _metrics_block(evaluated):
    if len(evaluated) == 0:
        return {"n": 0, "days": 0, "avg_fwd60": float("nan"), "dir_hit60": float("nan"),
                "stop_first": float("nan"), "avg_long_mfe": float("nan"),
                "avg_long_mae": float("nan"), "up2_60m": float("nan"),
                "reach_up15_eod": float("nan"), "n_negative_months": 0}
    return {
        "n": int(len(evaluated)),
        "days": int(evaluated["date"].nunique()),
        "avg_fwd60": float(evaluated["future_return_60m"].mean()),
        "dir_hit60": float(evaluated["direction_hit_60m"].mean()),
        "stop_first": float(evaluated["long_stop_first"].mean()),
        "avg_long_mfe": float(evaluated["long_mfe"].mean()),
        "avg_long_mae": float(evaluated["long_mae"].mean()),
        "up2_60m": float(evaluated["up2_60m"].mean()),
        "reach_up15_eod": float(evaluated["reach_up15_eod"].mean()),
        "n_negative_months": bls._n_negative_months(evaluated),
    }


def evaluate_combo(row, candidates, bars, args):
    """Re-derive one combo's fires (same offline path as the sweep) and collect
    every holdout gate metric + the paired time-of-day baseline."""
    combo = combo_from_row(row)
    delay = int(row["entry_delay_minutes"])
    fires = bls.select_fires(bls.apply_combo(candidates, combo), combo)
    train_sel = fires[fires["date"] <= args.train_end]
    hold_sel = fires[(fires["date"] >= args.validation_start)
                     & (fires["date"] <= args.validation_end)]

    train_eval = evaluate_long_alerts(train_sel, bars, entry_delay_minutes=delay)
    hold_eval = evaluate_long_alerts(hold_sel, bars, entry_delay_minutes=delay)

    # baseline days restricted to the holdout window: the paired comparison must
    # not borrow train-regime days
    bar_dates = bars["bar_start"].str.slice(0, 10)
    hold_bars = bars[(bar_dates >= args.validation_start) & (bar_dates <= args.validation_end)]
    baseline = baseline_same_time_long(hold_eval, hold_bars, entry_delay_minutes=delay)
    paired = paired_t_stat(hold_eval, baseline, column="future_return_60m")

    train_m = _metrics_block(train_eval)
    hold_m = _metrics_block(hold_eval)
    return {
        "combo_name": combo.name,
        "train_rank": int(row["train_rank"]),
        "bounce_prob_min": combo.bounce_prob_min,
        "down_floor": combo.down_floor,
        "upside_risk_min": combo.upside_risk_min,
        "long_risk_max": combo.long_risk_max,
        "train_n": train_m["n"],
        "train_avg_future_return_60m": train_m["avg_fwd60"],
        "train_dir_hit_60m_rate": train_m["dir_hit60"],
        "train_long_stop_first_rate": train_m["stop_first"],
        "holdout_n": hold_m["n"],
        "holdout_days": hold_m["days"],
        "holdout_avg_future_return_60m": hold_m["avg_fwd60"],
        "baseline_avg_future_return_60m": paired["mean_baseline"],
        "paired_mean_diff": paired["mean_diff"],
        "paired_t_stat": paired["t_stat"],
        "paired_n": paired["n"],
        "holdout_dir_hit_60m_rate": hold_m["dir_hit60"],
        "holdout_long_stop_first_rate": hold_m["stop_first"],
        "holdout_avg_long_mfe": hold_m["avg_long_mfe"],
        "holdout_avg_long_mae": hold_m["avg_long_mae"],
        "holdout_up2_60m_rate": hold_m["up2_60m"],
        "holdout_reach_up15_eod_rate": hold_m["reach_up15_eod"],
        "holdout_n_negative_months": hold_m["n_negative_months"],
    }


def decide_verdict(res):
    """GO / PARTIAL / NO-GO on the holdout metrics (see module constants)."""
    checks = {
        "n": res["holdout_n"] >= GO_MIN_FIRES and res["holdout_days"] >= GO_MIN_DAYS,
        "avg_fwd60": _ge(res["holdout_avg_future_return_60m"], GO_AVG_FWD60_MIN),
        "paired_edge": _ge(res["paired_mean_diff"], GO_PAIRED_EDGE_MIN),
        "t_stat": _ge(res["paired_t_stat"], GO_T_MIN),
        "dir_hit60": _ge(res["holdout_dir_hit_60m_rate"], GO_DIR_HIT60_MIN),
        "long_stop_first": _le(res["holdout_long_stop_first_rate"], GO_STOP_FIRST_MAX),
        "long_mae": _le(res["holdout_avg_long_mae"], GO_LONG_MAE_MAX),
        "up2_60m": _ge(res["holdout_up2_60m_rate"], GO_UP2_60M_MIN),
        "train_sign": _gt(res["train_avg_future_return_60m"], 0.0),
        "monthly": res["holdout_n_negative_months"] <= GO_MAX_NEGATIVE_MONTHS,
    }
    failed = [k for k, v in checks.items() if not v]
    beats_tod = _gt(res["paired_mean_diff"], 0.0)
    if not failed:
        verdict = "GO"
    elif beats_tod and len(failed) == 1 and failed[0] in PARTIAL_RELAXABLE:
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    return verdict, failed, beats_tod


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
        return (f"**Verdict: GO — {names} passed every holdout criterion.** The v1_short "
                "probability adds LONG dip-buy value out-of-sample; wiring it into any live "
                "decision-support layer is a separate, deliberate step.")
    if overall == "PARTIAL":
        names = ", ".join(f"`{r['combo_name']}`" for r in results if r["verdict"] == "PARTIAL")
        return (f"**Verdict: PARTIAL — {names} beat the time-of-day baseline but failed one of "
                "{t_stat, n, long_stop_first}.** The bounce-long reuse stays UNVALIDATED; "
                "nothing is wired live.")
    return ("**Verdict: NO-GO — no combination passed the holdout gate.** Reusing "
            "`short_adverse_prob` as a LONG dip-buy confirmation stays UNVALIDATED; "
            "nothing is wired live.")


def write_report(results, overall, args, sources, out_dir):
    lines = []
    lines.append("# Bounce-long (dip-buy) gate — holdout GO/NO-GO report\n")
    lines.append(_headline(overall, results) + "\n")
    lines.append(
        "**Reuse framing.** The gate buys dips: rows come from the pre-gate short replay "
        "candidates pool (down_score ≥ 3 dip minutes) and the confirmation knob "
        "`bounce_prob_min` reads the **v1_short** model output `short_adverse_prob` — "
        "P(rest-of-day upside MFE ≥ +1.5% from the decision bar), i.e. the SHORT's "
        "adverse-move probability — **as a LONG bounce probability**. The probability column "
        "keeps its `short_adverse_prob` name (it is that model's output). **Caveat:** the "
        f"train window (≤ {args.train_end}) is IN-SAMPLE for v1_short, so train numbers only "
        "rank combinations; the judgement below reads the holdout window only.\n")
    lines.append(
        f"The top {len(results)} train-ranked combinations from the sweep "
        f"(`{sources['summary']}`) were re-gated on the pre-gate replay candidates "
        f"(`{sources['candidates']}`, {sources['n_candidates']:,} rows) through the same offline "
        "path the sweep uses (dip floor → bounce confirmation → bounce-evidence floor → "
        "long-risk cap → session window → causal cooldown/daily-limit selection) and judged "
        f"**on the holdout window only** ({args.validation_start} → {args.validation_end}). "
        "Forward metrics follow the project execution model: entry = close of the decision + "
        "delay bar; raw close returns, so **positive = long profit**.\n")
    lines.append("Baseline the gate must beat:\n")
    lines.append(
        "1. **Time-of-day-matched** — unconditional LONG entry at the same HH:MM:SS on every "
        "other holdout day (paired per fire; edge = alert − baseline, positive = long edge).\n")

    if not results:
        lines.append("## No qualifying combinations\n")
        lines.append("The sweep produced no combination meeting the minimum samples "
                     "(train ≥ 30 fires; holdout ≥ 20 fires on ≥ 15 days), so nothing could be "
                     "judged. Verdict defaults to NO-GO.\n")
    else:
        lines.append(f"## 1. Top train-ranked combinations (train ≤ {args.train_end}, "
                     "in-sample for v1_short)\n")
        lines.append("| rank | combo | train n | avg fwd60 | dir-hit60 | stop-first |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            lines.append(f"| {r['train_rank']} | `{r['combo_name']}` | {r['train_n']} | "
                         f"{_pct(r['train_avg_future_return_60m'])} | "
                         f"{_rate(r['train_dir_hit_60m_rate'])} | "
                         f"{_rate(r['train_long_stop_first_rate'])} |")
        lines.append("")

        lines.append(f"## 2. Holdout metrics ({args.validation_start} → {args.validation_end})\n")
        lines.append("| combo | n | days | avg fwd60 | baseline fwd60 | paired edge | t | "
                     "dir-hit60 | stop-first | avg MAE | up2_60m | −months |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in results:
            lines.append(
                f"| `{r['combo_name']}` | {r['holdout_n']} | {r['holdout_days']} | "
                f"{_pct(r['holdout_avg_future_return_60m'])} | "
                f"{_pct(r['baseline_avg_future_return_60m'])} | "
                f"{_pct(r['paired_mean_diff'])} | {_num(r['paired_t_stat'])} | "
                f"{_rate(r['holdout_dir_hit_60m_rate'])} | "
                f"{_rate(r['holdout_long_stop_first_rate'])} | "
                f"{_pct(r['holdout_avg_long_mae'])} | {_rate(r['holdout_up2_60m_rate'])} | "
                f"{r['holdout_n_negative_months']} |")
        lines.append("")

    lines.append("## 3. Gate criteria (ALL must hold on holdout for GO)\n")
    lines.append(f"- n ≥ {GO_MIN_FIRES} evaluable fires on ≥ {GO_MIN_DAYS} distinct days")
    lines.append(f"- avg fwd60 ≥ {_pct(GO_AVG_FWD60_MIN)} AND paired edge ≥ "
                 f"{_pct(GO_PAIRED_EDGE_MIN)} AND paired t ≥ {GO_T_MIN:g}")
    lines.append(f"- dir-hit60 ≥ {_rate(GO_DIR_HIT60_MIN)}")
    lines.append(f"- long_stop_first ≤ {_rate(GO_STOP_FIRST_MAX)} AND avg long_mae ≤ "
                 f"{_pct(GO_LONG_MAE_MAX)}")
    lines.append(f"- up2_60m ≥ {_rate(GO_UP2_60M_MIN)}")
    lines.append(f"- train avg fwd60 > 0 (sign consistency) AND ≤ {GO_MAX_NEGATIVE_MONTHS} "
                 "negative holdout month(s)\n")
    lines.append("PARTIAL = beats the time-of-day baseline (positive paired edge) while exactly "
                 "one of {t_stat, n, long_stop_first} fails (every other criterion passes).\n")

    if results:
        lines.append("## 4. Verdicts\n")
        lines.append("| combo | verdict | beats ToD baseline | failed criteria |")
        lines.append("|---|---|---|---|")
        for r in results:
            failed = ", ".join(r["criteria_failed"]) if r["criteria_failed"] else "—"
            lines.append(f"| `{r['combo_name']}` | **{r['verdict']}** | "
                         f"{r['beats_tod_baseline']} | {failed} |")
        lines.append("")

    lines.append("## Status\n")
    if overall == "GO":
        lines.append("- The GO combination is recorded here as validated research evidence; "
                     "promoting it into any live decision-support layer is a separate, "
                     "deliberate edit.")
    else:
        lines.append("- The bounce-long reuse of `short_adverse_prob` stays UNVALIDATED; "
                     "nothing is wired live.")
        lines.append("- Project convention: the failed configuration is documented here and NOT "
                     "wired live.")
    lines.append(f"- Generated {now_kst_text()} by scripts/evaluate_bounce_long_gate.py "
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
    if "long_risk_prob" not in candidates.columns:
        candidates = candidates.assign(long_risk_prob=float("nan"))

    needed = {"combo_name", "split", "min_sample_ok", "train_rank", "entry_delay_minutes"}
    if not needed.issubset(summary.columns):
        raise SystemExit(f"sweep summary {summary_src} is missing columns "
                         f"{sorted(needed - set(summary.columns))} - rerun bounce_long_sweep.py.")

    train_rows = summary[(summary["split"] == "train")
                         & (summary["min_sample_ok"] == 1)
                         & summary["train_rank"].notna()]
    top = train_rows.sort_values("train_rank").head(args.top)

    print(f"[{now_kst_text()}] bounce-long gate judgement summary={summary_src} "
          f"candidates={cand_src} top={len(top)}/{args.top}")
    print(f"  holdout: {args.validation_start} ~ {args.validation_end} "
          f"(train ranking window <= {args.train_end}; in-sample for v1_short)")

    results = []
    for row in top.to_dict("records"):
        res = evaluate_combo(row, candidates, bars, args)
        verdict, failed, beats_tod = decide_verdict(res)
        res.update({"verdict": verdict, "criteria_failed": failed,
                    "beats_tod_baseline": beats_tod})
        results.append(res)
        print(f"  #{res['train_rank']} {res['combo_name']:<36} {verdict:<8} "
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
        print(f"\nVERDICT: GO — {winners} passed every holdout criterion; the v1_short "
              f"probability adds LONG dip-buy value out-of-sample (promotion is a separate, "
              f"deliberate step).")
    elif overall == "PARTIAL":
        partials = ", ".join(r["combo_name"] for r in results if r["verdict"] == "PARTIAL")
        print(f"\nVERDICT: PARTIAL — {partials} beat the time-of-day baseline but failed one of "
              f"t_stat/n/long_stop_first; the bounce-long reuse stays UNVALIDATED.")
    else:
        print("\nVERDICT: NO-GO — no combination passed the holdout gate; reusing "
              "short_adverse_prob as a LONG dip-buy confirmation stays UNVALIDATED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
