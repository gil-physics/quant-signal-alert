"""Bounce-long (dip-buy) threshold sweep -- tune on train, confirm on holdout.

Offline study of REUSING the Step 15 v1_short model output as a LONG entry
confirmation: ``short_adverse_prob`` is P(rest-of-day upside MFE >= +1.5% from
the decision bar) -- the short's adverse-move probability, which from the long
perspective is a DIP-BOUNCE probability. The hypothesis: enter LONG when the
bar is a dip (the pre-gate ``short_replay_candidates`` pool, down_score >= 3)
AND the model says a bounce is likely (short_adverse_prob >= bounce_prob_min).

The probability column keeps its ``short_adverse_prob`` name everywhere (it is
that model's output); only the gate knob is named ``bounce_prob_min``. Grid
(5 x 3 x 3 x 3 = 135 combinations):

    bounce_prob_min  {None, 0.45, 0.55, 0.60, 0.65}   short_adverse_prob >=
    down_floor       {3, 4, 5}                        down_score >=
    upside_risk_min  {None, 1.5, 3.0}                 upside_risk_score >=
    long_risk_max    {None, 0.60, 0.45}               long_risk_prob <=

``bounce_prob_min=None`` rows measure the model's contribution against
rule-only dip buying; a NaN ``short_adverse_prob`` (or ``long_risk_prob``) row
fails any configured cut on that column (conservative, mirroring the short
sweep's NaN-veto rule). Selection is identical to the short sweep: session
09:10-15:00, cooldown 15m, daily limit 5, causal via
directional_alert.select_online_alerts, entry delay 3m.

Overfitting guard: combinations are RANKED on the train window ONLY (avg
future_return_60m DESC -> dir_hit_60m DESC -> long_stop_first ASC; the LONG
sign, so bigger is better on returns); holdout rows are written alongside but
never drive the ranking. NOTE the train window is IN-SAMPLE for the v1_short
model (it was trained there), so train numbers only rank -- the GO/NO-GO
judgement on the top combinations lives in
scripts/evaluate_bounce_long_gate.py and reads the holdout window only.
Minimum samples: train >= 30 evaluable fires and holdout >= 20 evaluable fires
on >= 15 distinct days, else min_sample_ok=0 and the combo is excluded from
ranking.

Research only -- no orders, no Kiwoom, no order API.

    ..\\Python311-64\\python.exe scripts\\bounce_long_sweep.py
    ..\\Python311-64\\python.exe scripts\\bounce_long_sweep.py --train-end 2026-02-28 --validation-start 2026-03-01 --validation-end 2026-06-26
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
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
from directional_alert import select_online_alerts
from settings import load_data_config
from short_entry_eval import LONG_EVAL_COLUMNS, evaluate_long_alerts
from utils import now_kst_text

SUMMARY_TABLE = "bounce_long_sweep_summary"

# grid: 5 * 3 * 3 * 3 = 135 combinations; down_floor 3 equals the candidates'
# pre-gate floor, so that setting buys the whole dip pool.
BOUNCE_PROB_GRID = (None, 0.45, 0.55, 0.60, 0.65)
DOWN_FLOOR_GRID = (3.0, 4.0, 5.0)
UPSIDE_RISK_GRID = (None, 1.5, 3.0)
LONG_RISK_GRID = (None, 0.60, 0.45)

# selection defaults (identical to the short sweep / ShortThresholds defaults)
DEFAULT_COOLDOWN_MIN = 15
DEFAULT_DAILY_LIMIT = 5
DEFAULT_MIN_TIME = "09:10:00"
DEFAULT_MAX_TIME = "15:00:00"

SUMMARY_COLUMNS = [
    "combo_name", "split",
    "bounce_prob_min", "down_floor", "upside_risk_min", "long_risk_max",
    "cooldown_minutes", "daily_alert_limit",
    "min_time", "max_time", "entry_delay_minutes",
    "n_selected", "n_evaluated", "n_days",
    "avg_future_return_30m", "avg_future_return_60m", "avg_future_return_120m",
    "avg_future_return_eod",
    "dir_hit_30m_rate", "dir_hit_60m_rate", "dir_hit_120m_rate",
    "long_stop_first_rate", "avg_long_mfe", "avg_long_mae",
    "up2_60m_rate", "reach_up15_eod_rate", "n_negative_months",
    "min_sample_ok", "train_rank",
]


@dataclass(frozen=True)
class BounceCombo:
    """One grid point of the bounce-long gate (mirror role of ShortThresholds).

    ``name`` tags every summary row so a result can always be traced back to
    the knob combination that produced it. ``bounce_prob_min`` reads the
    v1_short ``short_adverse_prob`` column AS a bounce probability; None
    disables the model confirmation (rule-only dip buying).
    """

    name: str
    bounce_prob_min: float | None
    down_floor: float
    upside_risk_min: float | None
    long_risk_max: float | None
    daily_alert_limit: int = DEFAULT_DAILY_LIMIT
    cooldown_minutes: int = DEFAULT_COOLDOWN_MIN
    min_time: str = DEFAULT_MIN_TIME
    max_time: str = DEFAULT_MAX_TIME


def parse_args():
    p = argparse.ArgumentParser(
        description="Bounce-long sweep: reuse v1_short's short_adverse_prob as a LONG "
                    "dip-buy confirmation over short_replay_candidates "
                    "(train ranking -> holdout confirmation; research only).")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--candidates", default=None,
                   help="explicit candidates parquet (default: DB table "
                        f"{sgs.CANDIDATES_TABLE}, then {sgs.CANDIDATES_PARQUET}).")
    p.add_argument("--train-end", default="2026-02-28",
                   help="train split = dates <= this (ranking window; IN-SAMPLE for v1_short).")
    p.add_argument("--validation-start", default="2026-03-01")
    p.add_argument("--validation-end", default="2026-06-26")
    p.add_argument("--entry-delay", type=int, default=3,
                   help="entry = close of decision + N minutes bar (project standard 3).")
    p.add_argument("--cooldown-min", type=int, default=DEFAULT_COOLDOWN_MIN,
                   help=f"minutes between fires (default {DEFAULT_COOLDOWN_MIN}).")
    p.add_argument("--daily-limit", "--daily-alert-limit", dest="daily_limit", type=int,
                   default=DEFAULT_DAILY_LIMIT,
                   help=f"max fires per day (default {DEFAULT_DAILY_LIMIT}; "
                        "--daily-alert-limit is the runner-compatible alias).")
    p.add_argument("--min-time", default=DEFAULT_MIN_TIME,
                   help=f"session start (default {DEFAULT_MIN_TIME}).")
    p.add_argument("--max-time", default=DEFAULT_MAX_TIME,
                   help=f"session end, exclusive (default {DEFAULT_MAX_TIME}).")
    p.add_argument("--min-train-fires", type=int, default=30,
                   help="ranking floor: evaluable train fires (default 30).")
    p.add_argument("--min-holdout-fires", type=int, default=20,
                   help="ranking floor: evaluable holdout fires (default 20).")
    p.add_argument("--min-holdout-days", type=int, default=15,
                   help="ranking floor: distinct holdout fire days (default 15).")
    p.add_argument("--stamp", default=None,
                   help="output filename stamp (default: now-KST, threshold_sweep convention).")
    p.add_argument("--out-dir", default=str(PROCESSED_DIR),
                   help="directory for the summary CSV (default data/processed).")
    return p.parse_args()


def combo_name(bounce, down, urisk, lrisk):
    return (f"bp{sgs._fmt_knob(bounce)}_down{down:g}"
            f"_ur{sgs._fmt_knob(urisk)}_lr{sgs._fmt_knob(lrisk)}")


def make_combo(bounce, down, urisk, lrisk, *, daily_limit, cooldown_min, min_time, max_time):
    """One grid point as a BounceCombo (name = the combo's provenance tag)."""
    return BounceCombo(
        name=combo_name(bounce, down, urisk, lrisk),
        bounce_prob_min=None if bounce is None else float(bounce),
        down_floor=float(down),
        upside_risk_min=None if urisk is None else float(urisk),
        long_risk_max=None if lrisk is None else float(lrisk),
        daily_alert_limit=int(daily_limit),
        cooldown_minutes=int(cooldown_min),
        min_time=str(min_time),
        max_time=str(max_time),
    )


def apply_combo(candidates, combo: BounceCombo):
    """Offline pre-selection checks for one combo: dip floor, model bounce
    confirmation, bounce-evidence floor, long-risk cap, session window.

    NaN handling is conservative and mirrors the short sweep's veto rule: a NaN
    ``short_adverse_prob`` fails any configured ``bounce_prob_min`` (an
    unscored row never confirms a bounce) and a NaN ``long_risk_prob`` fails
    any configured ``long_risk_max`` (unknown long risk is not assumed low).
    """
    mask = candidates["down_score"] >= combo.down_floor
    if combo.bounce_prob_min is not None:
        # NaN fails `>=` -> excluded
        mask &= candidates["short_adverse_prob"] >= combo.bounce_prob_min
    if combo.upside_risk_min is not None:
        mask &= candidates["upside_risk_score"] >= combo.upside_risk_min
    if combo.long_risk_max is not None:
        # NaN fails `<=` -> excluded
        mask &= candidates["long_risk_prob"] <= combo.long_risk_max
    tod = candidates["time"]
    mask &= (tod >= combo.min_time) & (tod < combo.max_time)
    return candidates.loc[mask.fillna(False)]


def select_fires(gated, combo: BounceCombo):
    """Causal cooldown + daily-limit selection (owned by directional_alert)."""
    return select_online_alerts(gated, daily_alert_limit=combo.daily_alert_limit,
                                cooldown_minutes=combo.cooldown_minutes)


def _n_negative_months(evaluated):
    """Months whose avg future_return_60m is negative (NaN returns excluded)."""
    if len(evaluated) == 0:
        return 0
    m = evaluated.dropna(subset=["future_return_60m"])
    if m.empty:
        return 0
    months = m.groupby(m["date"].str.slice(0, 7))["future_return_60m"].mean()
    return int((months < 0).sum())


def _summary_row(combo, split, selected, joined, entry_delay):
    def _mean(col):
        return float(joined[col].mean()) if len(joined) else float("nan")

    return {
        "combo_name": combo.name,
        "split": split,
        "bounce_prob_min": combo.bounce_prob_min,
        "down_floor": combo.down_floor,
        "upside_risk_min": combo.upside_risk_min,
        "long_risk_max": combo.long_risk_max,
        "cooldown_minutes": combo.cooldown_minutes,
        "daily_alert_limit": combo.daily_alert_limit,
        "min_time": combo.min_time,
        "max_time": combo.max_time,
        "entry_delay_minutes": int(entry_delay),
        "n_selected": int(len(selected)),
        "n_evaluated": int(len(joined)),
        "n_days": int(joined["date"].nunique()) if len(joined) else 0,
        "avg_future_return_30m": _mean("future_return_30m"),
        "avg_future_return_60m": _mean("future_return_60m"),
        "avg_future_return_120m": _mean("future_return_120m"),
        "avg_future_return_eod": _mean("future_return_eod"),
        "dir_hit_30m_rate": _mean("direction_hit_30m"),
        "dir_hit_60m_rate": _mean("direction_hit_60m"),
        "dir_hit_120m_rate": _mean("direction_hit_120m"),
        "long_stop_first_rate": _mean("long_stop_first"),
        "avg_long_mfe": _mean("long_mfe"),
        "avg_long_mae": _mean("long_mae"),
        "up2_60m_rate": _mean("up2_60m"),
        "reach_up15_eod_rate": _mean("reach_up15_eod"),
        "n_negative_months": _n_negative_months(joined),
    }


def _pct(x):
    return f"{x * 100:+.2f}%" if x == x else "n/a"


def _rate(x):
    return f"{x * 100:.1f}%" if x == x else "n/a"


def main():
    args = parse_args()
    ensure_directories()
    source = load_data_config().model_source

    combos = [make_combo(b, d, u, l,
                         daily_limit=args.daily_limit, cooldown_min=args.cooldown_min,
                         min_time=args.min_time, max_time=args.max_time)
              for b in BOUNCE_PROB_GRID for d in DOWN_FLOOR_GRID
              for u in UPSIDE_RISK_GRID for l in LONG_RISK_GRID]

    conn = sqlite3.connect(args.db)
    try:
        candidates, cand_src = sgs.load_candidates(conn, args.candidates)
        bars = sgs.load_bars(conn, source, candidates["symbol"].unique())

        print(f"[{now_kst_text()}] bounce-long sweep combos={len(combos)} "
              f"candidates={len(candidates)} days={candidates['date'].nunique()} "
              f"symbols={sorted(candidates['symbol'].unique())} source={cand_src}")
        print(f"  framing: LONG entries on dip minutes; short_adverse_prob reused as "
              f"bounce probability (knob: bounce_prob_min)")
        print(f"  train: <= {args.train_end} (ranking only; IN-SAMPLE for v1_short) | "
              f"holdout: {args.validation_start} ~ {args.validation_end}")
        print(f"  selection: cooldown={args.cooldown_min}m daily_limit={args.daily_limit} "
              f"session={args.min_time}-{args.max_time} entry_delay={args.entry_delay}m")
        print(f"  min samples: train>={args.min_train_fires} fires, "
              f"holdout>={args.min_holdout_fires} fires on >={args.min_holdout_days} days")
        if not candidates["short_adverse_prob"].notna().any():
            print("  note: candidates carry no short_adverse_prob scores (rule-only build); "
                  "bounce_prob_min combos can never fire and will not qualify")
        if "long_risk_prob" not in candidates.columns:
            candidates = candidates.assign(long_risk_prob=float("nan"))
            print("  note: candidates carry no long_risk_prob column; "
                  "long_risk_max combos can never fire and will not qualify")

        # pass 1: gate + causal selection per combo (cheap, vectorized + per-day loop)
        selected_by_combo = []
        for i, combo in enumerate(combos, 1):
            fires = select_fires(apply_combo(candidates, combo), combo)
            selected_by_combo.append((combo, fires))
            if i % 50 == 0 or i == len(combos):
                print(f"  [{i}/{len(combos)}] combos gated+selected")

        # pass 2: forward metrics ONCE for the union of fired rows (selection never
        # reads the forward columns, so re-joining them per combo is leakage-free)
        key_cols = ["symbol", "decision_time", "close", "date", "time"]
        fired = [f[key_cols] for _, f in selected_by_combo if len(f)]
        union = (pd.concat(fired, ignore_index=True).drop_duplicates(["symbol", "decision_time"])
                 if fired else pd.DataFrame(columns=key_cols))
        print(f"  evaluating {len(union)} unique fired rows "
              f"(entry = decision + {args.entry_delay}m bar close, LONG metrics)")
        evaluated = evaluate_long_alerts(union, bars, entry_delay_minutes=args.entry_delay)
        metric_cols = ["symbol", "decision_time"] + LONG_EVAL_COLUMNS
        metrics = evaluated[metric_cols] if len(evaluated) else pd.DataFrame(columns=metric_cols)

        # pass 3: per-combo, per-split aggregation
        rows = []
        for combo, fires in selected_by_combo:
            train_sel = fires[fires["date"] <= args.train_end]
            hold_sel = fires[(fires["date"] >= args.validation_start)
                             & (fires["date"] <= args.validation_end)]
            for split, part in (("train", train_sel), ("holdout", hold_sel)):
                joined = part.merge(metrics, on=["symbol", "decision_time"],
                                    how="inner", validate="one_to_one")
                rows.append(_summary_row(combo, split, part, joined, args.entry_delay))
        summary = pd.DataFrame(rows)

        # ranking on TRAIN only (LONG sign: avg fwd60 DESC); both splits must
        # clear the minimum samples
        train_view = summary[summary["split"] == "train"].set_index("combo_name")
        hold_view = summary[summary["split"] == "holdout"].set_index("combo_name")
        ok = ((train_view["n_evaluated"] >= args.min_train_fires)
              & (hold_view["n_evaluated"] >= args.min_holdout_fires)
              & (hold_view["n_days"] >= args.min_holdout_days))
        summary["min_sample_ok"] = summary["combo_name"].map(ok).fillna(False).astype(int)
        ranked = train_view[ok].sort_values(
            ["avg_future_return_60m", "dir_hit_60m_rate", "long_stop_first_rate"],
            ascending=[False, False, True], na_position="last")
        rank_map = {name: i + 1 for i, name in enumerate(ranked.index)}
        summary["train_rank"] = summary["combo_name"].map(rank_map)
        summary["_split_order"] = (summary["split"] == "holdout").astype(int)
        summary = (summary.sort_values(["train_rank", "combo_name", "_split_order"],
                                       na_position="last")
                   .drop(columns=["_split_order"]).reset_index(drop=True))[SUMMARY_COLUMNS]

        summary.to_sql(SUMMARY_TABLE, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    stamp = args.stamp or now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"bounce_long_sweep_summary_{stamp}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n  qualifying combos: {len(rank_map)}/{len(combos)}")
    top_names = [n for n, _ in sorted(rank_map.items(), key=lambda kv: kv[1])[:10]]
    if top_names:
        print("  top train-ranked (train | holdout):")
        for name in top_names:
            t, h = train_view.loc[name], hold_view.loc[name]
            print(f"   #{rank_map[name]:>2} {name:<36} "
                  f"train n={int(t['n_evaluated']):>4} fwd60={_pct(t['avg_future_return_60m'])} "
                  f"hit={_rate(t['dir_hit_60m_rate'])} stop={_rate(t['long_stop_first_rate'])} | "
                  f"holdout n={int(h['n_evaluated']):>4} fwd60={_pct(h['avg_future_return_60m'])} "
                  f"hit={_rate(h['dir_hit_60m_rate'])} stop={_rate(h['long_stop_first_rate'])}")
    else:
        print("  no combination met the minimum sample sizes")

    print(f"\nsummary -> {summary_path}")
    print(f"db table -> {SUMMARY_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
