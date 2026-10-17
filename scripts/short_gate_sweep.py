"""Step 15: SHORT_ENTRY threshold sweep -- tune on train, confirm on holdout.

Re-gates the pre-gate ``short_replay_candidates`` rows (built by
scripts/build_short_replay_alerts.py, ideally with --cross-symbol) against the
full threshold grid and mirrors the ONLINE firing path offline for every
combination:

    short_entry.short_entry_gate (rule gate) & alert_type in DOWN_TYPES
      -> column-wise model veto (short_adverse_prob < short_adverse_prob_max;
         NaN is vetoed, same conservative rule as ShortFireGate)
      -> session window (min_time <= time < max_time)
      -> causal cooldown / daily-limit selection
         (directional_alert.select_online_alerts -- no future row influences
          an earlier firing decision)

The selected fires are scored with the short-direction forward metrics
(src/short_entry_eval.py: negative future_return = short profit).

Overfitting guard (plan section 5): combinations are RANKED on the train window
ONLY (avg future_return_60m asc -> dir_hit_60m desc -> stop_first asc); holdout
rows are written alongside but never drive the ranking. Minimum samples: train
>= 30 evaluable fires and holdout >= 20 evaluable fires on >= 15 distinct days,
else the combination is excluded from ranking (min_sample_ok=0). The GO/NO-GO
judgement on the top train-ranked combinations lives in
scripts/evaluate_short_entry_gate.py.

Research only -- no orders, no Kiwoom, no order API.

    ..\\Python311-64\\python.exe scripts\\short_gate_sweep.py
    ..\\Python311-64\\python.exe scripts\\short_gate_sweep.py --train-end 2026-02-28 --validation-start 2026-03-01 --validation-end 2026-06-26
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import replace
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

from config import DB_PATH, PROCESSED_DIR, ensure_directories
from directional_alert import select_online_alerts
from settings import load_data_config
from short_entry import DOWN_TYPES, ShortThresholds, short_entry_gate
from short_entry_eval import SHORT_EVAL_COLUMNS, evaluate_short_alerts
from utils import now_kst_text

CANDIDATES_TABLE = "short_replay_candidates"
CANDIDATES_PARQUET = PROCESSED_DIR / "short_replay_candidates.parquet"
SUMMARY_TABLE = "short_gate_sweep_summary"

# grid (plan section 5): 3 * 3 * 4 * 3 * 4 = 432 combinations; the candidates'
# down_score >= 3 floor sits below the lowest down_threshold, so no gated row
# is missing from the dump.
DOWN_GRID = (4.0, 5.0, 6.0)
UP_LIMIT_GRID = (0.0, 1.0, 2.0)
MARGIN_GRID = (2.0, 3.0, 4.0, 5.0)
RANK_GRID = (None, 5, 7)
VETO_GRID = (None, 0.60, 0.45, 0.35)

SUMMARY_COLUMNS = [
    "combo_name", "split",
    "down_threshold", "up_risk_limit", "score_margin", "bottom_rank_limit",
    "short_adverse_prob_max", "cooldown_minutes", "daily_alert_limit",
    "min_time", "max_time", "entry_delay_minutes",
    "n_selected", "n_evaluated", "n_days",
    "avg_future_return_30m", "avg_future_return_60m", "avg_future_return_120m",
    "avg_future_return_eod",
    "dir_hit_30m_rate", "dir_hit_60m_rate", "dir_hit_120m_rate",
    "stop_first_rate", "avg_short_mfe", "avg_short_mae",
    "down2_60m_rate", "reach_dn15_eod_rate", "n_positive_months",
    "min_sample_ok", "train_rank",
]

_D = ShortThresholds()   # dataclass defaults drive the selection CLI defaults


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 15: SHORT_ENTRY threshold sweep over short_replay_candidates "
                    "(train tuning -> holdout confirmation; research only).")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--candidates", default=None,
                   help="explicit candidates parquet (default: DB table "
                        f"{CANDIDATES_TABLE}, then {CANDIDATES_PARQUET}).")
    p.add_argument("--train-end", default="2026-02-28",
                   help="train split = dates <= this (ranking window).")
    p.add_argument("--validation-start", default="2026-03-01")
    p.add_argument("--validation-end", default="2026-06-26")
    p.add_argument("--entry-delay", type=int, default=3,
                   help="entry = close of decision + N minutes bar (project standard 3).")
    p.add_argument("--cooldown-min", type=int, default=_D.cooldown_minutes,
                   help=f"minutes between fires (default {_D.cooldown_minutes}).")
    p.add_argument("--daily-limit", "--daily-alert-limit", dest="daily_limit", type=int,
                   default=_D.daily_alert_limit,
                   help=f"max fires per day (default {_D.daily_alert_limit}; "
                        "--daily-alert-limit is the runner-compatible alias).")
    p.add_argument("--min-time", default=_D.min_time, help=f"session start (default {_D.min_time}).")
    p.add_argument("--max-time", default=_D.max_time, help=f"session end, exclusive (default {_D.max_time}).")
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


def _table_exists(conn, name):
    return bool(conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()[0])


def load_candidates(conn, parquet_override=None):
    """Pre-gate candidate rows: explicit parquet override, else the DB table,
    else the default parquet dump. Guarantees date/time columns and unique
    (symbol, decision_time) keys."""
    df, src = None, None
    if parquet_override:
        df, src = pd.read_parquet(parquet_override), str(parquet_override)
    elif conn is not None and _table_exists(conn, CANDIDATES_TABLE):
        df, src = pd.read_sql_query(f"SELECT * FROM {CANDIDATES_TABLE}", conn), f"db:{CANDIDATES_TABLE}"
    if (df is None or df.empty) and CANDIDATES_PARQUET.exists():
        df, src = pd.read_parquet(CANDIDATES_PARQUET), str(CANDIDATES_PARQUET)
    if df is None or df.empty:
        raise SystemExit("short_replay_candidates is empty - run "
                         "scripts/build_short_replay_alerts.py --cross-symbol first.")
    if "date" not in df.columns:
        df["date"] = df["decision_time"].str.slice(0, 10)
    if "time" not in df.columns:
        df["time"] = df["decision_time"].str.slice(11, 19)
    return df.drop_duplicates(["symbol", "decision_time"]).reset_index(drop=True), src


def load_bars(conn, source, symbols):
    bars = pd.read_sql_query(
        "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
        conn, params=(source,))
    bars = bars[bars["symbol"].isin(set(symbols))].reset_index(drop=True)
    if bars.empty:
        raise SystemExit(f"bars_1m {source} is empty for the candidate symbols.")
    return bars


def _fmt_knob(value):
    return "None" if value is None else f"{value:g}"


def combo_name(down, up, margin, rank, veto):
    return (f"down{down:g}_up{up:g}_m{margin:g}"
            f"_rank{_fmt_knob(rank)}_veto{_fmt_knob(veto)}")


def combo_thresholds(down, up, margin, rank, veto, *, base):
    """One grid point as a ShortThresholds (name = the combo's provenance tag)."""
    return replace(
        base,
        name=combo_name(down, up, margin, rank, veto),
        down_threshold=float(down),
        up_risk_limit=None if up is None else float(up),
        score_margin=float(margin),
        bottom_rank_limit=None if rank is None else int(rank),
        short_adverse_prob_max=None if veto is None else float(veto),
    )


def apply_combo(candidates, thr):
    """Offline mirror of the online DOWN-entry firing path: the row must classify
    as a DOWN type (alert_type in DOWN_TYPES) AND pass the rule gate -- this is
    exactly what short_alert_fire_gate does for the entry types -- then the
    column-wise model veto (NaN short_adverse_prob is vetoed, matching
    ShortFireGate's conservative rule), then the session window. This sweep tunes
    the DOWN-entry thresholds only; the informational alert types (SHORT_RISK /
    WEAKENING / VOLATILITY) fire on fixed constants and are not swept here."""
    mask = short_entry_gate(candidates, thr) & candidates["alert_type"].isin(DOWN_TYPES)
    if thr.short_adverse_prob_max is not None:
        # NaN fails `<` -> vetoed, same as the online gate
        mask &= candidates["short_adverse_prob"] < thr.short_adverse_prob_max
    tod = candidates["time"]
    mask &= (tod >= thr.min_time) & (tod < thr.max_time)
    return candidates.loc[mask]


def select_fires(gated, thr):
    """Causal cooldown + daily-limit selection (owned by directional_alert).
    ``daily_alert_limit=None`` means no cap (mirror of the live feed's default);
    select_online_alerts needs a number, so pass one larger than any day."""
    daily = thr.daily_alert_limit if thr.daily_alert_limit is not None else len(gated) + 1
    return select_online_alerts(gated, daily_alert_limit=daily,
                                cooldown_minutes=thr.cooldown_minutes)


def _n_positive_months(evaluated):
    """Months whose avg future_return_60m is positive (NaN returns excluded)."""
    if len(evaluated) == 0:
        return 0
    m = evaluated.dropna(subset=["future_return_60m"])
    if m.empty:
        return 0
    months = m.groupby(m["date"].str.slice(0, 7))["future_return_60m"].mean()
    return int((months > 0).sum())


def _summary_row(thr, split, selected, joined, entry_delay):
    def _mean(col):
        return float(joined[col].mean()) if len(joined) else float("nan")

    return {
        "combo_name": thr.name,
        "split": split,
        "down_threshold": thr.down_threshold,
        "up_risk_limit": thr.up_risk_limit,
        "score_margin": thr.score_margin,
        "bottom_rank_limit": thr.bottom_rank_limit,
        "short_adverse_prob_max": thr.short_adverse_prob_max,
        "cooldown_minutes": thr.cooldown_minutes,
        "daily_alert_limit": thr.daily_alert_limit,
        "min_time": thr.min_time,
        "max_time": thr.max_time,
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
        "stop_first_rate": _mean("short_stop_first"),
        "avg_short_mfe": _mean("short_mfe"),
        "avg_short_mae": _mean("short_mae"),
        "down2_60m_rate": _mean("down2_60m"),
        "reach_dn15_eod_rate": _mean("reach_dn15_eod"),
        "n_positive_months": _n_positive_months(joined),
    }


def _pct(x):
    return f"{x * 100:+.2f}%" if x == x else "n/a"


def _rate(x):
    return f"{x * 100:.1f}%" if x == x else "n/a"


def main():
    args = parse_args()
    ensure_directories()
    source = load_data_config().model_source

    combos = [(d, u, m, r, v)
              for d in DOWN_GRID for u in UP_LIMIT_GRID for m in MARGIN_GRID
              for r in RANK_GRID for v in VETO_GRID]
    base_thr = replace(_D, daily_alert_limit=args.daily_limit, cooldown_minutes=args.cooldown_min,
                       min_time=args.min_time, max_time=args.max_time)

    conn = sqlite3.connect(args.db)
    try:
        candidates, cand_src = load_candidates(conn, args.candidates)
        bars = load_bars(conn, source, candidates["symbol"].unique())

        print(f"[{now_kst_text()}] short gate sweep combos={len(combos)} "
              f"candidates={len(candidates)} days={candidates['date'].nunique()} "
              f"symbols={sorted(candidates['symbol'].unique())} source={cand_src}")
        print(f"  train: <= {args.train_end} | holdout: {args.validation_start} ~ {args.validation_end}")
        print(f"  selection: cooldown={args.cooldown_min}m daily_limit={args.daily_limit} "
              f"session={args.min_time}-{args.max_time} entry_delay={args.entry_delay}m")
        print(f"  min samples: train>={args.min_train_fires} fires, "
              f"holdout>={args.min_holdout_fires} fires on >={args.min_holdout_days} days")
        if "cross_symbol" in candidates.columns and not candidates["cross_symbol"].astype(bool).any():
            print("  note: candidates were built WITHOUT --cross-symbol (neutral mid ranks); "
                  "bottom-rank combos can never fire and will not qualify")
        if not candidates["short_adverse_prob"].notna().any():
            print("  note: candidates carry no short_adverse_prob scores (rule-only build); "
                  "veto combos can never fire and will not qualify")

        # pass 1: gate + causal selection per combo (cheap, vectorized + per-day loop)
        selected_by_combo = []
        for i, (d, u, m, r, v) in enumerate(combos, 1):
            thr = combo_thresholds(d, u, m, r, v, base=base_thr)
            fires = select_fires(apply_combo(candidates, thr), thr)
            selected_by_combo.append((thr, fires))
            if i % 100 == 0 or i == len(combos):
                print(f"  [{i}/{len(combos)}] combos gated+selected")

        # pass 2: forward metrics ONCE for the union of fired rows (selection never
        # reads the forward columns, so re-joining them per combo is leakage-free)
        key_cols = ["symbol", "decision_time", "close", "date", "time"]
        fired = [f[key_cols] for _, f in selected_by_combo if len(f)]
        union = (pd.concat(fired, ignore_index=True).drop_duplicates(["symbol", "decision_time"])
                 if fired else pd.DataFrame(columns=key_cols))
        print(f"  evaluating {len(union)} unique fired rows "
              f"(entry = decision + {args.entry_delay}m bar close)")
        evaluated = evaluate_short_alerts(union, bars, entry_delay_minutes=args.entry_delay)
        metric_cols = ["symbol", "decision_time"] + SHORT_EVAL_COLUMNS
        metrics = evaluated[metric_cols] if len(evaluated) else pd.DataFrame(columns=metric_cols)

        # pass 3: per-combo, per-split aggregation
        rows = []
        for thr, fires in selected_by_combo:
            train_sel = fires[fires["date"] <= args.train_end]
            hold_sel = fires[(fires["date"] >= args.validation_start)
                             & (fires["date"] <= args.validation_end)]
            for split, part in (("train", train_sel), ("holdout", hold_sel)):
                joined = part.merge(metrics, on=["symbol", "decision_time"],
                                    how="inner", validate="one_to_one")
                rows.append(_summary_row(thr, split, part, joined, args.entry_delay))
        summary = pd.DataFrame(rows)

        # ranking on TRAIN only; both splits must clear the minimum samples
        train_view = summary[summary["split"] == "train"].set_index("combo_name")
        hold_view = summary[summary["split"] == "holdout"].set_index("combo_name")
        ok = ((train_view["n_evaluated"] >= args.min_train_fires)
              & (hold_view["n_evaluated"] >= args.min_holdout_fires)
              & (hold_view["n_days"] >= args.min_holdout_days))
        summary["min_sample_ok"] = summary["combo_name"].map(ok).fillna(False).astype(int)
        ranked = train_view[ok].sort_values(
            ["avg_future_return_60m", "dir_hit_60m_rate", "stop_first_rate"],
            ascending=[True, False, True], na_position="last")
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
    summary_path = out_dir / f"short_gate_sweep_summary_{stamp}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n  qualifying combos: {len(rank_map)}/{len(combos)}")
    top_names = [n for n, _ in sorted(rank_map.items(), key=lambda kv: kv[1])[:10]]
    if top_names:
        print("  top train-ranked (train | holdout):")
        for name in top_names:
            t, h = train_view.loc[name], hold_view.loc[name]
            print(f"   #{rank_map[name]:>2} {name:<40} "
                  f"train n={int(t['n_evaluated']):>4} fwd60={_pct(t['avg_future_return_60m'])} "
                  f"hit={_rate(t['dir_hit_60m_rate'])} stop={_rate(t['stop_first_rate'])} | "
                  f"holdout n={int(h['n_evaluated']):>4} fwd60={_pct(h['avg_future_return_60m'])} "
                  f"hit={_rate(h['dir_hit_60m_rate'])} stop={_rate(h['stop_first_rate'])}")
    else:
        print("  no combination met the minimum sample sizes")

    print(f"\nsummary -> {summary_path}")
    print(f"db table -> {SUMMARY_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
