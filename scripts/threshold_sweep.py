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

import pandas as pd

from config import DB_PATH, PROCESSED_DIR, THRESHOLD_SWEEP_LOG_PATH, ensure_directories
from daily_topk_backtest import quality_exclusion_pairs
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from threshold_sweep import (
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    build_default_specs,
    prepare_base_frame,
    run_threshold_sweep,
    stable_validation_view,
)
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 13: threshold sweep for rule_signals using train/validation splits."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--quality-report", default=None,
                        help="Optional Step 6 CSV; default = latest quality_report_*.csv.")
    parser.add_argument("--ignore-quality-report", action="store_true")
    parser.add_argument("--keep-outlier-dates", action="store_true")
    parser.add_argument("--min-time", default="09:10:00")
    parser.add_argument("--max-time", default="15:00:00")
    parser.add_argument("--cooldown-minutes", type=int, default=60)
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--validation-start", default=VALIDATION_START)
    parser.add_argument("--validation-end", default=VALIDATION_END)
    parser.add_argument("--score-grid-only", action="store_true")
    parser.add_argument("--percentiles-only", action="store_true")
    parser.add_argument("--min-validation-signals", type=int, default=10)
    return parser.parse_args()


def _latest_quality_report():
    reports = sorted(PROCESSED_DIR.glob("quality_report_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _load_quality_report(path):
    if path is None:
        return None
    return pd.read_csv(path)


def _fmt_pct(value):
    return f"{float(value) * 100:.2f}%"


def _print_compact(rows, title):
    if rows.empty:
        print(f"\n{title}: none")
        return
    print(f"\n{title}:")
    for row in rows.head(10).to_dict("records"):
        print(
            f"  {row['threshold_name']:<42} "
            f"n={int(row['signal_count']):>4} day={row['avg_signals_per_day']:.2f} "
            f"hit2={_fmt_pct(row['hit_up2_rate'])} hit3={_fmt_pct(row['hit_up3_rate'])} "
            f"hit5={_fmt_pct(row['hit_up5_rate'])} mfe={_fmt_pct(row['average_MFE'])} "
            f"mae={_fmt_pct(row['average_MAE'])} stop={_fmt_pct(row['stop_first_rate'])} "
            f"r30={_fmt_pct(row['return_30m'])} r60={_fmt_pct(row['return_60m'])} "
            f"conc={_fmt_pct(row['symbol_concentration'])}"
        )


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(THRESHOLD_SWEEP_LOG_PATH, name="threshold_sweep")

    model_source = load_data_config().model_source
    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    quality_path = None
    if not args.ignore_quality_report:
        quality_path = Path(args.quality_report) if args.quality_report else _latest_quality_report()

    include_score_grid = not args.percentiles_only
    include_percentiles = not args.score_grid_only
    specs = build_default_specs(
        include_score_grid=include_score_grid,
        include_percentiles=include_percentiles,
    )

    print(
        f"[{now_kst_text()}] threshold sweep symbols={len(symbols)} "
        f"source={model_source} specs={len(specs)}"
    )
    print(f"  train: {args.train_start} ~ {args.train_end}")
    print(f"  validation: {args.validation_start} ~ {args.validation_end}")
    print(f"  quality report: {quality_path if quality_path else 'none'}")
    logger.info(
        "threshold sweep start symbols=%s source=%s specs=%s train=%s..%s validation=%s..%s",
        symbols,
        model_source,
        len(specs),
        args.train_start,
        args.train_end,
        args.validation_start,
        args.validation_end,
    )

    conn = sqlite3.connect(args.db)
    try:
        rule_signals = pd.read_sql_query(
            "SELECT * FROM rule_signals WHERE source = ?", conn, params=(model_source,)
        )
        labels = pd.read_sql_query(
            "SELECT * FROM labels WHERE source = ?", conn, params=(model_source,)
        )
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, open, high, low, close, volume "
            "FROM bars_1m WHERE source = ?",
            conn,
            params=(model_source,),
        )

        rule_signals = rule_signals[rule_signals["symbol"].isin(symbols)].reset_index(drop=True)
        labels = labels[labels["symbol"].isin(symbols)].reset_index(drop=True)
        bars = bars[bars["symbol"].isin(symbols)].reset_index(drop=True)
        if rule_signals.empty:
            raise SystemExit("rule_signals is empty - run build_rule_signals.py first.")
        if labels.empty:
            raise SystemExit("labels is empty - run make_labels.py first.")
        if bars.empty:
            raise SystemExit(f"bars_1m {model_source} is empty for active symbols.")

        quality_report = _load_quality_report(quality_path)
        fail_pairs, outlier_pairs = quality_exclusion_pairs(
            quality_report,
            exclude_outlier_dates=not args.keep_outlier_dates,
        )
        base, base_stats = prepare_base_frame(
            rule_signals,
            labels,
            symbols=symbols,
            min_time=args.min_time,
            max_time=args.max_time,
            fail_quality_pairs=fail_pairs,
            outlier_pairs=outlier_pairs,
        )

        summary, evaluated = run_threshold_sweep(
            base,
            bars,
            specs,
            base_stats=base_stats,
            train_start=args.train_start,
            train_end=args.train_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            cooldown_minutes=args.cooldown_minutes,
        )

        summary.to_sql("threshold_sweep_summary", conn, if_exists="replace", index=False)
        evaluated.to_sql("threshold_sweep_signals", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    stamp = now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
    summary_path = PROCESSED_DIR / f"threshold_sweep_summary_{stamp}.csv"
    signals_path = PROCESSED_DIR / f"threshold_sweep_signals_{stamp}.parquet"
    stable_path = PROCESSED_DIR / f"threshold_sweep_stable_{stamp}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    evaluated.to_parquet(signals_path, index=False)

    stable = stable_validation_view(summary, min_validation_signals=args.min_validation_signals)
    stable.to_csv(stable_path, index=False, encoding="utf-8-sig")

    validation = summary[summary["train_or_validation"] == "validation"].copy()
    baseline = validation[
        (validation["is_baseline"] == 1)
        & (validation["top_k"] == 1)
        & (validation["entry_delay"] == 3)
        & (validation["threshold_name"].str.startswith("score_"))
    ].head(1)

    print(
        f"\nbase rows: raw={base_stats.get('raw_rows', 0)} "
        f"time={base_stats.get('after_time_filter', 0)} "
        f"quality={base_stats.get('after_quality_filter', 0)}"
    )
    _print_compact(baseline, "baseline validation conservative + entry_delay_3m + top1")
    _print_compact(
        validation[
            (validation["signal_count"] >= args.min_validation_signals)
            & (validation["avg_signals_per_day"] <= 2)
        ].sort_values(
            ["hit_up3_rate", "up3_120m_rate", "stop_first_rate", "return_60m", "signal_count"],
            ascending=[False, False, True, False, False],
        ),
        "validation candidates sorted by balanced Step 13 criteria",
    )

    print(f"\nsummary -> {summary_path}")
    print(f"signals -> {signals_path}")
    print(f"stable train/validation view -> {stable_path}")
    print("db tables -> threshold_sweep_summary, threshold_sweep_signals")
    logger.info(
        "threshold sweep end summary_rows=%s signals=%s summary=%s",
        len(summary),
        len(evaluated),
        summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
