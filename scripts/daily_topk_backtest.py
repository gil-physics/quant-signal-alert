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

from config import BACKTEST_LOG_PATH, DB_PATH, PROCESSED_DIR, ensure_directories
from daily_topk_backtest import (
    THRESHOLD_CONFIGS,
    join_signals_labels,
    quality_exclusion_pairs,
    run_backtests,
    threshold_config_by_name,
)
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 12: daily top1/top2 backtest from rule_signals + labels."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--threshold-set", default="all",
                        help="all, conservative, medium, or aggressive.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cooldown-minutes", type=int, default=60)
    parser.add_argument("--min-time", default="09:10:00")
    parser.add_argument("--max-time", default="15:00:00")
    parser.add_argument("--quality-report", default=None,
                        help="Optional Step 6 CSV; default = latest data/processed/quality_report_*.csv if present.")
    parser.add_argument("--ignore-quality-report", action="store_true")
    parser.add_argument("--keep-outlier-dates", action="store_true",
                        help="Do not exclude dates with spike/range_outlier in the Step 6 report.")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def _latest_quality_report():
    reports = sorted(PROCESSED_DIR.glob("quality_report_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _load_quality_report(path):
    if path is None:
        return None
    import pandas as pd

    return pd.read_csv(path)


def _thresholds(name):
    if name == "all":
        return THRESHOLD_CONFIGS
    return [threshold_config_by_name(name)]


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(BACKTEST_LOG_PATH, name="backtest")

    import pandas as pd

    model_source = load_data_config().model_source
    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    quality_path = None
    if not args.ignore_quality_report:
        quality_path = Path(args.quality_report) if args.quality_report else _latest_quality_report()

    print(
        f"[{now_kst_text()}] daily top-k backtest symbols={symbols} "
        f"source={model_source} thresholds={args.threshold_set} top_k={args.top_k}"
    )
    if quality_path:
        print(f"  quality report: {quality_path}")
    else:
        print("  quality report: none")
    logger.info(
        "backtest start symbols=%s source=%s threshold_set=%s top_k=%s cooldown=%s",
        symbols, model_source, args.threshold_set, args.top_k, args.cooldown_minutes,
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
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
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
            raise SystemExit("bars_1m al_regular is empty for active symbols.")

        joined = join_signals_labels(rule_signals, labels, symbols=symbols)
        quality_report = _load_quality_report(quality_path)
        fail_pairs, outlier_pairs = quality_exclusion_pairs(
            quality_report, exclude_outlier_dates=not args.keep_outlier_dates
        )

        signals, daily_summary, threshold_summary = run_backtests(
            joined,
            bars,
            threshold_configs=_thresholds(args.threshold_set),
            top_k=args.top_k,
            cooldown_minutes=args.cooldown_minutes,
            min_time=args.min_time,
            max_time=args.max_time,
            fail_quality_pairs=fail_pairs,
            outlier_pairs=outlier_pairs,
        )

        parquet_path = PROCESSED_DIR / "backtest_signals.parquet"
        signals.to_parquet(parquet_path, index=False)
        signals.to_sql("backtest_signals", conn, if_exists="replace", index=False)
        daily_summary.to_sql("backtest_daily_summary", conn, if_exists="replace", index=False)
        threshold_summary.to_sql("backtest_threshold_summary", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    stamp = now_kst_text().replace(":", "").replace(" ", "_").replace("-", "")
    daily_path = PROCESSED_DIR / f"daily_topk_summary_{stamp}.csv"
    threshold_path = PROCESSED_DIR / f"daily_topk_threshold_summary_{stamp}.csv"
    daily_summary.to_csv(daily_path, index=False, encoding="utf-8-sig")
    threshold_summary.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    print("\nthreshold summary:")
    for row in threshold_summary.to_dict("records"):
        print(
            f"  {row['threshold_name']:<12} raw={row['raw_rows']} "
            f"time={row['after_time_filter']} quality={row['after_quality_filter']} "
            f"both={row['after_both_exclusion']} score={row['after_score_filter']} "
            f"cooldown={row['after_cooldown']} final={row['final_topk']} "
            f"hit2={row['hit_up2_rate']:.2%} hit3={row['hit_up3_rate']:.2%} "
            f"hit5={row['hit_up5_rate']:.2%} mfe={row['average_mfe']:.2%} "
            f"mae={row['average_mae']:.2%} stop_first={row['stop_first_rate']:.2%}"
        )
        print(
            f"    top1 hit2={row['top1_hit_up2_rate']:.2%} "
            f"hit3={row['top1_hit_up3_rate']:.2%} hit5={row['top1_hit_up5_rate']:.2%} "
            f"mfe={row['top1_average_mfe']:.2%} mae={row['top1_average_mae']:.2%} "
            f"stop={row['top1_stop_first_rate']:.2%} eod={row['top1_return_eod']:.2%}"
        )
        print(
            f"    top2 hit2={row['top2_hit_up2_rate']:.2%} "
            f"hit3={row['top2_hit_up3_rate']:.2%} hit5={row['top2_hit_up5_rate']:.2%} "
            f"mfe={row['top2_average_mfe']:.2%} mae={row['top2_average_mae']:.2%} "
            f"stop={row['top2_stop_first_rate']:.2%} eod={row['top2_return_eod']:.2%}"
        )
    print(f"\nbacktest_signals -> db + {parquet_path}")
    print(f"daily summary -> {daily_path}")
    print(f"threshold summary -> {threshold_path}")
    logger.info(
        "backtest end signals=%s daily_rows=%s threshold_rows=%s",
        len(signals), len(daily_summary), len(threshold_summary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
