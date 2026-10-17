"""Step 13_6: build a directional-alert evaluation report.

This script intentionally reuses Step 13_5 alert classification without editing
those files. It writes fixed report artifact names requested by step13_6.md.
"""

from __future__ import annotations

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

from config import DB_PATH, PROCESSED_DIR, ensure_directories
from daily_topk_backtest import quality_exclusion_pairs
from directional_alert_eval import (
    LABEL_COLUMNS,
    MICRO_COLUMNS,
    ReportConfig,
    build_alert_catalog,
    build_quality_checks,
    evaluate_alerts,
    select_for_evaluation,
    summarize_alerts,
)
from settings import load_active_symbols, load_data_config
from utils import now_kst_text


def _latest_quality_report():
    reports = sorted(PROCESSED_DIR.glob("quality_report_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _backup_existing_table(conn, table_name, stamp):
    exists = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()[0]
    if not exists:
        return None
    backup_name = f"{table_name}_backup_before_step13_6_{stamp}"
    conn.execute(f"DROP TABLE IF EXISTS {backup_name}")
    conn.execute(f"CREATE TABLE {backup_name} AS SELECT * FROM {table_name}")
    return backup_name


def main() -> int:
    ensure_directories()
    symbols = load_active_symbols()
    model_source = load_data_config().model_source
    config = ReportConfig()
    created_at = now_kst_text()
    stamp = created_at.replace(":", "").replace(" ", "_").replace("-", "")

    conn = sqlite3.connect(DB_PATH)
    try:
        rule_signals = pd.read_sql_query(
            "SELECT * FROM rule_signals WHERE source = ?",
            conn,
            params=(model_source,),
        )
        micro_cols = ", ".join(MICRO_COLUMNS)
        micro = pd.read_sql_query(
            f"SELECT symbol, decision_time, {micro_cols} FROM features_merged WHERE source = ?",
            conn,
            params=(model_source,),
        )
        label_cols = ", ".join(LABEL_COLUMNS)
        labels = pd.read_sql_query(
            f"SELECT symbol, decision_time, {label_cols} FROM labels WHERE source = ?",
            conn,
            params=(model_source,),
        )
        bars = pd.read_sql_query(
            "SELECT symbol, bar_start, high, low, close FROM bars_1m WHERE source = ?",
            conn,
            params=(model_source,),
        )
    finally:
        conn.close()

    for frame in (rule_signals, micro, labels, bars):
        frame.drop(frame.index[~frame["symbol"].isin(symbols)], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    quality_path = _latest_quality_report()
    quality_report = pd.read_csv(quality_path) if quality_path else None
    fail_pairs, outlier_pairs = quality_exclusion_pairs(quality_report)

    alerts, filter_stats = build_alert_catalog(
        rule_signals,
        micro,
        created_at=created_at,
        min_time=config.min_time,
        max_time=config.max_time,
        fail_quality_pairs=fail_pairs,
        outlier_pairs=outlier_pairs,
    )
    selected = select_for_evaluation(alerts, config=config)
    evaluated = evaluate_alerts(
        selected,
        bars,
        labels,
        entry_delay_minutes=config.entry_delay_minutes,
    )

    eval_summary = summarize_alerts(evaluated, by=["selection_mode"])
    grade_summary = summarize_alerts(evaluated, by=["selection_mode", "alert_grade"])
    type_summary = summarize_alerts(evaluated, by=["selection_mode", "alert_type"])
    checks = build_quality_checks(grade_summary, type_summary)

    # Fixed artifact names requested by step13_6.md.
    alerts_path = PROCESSED_DIR / "directional_alerts.parquet"
    eval_path = PROCESSED_DIR / "alert_evaluation_summary.csv"
    grade_path = PROCESSED_DIR / "alert_grade_summary.csv"
    type_path = PROCESSED_DIR / "alert_type_summary.csv"
    checks_path = PROCESSED_DIR / "alert_quality_checks.csv"

    alerts.to_parquet(alerts_path, index=False)
    eval_summary.to_csv(eval_path, index=False, encoding="utf-8-sig")
    grade_summary.to_csv(grade_path, index=False, encoding="utf-8-sig")
    type_summary.to_csv(type_path, index=False, encoding="utf-8-sig")
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")

    conn = sqlite3.connect(DB_PATH)
    try:
        backup = _backup_existing_table(conn, "directional_alerts", stamp)
        alerts.to_sql("directional_alerts", conn, if_exists="replace", index=False)
        evaluated.to_sql("directional_alert_evaluation_results", conn, if_exists="replace", index=False)
        eval_summary.to_sql("alert_evaluation_summary", conn, if_exists="replace", index=False)
        grade_summary.to_sql("alert_grade_summary", conn, if_exists="replace", index=False)
        type_summary.to_sql("alert_type_summary", conn, if_exists="replace", index=False)
        checks.to_sql("alert_quality_checks", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print(f"[{created_at}] Step 13_6 directional alert report")
    print(f"  symbols={len(symbols)} source={model_source} quality_report={quality_path.name if quality_path else 'none'}")
    print(
        "  filters: "
        f"raw={filter_stats.get('raw_rows', 0)} "
        f"time={filter_stats.get('after_time_filter', 0)} "
        f"quality={filter_stats.get('after_quality_filter', 0)} "
        f"both={filter_stats.get('after_both_exclusion', 0)} "
        f"fire_gate={filter_stats.get('after_fire_gate', 0)}"
    )
    if backup:
        print(f"  preserved previous DB table as {backup}")
    print(f"  directional_alerts rows={len(alerts)} -> {alerts_path}")
    print(f"  evaluated rows={len(evaluated)} modes={sorted(evaluated['selection_mode'].unique().tolist()) if len(evaluated) else []}")
    print(f"  summaries -> {eval_path}, {grade_path}, {type_path}")
    print("\nquality checks:")
    print(checks.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
