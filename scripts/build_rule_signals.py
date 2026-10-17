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

from config import DB_PATH, PROCESSED_DIR, RULE_SIGNAL_LOG_PATH, ensure_directories
from rule_signal import DEFAULT_DOWN_THRESHOLD, DEFAULT_UP_THRESHOLD, build_rule_signals, summarize_rule_signals
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 11: rule-based signal scores from features_relative."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--up-threshold", type=float, default=DEFAULT_UP_THRESHOLD)
    parser.add_argument("--down-threshold", type=float, default=DEFAULT_DOWN_THRESHOLD)
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(RULE_SIGNAL_LOG_PATH, name="rule_signal")

    import pandas as pd

    model_source = load_data_config().model_source
    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    print(
        f"[{now_kst_text()}] rule signals symbols={symbols} "
        f"source={model_source} thresholds=({args.up_threshold},{args.down_threshold})"
    )
    logger.info(
        "rule signal start symbols=%s source=%s up_threshold=%s down_threshold=%s",
        symbols, model_source, args.up_threshold, args.down_threshold,
    )

    conn = sqlite3.connect(args.db)
    try:
        features = pd.read_sql_query(
            "SELECT * FROM features_relative WHERE source = ?", conn, params=(model_source,)
        )
        features = features[features["symbol"].isin(symbols)].reset_index(drop=True)
        if features.empty:
            raise SystemExit("features_relative is empty for active symbols - run build_relative_features.py first.")

        signals = build_rule_signals(features, args.up_threshold, args.down_threshold)
        signals = signals.sort_values(["decision_time", "symbol"]).reset_index(drop=True)

        nan_total = int(signals[["up_score", "down_score"]].isna().sum().sum())
        if nan_total:
            raise SystemExit(f"rule signal scores contain NaN: {nan_total}")
        dup = int(signals.duplicated(subset=["symbol", "decision_time"]).sum())
        if dup:
            raise SystemExit(f"rule_signals has duplicate (symbol, decision_time) rows: {dup}")

        summary = summarize_rule_signals(signals)
        parquet_path = PROCESSED_DIR / "rule_signals.parquet"
        signals.to_parquet(parquet_path, index=False)
        signals.to_sql("rule_signals", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    up_rate = summary["up_candidates"] / summary["rows"] if summary["rows"] else 0.0
    down_rate = summary["down_candidates"] / summary["rows"] if summary["rows"] else 0.0
    print(
        f"  rows={summary['rows']} symbols={summary['symbols']} "
        f"up_candidates={summary['up_candidates']} ({up_rate:.2%}) "
        f"down_candidates={summary['down_candidates']} ({down_rate:.2%}) "
        f"both={summary['both_candidates']} residual_nan={nan_total}"
    )
    print(f"[{now_kst_text()}] rule_signals -> db + {parquet_path}")
    logger.info(
        "rule signal end rows=%s up=%s down=%s both=%s nan=%s",
        summary["rows"], summary["up_candidates"], summary["down_candidates"],
        summary["both_candidates"], nan_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
