from __future__ import annotations

import argparse
import csv as csv_mod
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

from config import AL_REGULAR_SOURCE, DB_PATH, LABELS_LOG_PATH, PROCESSED_DIR, ensure_directories
from labeling import OUTPUT_COLUMNS, label_distribution, label_symbol
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 10: forward-looking labels from features_relative decisions + bars_1m al_regular prices."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(LABELS_LOG_PATH, name="labels")

    import pandas as pd

    model_source = load_data_config().model_source  # al_regular (price source for labels)
    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    print(f"[{now_kst_text()}] make labels symbols={symbols} price_source={model_source}")
    logger.info("labels start symbols=%s price_source=%s", symbols, model_source)

    conn = sqlite3.connect(args.db)
    try:
        frames = []
        for symbol in symbols:
            decisions = pd.read_sql_query(
                "SELECT decision_time, close FROM features_relative WHERE symbol = ? ORDER BY decision_time",
                conn, params=(symbol,),
            )
            bars = pd.read_sql_query(
                "SELECT bar_start, bar_end, high, low FROM bars_1m WHERE symbol = ? AND source = ?",
                conn, params=(symbol, model_source),
            )
            if decisions.empty:
                print(f"[{now_kst_text()}] SKIP {symbol}: no features_relative decisions")
                logger.warning("skip symbol=%s no decisions", symbol)
                continue
            if bars.empty:
                raise SystemExit(f"{symbol}: no {model_source} bars_1m for label prices.")
            lab = label_symbol(symbol, decisions, bars)
            print(f"[{now_kst_text()}] {symbol}: decisions={len(decisions)} labels={len(lab)}")
            frames.append(lab)

        labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
        if labels.empty:
            raise SystemExit("no labels produced — run build_relative_features.py first.")

        # provenance + integrity
        labels.insert(1, "source", model_source)
        nan_total = int(labels[[c for c in labels.columns if c != "up5_event"]].isna().sum().sum())
        labels = labels.sort_values(["symbol", "decision_time"]).reset_index(drop=True)

        parquet_path = PROCESSED_DIR / "labels.parquet"
        labels.to_parquet(parquet_path, index=False)
        labels.to_sql("labels", conn, if_exists="replace", index=False)

        dist = label_distribution(labels)
    finally:
        conn.close()

    report_path = PROCESSED_DIR / ("label_distribution_" + now_kst_text().replace(":", "").replace(" ", "_").replace("-", "") + ".csv")
    with open(report_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=["label", "eligible", "positive", "positive_rate"])
        writer.writeheader()
        writer.writerows(dist)

    print(f"\nlabels: rows={len(labels)} symbols={labels['symbol'].nunique()} cols={len(labels.columns)} residual_nan={nan_total}")
    print("label distribution (positive rate over eligible rows):")
    for r in dist:
        print(f"  {r['label']:<18} {r['positive_rate']*100:6.2f}%  ({r['positive']}/{r['eligible']})")
    print(f"\nlabels -> db + {parquet_path}")
    print(f"distribution -> {report_path}")
    logger.info("labels end rows=%s nan=%s report=%s", len(labels), nan_total, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
