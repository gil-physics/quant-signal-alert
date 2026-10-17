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

from config import DB_PATH, MERGE_LOG_PATH, PROCESSED_DIR, ensure_directories, resampled_source
from feature_merge import leakage_violations, merge_asof_features
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8: leakage-safe as-of merge of features_1m/3m/5m into features_merged."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--drop-incomplete", action="store_true",
                        help="Drop decision rows that still lack any 3m/5m feature (default: keep + report).")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def _load(conn, table, source, symbols):
    import pandas as pd

    df = pd.read_sql_query(f"SELECT * FROM {table} WHERE source = ?", conn, params=(source,))
    return df[df["symbol"].isin(symbols)].reset_index(drop=True)


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(MERGE_LOG_PATH, name="merge")

    import pandas as pd

    data_cfg = load_data_config()
    model_source = data_cfg.model_source
    resampled = resampled_source(model_source)

    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    print(f"[{now_kst_text()}] merge features symbols={symbols} 1m={model_source} 3m/5m={resampled}")
    logger.info("merge start symbols=%s model_source=%s resampled=%s", symbols, model_source, resampled)

    conn = sqlite3.connect(args.db)
    try:
        f1 = _load(conn, "features_1m", model_source, symbols)
        f3 = _load(conn, "features_3m", resampled, symbols)
        f5 = _load(conn, "features_5m", resampled, symbols)
        print(f"  loaded features_1m={len(f1)} features_3m={len(f3)} features_5m={len(f5)}")
        if f1.empty:
            raise SystemExit("features_1m is empty — run build_features.py first.")

        merged = merge_asof_features(f1, f3, f5)

        # Leakage check (instruction #8 / completion criterion)
        v3, v5 = leakage_violations(merged)
        print(f"  leakage violations: bar_end_3m>{'decision_time'}={v3}  bar_end_5m>decision_time={v5}")
        if v3 or v5:
            for col in ("bar_end_3m", "bar_end_5m"):
                bad = merged[(merged[col].notna()) & (pd.to_datetime(merged[col]) > pd.to_datetime(merged["decision_time"]))]
                if len(bad):
                    print(bad[["symbol", "decision_time", col]].head().to_string())
            raise SystemExit(f"LEAKAGE DETECTED: 3m={v3} 5m={v5} rows violate bar_end <= decision_time")
        print("  [OK] no leakage: every attached bar_end_3m/5m <= decision_time")

        # NaN report (instruction: 병합 후 잔여 NaN 확인). Early-day decision times have no
        # complete 3m/5m feature yet -> NaN attachments; this is availability, not leakage.
        tf3_cols = [c for c in merged.columns if c.endswith("_3m")]
        tf5_cols = [c for c in merged.columns if c.endswith("_5m")]
        rows = len(merged)
        miss3 = int(merged[tf3_cols[0]].isna().sum()) if tf3_cols else rows
        miss5 = int(merged[tf5_cols[0]].isna().sum()) if tf5_cols else rows
        full = merged.dropna()
        print(f"  rows={rows} | no 3m yet={miss3} | no 5m yet={miss5} | fully-populated={len(full)}")
        logger.info("merged rows=%s no3m=%s no5m=%s full=%s leakage3=%s leakage5=%s",
                    rows, miss3, miss5, len(full), v3, v5)

        if args.drop_incomplete:
            before = len(merged)
            # drop on the attachment indicator (bar_end), not every _3m/_5m column: a row
            # whose 3m/5m bar is validly attached but has a NaN rolling feature is kept.
            indicators = [c for c in ("bar_end_3m", "bar_end_5m") if c in merged.columns]
            merged = merged.dropna(subset=indicators).reset_index(drop=True)
            print(f"  --drop-incomplete: kept {len(merged)} of {before} (dropped rows with no 3m/5m bar attached)")

        merged = merged.sort_values(["symbol", "decision_time"]).reset_index(drop=True)
        parquet_path = PROCESSED_DIR / "features_merged.parquet"
        merged.to_parquet(parquet_path, index=False)
        merged.to_sql("features_merged", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print(f"[{now_kst_text()}] features_merged: rows={len(merged)} cols={len(merged.columns)} "
          f"symbols={merged['symbol'].nunique()} -> db + {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
