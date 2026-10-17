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

import features_1m
import features_3m
import features_5m
from bars_reader import load_bars
from config import DB_PATH, FEATURES_LOG_PATH, PROCESSED_DIR, ensure_directories, resampled_source
from settings import load_active_symbols, load_data_config, normalize_symbol_code
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 7: build per-timeframe features (features_1m/3m/5m) for active symbols."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(FEATURES_LOG_PATH, name="features")

    data_cfg = load_data_config()
    model_source = data_cfg.model_source
    resampled = resampled_source(model_source)

    # (label, table, source, module, output_table)
    timeframes = [
        ("1m", "bars_1m", model_source, features_1m, "features_1m"),
        ("3m", "bars_3m", resampled, features_3m, "features_3m"),
        ("5m", "bars_5m", resampled, features_5m, "features_5m"),
    ]

    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()

    print(f"[{now_kst_text()}] build features symbols={symbols} model_source={model_source} resampled={resampled}")
    logger.info("features start symbols=%s model_source=%s resampled=%s", symbols, model_source, resampled)

    conn = sqlite3.connect(args.db)
    try:
        for label, table, source, module, out_table in timeframes:
            frames = []
            for symbol in symbols:
                df_bars = load_bars(symbol, source=source, conn=conn, table=table)
                if df_bars.empty:
                    print(f"[{now_kst_text()}] SKIP {symbol} {label}: no {source} bars")
                    logger.warning("skip symbol=%s timeframe=%s no bars source=%s", symbol, label, source)
                    continue
                frames.append(module.build(df_bars, source))

            id_cols = module.ID_COLUMNS
            feature_cols = module.FEATURE_COLUMNS
            if not frames:
                print(f"[{now_kst_text()}] {out_table}: no input — skipped")
                continue

            import pandas as pd

            combined = pd.concat(frames, ignore_index=True)
            rows_before = len(combined)

            # NaN report (instr #3) then drop warmup rolling NaNs (instr #4)
            nan_counts = combined[feature_cols].isna().sum()
            combined = combined.dropna(subset=feature_cols).reset_index(drop=True)
            rows_after = len(combined)
            remaining_nan = int(combined[feature_cols].isna().sum().sum())

            combined = combined.sort_values(id_cols[:2]).reset_index(drop=True)
            parquet_path = PROCESSED_DIR / f"{out_table}.parquet"
            combined.to_parquet(parquet_path, index=False)
            combined.to_sql(out_table, conn, if_exists="replace", index=False)

            worst = ", ".join(f"{c}={int(n)}" for c, n in nan_counts.sort_values(ascending=False).head(3).items() if n)
            print(
                f"[{now_kst_text()}] {out_table}: in={rows_before} out={rows_after} "
                f"dropped(warmup)={rows_before - rows_after} remaining_nan={remaining_nan} "
                f"symbols={combined['symbol'].nunique()} cols={len(combined.columns)}"
                + (f" | top NaN: {worst}" if worst else "")
            )
            logger.info(
                "%s built in=%s out=%s dropped=%s remaining_nan=%s parquet=%s",
                out_table, rows_before, rows_after, rows_before - rows_after, remaining_nan, parquet_path,
            )
    finally:
        conn.close()

    print(f"[{now_kst_text()}] features_1m / features_3m / features_5m written to db + data/processed/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
