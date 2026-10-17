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

from config import DB_PATH, PROCESSED_DIR, RELATIVE_LOG_PATH, ensure_directories
from relative_features import build_relative_features, partial_decision_times
from settings import (leader_symbols, load_active_symbols, load_data_config,
                      normalize_symbol_code)
from utils import now_kst_text, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 9: cross-symbol relative-strength features from features_merged."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated codes; default = active symbols.")
    parser.add_argument("--drop-partial", action="store_true",
                        help="Drop decision_times that lack the full active universe (default: keep + WARN report).")
    parser.add_argument("--db", default=str(DB_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    logger = setup_logger(RELATIVE_LOG_PATH, name="relative")

    import pandas as pd

    model_source = load_data_config().model_source
    if args.symbols:
        symbols = [normalize_symbol_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_active_symbols()
    active_total = len(symbols)
    # only leaders that are actually in the active universe contribute lead-lag columns
    leaders = [(lab, code) for lab, code in leader_symbols() if code in symbols]

    print(f"[{now_kst_text()}] relative features symbols={symbols} (active_total={active_total}) leaders={leaders}")
    logger.info("relative start symbols=%s active_total=%s leaders=%s", symbols, active_total, leaders)

    conn = sqlite3.connect(args.db)
    try:
        # Filter on source too so the universe is exactly one row per (symbol, decision_time).
        merged = pd.read_sql_query(
            "SELECT * FROM features_merged WHERE source = ?", conn, params=(model_source,)
        )
        merged = merged[merged["symbol"].isin(symbols)].reset_index(drop=True)
        if merged.empty:
            raise SystemExit("features_merged is empty for the active symbols — run merge_features.py first.")
        dup = int(merged.duplicated(subset=["symbol", "decision_time"]).sum())
        if dup:
            raise SystemExit(f"features_merged has {dup} duplicate (symbol, decision_time) rows — cannot build a clean cross-section.")

        out = build_relative_features(merged, leaders=leaders)

        # partial decision_times (Step 9 #8): fewer symbols than the full active universe
        partial = partial_decision_times(out, active_total)
        print(f"  decision_times total={out['decision_time'].nunique()} partial(<{active_total} symbols)={len(partial)}")
        if len(partial):
            logger.warning("partial decision_times count=%s (e.g. %s)", len(partial), list(partial.index[:3]))
            print(f"  WARN: {len(partial)} decision_times have < {active_total} active symbols "
                  f"(e.g. {list(partial.index[:3])})")
        if args.drop_partial and len(partial):
            before = len(out)
            out = out[~out["decision_time"].isin(partial.index)].reset_index(drop=True)
            print(f"  --drop-partial: kept {len(out)} of {before} rows")

        # provenance + sanity
        out.insert(1, "source", model_source)
        rank_min, rank_max = int(out["rank_ret_20m_1m"].min()), int(out["rank_ret_20m_1m"].max())
        nan_total = int(out.isna().sum().sum())
        bad_rank = int(((out["rank_ret_20m_1m"] < 1) | (out["rank_ret_20m_1m"] > out["active_symbol_count"])).sum())
        print(f"  rows={len(out)} symbols={out['symbol'].nunique()} "
              f"rank_ret_20m_1m∈[{rank_min},{rank_max}] out-of-range={bad_rank} residual_nan={nan_total}")

        out = out.sort_values(["decision_time", "rank_ret_20m_1m"]).reset_index(drop=True)
        parquet_path = PROCESSED_DIR / "features_relative.parquet"
        out.to_parquet(parquet_path, index=False)
        out.to_sql("features_relative", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    print(f"[{now_kst_text()}] features_relative: rows={len(out)} cols={len(out.columns)} -> db + {parquet_path}")
    logger.info("relative end rows=%s cols=%s out_of_range_rank=%s nan=%s",
                len(out), len(out.columns), bad_rank, nan_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
