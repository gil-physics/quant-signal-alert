"""Export the minimal sqlite the TCN trainer needs, for GPU-server training.

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. This script only
COPIES rows out of the local market DB into a new, much smaller sqlite file that can
be shipped to a remote GPU box; nothing in the source DB is modified.

Exactly the six tables scripts/train_tcn_risk_model.py reads are exported, filtered
to the configured model source + active symbols (everything else -- al_raw,
single_regular, resampled bars, realtime tables, prediction mirrors -- is dropped):
  directional_alerts, features_merged, features_relative, labels, bars_1m, features_1m

On the server, point the trainer at it:
  python scripts/train_tcn_risk_model.py --tcn-version v4_tcn_60m --device cuda:0 \\
      --db db/market_data_training.sqlite

Run:
  ..\\Python311-64\\python.exe scripts\\export_tcn_training_db.py
"""

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

from config import DB_DIR, DB_PATH
from settings import load_active_symbols, load_data_config
from utils import now_kst_text

EXPORT_TABLES = ("directional_alerts", "features_merged", "features_relative",
                 "labels", "bars_1m", "features_1m")


def parse_args():
    p = argparse.ArgumentParser(description="Export the minimal TCN-training sqlite.")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--out", default=str(DB_DIR / "market_data_training.sqlite"))
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file.")
    p.add_argument("--sources", default="",
                   help="comma-separated bars_1m source tags to export (default: the "
                        "configured model source only). The touch study needs "
                        "'al_regular,al_raw' -- its carry-variant MA warmup reads "
                        "premarket al_raw bars.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} already exists - pass --force to overwrite.")
        out.unlink()
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    ph = ",".join("?" for _ in symbols)
    # bars_1m may carry extra sources (touch study: al_raw premarket warmup); the
    # per-bar feature/label tables always ship the model source only.
    bar_sources = ([s.strip() for s in args.sources.split(",") if s.strip()]
                   or [model_source])

    print(f"[{now_kst_text()}] export training DB source={model_source} "
          f"bars_sources={bar_sources} symbols={len(symbols)} tables={len(EXPORT_TABLES)}")
    conn = sqlite3.connect(str(out))
    try:
        conn.execute("ATTACH DATABASE ? AS src", (str(Path(args.db)),))
        for table in EXPORT_TABLES:
            exists = conn.execute(
                "SELECT count(*) FROM src.sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()[0]
            if not exists:
                raise SystemExit(f"source DB has no table {table!r} - wrong --db?")
            sources = bar_sources if table == "bars_1m" else [model_source]
            sph = ",".join("?" for _ in sources)
            conn.execute(
                f"CREATE TABLE {table} AS SELECT * FROM src.{table} "
                f"WHERE source IN ({sph}) AND symbol IN ({ph})",
                (*sources, *symbols))
            n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"  {table:>20}: {n:,} rows")
        conn.commit()
        conn.execute("DETACH DATABASE src")
    finally:
        conn.close()

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nexport -> {out} ({size_mb:,.0f} MB)")
    print("server usage: python scripts/train_tcn_risk_model.py --tcn-version v4_tcn_60m "
          "--device cuda:0 --db db/market_data_training.sqlite ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
