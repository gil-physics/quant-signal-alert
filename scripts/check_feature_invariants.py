"""Lock the Step 7 feature invariants with explicit SQL checks.

  features_1m.source = config.data.model_source            (al_regular)
  features_3m.source = resampled_source(model_source)       (resampled_from_al_regular)
  features_5m.source = resampled_source(model_source)
  features_1m.decision_time == features_1m.bar_end_1m       (completely equal)

Exits non-zero if any invariant is violated (CI-friendly).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DB_PATH, resampled_source
from settings import load_data_config


def main() -> int:
    model_source = load_data_config().model_source
    resampled = resampled_source(model_source)
    expected_source = {
        "features_1m": model_source,
        "features_3m": resampled,
        "features_5m": resampled,
    }

    conn = sqlite3.connect(DB_PATH)
    checks = []  # (name, ok, detail)
    try:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        for table, want in expected_source.items():
            if table not in existing:
                checks.append((f"{table}.source = {want!r}", False, "table missing — run build_features.py"))
                continue
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            sources = sorted(r[0] for r in conn.execute(f"SELECT DISTINCT source FROM {table}"))
            ok = total > 0 and sources == [want]
            checks.append((f"{table}.source = {want!r}", ok, f"rows={total} distinct_source={sources}"))

        # decision_time must equal bar_end_1m on every features_1m row
        if "features_1m" in existing:
            bad = conn.execute(
                "SELECT COUNT(*) FROM features_1m WHERE decision_time != bar_end_1m"
            ).fetchone()[0]
            checks.append(("features_1m.decision_time == bar_end_1m", bad == 0, f"mismatches={bad}"))
        else:
            checks.append(("features_1m.decision_time == bar_end_1m", False, "table missing"))
    finally:
        conn.close()

    print("Step 7 feature invariants:")
    all_ok = True
    for name, ok, detail in checks:
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print("RESULT:", "ALL PASS" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
