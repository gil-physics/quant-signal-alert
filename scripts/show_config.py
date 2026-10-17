from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Korean symbol names need UTF-8 even on a cp949 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from settings import default_symbol, load_app_config, load_symbols


def main() -> int:
    config = load_app_config()
    symbols = load_symbols()
    actives = [s for s in symbols if s.active]

    print("=== Step 0: project baseline ===")
    print("session:")
    print(f"  train_start = {config.session.train_start}")
    print(f"  train_end   = {config.session.train_end}")
    print(f"  raw_start   = {config.session.raw_start}")
    print(f"  raw_end     = {config.session.raw_end}")
    print("signals:")
    print(f"  max_signals_per_day                 = {config.signals.max_signals_per_day}")
    print(f"  cooldown_minutes_per_symbol         = {config.signals.cooldown_minutes_per_symbol}")
    print(f"  min_minutes_after_open              = {config.signals.min_minutes_after_open}")
    print(f"  no_new_signal_minutes_before_close  = {config.signals.no_new_signal_minutes_before_close}")

    print("\n=== Step 1: fixed symbol list ===")
    for s in symbols:
        flag = "ON " if s.active else "off"
        print(f"  [{flag}] {s.code}  {s.name}  ({s.sector})")
    print(f"\nactive count = {len(actives)}")
    print(f"default symbol (first active) = {default_symbol()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
