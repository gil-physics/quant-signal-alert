"""Tests for causal online top1 selection (src/online_select.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from online_select import select_online_top1  # noqa: E402


def _cand(date, t, margin):
    return {"symbol": "AAA", "date": date, "decision_time": f"{date} {t}:00",
            "score_margin": margin, "up_score": margin, "close": 100}


class OnlineTop1Tests(unittest.TestCase):
    def test_picks_earliest_not_highest_score(self):
        # later signal has a higher score, but online must commit to the earliest.
        cands = pd.DataFrame([
            _cand("2026-01-05", "10:00", 3),
            _cand("2026-01-05", "11:00", 9),   # higher score, but later -> not picked
            _cand("2026-01-06", "09:30", 5),
        ])
        out = select_online_top1(cands)
        self.assertEqual(len(out), 2)                       # one per day
        first = out[out["date"] == "2026-01-05"].iloc[0]
        self.assertEqual(first["decision_time"], "2026-01-05 10:00:00")
        self.assertEqual(first["score_margin"], 3)          # earliest, not the 9
        self.assertTrue((out["daily_rank"] == 1).all())

    def test_empty(self):
        self.assertEqual(len(select_online_top1(pd.DataFrame())), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
