"""Live paper-trading ledger for the online top1 rule (v2_daily).

Each trading day yields at most ONE trade: the first qualifying breakout of the
day (causal), entered 3 minutes later, managed with a +3% / -1.5% first-touch
bracket held to EOD. Because the bracket is intraday, every trade closes the same
day, so the ledger is a day-by-day record of realised paper P&L (net of cost).

Deterministic from the DB, so re-running reproduces/extends the ledger as new bars
arrive (run after each session close). Run (cwd = quant):
    ..\\Python311-64\\python.exe scripts\\paper_track.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
for p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

import online_top1_eval as ote
from bracket_eval import prepare_entry, simulate_bracket
from online_select import select_online_top1

TARGET, STOP, DELAY, COST = ote.TARGET, ote.STOP, ote.DELAY, ote.DEFAULT_COST
LEDGER_PATH = ROOT_DIR / "data" / "processed" / "paper_ledger.csv"
OUTCOME_LABEL = {"target": "TARGET +3%", "stop": "STOP -1.5%", "time": "EOD close"}


def _build_ledger(online, by_day):
    rows = []
    cols = online[["symbol", "date", "decision_time", "_decision_dt", "close"]]
    for sym, date, dtime, dec_dt, close_t in cols.itertuples(index=False, name=None):
        arr = by_day.get((sym, date))
        if arr is None:
            continue
        rec = prepare_entry(arr, dec_dt, float(close_t), delay_bars=DELAY)
        if rec is None:
            continue
        ep, h, l, c, off = rec
        br = simulate_bracket(ep, h, l, c, off, target_pct=TARGET, stop_pct=STOP, time_exit_min=float("inf"))
        if br is None:
            continue
        gross = br["realized_return"]
        rows.append({
            "trade_date": date, "symbol": sym, "signal_time": dtime,
            "entry_delay_min": DELAY, "entry_price": round(ep, 2),
            "target_price": round(ep * (1 + TARGET), 2), "stop_price": round(ep * (1 - STOP), 2),
            "exit_reason": OUTCOME_LABEL[br["outcome"]], "hold_min": round(br["exit_offset_min"], 0),
            "mfe": round(br["mfe"], 5), "mae": round(br["mae"], 5),
            "gross_return": round(gross, 5), "cost": COST, "net_return": round(gross - COST, 5),
        })
    led = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    led["cum_net"] = led["net_return"].cumsum().round(5)
    return led


def main():
    candidates, by_day, symbols, model_source = ote._candidates()
    online = select_online_top1(candidates)
    led = _build_ledger(online, by_day)
    led.to_csv(LEDGER_PATH, index=False, encoding="utf-8-sig")

    n = len(led)
    if n == 0:
        print("no paper trades produced.")
        return
    wins = led[led["net_return"] > 0]
    print(f"paper ledger ({model_source}, {len(symbols)} symbols, entry_delay_{DELAY}m, "
          f"+{TARGET*100:.0f}%/-{STOP*100:.1f}%, cost {COST*100:.1f}% round-trip)")
    print(f"  trades={n}  period {led['trade_date'].iloc[0]} .. {led['trade_date'].iloc[-1]}")
    print(f"  win_rate(net)={len(wins)/n*100:.1f}%  TARGET={ (led['exit_reason']=='TARGET +3%').mean()*100:.1f}%  "
          f"STOP={ (led['exit_reason']=='STOP -1.5%').mean()*100:.1f}%")
    print(f"  gross_mean={led['gross_return'].mean()*100:.3f}%  net_mean={led['net_return'].mean()*100:.3f}%  "
          f"cumulative_net={led['cum_net'].iloc[-1]*100:.2f}% (sum of per-trade net, 1 unit/trade)")

    print("\n  last 10 paper trades:")
    show = led.tail(10)[["trade_date", "symbol", "signal_time", "entry_price", "exit_reason",
                         "net_return", "cum_net"]]
    for _, r in show.iterrows():
        print(f"    {r['trade_date']}  {r['symbol']}  sig {r['signal_time'][11:16]}  "
              f"entry {r['entry_price']:>9.1f}  {r['exit_reason']:<11}  "
              f"net {r['net_return']*100:>6.2f}%  cum {r['cum_net']*100:>7.2f}%")

    latest = led.iloc[-1]
    print(f"\n  latest session ({latest['trade_date']}): {latest['symbol']} signal "
          f"{latest['signal_time'][11:16]} -> {latest['exit_reason']} (net {latest['net_return']*100:+.2f}%)")
    print(f"\nledger -> {LEDGER_PATH}")


if __name__ == "__main__":
    main()
