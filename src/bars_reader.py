"""Step 4: canonical reader for downstream steps.

Source policy (config.yaml `data` section):
  al_raw          Integrated quote (통합시세), full 08:00-19:59. RAW storage only.
                  Not for modeling directly (contains pre/post-market).
  al_regular      Integrated quote filtered to the regular session 09:00-15:30.
                  ** This is the model input. ** All downstream steps read this by
                  default via load_bars()/load_model_bars().
  single_regular  Single-exchange regular file. COMPARISON only (sanity-check the
                  integrated feed); never the model input.

`config.data.model_source` is fixed to `al_regular`, and load_bars() defaults its
`source` to that value, so later steps read al_regular unless they explicitly ask
for another source.
"""

from __future__ import annotations

import sqlite3
from typing import Optional, Tuple

from config import DB_PATH
from settings import load_data_config

_COLUMNS = ("symbol", "bar_start", "bar_end", "open", "high", "low", "close", "volume", "source")


def _build_query(table: str, date_range: Optional[Tuple[Optional[str], Optional[str]]]):
    sql = (
        f"SELECT {', '.join(_COLUMNS)} FROM {table} WHERE symbol = ? AND source = ?"
    )
    params = []
    if date_range is not None:
        lo, hi = date_range
        if lo:
            sql += " AND substr(bar_start, 1, 10) >= ?"
            params.append(lo)
        if hi:
            sql += " AND substr(bar_start, 1, 10) <= ?"
            params.append(hi)
    sql += " ORDER BY bar_start"
    return sql, params


def load_bars(
    symbol: str,
    *,
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
    db_path=DB_PATH,
    table: str = "bars_1m",
):
    """Load bars for ``symbol`` as a pandas DataFrame.

    ``source`` defaults to config.data.model_source (al_regular) so downstream
    steps read the model series unless they explicitly pass another source.
    """
    import pandas as pd  # lazy: keeps this module importable in the 32-bit runtime

    if source is None:
        source = load_data_config().model_source

    date_range = (start_date, end_date) if (start_date or end_date) else None
    sql, extra = _build_query(table, date_range)
    params = [symbol, source, *extra]

    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        if own_conn:
            conn.close()


def load_model_bars(symbol: str, **kwargs):
    """Explicit alias: load the model source (al_regular) for ``symbol``."""
    kwargs.pop("source", None)
    return load_bars(symbol, source=load_data_config().model_source, **kwargs)
