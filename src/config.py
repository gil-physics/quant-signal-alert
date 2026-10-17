import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
DB_DIR = ROOT_DIR / "db"
LOG_DIR = ROOT_DIR / "logs"

CONFIG_DIR = ROOT_DIR / "config"
APP_CONFIG_PATH = CONFIG_DIR / "config.yaml"
SYMBOLS_CONFIG_PATH = CONFIG_DIR / "symbols_test.yaml"

DB_PATH = DB_DIR / "market_data.sqlite"
LOG_PATH = LOG_DIR / "collector.log"


def connect(db_path=DB_PATH, *, timeout: float = 30.0) -> sqlite3.Connection:
    """Open the shared market DB tuned for the realtime collector/engine split.

    The 32-bit Kiwoom collector (writer) and the 64-bit engine (reader + occasional
    alert writer) hit the SAME sqlite file concurrently. Under the default rollback
    journal ('delete') a reader and the writer are mutually exclusive, so a poll/read
    blocks the collector's bar write -- and vice versa -- until one side exceeds its
    busy timeout and raises ``database is locked``. WAL lets many readers run
    alongside the single writer without blocking; a long busy_timeout makes the rare
    writer-vs-writer overlap wait it out instead of erroring.

    WAL is a persistent property of the file (one connection flips it for every
    process); busy_timeout / synchronous are per-connection and re-applied on open.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def raw_parquet_path(symbol: str) -> Path:
    """Per-symbol raw 1-minute parquet path. The symbol code is never hard-coded."""
    return RAW_DIR / f"kiwoom_{symbol}_1m_raw.parquet"


def feature_parquet_path(symbol: str) -> Path:
    """Per-symbol feature parquet path."""
    return PROCESSED_DIR / f"features_{symbol}_1m.parquet"


REALTIME_LOG_PATH = LOG_DIR / "realtime.log"
COMPARE_LOG_PATH = LOG_DIR / "compare.log"
FEATURE_LOG_PATH = LOG_DIR / "features.log"
IMPORT_LOG_PATH = LOG_DIR / "import.log"
COLLECT_LOG_PATH = LOG_DIR / "collect_1m.log"
VALIDATE_LOG_PATH = LOG_DIR / "validate.log"
RESAMPLE_LOG_PATH = LOG_DIR / "resample.log"
QUALITY_LOG_PATH = LOG_DIR / "quality.log"
FEATURES_LOG_PATH = LOG_DIR / "features_build.log"
MERGE_LOG_PATH = LOG_DIR / "merge.log"
RELATIVE_LOG_PATH = LOG_DIR / "relative.log"
LABELS_LOG_PATH = LOG_DIR / "labels.log"
RULE_SIGNAL_LOG_PATH = LOG_DIR / "rule_signal.log"
BACKTEST_LOG_PATH = LOG_DIR / "backtest.log"
THRESHOLD_SWEEP_LOG_PATH = LOG_DIR / "threshold_sweep.log"

# Raw CSV minute data (1-minute bars) lives next to the project root, not inside it.
MINUTE_DATA_DIR = ROOT_DIR.parent / "minute_data"


def minute_csv_path(symbol: str, *, integrated: bool = False, data_dir: Path | None = None) -> Path:
    """Default CSV path for a symbol.

    integrated=False -> regular-session file '<symbol>_1min.csv'
    integrated=True  -> consolidated/integrated-quote file '<symbol>_AL_1min.csv'
    """
    base = data_dir or MINUTE_DATA_DIR
    suffix = "_AL_1min.csv" if integrated else "_1min.csv"
    return base / f"{symbol}{suffix}"

DEFAULT_TICK = "1"
DEFAULT_MAX_PAGES = 10
DEFAULT_SCREEN_NO = "1000"
DEFAULT_REQUEST_DELAY_SEC = 3.6
DEFAULT_SOURCE = "kiwoom_opt10080"
REALTIME_SOURCE = "kiwoom_realtime"

MINUTE_BARS_DDL = """
CREATE TABLE IF NOT EXISTS minute_bars (
    symbol TEXT NOT NULL,
    ts_kst TEXT NOT NULL,
    open INTEGER NOT NULL,
    high INTEGER NOT NULL,
    low INTEGER NOT NULL,
    close INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, ts_kst)
);
"""

REALTIME_MINUTE_BARS_DDL = """
CREATE TABLE IF NOT EXISTS realtime_minute_bars (
    symbol TEXT NOT NULL,
    ts_kst TEXT NOT NULL,
    open INTEGER NOT NULL,
    high INTEGER NOT NULL,
    low INTEGER NOT NULL,
    close INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, ts_kst)
);
"""

MINUTE_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS minute_features (
    symbol TEXT NOT NULL,
    ts_kst TEXT NOT NULL,
    ret_1m REAL,
    ret_3m REAL,
    ret_5m REAL,
    ret_10m REAL,
    ret_20m REAL,
    ret_60m REAL,
    range_1m REAL,
    body REAL,
    upper_shadow REAL,
    lower_shadow REAL,
    vol_5m REAL,
    vol_20m REAL,
    vol_60m REAL,
    volume_1m INTEGER,
    volume_change_1m REAL,
    volume_ma_5m REAL,
    volume_ma_20m REAL,
    volume_z_20m REAL,
    volume_z_60m REAL,
    minutes_from_open INTEGER,
    minutes_to_close INTEGER,
    session_bucket TEXT,
    return_from_open REAL,
    distance_from_day_high REAL,
    distance_from_day_low REAL,
    vwap REAL,
    vwap_distance REAL,
    created_at TEXT,
    PRIMARY KEY (symbol, ts_kst)
);
"""


# --- Step 2: unified, symbol-keyed bar tables ---
# The integrated-quote (통합시세) feed is the basis. Each symbol carries up to
# three source-tagged series that overlap in (symbol, bar_start):
#   al_raw         full AL file 08:00-19:59 (raw storage)
#   al_regular     AL filtered to the regular session (model candidate)
#   single_regular single-exchange regular file (comparison only)
# Because these series share bar_start values, `source` is part of the primary
# key so they coexist without collisions. The required completion query
#   SELECT symbol, COUNT(*) FROM bars_1m GROUP BY symbol
# still works unchanged.
def _bars_ddl(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    symbol TEXT NOT NULL,
    bar_start TEXT NOT NULL,
    bar_end TEXT NOT NULL,
    open INTEGER NOT NULL,
    high INTEGER NOT NULL,
    low INTEGER NOT NULL,
    close INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, bar_start, source)
);
"""


BARS_1M_DDL = _bars_ddl("bars_1m")
BARS_3M_DDL = _bars_ddl("bars_3m")
BARS_5M_DDL = _bars_ddl("bars_5m")
BARS_TABLES = {"1m": "bars_1m", "3m": "bars_3m", "5m": "bars_5m"}

# Source tags for CSV imports.
AL_RAW_SOURCE = "al_raw"             # full integrated-quote file, 08:00-19:59
AL_REGULAR_SOURCE = "al_regular"     # integrated quote filtered to the regular session
SINGLE_REGULAR_SOURCE = "single_regular"  # single-exchange regular file (comparison)
RESAMPLED_SOURCE_PREFIX = "resampled_from_"  # 3m/5m bars derived from a 1m source


def resampled_source(base_source: str) -> str:
    """Source tag for resampled bars, encoding the 1m base source.

    e.g. resampled_source("al_regular") -> "resampled_from_al_regular". This keeps
    the (symbol, bar_start, source) primary key collision-free if other base
    sources (e.g. al_raw) are ever resampled too.
    """
    return f"{RESAMPLED_SOURCE_PREFIX}{base_source}"


def ensure_directories() -> None:
    for path in [
        CONFIG_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        LABELS_DIR,
        DB_DIR,
        LOG_DIR,
        ROOT_DIR / "src",
        ROOT_DIR / "scripts",
    ]:
        path.mkdir(parents=True, exist_ok=True)
