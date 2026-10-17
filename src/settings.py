"""Configuration loader for Step 0 (project baseline) and Step 1 (symbol list).

This module is the single entry point for reading ``config/config.yaml`` and
``config/symbols_test.yaml``. Source code must never hard-code a symbol code;
instead call :func:`load_symbols` / :func:`default_symbol`.

YAML parsing prefers PyYAML. If PyYAML is not importable (e.g. a stripped-down
runtime), a minimal built-in parser handles the small, well-defined subset of
YAML used by these config files so the collectors keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import APP_CONFIG_PATH, SYMBOLS_CONFIG_PATH

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Typed views over the config files
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Symbol:
    code: str
    name: str
    sector: str
    active: bool


@dataclass(frozen=True)
class SessionConfig:
    train_start: str
    train_end: str
    raw_start: str
    raw_end: str


@dataclass(frozen=True)
class SignalConfig:
    max_signals_per_day: int
    cooldown_minutes_per_symbol: int
    min_minutes_after_open: int
    no_new_signal_minutes_before_close: int


@dataclass(frozen=True)
class DataConfig:
    """Step 2/3 data classification: which source feeds models, raw, comparison."""

    model_source: str
    raw_source: str
    comparison_source: str


@dataclass(frozen=True)
class AppConfig:
    session: SessionConfig
    signals: SignalConfig


# --------------------------------------------------------------------------- #
# YAML loading (PyYAML preferred, built-in fallback otherwise)
# --------------------------------------------------------------------------- #
def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML mapping from ``path`` and return it as a dict."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"config file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = _parse_simple_yaml(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at the top level of {file_path}")
    return data


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", ""):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _strip_comment(line: str) -> str:
    in_single = in_double = False
    out: List[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out)


def _parse_node(lines: List[tuple], i: int, indent: int):
    """Parse a block at column ``indent`` starting at line index ``i``.

    Handles nested mappings and sequences of mappings/scalars with consistent
    space indentation. Returns ``(value, next_index)``.
    """
    if lines[i][1].startswith("- "):
        result: List[Any] = []
        while (
            i < len(lines)
            and lines[i][0] == indent
            and lines[i][1].startswith("- ")
        ):
            inline = lines[i][1][2:]
            item_indent = indent + 2
            sub: List[tuple] = [(item_indent, inline)]
            i += 1
            while i < len(lines) and lines[i][0] >= item_indent:
                sub.append((lines[i][0], lines[i][1]))
                i += 1
            value, _ = _parse_node(sub, 0, item_indent)
            result.append(value)
        return result, i

    mapping: Dict[str, Any] = {}
    while (
        i < len(lines)
        and lines[i][0] == indent
        and not lines[i][1].startswith("- ")
    ):
        key, sep, rest = lines[i][1].partition(":")
        if not sep:
            raise ValueError(f"invalid YAML line (expected 'key: value'): {lines[i][1]!r}")
        key = key.strip()
        rest = rest.strip()
        i += 1
        if rest == "":
            if i < len(lines) and lines[i][0] > indent:
                value, i = _parse_node(lines, i, lines[i][0])
                mapping[key] = value
            else:
                mapping[key] = None
        else:
            mapping[key] = _parse_scalar(rest)
    return mapping, i


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Minimal YAML parser for the config subset (fallback when PyYAML absent)."""
    lines: List[tuple] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if stripped.strip() == "":
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}
    value, _ = _parse_node(lines, 0, lines[0][0])
    if not isinstance(value, dict):
        raise ValueError("top-level YAML must be a mapping")
    return value


# --------------------------------------------------------------------------- #
# Step 1: symbol list
# --------------------------------------------------------------------------- #
def load_symbols(path: Optional[PathLike] = None, *, active_only: bool = False) -> List[Symbol]:
    """Load the fixed symbol list. Pass ``active_only=True`` for active symbols."""
    data = load_yaml(path or SYMBOLS_CONFIG_PATH)
    raw_symbols = data.get("symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ValueError("symbols config must contain a non-empty 'symbols' list")

    symbols: List[Symbol] = []
    seen_codes = set()
    for index, entry in enumerate(raw_symbols):
        if not isinstance(entry, dict):
            raise ValueError(f"symbols[{index}] must be a mapping")
        missing = [k for k in ("code", "name", "sector", "active") if k not in entry]
        if missing:
            raise ValueError(f"symbols[{index}] missing keys: {missing}")
        if not isinstance(entry["code"], str):
            raise ValueError(
                f"symbols[{index}] code must be a quoted string (got {entry['code']!r}); "
                "wrap the code in quotes in YAML to preserve leading zeros"
            )
        code = entry["code"].strip()
        if not code:
            raise ValueError(f"symbols[{index}] has an empty code")
        if code in seen_codes:
            raise ValueError(f"duplicate symbol code: {code}")
        seen_codes.add(code)
        symbols.append(
            Symbol(
                code=code,
                name=str(entry["name"]),
                sector=str(entry["sector"]),
                active=bool(entry["active"]),
            )
        )

    if active_only:
        symbols = [s for s in symbols if s.active]
    return symbols


def active_symbols(path: Optional[PathLike] = None) -> List[Symbol]:
    """Return only the symbols flagged ``active: true``, preserving file order."""
    return load_symbols(path, active_only=True)


def active_symbol_codes(path: Optional[PathLike] = None) -> List[str]:
    """Convenience: the codes of all active symbols, in file order."""
    return [s.code for s in active_symbols(path)]


def default_symbol(path: Optional[PathLike] = None) -> str:
    """The first active symbol code; used as the default target for single-symbol runs."""
    actives = active_symbols(path)
    if not actives:
        raise ValueError(
            "no active symbols found; set active: true for at least one entry in the symbols config"
        )
    return actives[0].code


def load_active_symbols(path: Optional[PathLike] = None) -> List[str]:
    """Active symbol codes, in file order — the iteration target for multi-symbol runs."""
    return active_symbol_codes(path)


def realtime_default_symbol(config_path: Optional[PathLike] = None,
                            symbols_path: Optional[PathLike] = None) -> str:
    """Default single-stock target for the realtime trend-signal feed.

    Reads ``realtime.default_symbol`` from config.yaml (kept in config, never
    hard-coded in source); falls back to the first active symbol if unset.
    """
    data = load_yaml(config_path or APP_CONFIG_PATH)
    section = data.get("realtime") or {}
    code = section.get("default_symbol")
    if code:
        return normalize_symbol_code(str(code))
    return default_symbol(symbols_path)


def leader_symbols(config_path: Optional[PathLike] = None) -> List[tuple]:
    """Universe leaders for the Step 9 cross-symbol lead-lag features (risk model
    v2.1x). Reads ``relative.leaders`` (a list of ``{label, code}`` mappings) from
    config.yaml (codes kept in config, never hard-coded in source). Returns a list of
    ``(label, normalized_code)`` tuples; empty when unset so no lead-lag columns are
    produced.
    """
    data = load_yaml(config_path or APP_CONFIG_PATH)
    section = data.get("relative") or {}
    raw = section.get("leaders") or []
    out: List[tuple] = []
    for entry in raw:
        if isinstance(entry, dict) and entry.get("code") and entry.get("label"):
            out.append((str(entry["label"]).strip(), normalize_symbol_code(str(entry["code"]))))
    return out


def normalize_symbol_code(code: str) -> str:
    """Korean stock codes are 6 digits. Zero-pad a numeric code shorter than 6
    digits that lost a leading zero (a common gotcha when codes pass through ints
    or spreadsheets). Non-numeric codes are returned stripped, unchanged."""
    text = str(code).strip()
    if text.isdigit() and len(text) < 6:
        return text.zfill(6)
    return text


# --------------------------------------------------------------------------- #
# Step 0: project baseline (session + signals)
# --------------------------------------------------------------------------- #
def load_app_config(path: Optional[PathLike] = None) -> AppConfig:
    data = load_yaml(path or APP_CONFIG_PATH)

    session_raw = data.get("session")
    if not isinstance(session_raw, dict):
        raise ValueError("config must contain a 'session' mapping")
    session = SessionConfig(
        train_start=str(session_raw["train_start"]),
        train_end=str(session_raw["train_end"]),
        raw_start=str(session_raw["raw_start"]),
        raw_end=str(session_raw["raw_end"]),
    )

    signals_raw = data.get("signals")
    if not isinstance(signals_raw, dict):
        raise ValueError("config must contain a 'signals' mapping")
    signals = SignalConfig(
        max_signals_per_day=int(signals_raw["max_signals_per_day"]),
        cooldown_minutes_per_symbol=int(signals_raw["cooldown_minutes_per_symbol"]),
        min_minutes_after_open=int(signals_raw["min_minutes_after_open"]),
        no_new_signal_minutes_before_close=int(signals_raw["no_new_signal_minutes_before_close"]),
    )
    return AppConfig(session=session, signals=signals)


def load_session_config(path: Optional[PathLike] = None) -> SessionConfig:
    return load_app_config(path).session


def load_signal_config(path: Optional[PathLike] = None) -> SignalConfig:
    return load_app_config(path).signals


def load_data_config(path: Optional[PathLike] = None) -> DataConfig:
    """Step 2/3 data classification from config.yaml's `data` section.

    Falls back to the established source tags if the section is absent.
    """
    data = load_yaml(path or APP_CONFIG_PATH)
    section = data.get("data") or {}
    return DataConfig(
        model_source=str(section.get("model_source", "al_regular")),
        raw_source=str(section.get("raw_source", "al_raw")),
        comparison_source=str(section.get("comparison_source", "single_regular")),
    )
