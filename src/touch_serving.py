"""Step 22: touch-model serving adapter + shared serving-frame glue.

Live decision-support ONLY -- this module scores breakout events and maps scores
to frozen OOF grades; it NEVER places, cancels, or sends any order (no order API
is imported or called). Torch stays lazy (tcn_model imports it inside functions),
so importing this module is dependency-light.

Shared-vs-separated contract (user directive #4):
  SHARED     bar ingest + causal feature generation (realtime_features), the
             60-bar window assembly (tcn_data.build_bar_index / gather_windows --
             the house per-bar batch recompute IS the ring buffer), session/mask
             handling (gather_windows day-clamp + pad-mask channel), and the
             schema/coverage validation helpers here.
  SEPARATED  TouchModelAdapter owns the touch model's artifacts, per-seed
             preprocessors, fixed feature lists and failure surface. A future
             risk-model sequence adapter (Phase-2) sits BESIDE this class with its
             own scalers/feature lists/logging -- the models are never coupled.

Failure isolation (user modification #7): every public entry point RAISES on any
problem (missing artifact, universe drift, schema mismatch, uncovered anchor);
callers must wrap scoring per event batch so a scoring failure only disables the
grade line -- the tier-1 breakout alert never depends on this module succeeding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import tcn_data
import tcn_model as tm
import touch_model as tmm
from feature_lib import FEATURE_COLUMNS_1M
from relative_features import RANK_INPUTS, build_relative_features

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
GRADE_MAP_PATH = _MODELS_DIR / "touch_grade_map.json"
SERVING_VERSION = "touch_cnn_prod"


# --------------------------------------------------------------------------- #
# shared serving-frame glue (feature build stays in realtime_features)
# --------------------------------------------------------------------------- #
def feature_covered_rows(frame):
    """Replicate the features_1m ETL row rule (scripts/build_features.py drops
    warmup rows on FEATURE_COLUMNS_1M -- ret_60m/volume_z_60m bind, so the first
    covered bar is ~10:01). The offline features_1m / features_merged /
    features_relative tables all share this row set, so applying the same dropna
    to the serving-built merged frame reproduces the offline coverage exactly
    (window padding parity included)."""
    missing = [c for c in FEATURE_COLUMNS_1M if c not in frame.columns]
    if missing:
        raise ValueError(f"serving frame lacks features_1m columns: {missing}")
    return frame.dropna(subset=list(FEATURE_COLUMNS_1M)).reset_index(drop=True)


def build_serving_frames(bars_window, *, source):
    """(covered_merged, relatives) for ALL symbols in a multi-day 1m bars window.

    covered_merged = realtime_features._build_merged output restricted to the
    feature-covered rows (see feature_covered_rows); it carries every
    features_merged column AND the 13 sequence channels, so it feeds BOTH
    touch_model.join_event_features and tcn_data.build_bar_index. relatives =
    the 8 cross-sectional columns per (symbol, decision_time) over the covered
    rows (RANK_INPUTS is a subset of the coverage rule, so no extra drop)."""
    from realtime_features import _build_merged   # shared builder (directive #4)

    merged = _build_merged(bars_window, source)
    if merged is None or len(merged) == 0:
        return pd.DataFrame(), pd.DataFrame()
    covered = feature_covered_rows(merged)
    if covered.empty:
        return covered, covered.iloc[0:0]
    nonan = covered.dropna(subset=list(RANK_INPUTS))
    rel = build_relative_features(nonan) if len(nonan) else nonan
    keep = ["symbol", "decision_time"] + list(tmm.RELATIVE_FEATURES_V1)
    missing = [c for c in keep if c not in rel.columns]
    if missing:
        raise ValueError(f"relative frame lacks fixed-list columns: {missing}")
    return covered, rel[keep].reset_index(drop=True)


def validate_serving_schema(frame, required, *, tag):
    """Presence + numeric-dtype check for a declared fixed column list."""
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{tag}: missing columns {missing}")
    bad = [c for c in required
           if not np.issubdtype(np.asarray(frame[c]).dtype, np.number)]
    if bad:
        raise ValueError(f"{tag}: non-numeric columns {bad}")


# --------------------------------------------------------------------------- #
# frozen grade map
# --------------------------------------------------------------------------- #
def load_grade_map(path=None):
    """Load the FROZEN OOF grade map (models/touch_grade_map.json). Raises with a
    build hint when absent -- the runner then degrades to breakout-only mode."""
    p = Path(path) if path else GRADE_MAP_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"grade map not found at {p} -- build it from the 5-seed OOF parquets "
            "first (scripts/build_touch_grade_map.py).")
    gm = json.loads(p.read_text(encoding="utf-8"))
    for key in ("boundaries", "thresholds", "grades", "target"):
        if key not in gm:
            raise ValueError(f"grade map {p} lacks {key!r} -- rebuild it.")
    if gm["target"] != tmm.TOUCH_TARGET:
        raise ValueError(f"grade map target {gm['target']!r} != {tmm.TOUCH_TARGET!r}")
    return gm


def grade_display(grade_map, grade):
    """(oof_touch_rate, n) for one grade letter -- what the alert line cites."""
    st = grade_map["grades"][grade]
    return float(st["touch_rate"]), int(st["n"])


# --------------------------------------------------------------------------- #
# the touch adapter (separated per user directive #4)
# --------------------------------------------------------------------------- #
def resolve_touch_model_path(version=SERVING_VERSION, seed=42, models_dir=None, *,
                             artifact_suffix=""):
    """Prod artifact path from the touch registry; ``artifact_suffix`` namespaces
    smoke/diagnostic weight sets (inserted before _seed, mirroring the trainer)."""
    cfg = tmm.touch_version_config(version)
    pattern = cfg["filename_pattern"]
    if artifact_suffix:
        pattern = pattern.replace("_seed{seed}", f"{artifact_suffix}_seed{{seed}}")
    return Path(models_dir) / pattern.format(seed=seed) if models_dir \
        else _MODELS_DIR / pattern.format(seed=seed)


class TouchModelAdapter:
    """Loads the prod 5-seed ensemble and scores touch events.

    Scoring contract: ``score_events(events, features_frame, relatives)`` where
    events = touch_model.build_touch_setups rows (feature-covered ONLY -- the
    caller gates coverage) and the frames come from build_serving_frames (live)
    or the DB tables (offline reference). Returns the mean-of-seeds sigmoid
    probability per event, raw and UNCALIBRATED -- display via the frozen grade
    map, never as "확률 X%".
    """

    def __init__(self, *, version=SERVING_VERSION, models_dir=None,
                 artifact_suffix="", device="cpu", expected_symbols=None):
        cfg = tmm.touch_version_config(version)
        self.version = version
        self.device = device
        self.seeds = tuple(cfg["seeds"])
        self.arts = []
        for seed in self.seeds:
            path = resolve_touch_model_path(version, seed, models_dir,
                                            artifact_suffix=artifact_suffix)
            self.arts.append(tm.load_tcn(path))
        metas = [a["metadata"] for a in self.arts]
        symbol_sets = {tuple(m.get("symbols") or ()) for m in metas}
        if len(symbol_sets) != 1 or symbol_sets == {()}:
            raise ValueError(
                f"{version}: seed artifacts disagree on (or lack) the frozen "
                "symbol list -- retrain the prod ensemble (--mode prod embeds it).")
        self.symbols = list(symbol_sets.pop())
        static_cols = tmm.static_features_v1(self.symbols)
        for m in metas:
            frozen = list(m.get("static_feature_columns") or [])
            if frozen and frozen != static_cols:
                raise ValueError(f"{version}: artifact static feature list drifted "
                                 "from static_features_v1(symbols).")
        for a in self.arts:
            if list(a["static_pre"].feature_columns) != static_cols:
                raise ValueError(f"{version}: static preprocessor columns do not "
                                 "match static_features_v1(symbols).")
        if expected_symbols is not None:
            now = sorted(str(s) for s in expected_symbols)
            if now != sorted(self.symbols):
                raise ValueError(
                    f"{version}: active symbol set changed since training "
                    f"(trained {len(self.symbols)}, now {len(now)}) -- the symbol "
                    "one-hots fix static_dim, so the prod ensemble must be "
                    "retrained before serving the new universe.")
        self.static_cols = static_cols
        self.seq_config = self.arts[0]["seq_config"]

    def score_events(self, events, features_frame, relatives):
        """Mean-of-seeds probabilities for the (feature-covered) events.

        Raises on ANY problem; the caller isolates failures per batch so tier-1
        breakout alerts survive a scoring outage (user modification #7)."""
        if len(events) == 0:
            return np.empty(0, dtype=float)
        validate_serving_schema(
            features_frame, list(self.seq_config.seq_feature_columns),
            tag="features_frame(seq)")
        joined = tmm.join_event_features(events.reset_index(drop=True),
                                         features_frame, relatives, self.symbols)
        bar_index = tcn_data.build_bar_index(features_frame, symbols=self.symbols,
                                             seq_config=self.seq_config)
        per_seed = []
        for art in self.arts:
            per_seed.append(np.asarray(tm.predict_sequence_model(
                art["state_dict"], joined, bar_index, art["seq_pre"],
                art["static_pre"], seq_config=art["seq_config"],
                tcn_config=art["tcn_config"], device=self.device), dtype=float))
        return tm.ensemble_probs(per_seed)
