"""Step 7-1: 1-minute features (built from bars_1m, source=al_regular)."""

from __future__ import annotations

from feature_lib import COLUMNS_1M, FEATURE_COLUMNS_1M, ID_COLUMNS_1M, build_minute_features

ID_COLUMNS = ID_COLUMNS_1M
FEATURE_COLUMNS = FEATURE_COLUMNS_1M
COLUMNS = COLUMNS_1M


def build(df_1m, source):
    """Return the features_1m frame for the given 1-minute bars (tagged with source)."""
    return build_minute_features(df_1m, source)
