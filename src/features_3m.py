"""Step 7-2: 3-minute features (built from bars_3m, source=resampled_from_al_regular)."""

from __future__ import annotations

from feature_lib import build_bar_features, feature_columns_bar, id_columns_bar

SUFFIX = "3m"
ID_COLUMNS = id_columns_bar(SUFFIX)
FEATURE_COLUMNS = feature_columns_bar(SUFFIX)
COLUMNS = ID_COLUMNS + FEATURE_COLUMNS


def build(df_3m, source):
    """Return the features_3m frame for the given 3-minute bars (tagged with source)."""
    return build_bar_features(df_3m, SUFFIX, source)
