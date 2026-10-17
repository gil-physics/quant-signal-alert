"""Step 7-3: 5-minute features (built from bars_5m, source=resampled_from_al_regular).

Same construction as 3m, with the 5m suffix.
"""

from __future__ import annotations

from feature_lib import build_bar_features, feature_columns_bar, id_columns_bar

SUFFIX = "5m"
ID_COLUMNS = id_columns_bar(SUFFIX)
FEATURE_COLUMNS = feature_columns_bar(SUFFIX)
COLUMNS = ID_COLUMNS + FEATURE_COLUMNS


def build(df_5m, source):
    """Return the features_5m frame for the given 5-minute bars (tagged with source)."""
    return build_bar_features(df_5m, SUFFIX, source)
