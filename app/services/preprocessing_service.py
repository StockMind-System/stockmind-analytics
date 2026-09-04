from collections.abc import Mapping, Sequence

import pandas as pd

from app.ml.feature_engineering import build_feature_matrix


def preprocess_records(
    records: Sequence[Mapping[str, object]],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    return build_feature_matrix(records, expected_columns)

