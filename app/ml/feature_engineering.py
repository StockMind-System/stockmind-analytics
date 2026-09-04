from collections.abc import Iterable, Mapping

import pandas as pd


def build_feature_matrix(
    records: Iterable[Mapping[str, object]],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert API records into a numeric matrix suitable for model input."""
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("At least one record is required")

    numeric_columns = frame.select_dtypes(include="number").columns
    frame[numeric_columns] = frame[numeric_columns].fillna(
        frame[numeric_columns].median()
    )
    frame = pd.get_dummies(frame, dummy_na=True, dtype=float).fillna(0.0)

    if expected_columns is not None:
        frame = frame.reindex(columns=expected_columns, fill_value=0.0)

    return frame.astype(float)

