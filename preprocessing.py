import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _is_identifier(column: object, series: pd.Series) -> bool:
    name = str(column).strip().lower()
    explicit_id_name = bool(
        re.search(r"(^id$|_id$|^id_|identifier$|uuid$)", name)
    )
    mostly_unique = (
        len(series) >= 20
        and series.nunique(dropna=True) / len(series) >= 0.98
    )
    text_identifier = mostly_unique and not is_numeric_dtype(series.dtype)
    return explicit_id_name or text_identifier or (name.endswith("number") and mostly_unique)


def _normalize_categories(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.replace(
        {
            "m": "male",
            "f": "female",
            "y": "yes",
            "n": "no",
            "true": "yes",
            "false": "no",
        }
    )


def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    return_report: bool = False,
):
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    target = df[target_column].copy() if target_column else None
    features = df.drop(columns=[target_column]) if target_column else df.copy()
    identifier_cols = [
        column for column in features.columns if _is_identifier(column, features[column])
    ]

    categorical_cols = [
        column
        for column in features.columns
        if column not in identifier_cols
        and (
            not is_numeric_dtype(features[column].dtype)
            or is_bool_dtype(features[column].dtype)
        )
    ]
    numerical_cols = [
        column
        for column in features.columns
        if column not in identifier_cols
        and column not in categorical_cols
        and is_numeric_dtype(features[column].dtype)
        and not is_bool_dtype(features[column].dtype)
    ]

    for column in categorical_cols:
        features[column] = _normalize_categories(features[column])

    if categorical_cols:
        features = pd.get_dummies(
            features,
            columns=categorical_cols,
            prefix_sep="__",
            dtype="int8",
        )

    scaled_cols = []
    for column in numerical_cols:
        numeric = pd.to_numeric(features[column], errors="coerce").astype(float)
        std = float(numeric.std(ddof=0))
        if np.isfinite(std) and std > 0:
            features[column] = (numeric - float(numeric.mean())) / std
            scaled_cols.append(str(column))
        else:
            features[column] = 0.0

    processed = features.drop(columns=identifier_cols, errors="ignore")
    if target_column:
        processed[target_column] = target

    report = {
        "encoded_columns": [str(column) for column in categorical_cols],
        "scaled_columns": scaled_cols,
        "excluded_columns": [str(column) for column in identifier_cols],
        "target_column": target_column,
        "target_preserved": bool(target_column),
    }

    if return_report:
        return processed, report
    return processed, report["encoded_columns"], scaled_cols
