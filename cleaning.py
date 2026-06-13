import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


NULL_TEXT_VALUES = {
    "",
    "-",
    "--",
    "?",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
}

CATEGORY_ALIASES = {
    "eastern": "east",
    "nort": "north",
    "northern": "north",
    "southern": "south",
    "western": "west",
    "prem": "premium",
    "m": "male",
    "f": "female",
    "y": "yes",
    "n": "no",
    "true": "yes",
    "false": "no",
}


def _is_identifier_name(column: object) -> bool:
    name = str(column).strip().lower()
    return bool(
        re.search(
            r"(^id$|_id$|^id_|identifier$|uuid$|zip|postal|phone|code$)",
            name,
        )
    )


def _standardize_text(series: pd.Series) -> pd.Series:
    standardized = series.astype("string").str.strip().str.lower()
    standardized = standardized.mask(standardized.isin(NULL_TEXT_VALUES), pd.NA)
    return standardized.replace(CATEGORY_ALIASES)


def _coerce_numeric_like(column: object, series: pd.Series) -> pd.Series:
    if _is_identifier_name(column):
        return series

    standardized = _standardize_text(series)
    candidate = standardized.str.replace(r"[$,]", "", regex=True)
    parsed = pd.to_numeric(candidate, errors="coerce")
    non_null_count = int(standardized.notna().sum())

    if non_null_count and parsed.notna().sum() / non_null_count >= 0.80:
        return parsed
    return standardized


def _fill_value(series: pd.Series):
    non_null = series.dropna()

    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        if non_null.empty:
            return 0
        # Median is less sensitive to outliers than mean imputation.
        return non_null.median()

    if non_null.empty:
        return "unknown"

    modes = non_null.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else non_null.iloc[0]


def clean_dataset(df: pd.DataFrame, return_report: bool = False):
    cleaned = df.copy()
    rows_before = len(cleaned)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    converted_numeric_columns = []
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        converted = _coerce_numeric_like(column, cleaned[column])
        if is_numeric_dtype(converted.dtype):
            converted_numeric_columns.append(str(column))
        cleaned[column] = converted

    null_before = int(cleaned.isna().sum().sum())
    duplicates_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    imputed_columns = {}
    for column in cleaned.columns:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count:
            cleaned[column] = cleaned[column].fillna(_fill_value(cleaned[column]))
            imputed_columns[str(column)] = missing_count

    report = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "null_before": null_before,
        "null_after": int(cleaned.isna().sum().sum()),
        "imputed_columns": imputed_columns,
        "converted_numeric_columns": converted_numeric_columns,
    }

    if return_report:
        return cleaned, report
    return cleaned
