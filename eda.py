import base64
import io
import threading

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MAX_PLOT_ROWS = 5_000
MAX_PLOT_COLUMNS = 12
_PLOT_LOCK = threading.Lock()


def _encode_figure(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _outlier_count(numeric_df: pd.DataFrame) -> int:
    if numeric_df.empty:
        return 0
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1
    mask = numeric_df.lt(q1 - 1.5 * iqr) | numeric_df.gt(q3 + 1.5 * iqr)
    return int(mask.sum().sum())


def _target_column(df: pd.DataFrame):
    candidates = []
    for column in df.columns:
        unique = int(df[column].nunique(dropna=True))
        if 1 < unique <= 10 and not str(column).lower().endswith("id"):
            priority = 0 if not pd.api.types.is_numeric_dtype(df[column]) else 1
            candidates.append((priority, unique, column))
    return min(candidates, default=(None, None, None))[2]


def dataset_info(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    plot_numeric = numeric_df.iloc[:MAX_PLOT_ROWS, :MAX_PLOT_COLUMNS]
    target_col = _target_column(df)

    stats = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
            if count
        },
        "column_names": [str(column) for column in df.columns],
        "outlier_count": _outlier_count(numeric_df),
        "visuals": {},
        "target_col": str(target_col) if target_col is not None else None,
    }

    with _PLOT_LOCK:
        if not plot_numeric.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=plot_numeric, ax=ax)
            ax.set_title("Numeric distributions and outliers")
            ax.tick_params(axis="x", rotation=30)
            stats["visuals"]["boxplot"] = _encode_figure(fig)

        usable_numeric = [
            column for column in plot_numeric if plot_numeric[column].nunique(dropna=True) > 1
        ]
        if len(usable_numeric) >= 2:
            x_col, y_col = usable_numeric[:2]
            relationship_data = plot_numeric[[x_col, y_col]].dropna()
            if len(relationship_data) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(
                    x=x_col,
                    y=y_col,
                    data=relationship_data,
                    scatter_kws={"alpha": 0.45, "s": 24},
                    line_kws={"color": "#dc2626"},
                    ax=ax,
                )
                ax.set_title(f"Relationship: {x_col} vs {y_col}")
                stats["visuals"]["relationship"] = _encode_figure(fig)

        if target_col is not None:
            target_data = df[[target_col]].dropna().iloc[:MAX_PLOT_ROWS]
            if not target_data.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                order = target_data[target_col].value_counts().index
                sns.countplot(data=target_data, x=target_col, order=order, color="#2563eb", ax=ax)
                ax.set_title(f"Distribution of {target_col}")
                ax.tick_params(axis="x", rotation=30)
                stats["visuals"]["countplot"] = _encode_figure(fig)

        correlation_data = plot_numeric[usable_numeric]
        if correlation_data.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                correlation_data.corr(),
                annot=correlation_data.shape[1] <= 8,
                cmap="RdYlGn",
                fmt=".2f",
                vmin=-1,
                vmax=1,
                ax=ax,
            )
            ax.set_title("Feature correlation matrix")
            stats["visuals"]["heatmap"] = _encode_figure(fig)

    return stats
