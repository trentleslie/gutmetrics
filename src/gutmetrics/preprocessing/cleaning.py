"""Module for cleaning and validating metabolomics and microbiome data."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


def standardize_index(
    df: pd.DataFrame,
    index_col: str = "public_client_id",
    index_type: str = "float64",
) -> pd.DataFrame:
    """Standardize DataFrame index to consistent format.

    Args:
    ----
        df: Input DataFrame
        index_col: Column to use as index
        index_type: Type to cast index to

    Returns:
    -------
        DataFrame with standardized index

    Raises:
    ------
        ValueError: If DataFrame is empty or contains duplicate IDs

    """
    if df.empty:
        msg = "Cannot standardize index of empty DataFrame"
        raise ValueError(msg)

    # Check for duplicate IDs
    if df[index_col].duplicated().any():
        msg = "Duplicate client IDs found"
        raise ValueError(msg)

    df_copy = df.copy()
    if df_copy.index.name != index_col and index_col in df_copy.columns:
        df_copy = df_copy.set_index(index_col)
    df_copy.index = df_copy.index.astype(index_type)
    return df_copy


def remove_outliers(
    data: pd.DataFrame, column: str, n_std: float = 3.0
) -> pd.DataFrame:
    """Remove outliers from a DataFrame column using the IQR method.

    Args:
        data: Input DataFrame
        column: Column name to check for outliers
        n_std: Number of standard deviations to use as threshold

    Returns:
        DataFrame with outliers removed
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - n_std * iqr
    upper_bound = q3 + n_std * iqr

    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


def validate_metabolomics_data(
    df: pd.DataFrame,
    required_cols: Sequence[str] | None = None,
) -> bool:
    """Validate metabolomics data format and content.

    Args:
    ----
        df: Metabolomics DataFrame to validate
        required_cols: List of required columns

    Returns:
    -------
        True if valid

    Raises:
    ------
        ValueError: If validation fails or DataFrame is empty

    """
    if df.empty:
        msg = "Cannot validate empty DataFrame"
        raise ValueError(msg)

    if required_cols is None:
        required_cols = ["shannon", "PD_whole_tree", "chao1"]

    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        msg = f"Missing required columns: {', '.join(missing_cols)}"
        raise ValueError(msg)

    # Check for all numeric data in metabolite columns
    metabolite_cols = [col for col in df.columns if col not in required_cols]
    non_numeric = (
        df[metabolite_cols].select_dtypes(exclude=["float64", "int64"]).columns
    )
    if len(non_numeric) > 0:
        msg = f"Non-numeric data found in columns: {', '.join(non_numeric)}"
        raise ValueError(msg)

    return True


def validate_microbiome_data(df: pd.DataFrame, min_reads: int = 30000) -> bool:
    """Validate microbiome data format and content.

    Args:
    ----
        df: Microbiome DataFrame to validate
        min_reads: Minimum number of reads required

    Returns:
    -------
        True if valid

    Raises:
    ------
        ValueError: If validation fails or DataFrame is empty

    """
    if df.empty:
        msg = "Cannot validate empty DataFrame"
        raise ValueError(msg)

    # Check for OTU columns
    if not any("bacteria" in col.lower() for col in df.columns):
        msg = "No bacterial OTU columns found"
        raise ValueError(msg)

    # Check read depth
    if "total_reads" in df.columns:
        low_reads = df[df["total_reads"] < min_reads].index.tolist()
        if low_reads:
            msg = f"Samples {low_reads} have fewer than {min_reads} reads"
            raise ValueError(msg)

    # Check for normalized values
    otu_cols = [col for col in df.columns if "bacteria" in col.lower()]
    if TYPE_CHECKING:
        sums: NDArray[np.float64] = df[otu_cols].sum(axis=1).to_numpy()
    else:
        sums = df[otu_cols].sum(axis=1).to_numpy()

    if not np.allclose(sums, 1.0):
        msg = "OTU abundances are not normalized to sum to 1"
        raise ValueError(msg)

    return True


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Clean metadata DataFrame.

    Args:
    ----
        df: DataFrame to clean

    Returns:
    -------
        Cleaned DataFrame

    Raises:
    ------
        ValueError: If required columns are missing
    """
    required_cols = [
        "bacteria_1",
        "bacteria_2",
        "total_reads",
    ]  # Example required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        msg = "Missing required columns"
        raise ValueError(msg)
    return df
