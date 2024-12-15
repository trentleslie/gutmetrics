"""Module for scaling and normalizing data."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_metabolomics(
    data: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    *,  # Force copy as keyword-only argument
    copy: bool = True,
) -> pd.DataFrame:
    """Scale metabolomics data using StandardScaler while preserving metadata.

    Args:
    ----
        data: Input DataFrame with metabolomics data
        exclude_cols: Columns to exclude from scaling (e.g. metadata)
        copy: Whether to copy the data before scaling

    Returns:
    -------
        Scaled DataFrame with same structure as input

    """
    if exclude_cols is None:
        exclude_cols = ["shannon", "PD_whole_tree", "chao1", "BMI", "Age", "sex"]

    # Create copy if requested
    if copy:
        data = data.copy()

    # Separate features to scale from excluded columns
    cols_to_scale = [col for col in data.columns if col not in exclude_cols]

    # Initialize and fit scaler
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

    # Scale features
    if cols_to_scale:
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data


def scale_proteomics(
    data: pd.DataFrame,
    metadata_cols: list[str] | None = None,
    *,  # Force copy as keyword-only argument
    copy: bool = True,
) -> pd.DataFrame:
    """Scale proteomics data while preserving metadata columns.

    Args:
    ----
        data: Input DataFrame with proteomics data
        metadata_cols: Metadata columns to exclude from scaling
        copy: Whether to copy the data before scaling

    Returns:
    -------
        Scaled DataFrame with same structure as input

    """
    if metadata_cols is None:
        metadata_cols = ["shannon", "sex", "age"]

    if copy:
        data = data.copy()

    # Get protein columns
    protein_cols = [col for col in data.columns if col not in metadata_cols]

    # Scale protein data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    if protein_cols:
        data[protein_cols] = scaler.fit_transform(data[protein_cols])

    return data


def scale_clinical_labs(
    data: pd.DataFrame,
    metadata_cols: list[str] | None = None,
    *,  # Force copy as keyword-only argument
    copy: bool = True,
) -> pd.DataFrame:
    """Scale clinical laboratory data while preserving metadata.

    Args:
    ----
        data: Input DataFrame with clinical lab data
        metadata_cols: Metadata columns to exclude from scaling
        copy: Whether to copy the data before scaling

    Returns:
    -------
        Scaled DataFrame with same structure as input

    """
    if metadata_cols is None:
        metadata_cols = ["shannon"]

    if copy:
        data = data.copy()

    # Get clinical lab columns
    lab_cols = [col for col in data.columns if col not in metadata_cols]

    # Scale lab data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    if lab_cols:
        data[lab_cols] = scaler.fit_transform(data[lab_cols])

    return data


def scale_and_combine_omics(
    metabolomics_data: pd.DataFrame,
    proteomics_data: pd.DataFrame | None = None,
    clinical_data: pd.DataFrame | None = None,
    join: str = "inner",
) -> pd.DataFrame:
    """Scale and combine multiple omics data types.

    Args:
    ----
        metabolomics_data: Metabolomics DataFrame
        proteomics_data: Optional proteomics DataFrame
        clinical_data: Optional clinical labs DataFrame
        join: How to join DataFrames ('inner' or 'outer')

    Returns:
    -------
        Combined scaled DataFrame

    """
    # Scale each datatype
    scaled_metabolomics = scale_metabolomics(metabolomics_data)

    dfs_to_merge = [scaled_metabolomics]

    if proteomics_data is not None:
        scaled_proteomics = scale_proteomics(proteomics_data)
        dfs_to_merge.append(scaled_proteomics)

    if clinical_data is not None:
        scaled_clinical = scale_clinical_labs(clinical_data)
        dfs_to_merge.append(scaled_clinical)

    # Merge all available data
    if len(dfs_to_merge) > 1:
        merged_data = dfs_to_merge[0]
        for next_data in dfs_to_merge[1:]:
            merged_data = merged_data.merge(
                next_data,
                left_index=True,
                right_index=True,
                how=join,
            )
        return merged_data

    return dfs_to_merge[0]


def get_scaled_feature_names(
    data: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """Get names of scaled features excluding metadata columns.

    Args:
    ----
        data: Input DataFrame
        exclude_cols: Columns to exclude

    Returns:
    -------
        List of scaled feature column names

    """
    if exclude_cols is None:
        exclude_cols = ["shannon", "PD_whole_tree", "chao1", "BMI", "Age", "sex"]

    return [col for col in data.columns if col not in exclude_cols]
