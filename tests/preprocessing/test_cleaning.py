"""Tests for data cleaning functions."""

import pandas as pd
import pytest
from gutmetrics.preprocessing.cleaning import (
    clean_metadata,
    remove_outliers,
    standardize_index,
    validate_metabolomics_data,
    validate_microbiome_data,
)

# Constants for test values
MIN_READS = 30000
STD_THRESHOLD = 3.0
EXPECTED_SUM = 100.0
EXPECTED_NON_OUTLIER_COUNT = 4  # Number of values expected after removing outliers


@pytest.fixture()
def sample_data() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    data = {
        "public_client_id": [1, 2, 3],
        "value": [10.0, 20.0, 30.0],
    }
    return pd.DataFrame(data)


def test_standardize_index(sample_data: pd.DataFrame) -> None:
    """Test standardize_index function."""
    result = standardize_index(sample_data)
    assert result.index.name == "public_client_id"
    assert result.index.dtype == "float64"
    assert (
        len(result.columns) == 1
    )  # Changed from 4 to 1 since 'public_client_id' becomes the index


def test_remove_outliers() -> None:
    """Test remove_outliers function."""
    test_data = pd.DataFrame({"value": [1.0, 2.0, 100.0, 3.0, 2.5]})
    result = remove_outliers(test_data, "value", n_std=STD_THRESHOLD)
    assert (
        len(result) == EXPECTED_NON_OUTLIER_COUNT
    )  # Using constant instead of magic number
    assert result["value"].to_numpy().max() <= EXPECTED_SUM


def test_validate_metabolomics_empty() -> None:
    """Test validate_metabolomics_data with empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Cannot validate empty DataFrame"):
        validate_metabolomics_data(empty_df)


def test_validate_metabolomics_data() -> None:
    """Test validate_metabolomics_data function."""
    test_data = pd.DataFrame(
        {
            "shannon": [1.0, 2.0],
            "PD_whole_tree": [0.5, 0.6],
            "chao1": [100.0, 120.0],
            "metabolite1": [0.1, 0.2],
        },
    )
    assert validate_metabolomics_data(test_data)


def test_validate_microbiome_data() -> None:
    """Test validate_microbiome_data function."""
    test_data = pd.DataFrame(
        {
            "bacteria_1": [0.5, 0.6],
            "bacteria_2": [0.5, 0.4],
            "total_reads": [MIN_READS, MIN_READS + 1000],
        },
    )
    cleaned = clean_metadata(test_data)
    assert "bacteria_1" in cleaned.columns
    assert "bacteria_2" in cleaned.columns
    assert cleaned["bacteria_1"].dtype == "float64"
    assert cleaned["bacteria_2"].dtype == "float64"
    assert validate_microbiome_data(cleaned)


def test_validate_metabolomics_invalid_data() -> None:
    """Test validate_metabolomics_data with invalid data."""
    test_data = pd.DataFrame(
        {
            "invalid_column": [1.0, 2.0],
            "another_invalid": [0.5, 0.6],
        }
    )
    with pytest.raises(
        ValueError, match="Missing required columns: shannon, PD_whole_tree, chao1"
    ):
        validate_metabolomics_data(test_data)


def test_validate_microbiome_invalid_reads() -> None:
    """Test validate_microbiome_data with insufficient reads."""
    test_data = pd.DataFrame(
        {
            "bacteria_1": [0.5, 0.6],
            "bacteria_2": [0.5, 0.4],
            "total_reads": [MIN_READS - 1000, MIN_READS - 500],  # Below minimum
        }
    )
    cleaned = clean_metadata(test_data)
    with pytest.raises(
        ValueError, match=r"Samples \[0, 1\] have fewer than 30000 reads"
    ):
        validate_microbiome_data(cleaned)


def test_clean_metadata_missing_columns() -> None:
    """Test clean_metadata with missing required columns."""
    test_data = pd.DataFrame(
        {
            "some_column": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_metadata(test_data)


def test_standardize_index_duplicate_ids() -> None:
    """Test standardize_index with duplicate IDs."""
    data = pd.DataFrame(
        {
            "public_client_id": [1, 1, 2],  # Duplicate ID
            "value": [10.0, 20.0, 30.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate client IDs found"):
        standardize_index(data)


def test_remove_outliers_all_same() -> None:
    """Test remove_outliers with no variation in data."""
    test_data = pd.DataFrame({"value": [1.0, 1.0, 1.0]})
    result = remove_outliers(test_data, "value", n_std=STD_THRESHOLD)
    assert len(result) == 3
    assert (result["value"] == 1.0).all()
