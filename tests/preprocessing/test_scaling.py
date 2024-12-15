"""Unit tests for scaling module."""

import numpy as np
import pandas as pd
import pytest

from gutmetrics.preprocessing.scaling import (
    scale_metabolomics,
    scale_proteomics,
    scale_clinical_labs,
    scale_and_combine_omics,
    get_scaled_feature_names,
)


@pytest.fixture
def sample_metabolomics_data() -> pd.DataFrame:
    """Create sample metabolomics data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "met1": np.random.normal(10, 2, n_samples),
            "met2": np.random.normal(20, 5, n_samples),
            "shannon": np.random.uniform(0.5, 0.7, n_samples),
            "BMI": np.random.normal(25, 3, n_samples),
            "Age": np.random.normal(40, 10, n_samples),
            "sex": np.random.binomial(1, 0.5, n_samples),
        },
        index=[f"p{i}" for i in range(n_samples)],
    )


@pytest.fixture
def sample_proteomics_data() -> pd.DataFrame:
    """Create sample proteomics data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "prot1": np.random.normal(100, 20, n_samples),
            "prot2": np.random.normal(200, 50, n_samples),
            "shannon": np.random.uniform(0.5, 0.7, n_samples),
            "sex": np.random.binomial(1, 0.5, n_samples),
            "age": np.random.normal(40, 10, n_samples),
        },
        index=[f"p{i}" for i in range(n_samples)],
    )


@pytest.fixture
def sample_clinical_data() -> pd.DataFrame:
    """Create sample clinical lab data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "lab1": np.random.normal(50, 10, n_samples),
            "lab2": np.random.normal(150, 30, n_samples),
            "shannon": np.random.uniform(0.5, 0.7, n_samples),
        },
        index=[f"p{i}" for i in range(n_samples)],
    )


def test_scale_metabolomics_basic(sample_metabolomics_data: pd.DataFrame) -> None:
    """Test basic scaling of metabolomics data."""
    scaled = scale_metabolomics(sample_metabolomics_data)

    # Check that metadata columns are unchanged
    pd.testing.assert_series_equal(
        scaled["shannon"],
        sample_metabolomics_data["shannon"],
    )

    # Check that features are scaled (mean ≈ 0, std ≈ 1)
    assert np.abs(scaled["met1"].mean()) < 1e-10
    assert np.abs(scaled["met2"].mean()) < 1e-10
    assert np.abs(scaled["met1"].std(ddof=0) - 1) < 1e-10
    assert np.abs(scaled["met2"].std(ddof=0) - 1) < 1e-10


def test_scale_metabolomics_copy(sample_metabolomics_data: pd.DataFrame) -> None:
    """Test copy parameter in scale_metabolomics."""
    # Test with copy=True
    original = sample_metabolomics_data.copy()
    scaled = scale_metabolomics(sample_metabolomics_data, copy=True)
    pd.testing.assert_frame_equal(sample_metabolomics_data, original)

    # Test with copy=False
    scaled_inplace = scale_metabolomics(sample_metabolomics_data, copy=False)
    assert scaled_inplace is sample_metabolomics_data


def test_scale_proteomics_basic(sample_proteomics_data: pd.DataFrame) -> None:
    """Test basic scaling of proteomics data."""
    scaled = scale_proteomics(sample_proteomics_data)

    # Check metadata preservation
    pd.testing.assert_series_equal(
        scaled["shannon"],
        sample_proteomics_data["shannon"],
    )

    # Check scaling
    assert np.abs(scaled["prot1"].mean()) < 1e-10
    assert np.abs(scaled["prot2"].mean()) < 1e-10
    assert np.abs(scaled["prot1"].std(ddof=0) - 1) < 1e-10
    assert np.abs(scaled["prot2"].std(ddof=0) - 1) < 1e-10


def test_scale_clinical_labs_basic(sample_clinical_data: pd.DataFrame) -> None:
    """Test basic scaling of clinical lab data."""
    scaled = scale_clinical_labs(sample_clinical_data)

    # Check metadata preservation
    pd.testing.assert_series_equal(
        scaled["shannon"],
        sample_clinical_data["shannon"],
    )

    # Check scaling
    assert np.abs(scaled["lab1"].mean()) < 1e-10
    assert np.abs(scaled["lab2"].mean()) < 1e-10
    assert np.abs(scaled["lab1"].std(ddof=0) - 1) < 1e-10
    assert np.abs(scaled["lab2"].std(ddof=0) - 1) < 1e-10


def test_scale_and_combine_omics_all(
    sample_metabolomics_data: pd.DataFrame,
    sample_proteomics_data: pd.DataFrame,
    sample_clinical_data: pd.DataFrame,
) -> None:
    """Test combining all three types of omics data."""
    combined = scale_and_combine_omics(
        sample_metabolomics_data,
        sample_proteomics_data,
        sample_clinical_data,
    )

    # Check that all feature columns are present
    expected_cols = {"met1", "met2", "prot1", "prot2", "lab1", "lab2"}
    assert all(col in combined.columns for col in expected_cols)

    # Check that features are scaled
    for col in expected_cols:
        assert np.abs(combined[col].mean()) < 1e-10
        assert np.abs(combined[col].std(ddof=0) - 1) < 1e-10


def test_scale_and_combine_omics_subset(
    sample_metabolomics_data: pd.DataFrame,
    sample_proteomics_data: pd.DataFrame,
) -> None:
    """Test combining just metabolomics and proteomics data."""
    combined = scale_and_combine_omics(
        sample_metabolomics_data,
        sample_proteomics_data,
        None,
    )

    # Check expected columns
    expected_cols = {"met1", "met2", "prot1", "prot2"}
    assert all(col in combined.columns for col in expected_cols)


def test_get_scaled_feature_names(sample_metabolomics_data: pd.DataFrame) -> None:
    """Test getting names of scaled features."""
    feature_names = get_scaled_feature_names(sample_metabolomics_data)
    assert set(feature_names) == {"met1", "met2"}

    # Test with custom exclude list
    custom_features = get_scaled_feature_names(
        sample_metabolomics_data,
        exclude_cols=["shannon", "met1"],
    )
    assert set(custom_features) == {"met2", "BMI", "Age", "sex"}
