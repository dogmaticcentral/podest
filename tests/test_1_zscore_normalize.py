# test_1_zscore_normalize.py
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from podest.main import _zscore_normalize


@pytest.mark.run(order=1)
def test_basic_normalization():
    """Test basic z-score normalization."""
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    normalized = _zscore_normalize(x)

    # Check that mean is approximately 0 and std is approximately 1
    assert_almost_equal(normalized.mean(), 0.0, decimal=10)
    assert_almost_equal(normalized.std(), 1.0, decimal=10)


@pytest.mark.run(order=2)
def test_empty_array():
    """Test with empty array."""
    x = np.array([])
    result = _zscore_normalize(x)
    assert result.size == 0


@pytest.mark.run(order=3)
def test_single_element():
    """Test with single element array."""
    x = np.array([5.0])
    with pytest.raises(ValueError, match="zero standard deviation"):
        _zscore_normalize(x)


@pytest.mark.run(order=4)
def test_constant_array():
    """Test with constant array (zero std)."""
    x = np.array([3, 3, 3, 3])
    with pytest.raises(ValueError, match="zero standard deviation"):
        _zscore_normalize(x)


@pytest.mark.run(order=5)
def test_list_input():
    """Test with list input."""
    x = [1, 2, 3, 4, 5]
    normalized = _zscore_normalize(x)
    assert_almost_equal(normalized.mean(), 0.0, decimal=10)
    assert_almost_equal(normalized.std(), 1.0, decimal=10)