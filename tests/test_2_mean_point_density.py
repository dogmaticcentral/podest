# test_2_mean_point_density.py
import pytest
import numpy as np
from podest.main import mean_point_density


@pytest.mark.run(order=1)
def test_basic_functionality():
    """Test basic point density calculation."""
    # Create a simple 2x2 grid of points
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])

    density = mean_point_density(x, y, radius=2.0)
    assert isinstance(density, float)
    assert density > 0


@pytest.mark.run(order=2)
def test_single_point():
    """Test with single point."""
    x = np.array([0])
    y = np.array([0])

    density = mean_point_density(x, y)
    assert density == 1.0  # Single point contributes weight 1 to itself


@pytest.mark.run(order=3)
def test_mismatched_lengths():
    """Test with mismatched array lengths."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2])

    with pytest.raises(ValueError, match="same length"):
        mean_point_density(x, y)


@pytest.mark.run(order=4)
def test_empty_arrays():
    """Test with empty arrays."""
    x = np.array([])
    y = np.array([])

    with pytest.raises(ValueError, match="cannot be empty"):
        mean_point_density(x, y)


@pytest.mark.run(order=5)
def test_radius_parameter():
    """Test that radius parameter does not affect results.
    (As long as it is much bigger than the kernel_scale.)
    """
    x = np.array([0, 10])  # Two distant points
    y = np.array([0, 0])

    density_large = mean_point_density(x, y, radius=20.0)
    density_small = mean_point_density(x, y, radius=1.0)

    # The radius should not matter, because the kernel_scale
    # decides on the "horizon" beyond which the neighbors become invisible
    assert abs(density_large - density_small) < 1.e-6

@pytest.mark.run(order=6)
def test_kernel_scale_parameter():
    """Test that kernel_scale parameter affects results."""
    x = np.array([0, 10])  # Two distant points
    y = np.array([0, 0])

    density_large = mean_point_density(x, y, kernel_scale=100)
    density_small = mean_point_density(x, y, kernel_scale=1.0)

    # The radius should not matter, because the kernel_scale
    # decides on the "horizon" beyond which the neighbors become invisible
    assert density_large > density_small


@pytest.mark.run(order=7)
def test_clustered_vs_scattered():
    """Test that clustered points have higher density."""
    # Clustered points
    x_clustered = np.array([0, 0.1, 0.2])
    y_clustered = np.array([0, 0.1, 0.2])

    # Scattered points
    x_scattered = np.array([0, 10, 20])
    y_scattered = np.array([0, 10, 20])

    density_clustered = mean_point_density(x_clustered, y_clustered)
    density_scattered = mean_point_density(x_scattered, y_scattered)

    assert density_clustered > density_scattered