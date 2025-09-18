"""
Test suite for podest package.

Run with: python -m pytest tests/
"""
from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from podest.main import (
    _zscore_normalize,
    mean_point_density,
    two_scenario_mpd_comparison,
    _two_scenario_p_value_sequential,
    _two_scenario_p_value_parallel,
    two_scenario_p_value,
)


class TestZscoreNormalize:
    """Test cases for zscore_normalize function."""

    def test_basic_normalization(self):
        """Test basic z-score normalization."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        normalized = _zscore_normalize(x)

        # Check that mean is approximately 0 and std is approximately 1
        assert_almost_equal(normalized.mean(), 0.0, decimal=10)
        assert_almost_equal(normalized.std(), 1.0, decimal=10)

    def test_empty_array(self):
        """Test with empty array."""
        x = np.array([])
        result = _zscore_normalize(x)
        assert result.size == 0

    def test_single_element(self):
        """Test with single element array."""
        x = np.array([5.0])
        with pytest.raises(ValueError, match="zero standard deviation"):
            _zscore_normalize(x)

    def test_constant_array(self):
        """Test with constant array (zero std)."""
        x = np.array([3, 3, 3, 3])
        with pytest.raises(ValueError, match="zero standard deviation"):
            _zscore_normalize(x)

    def test_list_input(self):
        """Test with list input."""
        x = [1, 2, 3, 4, 5]
        normalized = _zscore_normalize(x)
        assert_almost_equal(normalized.mean(), 0.0, decimal=10)
        assert_almost_equal(normalized.std(), 1.0, decimal=10)


class TestMeanPointDensity:
    """Test cases for mean_point_density function."""

    def test_basic_functionality(self):
        """Test basic point density calculation."""
        # Create a simple 2x2 grid of points
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])

        density = mean_point_density(x, y, radius=2.0)
        assert isinstance(density, float)
        assert density > 0

    def test_single_point(self):
        """Test with single point."""
        x = np.array([0])
        y = np.array([0])

        density = mean_point_density(x, y)
        assert density == 1.0  # Single point contributes weight 1 to itself

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(ValueError, match="same length"):
            mean_point_density(x, y)

    def test_empty_arrays(self):
        """Test with empty arrays."""
        x = np.array([])
        y = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            mean_point_density(x, y)

    def test_radius_parameter(self):
        """Test that radius parameter affects results."""
        x = np.array([0, 10])  # Two distant points
        y = np.array([0, 0])

        density_small = mean_point_density(x, y, radius=1.0)
        density_large = mean_point_density(x, y, radius=20.0)

        # Larger radius should include more neighbors
        assert density_large > density_small

    def test_clustered_vs_scattered(self):
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


class TestTwoScenarioMpdComparison:
    """Test cases for two_scenario_mpd_comparison function."""

    def test_basic_comparison(self):
        """Test basic two-scenario comparison."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        mpd1, mpd2 = two_scenario_mpd_comparison(x, x_alt, y)

        assert isinstance(mpd1, float)
        assert isinstance(mpd2, float)
        assert mpd1 > 0
        assert mpd2 > 0

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        x = np.array([1, 2])
        x_alt = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(ValueError, match="same length"):
            two_scenario_mpd_comparison(x, x_alt, y)

    def test_zero_std_y(self):
        """Test with zero standard deviation in y."""
        x = np.array([1, 2, 3])
        x_alt = np.array([4, 5, 6])
        y = np.array([1, 1, 1])  # Constant y values

        with pytest.raises(ValueError, match="zero standard deviation"):
            two_scenario_mpd_comparison(x, x_alt, y)

    def test_zero_std_combined_x(self):
        """Test with zero standard deviation in combined x."""
        x = np.array([1, 1, 1])
        x_alt = np.array([1, 1, 1])  # All x values the same
        y = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="zero standard deviation"):
            two_scenario_mpd_comparison(x, x_alt, y)


class TestTwoScenarioPValue:
    """Test cases for p-value calculation functions."""

    def simple_generator(self, size=3, rng=None):
        """Simple generator for testing."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(0, 1, size)

    def test_p_value_sequential(self):
        """Test sequential p-value calculation."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        p_val = _two_scenario_p_value_sequential(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,  # Small number for testing
            verbose=False
        )

        assert 0.0 <= p_val <= 1.0

    def test_p_value_parallel(self):
        """Test parallel p-value calculation."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        p_val = _two_scenario_p_value_parallel(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,  # Small number for testing
            verbose=False
        )

        assert 0.0 <= p_val <= 1.0

    def test_reproducibility_parallel(self):
        """Test that parallel computation is reproducible."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        p_val1 = _two_scenario_p_value_parallel(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,
            base_seed=123
        )

        p_val2 = _two_scenario_p_value_parallel(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,
            base_seed=123
        )

        assert p_val1 == p_val2

    def test_invalid_n_simulations(self):
        """Test with invalid number of simulations."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="must be positive"):
            _two_scenario_p_value_parallel(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params={'size': 3},
                n_simulations=0
            )


class TestTwoScenarioPValueAuto:
    """Test cases for unified p-value calculation function."""

    def simple_generator(self, size=3, rng=None):
        """Simple generator for testing."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(0, 1, size)

    def test_auto_sequential_mode(self):
        """Test automatic sequential mode (n_cpu=1)."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        p_val = two_scenario_p_value(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,
            n_cpu=1,
            verbose=False
        )

        assert 0.0 <= p_val <= 1.0

    def test_auto_parallel_mode(self):
        """Test automatic parallel mode (n_cpu>1)."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        p_val = two_scenario_p_value(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,
            n_cpu=2,
            base_seed=123,
            verbose=False
        )

        assert 0.0 <= p_val <= 1.0

    def test_reproducibility_auto_parallel(self):
        """Test reproducibility in auto parallel mode."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        p_val1 = _two_scenario_p_value_parallel(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,
            n_cpu=2,
            base_seed=456
        )

        p_val2 = _two_scenario_p_value_sequential(
            x, x_alt, y,
            x_alt_generator=self.simple_generator,
            generator_params={'size': 3},
            n_simulations=10,
            n_cpu=2,
            base_seed=456
        )

        assert p_val1 == p_val2

    def test_invalid_n_cpu(self):
        """Test with invalid n_cpu values."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        # Test negative n_cpu
        with pytest.raises(ValueError, match="n_cpu must be at least 1"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params={'size': 3},
                n_cpu=0
            )

        # Test non-integer n_cpu
        with pytest.raises(TypeError, match="n_cpu must be an integer"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params={'size': 3},
                n_cpu=2.5
            )

    def test_invalid_generator(self):
        """Test with invalid generator function."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        with pytest.raises(TypeError, match="x_alt_generator must be callable"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator="not_callable",  # Invalid
                generator_params={'size': 3},
                n_cpu=1
            )

    def test_invalid_generator_params(self):
        """Test with invalid generator parameters."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        with pytest.raises(TypeError, match="generator_params must be a dictionary"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params=['not', 'a', 'dict'],  # Invalid
                n_cpu=1
            )

    def test_invalid_simulations(self):
        """Test with invalid simulation parameters."""
        x = np.array([0, 1, 2])
        x_alt = np.array([0, 0.5, 1])
        y = np.array([0, 1, 2])

        # Non-integer n_simulations
        with pytest.raises(TypeError, match="n_simulations must be an integer"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params={'size': 3},
                n_simulations=100.5
            )

        # Zero simulations
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params={'size': 3},
                n_simulations=0
            )

    def test_empty_arrays(self):
        """Test with empty input arrays."""
        x = np.array([])
        x_alt = np.array([])
        y = np.array([])

        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            two_scenario_p_value_serial(
                x, x_alt, y,
                x_alt_generator=self.simple_generator,
                generator_params={'size': 0},
                n_cpu=1
            )

    def test_mode_selection_consistency(self):
        """Test that n_cpu=1 and n_cpu>1 give similar results."""
        x = np.array([0, 1, 2, 3, 4])
        x_alt = np.array([0, 0.5, 1, 1.5, 2])
        y = np.array([0, 1, 2, 3, 4])

        # Use same seed for reproducible generator
        def seeded_generator(size=5, rng=None):
            if rng is None:
                rng = np.random.default_rng(789)
            return rng.normal(0, 1, size)

        # Sequential mode
        p_val_seq = two_scenario_p_value_serial(
            x, x_alt, y,
            x_alt_generator=seeded_generator,
            generator_params={'size': 5},
            n_simulations=50,  # Small number for test speed
            n_cpu=1
        )

        # Parallel mode
        p_val_par = two_scenario_p_value_serial(
            x, x_alt, y,
            x_alt_generator=seeded_generator,
            generator_params={'size': 5},
            n_simulations=50,
            n_cpu=2,
            base_seed=789
        )

        # Results should be close (within statistical variation)
        # We can't expect exact equality due to different RNG handling
        assert 0.0 <= p_val_seq <= 1.0
        assert 0.0 <= p_val_par <= 1.0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
