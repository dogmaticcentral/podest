# test_5_two_scenario_pvalue_auto.py
# test_5_two_scenario_pvalue_auto.py
from random import randrange

import pytest
import numpy as np
from podest.main import (
    two_scenario_p_value,
    _two_scenario_p_value_parallel,
    _two_scenario_p_value_sequential,
)


def simple_generator(size=3, rng=None):
    """Simple generator for testing."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, 1, size)


@pytest.mark.run(order=1)
def test_auto_sequential_mode():
    """Test automatic sequential mode (n_cpu=1)."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    p_val = two_scenario_p_value(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,
        n_cpu=1,
        verbose=False,
    )

    assert 0.0 <= p_val <= 1.0


@pytest.mark.run(order=2)
def test_auto_parallel_mode():
    """Test automatic parallel mode (n_cpu>1)."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    p_val = two_scenario_p_value(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,
        n_cpu=2,
        base_seed=123,
        verbose=False,
    )

    assert 0.0 <= p_val <= 1.0


@pytest.mark.run(order=3)
def test_reproducibility_auto_parallel():
    """Test reproducibility in auto parallel mode."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    p_val1 = _two_scenario_p_value_parallel(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,
        n_workers=2,
        base_seed=456,
    )

    p_val2 = _two_scenario_p_value_parallel(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,
        n_workers=4,
        base_seed=456,
    )
    assert abs(p_val1 - p_val2) < 1.e-6


@pytest.mark.run(order=4)
def test_invalid_n_cpu():
    """Test with invalid n_cpu values."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    # Test negative n_cpu
    with pytest.raises(ValueError, match="n_cpu must be at least 1"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params={'size': 3},
            n_cpu=0
        )

    # Test non-integer n_cpu
    with pytest.raises(TypeError, match="n_cpu must be an integer"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params={'size': 3},
            n_cpu=2.5
        )


@pytest.mark.run(order=5)
def test_invalid_generator():
    """Test with invalid generator function."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    with pytest.raises(TypeError, match="x_alt_generator must be callable"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator="not_callable",  # Invalid
            generator_params={'size': 3},
        )


@pytest.mark.run(order=6)
def test_invalid_generator_params():
    """Test with invalid generator parameters."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    with pytest.raises(TypeError, match="generator_params must be a dictionary"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params=['not', 'a', 'dict'],  # Invalid
        )


@pytest.mark.run(order=7)
def test_invalid_simulations():
    """Test with invalid simulation parameters."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    # Non-integer n_simulations
    with pytest.raises(TypeError, match="n_simulations must be an integer"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params={'size': 3},
            n_simulations=100.5,
        )

    # Zero simulations
    with pytest.raises(ValueError, match="n_simulations must be positive"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params={'size': 3},
            n_simulations=0,
        )


@pytest.mark.run(order=8)
def test_empty_arrays():
    """Test with empty input arrays."""
    x = np.array([])
    x_alt = np.array([])
    y = np.array([])

    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        two_scenario_p_value(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params={'size': 0},
            n_cpu=1,
        )


@pytest.mark.run(order=9)
def test_mode_selection_consistency():
    """Test that n_cpu=1 and n_cpu>1 give similar results."""
    x = np.array([0, 1, 2, 3, 4])
    x_alt = np.array([0, 0.5, 1, 1.5, 2])
    y = np.array([0, 1, 2, 3, 4])

    # Use same seed for reproducible generator
    def x_alt_generator(rng, size=5):
        return rng.normal(0, 1, size)

    # Sequential mode
    p_val_seq = two_scenario_p_value(
        x,
        x_alt,
        y,
        x_alt_generator=x_alt_generator,
        generator_params={'size': 5},
        n_simulations=50,
        n_cpu=1,
        base_seed=789,
     )
    print()
    print(" ************************************ ")
    print()
    # Parallel mode
    p_val_par = two_scenario_p_value(
        x,
        x_alt,
        y,
        x_alt_generator=x_alt_generator,
        generator_params={'size': 5},
        n_simulations=50,
        n_cpu=2,
        base_seed=789,
    )

    # Results should be close (within statistical variation)
    # We can't expect exact equality due to different RNG handling
    assert 0.0 <= p_val_seq <= 1.0
    assert 0.0 <= p_val_par <= 1.0
    assert abs(p_val_seq - p_val_par) < 1.e-6
