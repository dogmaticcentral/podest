# test_4_two_scenario_pvalue.py
import pytest
import numpy as np
from podest.main import (
    _two_scenario_p_value_sequential,
    _two_scenario_p_value_parallel,
)

def simple_generator(size=3, rng=None):
    """Simple generator for testing."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, 1, size)


@pytest.mark.run(order=1)
def test_p_value_sequential():
    """Test sequential p-value calculation."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    p_val = _two_scenario_p_value_sequential(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,  # Small number for testing
        verbose=False,
    )

    assert 0.0 <= p_val <= 1.0


@pytest.mark.run(order=2)
def test_p_value_parallel():
    """Test parallel p-value calculation."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    p_val = _two_scenario_p_value_parallel(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,  # Small number for testing
        verbose=False,
    )

    assert 0.0 <= p_val <= 1.0


@pytest.mark.run(order=3)
def test_reproducibility_parallel():
    """Test that parallel computation is reproducible."""
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
        base_seed=123,
    )

    p_val2 = _two_scenario_p_value_parallel(
        x,
        x_alt,
        y,
        x_alt_generator=simple_generator,
        generator_params={'size': 3},
        n_simulations=10,
        base_seed=123,
    )

    assert p_val1 == p_val2


@pytest.mark.run(order=4)
def test_invalid_n_simulations():
    """Test with invalid number of simulations."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError, match="must be positive"):
        _two_scenario_p_value_parallel(
            x,
            x_alt,
            y,
            x_alt_generator=simple_generator,
            generator_params={'size': 3},
            n_simulations=0,
        )