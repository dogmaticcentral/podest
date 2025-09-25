# test_validate_generator_rng.py

import pytest
import inspect
import numpy as np
from typing import Callable, Dict, Any
from podest.main import _validate_generator_rng_compatibility
# -------------------------------------------------------------------
# Example x_alt_generator implementations for testing
# -------------------------------------------------------------------

def good_generator(size: int, rng: np.random.Generator):
    """A valid generator: uses provided rng correctly."""
    return rng.normal(size=size)

def missing_rng(size: int):
    """Invalid generator: does not accept rng."""
    return np.random.normal(size=size)

def bad_rng_callable(size: int, rng: np.random.Generator):
    """Invalid: treats rng as a callable instead of a Generator."""
    return rng(size)  # rng.__call__ doesn't exist -> TypeError

def bad_rng_method(size: int, rng: np.random.Generator):
    """Invalid: tries to use a non-existent method of rng."""
    return rng.not_a_method(size=size)  # AttributeError


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_good_generator_passes():
    params = {"size": 5}
    # Should not raise any error
    _validate_generator_rng_compatibility(good_generator, params)

def test_missing_rng_argument_raises_typeerror():
    params = {"size": 5}
    with pytest.raises(TypeError, match="must accept a keyword argument named 'rng'"):
        _validate_generator_rng_compatibility(missing_rng, params)

def test_bad_rng_callable_raises_typeerror():
    params = {"size": 5}
    with pytest.raises(TypeError, match="could not be called successfully"):
        _validate_generator_rng_compatibility(bad_rng_callable, params)

def test_bad_rng_method_raises_typeerror():
    params = {"size": 5}
    with pytest.raises(TypeError, match="incompatible way"):
        _validate_generator_rng_compatibility(bad_rng_method, params)

def test_good_generator_output_is_deterministic_with_fixed_seed():
    """Even though this is validation, check reproducibility with fixed seed 999."""
    params = {"size": 3}
    _validate_generator_rng_compatibility(good_generator, params)
    # Run generator directly with same seed to check determinism
    rng1 = np.random.default_rng(999)
    res1 = good_generator(size=3, rng=rng1)
    rng2 = np.random.default_rng(999)
    res2 = good_generator(size=3, rng=rng2)
    assert np.allclose(res1, res2), "Results should be identical with fixed seed"
