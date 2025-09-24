# test_3_two_scenario_mpd_comparison.py
import pytest
import numpy as np
from podest.main import two_scenario_mpd_comparison


@pytest.mark.run(order=1)
def test_basic_comparison():
    """Test basic two-scenario comparison."""
    x = np.array([0, 1, 2])
    x_alt = np.array([0, 0.5, 1])
    y = np.array([0, 1, 2])

    mpd1, mpd2 = two_scenario_mpd_comparison(x, x_alt, y)

    assert isinstance(mpd1, float)
    assert isinstance(mpd2, float)
    assert mpd1 > 0
    assert mpd2 > 0


@pytest.mark.run(order=2)
def test_mismatched_lengths():
    """Test with mismatched array lengths."""
    x = np.array([1, 2])
    x_alt = np.array([1, 2, 3])
    y = np.array([1, 2])

    with pytest.raises(ValueError, match="same length"):
        two_scenario_mpd_comparison(x, x_alt, y)


@pytest.mark.run(order=3)
def test_zero_std_y():
    """Test with zero standard deviation in y."""
    x = np.array([1, 2, 3])
    x_alt = np.array([4, 5, 6])
    y = np.array([1, 1, 1])  # Constant y values

    with pytest.raises(ValueError, match="zero standard deviation"):
        two_scenario_mpd_comparison(x, x_alt, y)


@pytest.mark.run(order=4)
def test_zero_std_combined_x():
    """Test with zero standard deviation in combined x."""
    x = np.array([1, 1, 1])
    x_alt = np.array([1, 1, 1])  # All x values the same
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="zero standard deviation"):
        two_scenario_mpd_comparison(x, x_alt, y)