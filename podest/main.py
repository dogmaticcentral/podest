"""
Point density statistical evaluation for 2D graphs.

This module provides functions for computing point density statistics and performing
statistical comparisons between different point distributions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree
import dask
from dask import delayed


def _zscore_normalize(x: ArrayLike) -> NDArray[np.float64]:
    """
    Normalize array using z-score standardization.

    Parameters
    ----------
    x : ArrayLike
        Input array to normalize.

    Returns
    -------
    NDArray[np.float64]
        Z-score normalized array.

    Raises
    ------
    ValueError
        If input array has zero standard deviation.
    """
    x_array = np.asarray(x, dtype=np.float64)
    if x_array.size == 0:
        return x_array

    std = x_array.std()
    if std == 0:
        raise ValueError("Cannot normalize array with zero standard deviation")

    return (x_array - x_array.mean()) / std


def mean_point_density(
        x: ArrayLike,
        y: ArrayLike,
        radius: float = 10.0,
        kernel_scale: float = 0.5
) -> float:
    """
    Compute the mean spatial density of 2D points using a Gaussian kernel.

    For each point, the density is estimated by summing exponentially weighted distances
    to its neighbors within a fixed radius. The weights decay with distance (Gaussian kernel),
    giving higher influence to closer points.

    Parameters
    ----------
    x : ArrayLike
        1D array of x-coordinates of the points.
    y : ArrayLike
        1D array of y-coordinates of the points (must match length of `x`).
    radius : float, default=10.0
        Search radius for neighbor detection.
    kernel_scale : float, default=0.5
        Scale parameter for Gaussian kernel weighting.

    Returns
    -------
    float
        Mean density of all points. Higher values indicate tighter clustering.

    Raises
    ------
    ValueError
        If x and y arrays have different lengths or are empty.

    Notes
    -----
    - Uses a KD-Tree for efficient neighbor searches from O(N²) to ~O(N log N).
    - Suitable for low-dimensional (2D/3D) data.

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> density = mean_point_density(x, y)
    >>> print(f"Mean point density: {density:.2f}")
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays must have the same length")
    if len(x_array) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Organize data into a 2D array (n points x 2 dimensions)
    points = np.column_stack((x_array, y_array))

    # Build KD-Tree for efficient neighbor searches
    tree = cKDTree(points)
    densities = np.zeros(points.shape[0], dtype=np.float64)

    for i, point in enumerate(points):
        # Find neighbors within radius
        neighbor_indices = tree.query_ball_point(point, r=radius)
        if len(neighbor_indices) == 0:
            continue

        distances = np.linalg.norm(points[neighbor_indices] - point, axis=1)
        weights = np.exp(-(distances / kernel_scale) ** 2)
        densities[i] = np.sum(weights)

    return float(densities.mean())


def two_scenario_mpd_comparison(
        x: ArrayLike,
        x_alt: ArrayLike,
        y: ArrayLike
) -> tuple[float, float]:
    """
    Compare mean point densities between two scenarios with normalized coordinates.

    Parameters
    ----------
    x : ArrayLike
        X-coordinates for the first scenario.
    x_alt : ArrayLike
        X-coordinates for the alternative scenario.
    y : ArrayLike
        Y-coordinates (shared between scenarios).

    Returns
    -------
    tuple[float, float]
        Mean point densities for (original scenario, alternative scenario).

    Raises
    ------
    ValueError
        If input arrays have inconsistent lengths or are empty.
    """
    x_array = np.asarray(x, dtype=np.float64)
    x_alt_array = np.asarray(x_alt, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays must have the same length")
    if len(x_alt_array) != len(y_array):
        raise ValueError("x_alt and y arrays must have the same length")

    # Normalize y-coordinates
    scale_y = np.std(y_array)
    if scale_y == 0:
        raise ValueError("Y coordinates have zero standard deviation")
    y_scaled = (y_array - np.mean(y_array)) / scale_y

    # Normalize x-coordinates using global statistics
    global_x = np.concatenate([x_array, x_alt_array])
    scale_x = np.std(global_x)
    if scale_x == 0:
        raise ValueError("Combined X coordinates have zero standard deviation")
    global_mean_x = np.mean(global_x)

    x_scaled = (x_array - global_mean_x) / scale_x
    x_alt_scaled = (x_alt_array - global_mean_x) / scale_x

    mpd = mean_point_density(x_scaled, y_scaled)
    mpd_alt = mean_point_density(x_alt_scaled, y_scaled)

    return mpd, mpd_alt


def _simulation_task(
        x: ArrayLike,
        y: ArrayLike,
        x_alt_generator: callable[..., ArrayLike],
        generator_params: dict[str, any],
        ref_difference: float,
        seed: int | None = None
) -> bool:
    """
    Perform a single simulation with an independent RNG instance.

    Parameters
    ----------
    x, y : ArrayLike
        Input data for comparison.
    x_alt_generator : Callable
        Function to generate alternative scenario data.
    generator_params : Dict[str, Any]
        Parameters for the generator function.
    ref_difference : float
        Reference difference to compare against.
    seed : Optional[int]
        Seed value to initialize an independent RNG stream.

    Returns
    -------
    bool
        Whether the simulated difference exceeds the reference.
    """
    # Create an independent RNG instance for this task
    rng = None if seed is None else np.random.default_rng(seed)
    modified_generator_params = generator_params.copy()
    modified_generator_params['rng'] = rng

    x_alt_simulated = x_alt_generator(**modified_generator_params)
    mean_density, mean_density_simulated = two_scenario_mpd_comparison(
        x, x_alt_simulated, y
    )
    diff = mean_density_simulated - mean_density

    return diff > ref_difference

# TODO - make number of cpus an argument
def _two_scenario_p_value_parallel(
        x: ArrayLike,
        x_alt: ArrayLike,
        y: ArrayLike,
        x_alt_generator: callable[..., ArrayLike],
        generator_params: dict[str, any],
        n_simulations: int = 1000,
        base_seed: int = 42,
        verbose: bool = False
) -> float:
    """
    Calculate p-value using parallel simulations with independent RNG streams.

    Parameters
    ----------
    x : ArrayLike
        X-coordinates for the original scenario.
    x_alt : ArrayLike
        X-coordinates for the alternative scenario.
    y : ArrayLike
        Y-coordinates (shared between scenarios).
    x_alt_generator : callable
        Function to generate alternative scenario data for simulations.
    generator_params : dict[str, any]
        Parameters for the generator function.
    n_simulations : int, default=1000
        Number of simulations to run.
    base_seed : int, default=42
        Base seed for reproducible independent RNG streams.
    verbose : bool, default=False
        Whether to print progress and debug information.

    Returns
    -------
    float
        P-value as the proportion of simulations exceeding reference difference.

    Raises
    ------
    ValueError
        If n_simulations is not positive.
    """
    if n_simulations <= 0:
        raise ValueError("Number of simulations must be positive")

    # Calculate reference difference once
    mean_density, mean_density_alt = two_scenario_mpd_comparison(x, x_alt, y)
    ref_difference = mean_density_alt - mean_density

    # Create independent seeds for each simulation
    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = seed_seq.spawn(n_simulations)

    # Create delayed tasks for parallel execution
    delayed_tasks = [
        delayed(_simulation_task)(
            x, y, x_alt_generator, generator_params, ref_difference,
            seed=int(child_seeds[i].generate_state(1)[0])
        )
        for i in range(n_simulations)
    ]

    # Compute results in parallel
    if verbose:
        try:
            from tqdm.auto import tqdm
            with tqdm(total=n_simulations, desc="Simulations") as pbar:
                results = dask.compute(*delayed_tasks, scheduler='processes')
                pbar.update(n_simulations)
        except ImportError:
            print("tqdm not available, running without progress bar...")
            results = dask.compute(*delayed_tasks, scheduler='processes')
    else:
        results = dask.compute(*delayed_tasks, scheduler='processes')

    # Count occurrences where difference exceeds reference
    number_of_bigger_density_occurrences = sum(results)

    if verbose:
        print(f"Reference Difference: {ref_difference:.6f}")
        print(f"Bigger Density Occurrences: {number_of_bigger_density_occurrences}")

    return number_of_bigger_density_occurrences / n_simulations

def _two_scenario_p_value_sequential(
        x: ArrayLike,
        x_alt: ArrayLike,
        y: ArrayLike,
        x_alt_generator: callable[..., ArrayLike],
        generator_params: dict[str, any],
        n_simulations: int = 1000,
        verbose: bool = False
) -> float:
    """
    Calculate p-value using sequential simulations.

    Parameters
    ----------
    x : ArrayLike
        X-coordinates for the original scenario.
    x_alt : ArrayLike
        X-coordinates for the alternative scenario.
    y : ArrayLike
        Y-coordinates (shared between scenarios).
    x_alt_generator : callable
        Function to generate alternative scenario data for simulations.
    generator_params : dict[str, any]
        Parameters for the generator function.
    n_simulations : int, default=1000
        Number of simulations to run.
    verbose : bool, default=False
        Whether to print debug information.

    Returns
    -------
    float
        P-value as the proportion of simulations exceeding reference difference.

    Raises
    ------
    ValueError
        If n_simulations is not positive.
    """
    if n_simulations <= 0:
        raise ValueError("Number of simulations must be positive")

    mean_density, mean_density_alt = two_scenario_mpd_comparison(x, x_alt, y)
    ref_difference = mean_density_alt - mean_density

    results = [
        _simulation_task(x, y, x_alt_generator, generator_params, ref_difference, None)
        for _ in range(n_simulations)
    ]
    number_of_bigger_density_occurrences = sum(results)

    if verbose:
        print(f"Reference Difference: {ref_difference:.6f}")
        print(f"Bigger Density Occurrences: {number_of_bigger_density_occurrences}")

    return number_of_bigger_density_occurrences / n_simulations

def two_scenario_p_value(
        x: ArrayLike,
        x_alt: ArrayLike,
        y: ArrayLike,
        x_alt_generator: callable[..., ArrayLike],
        generator_params: dict[str, any],
        n_simulations: int = 1000,
        n_cpu: int = 1,
        base_seed: int = 42,
        verbose: bool = False
) -> float:
    """
    Calculate p-value with automatic serial/parallel execution based on CPU count.

    This function serves as a unified entry point that automatically chooses between
    sequential and parallel execution based on the n_cpu parameter. For single CPU
    (n_cpu=1), it uses the optimized sequential implementation. For multiple CPUs
    (n_cpu>1), it leverages parallel processing for improved performance.

    Parameters
    ----------
    x : ArrayLike
        X-coordinates for the original scenario.
    x_alt : ArrayLike
        X-coordinates for the alternative scenario.
    y : ArrayLike
        Y-coordinates (shared between scenarios).
    x_alt_generator : callable
        Function to generate alternative scenario data for simulations.
        Must accept keyword arguments including 'rng' for random number generator.
    generator_params : dict[str, any]
        Parameters for the generator function. The function will add 'rng' parameter
        automatically for reproducible random number generation.
    n_simulations : int, default=1000
        Number of Monte Carlo simulations to run. Higher values provide more
        accurate p-value estimates but increase computation time.
    n_cpu : int, default=1
        Number of CPU cores to use for computation:
        - n_cpu=1: Use sequential processing (optimized for single-core)
        - n_cpu>1: Use parallel processing with Dask (utilizes multiple cores)
        - n_cpu<1: Invalid, will raise ValueError
    base_seed : int, default=42
        Base seed for reproducible random number generation. Only used in
        parallel mode to ensure reproducible results across runs.
    verbose : bool, default=False
        Whether to print progress information and diagnostic output.
        In parallel mode, shows progress bar if tqdm is available.

    Returns
    -------
    float
        P-value as the proportion of simulations where the generated alternative
        scenario has higher density difference than the observed difference.
        Values range from 0.0 to 1.0, where smaller values indicate stronger
        evidence against the null hypothesis.

    Raises
    ------
    ValueError
        If n_cpu < 1, n_simulations <= 0, or if input arrays have inconsistent
        dimensions.
    TypeError
        If x_alt_generator is not callable or generator_params is not a dictionary.

    Notes
    -----
    - Sequential mode (n_cpu=1) has lower overhead and is optimal for smaller
      datasets or when system resources are limited.
    - Parallel mode (n_cpu>1) provides speedup for computationally intensive
      simulations but has higher memory usage and startup overhead.
    - The parallel implementation uses independent random number streams to
      ensure statistical validity and reproducibility.
    - Progress tracking in parallel mode requires optional 'tqdm' dependency.

    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Simple generator function
    >>> def random_x_generator(size=100, noise=1.0, rng=None):
    ...     if rng is None:
    ...         rng = np.random.default_rng()
    ...     return rng.normal(0, noise, size)
    >>> 
    >>> # Sample data
    >>> x1 = np.random.normal(0, 1, 50)
    >>> x2 = np.random.normal(0, 2, 50)  # More spread out
    >>> y = np.random.normal(0, 1, 50)
    >>> 
    >>> # Sequential processing
    >>> p_val_seq = two_scenario_p_value(
    ...     x1, x2, y,
    ...     x_alt_generator=random_x_generator,
    ...     generator_params={'size': len(x1), 'noise': 1.5},
    ...     n_simulations=1000,
    ...     n_cpu=1
    ... )
    >>> 
    >>> # Parallel processing (4 cores)
    >>> p_val_par = two_scenario_p_value(
    ...     x1, x2, y,
    ...     x_alt_generator=random_x_generator,
    ...     generator_params={'size': len(x1), 'noise': 1.5},
    ...     n_simulations=1000,
    ...     n_cpu=4,
    ...     verbose=True
    ... )

    See Also
    --------
    two_scenario_p_value : Sequential implementation
    two_scenario_p_value_parallel : Parallel implementation
    mean_point_density : Core density calculation function
    """
    # Input validation with descriptive error messages
    if not isinstance(n_cpu, int):
        raise TypeError(f"n_cpu must be an integer, got {type(n_cpu).__name__}")

    if n_cpu < 1:
        raise ValueError(f"n_cpu must be at least 1, got {n_cpu}")

    if not isinstance(n_simulations, int):
        raise TypeError(f"n_simulations must be an integer, got {type(n_simulations).__name__}")

    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")

    if not callable(x_alt_generator):
        raise TypeError(f"x_alt_generator must be callable, got {type(x_alt_generator).__name__}")

    if not isinstance(generator_params, dict):
        raise TypeError(f"generator_params must be a dictionary, got {type(generator_params).__name__}")

    if not isinstance(base_seed, int):
        raise TypeError(f"base_seed must be an integer, got {type(base_seed).__name__}")

    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be a boolean, got {type(verbose).__name__}")

    # Validate array inputs early (let the underlying functions handle detailed validation)
    try:
        x_array = np.asarray(x)
        x_alt_array = np.asarray(x_alt)
        y_array = np.asarray(y)
    except Exception as e:
        raise ValueError(f"Failed to convert input arrays to numpy arrays: {e}")

    if x_array.size == 0 or x_alt_array.size == 0 or y_array.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Log execution mode for transparency
    if verbose:
        mode_str = "sequential" if n_cpu == 1 else f"parallel ({n_cpu} CPUs)"
        print(f"Running {n_simulations} simulations in {mode_str} mode...")

    # Route to appropriate implementation based on CPU count
    if n_cpu == 1:
        # Sequential execution - optimized for single core
        return _two_scenario_p_value_sequential(
            x=x,
            x_alt=x_alt,
            y=y,
            x_alt_generator=x_alt_generator,
            generator_params=generator_params,
            n_simulations=n_simulations,
            verbose=verbose
        )
    else:
        # Parallel execution - leverage multiple cores
        return _two_scenario_p_value_parallel(
            x=x,
            x_alt=x_alt,
            y=y,
            x_alt_generator=x_alt_generator,
            generator_params=generator_params,
            n_simulations=n_simulations,
            base_seed=base_seed,
            verbose=verbose
        )

