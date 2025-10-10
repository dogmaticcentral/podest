#' Point density statistical evaluation for 2D graphs.
#'
#' This module provides functions for computing point density statistics and performing
#' statistical comparisons between different point distributions.

library(FNN)  # For KNN/neighbor searches (alternative to scipy.spatial.cKDTree)

#' Normalize array using z-score standardization.
#'
#' @param x Numeric vector to normalize.
#' @return Z-score normalized vector.
#' @export
zscore_normalize <- function(x) {
  x <- as.numeric(x)
  if (length(x) == 0) {
    return(x)
  }

  std_val <- sd(x)
  if (std_val == 0) {
    stop("Cannot normalize array with zero standard deviation")
  }

  return((x - mean(x)) / std_val)
}

#' Compute the mean spatial density of 2D points using a Gaussian kernel.
#'
#' For each point, the density is estimated by summing exponentially weighted distances
#' to its neighbors within a fixed radius. The weights decay with distance (Gaussian kernel),
#' giving higher influence to closer points.
#'
#' @param x Numeric vector of x-coordinates of the points.
#' @param y Numeric vector of y-coordinates of the points (must match length of x).
#' @param radius Search radius for neighbor detection. Default is 10.0.
#' @param kernel_scale Scale parameter for Gaussian kernel weighting. Default is 0.5.
#' @return Mean density of all points. Higher values indicate tighter clustering.
#' @export
#'
#' @examples
#' x <- runif(100)
#' y <- runif(100)
#' density <- mean_point_density(x, y)
#' cat("Mean point density:", density, "\n")
mean_point_density <- function(x, y, radius = 10.0, kernel_scale = 0.5) {
  x <- as.numeric(x)
  y <- as.numeric(y)

  if (length(x) != length(y)) {
    stop("x and y arrays must have the same length")
  }
  if (length(x) == 0) {
    stop("Input arrays cannot be empty")
  }

  # Organize data into a matrix (n points x 2 dimensions)
  points <- cbind(x, y)
  n_points <- nrow(points)
  densities <- numeric(n_points)

  # For each point, find neighbors within radius and compute density
  for (i in 1:n_points) {
    point <- points[i, , drop = FALSE]

    # Calculate distances to all other points
    distances <- sqrt(rowSums((points - matrix(point, nrow = n_points, ncol = 2, byrow = TRUE))^2))

    # Find neighbors within radius
    neighbor_indices <- which(distances <= radius)

    if (length(neighbor_indices) == 0) {
      densities[i] <- 0
      next
    }

    # Calculate weights using Gaussian kernel
    neighbor_distances <- distances[neighbor_indices]
    weights <- exp(-(neighbor_distances / kernel_scale)^2)
    densities[i] <- sum(weights)
  }

  return(mean(densities))
}

#' Compare mean point densities between two scenarios with normalized coordinates.
#'
#' @param x X-coordinates for the first scenario.
#' @param x_alt X-coordinates for the alternative scenario.
#' @param y Y-coordinates (shared between scenarios).
#' @return List with mean point densities for original and alternative scenarios.
#' @export
two_scenario_mpd_comparison <- function(x, x_alt, y) {
  x <- as.numeric(x)
  x_alt <- as.numeric(x_alt)
  y <- as.numeric(y)

  if (length(x) != length(y)) {
    stop("x and y arrays must have the same length")
  }
  if (length(x_alt) != length(y)) {
    stop("x_alt and y arrays must have the same length")
  }

  # Normalize y-coordinates
  scale_y <- sd(y)
  if (scale_y == 0) {
    stop("Y coordinates have zero standard deviation")
  }
  y_scaled <- (y - mean(y)) / scale_y

  # Normalize x-coordinates using global statistics
  global_x <- c(x, x_alt)
  scale_x <- sd(global_x)
  if (scale_x == 0) {
    stop("Combined X coordinates have zero standard deviation")
  }
  global_mean_x <- mean(global_x)

  x_scaled <- (x - global_mean_x) / scale_x
  x_alt_scaled <- (x_alt - global_mean_x) / scale_x

  mpd <- mean_point_density(x_scaled, y_scaled)
  mpd_alt <- mean_point_density(x_alt_scaled, y_scaled)

  return(c(mpd, mpd_alt))
}

#' Perform a single simulation with an independent RNG instance.
#'
#' @param x Input x data for comparison.
#' @param y Input y data for comparison.
#' @param x_alt_generator Function to generate alternative scenario data.
#' @param generator_params List of parameters for the generator function.
#' @param ref_difference Reference difference to compare against.
#' @param seed Random seed for this simulation.
#' @return Boolean indicating whether the simulated difference exceeds the reference.
simulation_task <- function(x, y, x_alt_generator, generator_params, ref_difference, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Generate alternative scenario
  x_alt_simulated <- do.call(x_alt_generator, generator_params)

  # Compare densities
  densities <- two_scenario_mpd_comparison(x, x_alt_simulated, y)
  mean_density <- densities[1]
  mean_density_simulated <- densities[2]

  diff <- mean_density_simulated - mean_density
  return(diff > ref_difference)
}

#' Calculate p-value using parallel simulations (STUB IMPLEMENTATION).
#'
#' Note: This is a stub implementation. In a full implementation, you would use
#' R's parallel processing capabilities like parallel::mclapply() or future.apply.
#'
#' @param x X-coordinates for the original scenario.
#' @param x_alt X-coordinates for the alternative scenario.
#' @param y Y-coordinates (shared between scenarios).
#' @param x_alt_generator Function to generate alternative scenario data.
#' @param generator_params List of parameters for the generator function.
#' @param n_simulations Number of simulations to run. Default is 1000.
#' @param n_workers Number of worker processes (ignored in stub). Default is NULL.
#' @param base_seed Base seed for reproducible results. Default is 42.
#' @param verbose Whether to print progress information. Default is FALSE.
#' @return P-value as proportion of simulations exceeding reference difference.
two_scenario_p_value_parallel <- function(x, x_alt, y, x_alt_generator, generator_params,
                                        n_simulations = 1000, n_workers = NULL,
                                        base_seed = 42, verbose = FALSE) {
  if (n_simulations <= 0) {
    stop("Number of simulations must be positive")
  }

  if (!is.null(n_workers) && (!is.numeric(n_workers) || n_workers <= 0)) {
    stop("Number of workers must be NULL or a positive integer")
  }

  # Calculate reference difference once
  densities <- two_scenario_mpd_comparison(x, x_alt, y)
  mean_density <- densities[1]
  mean_density_alt <- densities[2]
  ref_difference <- mean_density_alt - mean_density

  if (verbose) {
    cat("STUB: Parallel implementation not yet implemented.\n")
    cat("Falling back to sequential processing...\n")
  }

  # STUB: For now, fall back to sequential implementation
  return(two_scenario_p_value_sequential(x, x_alt, y, x_alt_generator, generator_params,
                                       n_simulations, base_seed, verbose))
}

#' Calculate p-value using sequential simulations.
#'
#' @param x X-coordinates for the original scenario.
#' @param x_alt X-coordinates for the alternative scenario.
#' @param y Y-coordinates (shared between scenarios).
#' @param x_alt_generator Function to generate alternative scenario data.
#' @param generator_params List of parameters for the generator function.
#' @param n_simulations Number of simulations to run. Default is 1000.
#' @param base_seed Base seed for reproducible results. Default is 42.
#' @param verbose Whether to print debug information. Default is FALSE.
#' @return P-value as proportion of simulations exceeding reference difference.
#' @export
two_scenario_p_value_sequential <- function(x, x_alt, y, x_alt_generator, generator_params,
                                          n_simulations = 1000, base_seed = 42, verbose = FALSE) {
  if (n_simulations <= 0) {
    stop("Number of simulations must be positive")
  }

  # Calculate reference difference
  densities <- two_scenario_mpd_comparison(x, x_alt, y)
  mean_density <- densities[1]
  mean_density_alt <- densities[2]
  ref_difference <- mean_density_alt - mean_density

  # Set base seed for reproducibility
  set.seed(base_seed)

  # Run simulations
  results <- logical(n_simulations)
  for (i in 1:n_simulations) {
    # Use different seeds for each simulation while maintaining reproducibility
    sim_seed <- base_seed + i
    results[i] <- simulation_task(x, y, x_alt_generator, generator_params, ref_difference, sim_seed)
  }

  number_of_bigger_density_occurrences <- sum(results)

  if (verbose) {
    cat(sprintf("Reference Difference: %.6f\n", ref_difference))
    cat(sprintf("Bigger Density Occurrences: %d\n", number_of_bigger_density_occurrences))
  }

  return(number_of_bigger_density_occurrences / n_simulations)
}

#' Validate that generator function is compatible with required interface.
#'
#' @param x_alt_generator The generator function to test.
#' @param generator_params List of parameters for the generator function.
#' @return NULL if validation passes, stops execution with error if validation fails.
validate_generator_compatibility <- function(x_alt_generator, generator_params) {
  if (!is.function(x_alt_generator)) {
    stop("x_alt_generator must be a function")
  }

  if (!is.list(generator_params)) {
    stop("generator_params must be a list")
  }

  # Try to call the generator function with test parameters
  tryCatch({
    # Test call with provided parameters
    test_result <- do.call(x_alt_generator, generator_params)

    # Check that result is numeric
    if (!is.numeric(test_result)) {
      stop("Generator function must return numeric values")
    }
  }, error = function(e) {
    stop(paste("Validation Error: Generator function failed during test call.",
               "Original error:", e$message))
  })

  return(invisible(NULL))
}

#' Perform sanity checks on input parameters.
#'
#' @param n_cpu Number of CPU cores to use.
#' @param n_simulations Number of simulations.
#' @param x_alt_generator Generator function.
#' @param generator_params Parameters for generator.
#' @param base_seed Base seed value.
#' @param verbose Verbose flag.
#' @return NULL if all checks pass, stops execution with error otherwise.
sanity_checks <- function(n_cpu, n_simulations, x_alt_generator, generator_params, base_seed, verbose) {
  if (!is.numeric(n_cpu) || length(n_cpu) != 1 || n_cpu != as.integer(n_cpu)) {
    stop(paste("n_cpu must be an integer, got", class(n_cpu)[1]))
  }

  if (n_cpu < 1) {
    stop(paste("n_cpu must be at least 1, got", n_cpu))
  }

  if (!is.numeric(n_simulations) || length(n_simulations) != 1 || n_simulations != as.integer(n_simulations)) {
    stop(paste("n_simulations must be an integer, got", class(n_simulations)[1]))
  }

  if (n_simulations <= 0) {
    stop(paste("n_simulations must be positive, got", n_simulations))
  }

  if (!is.function(x_alt_generator)) {
    stop(paste("x_alt_generator must be callable, got", class(x_alt_generator)[1]))
  }

  if (!is.list(generator_params)) {
    stop(paste("generator_params must be a list, got", class(generator_params)[1]))
  }

  if (!is.numeric(base_seed) || length(base_seed) != 1 || base_seed != as.integer(base_seed)) {
    stop(paste("base_seed must be an integer, got", class(base_seed)[1]))
  }

  if (!is.logical(verbose) || length(verbose) != 1) {
    stop(paste("verbose must be a boolean, got", class(verbose)[1]))
  }

  # Validate generator compatibility
  validate_generator_compatibility(x_alt_generator, generator_params)

  return(invisible(NULL))
}

#' Calculate p-value with automatic serial/parallel execution based on CPU count.
#'
#' This function serves as a unified entry point that automatically chooses between
#' sequential and parallel execution based on the n_cpu parameter. For single CPU
#' (n_cpu=1), it uses the optimized sequential implementation. For multiple CPUs
#' (n_cpu>1), it leverages parallel processing for improved performance.
#'
#' @param x X-coordinates for the original scenario.
#' @param x_alt X-coordinates for the alternative scenario.
#' @param y Y-coordinates (shared between scenarios).
#' @param x_alt_generator Function to generate alternative scenario data.
#' @param generator_params List of parameters for the generator function.
#' @param n_simulations Number of Monte Carlo simulations. Default is 1000.
#' @param n_cpu Number of CPU cores to use. Default is 1.
#' @param base_seed Base seed for reproducible results. Default is 42.
#' @param verbose Whether to print progress information. Default is FALSE.
#' @return P-value as proportion of simulations exceeding reference difference.
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple generator function
#' random_x_generator <- function(size = 100, noise = 1.0) {
#'   rnorm(size, mean = 0, sd = noise)
#' }
#'
#' # Sample data
#' x1 <- rnorm(50, 0, 1)
#' x2 <- rnorm(50, 0, 2)  # More spread out
#' y <- rnorm(50, 0, 1)
#'
#' # Sequential processing
#' p_val_seq <- two_scenario_p_value(
#'   x1, x2, y,
#'   x_alt_generator = random_x_generator,
#'   generator_params = list(size = length(x1), noise = 1.5),
#'   n_simulations = 1000,
#'   n_cpu = 1
#' )
#'
#' # Parallel processing (4 cores) - currently falls back to sequential
#' p_val_par <- two_scenario_p_value(
#'   x1, x2, y,
#'   x_alt_generator = random_x_generator,
#'   generator_params = list(size = length(x1), noise = 1.5),
#'   n_simulations = 1000,
#'   n_cpu = 4,
#'   verbose = TRUE
#' )
#' }
two_scenario_p_value <- function(x, x_alt, y, x_alt_generator, generator_params,
                                n_simulations = 1000, n_cpu = 1, base_seed = 42, verbose = FALSE) {
  # Input validation
  sanity_checks(n_cpu, n_simulations, x_alt_generator, generator_params, base_seed, verbose)

  # Validate array inputs early
  tryCatch({
    x <- as.numeric(x)
    x_alt <- as.numeric(x_alt)
    y <- as.numeric(y)
  }, error = function(e) {
    stop(paste("Failed to convert input arrays to numeric:", e$message))
  })

  if (length(x) == 0 || length(x_alt) == 0 || length(y) == 0) {
    stop("Input arrays cannot be empty")
  }

  # Log execution mode for transparency
  if (verbose) {
    mode_str <- if (n_cpu == 1) "sequential" else paste0("parallel (", n_cpu, " CPUs)")
    cat(sprintf("Running %d simulations in %s mode...\n", n_simulations, mode_str))
  }

  # Route to appropriate implementation based on CPU count
  if (n_cpu == 1) {
    # Sequential execution - optimized for single core
    return(two_scenario_p_value_sequential(
      x = x,
      x_alt = x_alt,
      y = y,
      x_alt_generator = x_alt_generator,
      generator_params = generator_params,
      n_simulations = n_simulations,
      base_seed = base_seed,
      verbose = verbose
    ))
  } else {
    # Parallel execution - leverage multiple cores (currently stub)
    return(two_scenario_p_value_parallel(
      x = x,
      x_alt = x_alt,
      y = y,
      x_alt_generator = x_alt_generator,
      generator_params = generator_params,
      n_simulations = n_simulations,
      n_workers = n_cpu,
      base_seed = base_seed,
      verbose = verbose
    ))
  }
}