"""Markov-chain utilities used by experiments and metrics."""

from __future__ import annotations

import numpy as np


def is_stochastic_matrix(P: np.ndarray, tol: float = 1e-9) -> bool:
    """Return True when ``P`` is a finite row-stochastic square matrix."""
    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.all(np.isfinite(matrix)):
        return False
    if np.any(matrix < -tol):
        return False
    row_sums = matrix.sum(axis=1)
    return bool(np.all(np.abs(row_sums - 1.0) <= tol))


def normalize_rows(P: np.ndarray) -> np.ndarray:
    """Return a copy of ``P`` with each row normalized to sum to one.

    Rows with near-zero total mass are replaced by a uniform distribution.
    Negative values are clipped to zero before normalization.
    """
    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("P must be a 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P must contain only finite values")

    normalized = np.clip(matrix.copy(), 0.0, None)
    n_cols = normalized.shape[1]
    if n_cols == 0:
        raise ValueError("P must have at least one column")

    uniform_row = np.full(n_cols, 1.0 / n_cols, dtype=float)
    row_sums = normalized.sum(axis=1)
    tiny = 1e-15

    for i, row_sum in enumerate(row_sums):
        if row_sum <= tiny:
            normalized[i] = uniform_row
        else:
            normalized[i] /= row_sum
    return normalized


def make_ergodic(P: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Mix ``P`` with a uniform kernel to make it strictly positive for eps>0."""
    if eps < 0.0 or eps > 1.0:
        raise ValueError("eps must be in [0, 1]")

    base = normalize_rows(P)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("P must be a square matrix")

    n = base.shape[0]
    uniform_kernel = np.full((n, n), 1.0 / n, dtype=float)
    mixed = (1.0 - eps) * base + eps * uniform_kernel
    return normalize_rows(mixed)


def stationary_dist(
    P: np.ndarray, tol: float = 1e-12, max_iter: int = 1_000_000
) -> np.ndarray:
    """Compute stationary distribution ``pi`` for row-stochastic kernel ``P``."""
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P must contain only finite values")

    kernel = normalize_rows(matrix)
    n = kernel.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)

    converged = False
    for _ in range(max_iter):
        next_pi = pi @ kernel
        if np.linalg.norm(next_pi - pi, ord=1) <= tol:
            pi = next_pi
            converged = True
            break
        pi = next_pi

    if not converged:
        eigenvalues, eigenvectors = np.linalg.eig(kernel.T)
        idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        pi = np.real(eigenvectors[:, idx])
        pi = np.clip(pi, 0.0, None)

    pi = np.real(pi)
    pi = np.clip(pi, 0.0, None)
    total = float(np.sum(pi))
    if total <= 0.0 or not np.isfinite(total):
        pi = np.full(n, 1.0 / n, dtype=float)
    else:
        pi /= total
    return pi


def kernel_power(P: np.ndarray, tau: int) -> np.ndarray:
    """Return ``P`` to power ``tau`` using repeated squaring."""
    if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)):
        raise TypeError("tau must be an integer")
    if tau < 0:
        raise ValueError("tau must be >= 0")

    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be a square matrix")

    n = matrix.shape[0]
    if tau == 0:
        return np.eye(n, dtype=matrix.dtype)
    if tau == 1:
        return matrix.copy()

    result = np.eye(n, dtype=matrix.dtype)
    base = matrix.copy()
    exponent = int(tau)

    while exponent > 0:
        if exponent & 1:
            result = result @ base
        base = base @ base
        exponent >>= 1
    return result


def simulate_chain(
    P: np.ndarray, T: int, x0: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate states ``[x0, x1, ..., x_{T-1}]`` from transition kernel ``P``."""
    if isinstance(T, bool) or not isinstance(T, (int, np.integer)):
        raise TypeError("T must be an integer")
    if T < 1:
        raise ValueError("T must be >= 1")
    if isinstance(x0, bool) or not isinstance(x0, (int, np.integer)):
        raise TypeError("x0 must be an integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P must contain only finite values")

    n = matrix.shape[0]
    if not 0 <= int(x0) < n:
        raise ValueError("x0 must be in [0, n)")

    if is_stochastic_matrix(matrix):
        kernel = matrix
    else:
        # Robustness for lightly malformed kernels in experiment code.
        kernel = normalize_rows(matrix)

    states = np.empty(int(T), dtype=np.int64)
    states[0] = int(x0)
    for t in range(1, int(T)):
        states[t] = int(rng.choice(n, p=kernel[states[t - 1]]))
    return states
