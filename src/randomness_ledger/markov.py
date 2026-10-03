"""Markov-chain utilities used by experiments and metrics."""

from __future__ import annotations

import numpy as np


def validate_probability_vector(vec: np.ndarray, n: int, name: str) -> np.ndarray:
    """Check a probability law; allow only roundoff in its total mass."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (n,) or n < 1:
        raise ValueError(f"{name} must have shape ({n},) on a nonempty space")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(arr.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must sum to 1")
    return arr / arr.sum()


def validate_kernel(P: np.ndarray) -> np.ndarray:
    """Check a transition law without silently repairing invalid dynamics.

    Normalization removes only accepted floating-point row-sum roundoff.
    Use ``normalize_rows`` explicitly when constructing a kernel from weights.
    """
    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("P must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("P must be finite and nonnegative")
    totals = matrix.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("P rows must sum to 1")
    return matrix / totals


def validate_stationary(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Validate the supplied stationary law, including its invariance."""
    kernel = validate_kernel(P)
    law = validate_probability_vector(pi, kernel.shape[0], "pi_stationary")
    if np.linalg.norm(law @ kernel - law, ord=1) > 1e-10:
        raise ValueError("pi_stationary must be stationary for P")
    return law


def is_stochastic_matrix(P: np.ndarray, tol: float = 1e-9) -> bool:
    """Return True when ``P`` is a finite row-stochastic square matrix."""
    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return False
    if not np.all(np.isfinite(matrix)):
        return False
    if np.any(matrix < -tol):
        return False
    row_sums = matrix.sum(axis=1)
    return bool(np.all(np.abs(row_sums - 1.0) <= tol))


def normalize_rows(P: np.ndarray) -> np.ndarray:
    """Return a copy of ``P`` with each row normalized to sum to one.

    Rows with zero total mass are replaced by a uniform distribution.
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
    with np.errstate(over="ignore"):
        row_sums = normalized.sum(axis=1)

    for i, row_sum in enumerate(row_sums):
        if row_sum == 0.0:
            normalized[i] = uniform_row
        elif np.isfinite(row_sum):
            normalized[i] /= row_sum
        else:
            scaled = normalized[i] / normalized[i].max()
            normalized[i] = scaled / scaled.sum()
    return normalized


def make_ergodic(P: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Mix ``P`` with a uniform kernel to make it strictly positive for eps>0."""
    if not np.isfinite(eps) or eps < 0.0 or eps > 1.0:
        raise ValueError("eps must be in [0, 1]")

    base = normalize_rows(P)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("P must be a square matrix")

    n = base.shape[0]
    uniform_kernel = np.full((n, n), 1.0 / n, dtype=float)
    mixed = (1.0 - eps) * base + eps * uniform_kernel
    if eps > 0.0 and np.any(mixed <= 0.0):
        raise ValueError("eps is too small to represent a strictly positive mixed kernel")
    return normalize_rows(mixed)


def stationary_dist(
    P: np.ndarray, tol: float = 1e-12, max_iter: int = 1_000_000
) -> np.ndarray:
    """Compute a stationary law and check its residual before returning.

    Reducible kernels can have several stationary laws; no uniqueness is claimed.
    A constrained linear-system fallback handles nonconvergence and periodicity.
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be > 0")

    kernel = validate_kernel(P)
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
        system = np.vstack((kernel.T - np.eye(n), np.ones((1, n))))
        rhs = np.concatenate((np.zeros(n), [1.0]))
        pi, *_ = np.linalg.lstsq(system, rhs, rcond=None)

    # Only remove floating-point negative noise, never replace a failed solve
    # by the uniform law (which need not be stationary).
    if not np.all(np.isfinite(pi)) or np.min(pi) < -1e-10:
        raise RuntimeError("stationary solve produced an invalid probability law")
    pi = np.clip(pi, 0.0, None)
    total = float(np.sum(pi))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("stationary solve produced zero or nonfinite mass")
    pi /= total
    if np.linalg.norm(pi @ kernel - pi, ord=1) > max(tol, 1e-10):
        raise RuntimeError("stationary solve failed its invariance check")
    return pi


def kernel_power(P: np.ndarray, tau: int) -> np.ndarray:
    """Return ``P`` to power ``tau`` using repeated squaring."""
    if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)):
        raise TypeError("tau must be an integer")
    if tau < 0:
        raise ValueError("tau must be >= 0")

    matrix = validate_kernel(P)

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

    kernel = validate_kernel(P)

    n = kernel.shape[0]
    if not 0 <= int(x0) < n:
        raise ValueError("x0 must be in [0, n)")

    states = np.empty(int(T), dtype=np.int64)
    states[0] = int(x0)
    for t in range(1, int(T)):
        states[t] = int(rng.choice(n, p=kernel[states[t - 1]]))
    return states
