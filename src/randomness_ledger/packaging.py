"""Partition and lifting utilities for Markov coarse-graining."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from randomness_ledger.markov import kernel_power, normalize_rows, stationary_dist


def _validate_pi_map(pi_map: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """Validate and normalize partition labels to contiguous integers 0..k-1."""
    labels = np.asarray(pi_map)
    if labels.ndim != 1:
        raise ValueError("pi_map must be a 1D array")
    if labels.size == 0:
        raise ValueError("pi_map must be non-empty")
    if not np.all(np.isfinite(labels)):
        raise ValueError("pi_map must contain only finite values")
    if not np.all(np.equal(labels, np.floor(labels))):
        raise ValueError("pi_map entries must be integer-valued")

    pi_int = labels.astype(np.int64)
    if np.any(pi_int < 0):
        raise ValueError("pi_map entries must be >= 0")

    k = int(pi_int.max()) + 1
    counts = np.bincount(pi_int, minlength=k)
    if np.any(counts == 0):
        raise ValueError("pi_map labels must be contiguous 0..k-1 with no missing labels")
    return pi_int, k, counts


def _validate_prob_vector(name: str, vec: np.ndarray, size: int) -> np.ndarray:
    """Validate a probability vector with fixed size."""
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < -1e-15):
        raise ValueError(f"{name} must be nonnegative")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"{name} must sum to 1")
    return arr


def pushforward_dist(mu_micro: np.ndarray, pi_map: np.ndarray, k: int) -> np.ndarray:
    """Push micro distribution to macro labels via partition map."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if int(k) < 1:
        raise ValueError("k must be >= 1")

    pi_int, inferred_k, _ = _validate_pi_map(pi_map)
    if int(k) != inferred_k:
        raise ValueError(f"k mismatch: expected {inferred_k}, got {int(k)}")

    mu = np.asarray(mu_micro, dtype=float)
    if mu.ndim != 1 or mu.shape[0] != pi_int.shape[0]:
        raise ValueError("mu_micro must have shape (n,) matching pi_map")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu_micro must contain only finite values")

    return np.bincount(pi_int, weights=mu, minlength=int(k)).astype(float)


def uniform_lift(mu_macro: np.ndarray, pi_map: np.ndarray) -> np.ndarray:
    """Lift macro distribution uniformly within each partition fiber."""
    pi_int, k, counts = _validate_pi_map(pi_map)
    mu = _validate_prob_vector("mu_macro", mu_macro, k)

    mu_micro = mu[pi_int] / counts[pi_int]
    return mu_micro


def stationary_conditional_lift(
    mu_macro: np.ndarray, pi_map: np.ndarray, pi_stationary: np.ndarray
) -> np.ndarray:
    """Lift via stationary conditionals inside each fiber.

    If the stationary macro mass of a fiber is numerically zero, this falls back
    to a uniform allocation within that fiber for that macro label.
    """
    pi_int, k, counts = _validate_pi_map(pi_map)
    mu = _validate_prob_vector("mu_macro", mu_macro, k)
    pi_stat = _validate_prob_vector("pi_stationary", pi_stationary, pi_int.shape[0])

    pi_macro = pushforward_dist(pi_stat, pi_int, k)
    mu_micro = np.zeros_like(pi_stat)
    tiny = 1e-15

    for x in range(k):
        in_fiber = pi_int == x
        if pi_macro[x] > tiny:
            mu_micro[in_fiber] = mu[x] * (pi_stat[in_fiber] / pi_macro[x])
        else:
            mu_micro[in_fiber] = mu[x] / counts[x]

    total = float(mu_micro.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("lift produced invalid micro distribution")
    mu_micro /= total
    return mu_micro


def macro_kernel(
    P: np.ndarray,
    pi_map: np.ndarray,
    tau: int,
    lift: str = "uniform",
    pi_stationary: np.ndarray | None = None,
) -> np.ndarray:
    """Construct induced macro kernel using the selected lift operator."""
    if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)):
        raise TypeError("tau must be an integer")
    if tau < 0:
        raise ValueError("tau must be >= 0")
    if lift not in {"uniform", "stationary"}:
        raise ValueError("lift must be 'uniform' or 'stationary'")

    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P must contain only finite values")

    n = matrix.shape[0]
    pi_int, k, _ = _validate_pi_map(pi_map)
    if pi_int.shape[0] != n:
        raise ValueError("pi_map length must match P dimension")

    kernel = normalize_rows(matrix)
    P_tau = kernel_power(kernel, int(tau))

    if lift == "stationary":
        if pi_stationary is None:
            pi_stationary_vec = stationary_dist(kernel)
        else:
            pi_stationary_vec = _validate_prob_vector("pi_stationary", pi_stationary, n)

    macro_rows = []
    for x in range(k):
        mu_macro = np.zeros(k, dtype=float)
        mu_macro[x] = 1.0

        if lift == "uniform":
            mu_micro = uniform_lift(mu_macro, pi_int)
        else:
            mu_micro = stationary_conditional_lift(mu_macro, pi_int, pi_stationary_vec)

        mu_micro_next = mu_micro @ P_tau
        row = pushforward_dist(mu_micro_next, pi_int, k)
        macro_rows.append(row)

    macroP = np.vstack(macro_rows)
    row_sums = macroP.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError("macro kernel rows do not sum to 1 within tolerance")
    return macroP
