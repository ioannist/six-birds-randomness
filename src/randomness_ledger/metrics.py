"""Information-theoretic metrics for coarse-grained Markov dynamics."""

from __future__ import annotations

import numpy as np

from randomness_ledger.markov import kernel_power, normalize_rows, stationary_dist
from randomness_ledger.packaging import macro_kernel, pushforward_dist


def _validate_partition(pi_map: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Validate partition labels as contiguous integers 0..k-1."""
    labels = np.asarray(pi_map)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("pi_map must have shape (n,) matching P")
    if not np.all(np.isfinite(labels)):
        raise ValueError("pi_map must contain only finite values")
    if not np.all(labels == np.floor(labels)):
        raise ValueError("pi_map entries must be integer-valued")

    pi_int = labels.astype(np.int64)
    if np.any(pi_int < 0):
        raise ValueError("pi_map entries must be >= 0")

    k = int(pi_int.max()) + 1
    counts = np.bincount(pi_int, minlength=k)
    if np.any(counts == 0):
        raise ValueError("pi_map labels must be contiguous 0..k-1 with no gaps")
    return pi_int, k


def _validate_kernel(P: np.ndarray) -> np.ndarray:
    """Validate and normalize a square transition kernel."""
    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P must contain only finite values")
    return normalize_rows(matrix)


def _validate_prob_vector(vec: np.ndarray, n: int, name: str) -> np.ndarray:
    """Validate a probability vector."""
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < -1e-15):
        raise ValueError(f"{name} must be nonnegative")
    if not np.isclose(float(arr.sum()), 1.0, atol=1e-10):
        raise ValueError(f"{name} must sum to 1")
    return arr


def _row_entropies(rows: np.ndarray) -> np.ndarray:
    """Compute entropy for each row using natural logs.

    We mask entries with p == 0, so 0 * log(0) is treated as 0 and does not
    introduce NaNs in entropy/KL-style calculations.
    """
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("rows must be 2D")

    out = np.zeros(values.shape[0], dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        mask = row > 0.0
        out[i] = -np.sum(row[mask] * np.log(row[mask]))
    return out


def _pz_rows(P: np.ndarray, pi_map: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return all micro-to-macro destination rows ``p_z`` for ``P^tau``."""
    if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)):
        raise TypeError("tau must be an integer")
    if tau < 0:
        raise ValueError("tau must be >= 0")

    kernel = _validate_kernel(P)
    n = kernel.shape[0]
    pi_int, k = _validate_partition(pi_map, n)

    P_tau = kernel_power(kernel, int(tau))
    indicator = np.zeros((n, k), dtype=float)
    indicator[np.arange(n), pi_int] = 1.0
    pz = P_tau @ indicator
    return pz, pi_int, k


def _stationary_pi(P: np.ndarray, pi_stationary: np.ndarray | None) -> np.ndarray:
    """Get stationary micro distribution from argument or by computation."""
    kernel = _validate_kernel(P)
    n = kernel.shape[0]
    if pi_stationary is None:
        return stationary_dist(kernel)
    return _validate_prob_vector(pi_stationary, n, "pi_stationary")


def step_entropy(macroP: np.ndarray, macro_stationary: np.ndarray | None = None) -> float:
    """Compute stationary-weighted row entropy of a macro transition kernel."""
    macro_kernel_arr = _validate_kernel(macroP)
    k = macro_kernel_arr.shape[0]

    if macro_stationary is None:
        pi_macro = stationary_dist(macro_kernel_arr)
    else:
        pi_macro = _validate_prob_vector(macro_stationary, k, "macro_stationary")

    ent = _row_entropies(macro_kernel_arr)
    return float(np.sum(pi_macro * ent))


def route_mismatch(
    P: np.ndarray,
    pi_map: np.ndarray,
    tau: int,
    lift: str = "uniform",
    pi_stationary: np.ndarray | None = None,
    norm: str = "l1",
) -> float:
    """Measure within-macro heterogeneity relative to the induced macro kernel."""
    if norm not in {"l1", "tv"}:
        raise ValueError("norm must be 'l1' or 'tv'")

    pz, pi_int, _ = _pz_rows(P, pi_map, tau)
    kernel = _validate_kernel(P)
    pi = _stationary_pi(kernel, pi_stationary)
    hatP = macro_kernel(kernel, pi_int, int(tau), lift=lift, pi_stationary=pi)

    diffs = np.abs(pz - hatP[pi_int])
    distances = np.sum(diffs, axis=1)
    if norm == "tv":
        distances *= 0.5
    return float(np.sum(pi * distances))


def intrinsic_term(
    P: np.ndarray, pi_map: np.ndarray, tau: int, pi_stationary: np.ndarray | None = None
) -> float:
    """Compute ``H(Y_{t+tau} | X_t)`` under stationary micro weighting."""
    pz, _, _ = _pz_rows(P, pi_map, tau)
    pi = _stationary_pi(P, pi_stationary)
    ent = _row_entropies(pz)
    return float(np.sum(pi * ent))


def macro_cond_entropy(
    P: np.ndarray, pi_map: np.ndarray, tau: int, pi_stationary: np.ndarray | None = None
) -> float:
    """Compute ``H(Y_{t+tau} | Y_t)`` under stationary micro distribution."""
    pz, pi_int, k = _pz_rows(P, pi_map, tau)
    pi = _stationary_pi(P, pi_stationary)
    pi_macro = pushforward_dist(pi, pi_int, k)

    n = pi.shape[0]
    cond = np.zeros((k, n), dtype=float)
    counts = np.bincount(pi_int, minlength=k)
    tiny = 1e-15
    for x in range(k):
        mask = pi_int == x
        if pi_macro[x] > tiny:
            cond[x, mask] = pi[mask] / pi_macro[x]
        else:
            cond[x, mask] = 1.0 / counts[x]

    px = cond @ pz
    ent = _row_entropies(px)
    return float(np.sum(pi_macro * ent))


def closure_deficit(
    P: np.ndarray, pi_map: np.ndarray, tau: int, pi_stationary: np.ndarray | None = None
) -> float:
    """Compute ``I(X_t; Y_{t+tau} | Y_t)`` as stationary-weighted expected KL."""
    pz, pi_int, k = _pz_rows(P, pi_map, tau)
    pi = _stationary_pi(P, pi_stationary)
    pi_macro = pushforward_dist(pi, pi_int, k)

    n = pi.shape[0]
    cond = np.zeros((k, n), dtype=float)
    counts = np.bincount(pi_int, minlength=k)
    tiny = 1e-15
    for x in range(k):
        mask = pi_int == x
        if pi_macro[x] > tiny:
            cond[x, mask] = pi[mask] / pi_macro[x]
        else:
            cond[x, mask] = 1.0 / counts[x]

    px = cond @ pz
    qz = px[pi_int]

    kl = np.zeros(n, dtype=float)
    for z in range(n):
        p = pz[z]
        q = qz[z]
        # Mask p == 0 terms so only support of p contributes to KL(p || q).
        mask = p > 0.0
        if np.any(q[mask] <= 0.0):
            kl[z] = np.inf
        else:
            kl[z] = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
    return float(np.sum(pi * kl))


def decomposition_check(
    P: np.ndarray, pi_map: np.ndarray, tau: int, pi_stationary: np.ndarray | None = None
) -> float:
    """Return residual of ``H(Y|Y)-H(Y|X)-I(X;Y|Y)`` for the selected lag."""
    hyy = macro_cond_entropy(P, pi_map, tau, pi_stationary=pi_stationary)
    hyx = intrinsic_term(P, pi_map, tau, pi_stationary=pi_stationary)
    cd = closure_deficit(P, pi_map, tau, pi_stationary=pi_stationary)
    return float(hyy - hyx - cd)
