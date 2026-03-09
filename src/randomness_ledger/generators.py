"""Synthetic micro-chain generators for coarse-graining benchmarks."""

from __future__ import annotations

import numpy as np

from randomness_ledger.markov import make_ergodic, normalize_rows


def _validate_positive_sizes(name: str, sizes: list[int]) -> list[int]:
    """Validate a non-empty list of positive integer sizes."""
    if not isinstance(sizes, list) or len(sizes) == 0:
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    out: list[int] = []
    for s in sizes:
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)) or int(s) <= 0:
            raise ValueError(f"{name} entries must be positive integers")
        out.append(int(s))
    return out


def _validate_seed(seed: int) -> int:
    """Validate integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return int(seed)


def _validate_eps(eps: float) -> float:
    """Validate strictly positive ergodic mixing coefficient."""
    eps_val = float(eps)
    if not np.isfinite(eps_val) or not (0.0 < eps_val <= 1.0):
        raise ValueError("aperiodic_eps must be in (0, 1]")
    return eps_val


def _build_partition(fiber_sizes: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build contiguous partition labels and per-fiber index arrays."""
    sizes = _validate_positive_sizes("fiber_sizes", fiber_sizes)
    n_macro = len(sizes)
    n_micro = int(sum(sizes))
    pi_map = np.empty(n_micro, dtype=np.int64)
    fibers: list[np.ndarray] = []

    cursor = 0
    for label, size in enumerate(sizes):
        inds = np.arange(cursor, cursor + size, dtype=np.int64)
        pi_map[inds] = label
        fibers.append(inds)
        cursor += size

    if int(pi_map.max()) + 1 != n_macro:
        raise RuntimeError("internal partition construction failed")
    return pi_map, fibers


def _sample_macro_kernel(rng: np.random.Generator, k: int) -> np.ndarray:
    """Sample a strictly positive macro kernel with Dirichlet rows."""
    return rng.dirichlet(np.ones(k), size=k)


def _row_from_macro_masses(
    rng: np.random.Generator, macro_masses: np.ndarray, fibers: list[np.ndarray], n: int
) -> np.ndarray:
    """Distribute macro masses across destination micro states by fiber."""
    row = np.zeros(n, dtype=float)
    for b, inds in enumerate(fibers):
        weights = rng.dirichlet(np.ones(len(inds)))
        row[inds] = float(macro_masses[b]) * weights
    return row


def gen_exactly_lumpable(
    n_macro: int, fiber_sizes: list[int], seed: int, aperiodic_eps: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a strictly positive micro kernel that is exactly lumpable at tau=1."""
    if isinstance(n_macro, bool) or not isinstance(n_macro, (int, np.integer)):
        raise TypeError("n_macro must be an integer")
    n_macro_int = int(n_macro)
    if n_macro_int < 1:
        raise ValueError("n_macro must be >= 1")

    sizes = _validate_positive_sizes("fiber_sizes", fiber_sizes)
    if len(sizes) != n_macro_int:
        raise ValueError("len(fiber_sizes) must equal n_macro")

    eps = _validate_eps(aperiodic_eps)
    rng = np.random.default_rng(_validate_seed(seed))
    pi_map, fibers = _build_partition(sizes)
    n_micro = len(pi_map)
    K = _sample_macro_kernel(rng, n_macro_int)

    P = np.zeros((n_micro, n_micro), dtype=float)
    for a, src_inds in enumerate(fibers):
        for i in src_inds:
            P[i] = _row_from_macro_masses(rng, K[a], fibers, n_micro)

    P = normalize_rows(P)
    P = make_ergodic(P, eps=eps)
    meta = {
        "kind": "exactly_lumpable",
        "seed": int(seed),
        "n_macro": n_macro_int,
        "fiber_sizes": sizes,
        "n_micro": n_micro,
        "aperiodic_eps": eps,
        "K": K,
    }
    return P, pi_map, meta


def gen_perturbed_lumpable(
    n_macro: int,
    fiber_sizes: list[int],
    seed: int,
    aperiodic_eps: float,
    heterogeneity_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a perturbed-lumpable kernel with micro-specific macro transitions."""
    if isinstance(n_macro, bool) or not isinstance(n_macro, (int, np.integer)):
        raise TypeError("n_macro must be an integer")
    n_macro_int = int(n_macro)
    if n_macro_int < 1:
        raise ValueError("n_macro must be >= 1")

    sizes = _validate_positive_sizes("fiber_sizes", fiber_sizes)
    if len(sizes) != n_macro_int:
        raise ValueError("len(fiber_sizes) must equal n_macro")

    eps = _validate_eps(aperiodic_eps)
    alpha = float(np.clip(heterogeneity_alpha, 0.0, 1.0))
    rng = np.random.default_rng(_validate_seed(seed))
    pi_map, fibers = _build_partition(sizes)
    n_micro = len(pi_map)
    K_base = _sample_macro_kernel(rng, n_macro_int)

    P = np.zeros((n_micro, n_micro), dtype=float)
    for a, src_inds in enumerate(fibers):
        for i in src_inds:
            r = rng.dirichlet(np.ones(n_macro_int))
            K_i = (1.0 - alpha) * K_base[a] + alpha * r
            P[i] = _row_from_macro_masses(rng, K_i, fibers, n_micro)

    P = normalize_rows(P)
    P = make_ergodic(P, eps=eps)
    meta = {
        "kind": "perturbed_lumpable",
        "seed": int(seed),
        "n_macro": n_macro_int,
        "fiber_sizes": sizes,
        "n_micro": n_micro,
        "aperiodic_eps": eps,
        "heterogeneity_alpha": alpha,
    }
    return P, pi_map, meta


def gen_metastable(
    block_sizes: list[int], p_in: float, p_out: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a strictly positive metastable block-structured micro chain."""
    sizes = _validate_positive_sizes("block_sizes", block_sizes)
    p_in_val = float(p_in)
    p_out_val = float(p_out)
    if not np.isfinite(p_in_val) or p_in_val < 0.0:
        raise ValueError("p_in must be finite and >= 0")
    if not np.isfinite(p_out_val) or p_out_val < 0.0:
        raise ValueError("p_out must be finite and >= 0")
    if p_in_val == 0.0 and p_out_val == 0.0:
        raise ValueError("p_in and p_out cannot both be zero")

    ergodic_eps = 1e-3
    rng = np.random.default_rng(_validate_seed(seed))
    pi_map, blocks = _build_partition(sizes)
    n_micro = len(pi_map)
    n_blocks = len(blocks)

    within_kernels = [
        rng.dirichlet(np.ones(len(block_inds)), size=len(block_inds))
        for block_inds in blocks
    ]

    P = np.zeros((n_micro, n_micro), dtype=float)
    for b, block_inds in enumerate(blocks):
        local_kernel = within_kernels[b]
        outside_inds = np.setdiff1d(np.arange(n_micro, dtype=np.int64), block_inds)

        for local_row, i in enumerate(block_inds):
            w_in = p_in_val * (0.75 + 0.5 * float(rng.random()))
            w_out = p_out_val * (0.75 + 0.5 * float(rng.random()))

            row = np.zeros(n_micro, dtype=float)
            row[block_inds] += w_in * local_kernel[local_row]
            if len(outside_inds) > 0:
                row[outside_inds] += w_out * (1.0 / len(outside_inds))
            else:
                row[block_inds] += w_out * local_kernel[local_row]

            row_sum = float(row.sum())
            if row_sum <= 0.0:
                raise ValueError("constructed a zero row; check p_in/p_out settings")
            P[i] = row / row_sum

    P = normalize_rows(P)
    P = make_ergodic(P, eps=ergodic_eps)
    meta = {
        "kind": "metastable",
        "seed": int(seed),
        "block_sizes": sizes,
        "n_micro": n_micro,
        "p_in": p_in_val,
        "p_out": p_out_val,
        "ergodic_eps": ergodic_eps,
    }
    return P, pi_map, meta


def gen_hidden_types(
    n_macro: int, fiber_sizes: list[int], type_split: float, seed: int, strength: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a chain with hidden within-fiber types driving macro futures."""
    if isinstance(n_macro, bool) or not isinstance(n_macro, (int, np.integer)):
        raise TypeError("n_macro must be an integer")
    n_macro_int = int(n_macro)
    if n_macro_int < 1:
        raise ValueError("n_macro must be >= 1")

    sizes = _validate_positive_sizes("fiber_sizes", fiber_sizes)
    if len(sizes) != n_macro_int:
        raise ValueError("len(fiber_sizes) must equal n_macro")

    split = float(np.clip(type_split, 0.0, 1.0))
    s = float(np.clip(strength, 0.0, 1.0))
    rng = np.random.default_rng(_validate_seed(seed))
    pi_map, fibers = _build_partition(sizes)
    n_micro = len(pi_map)

    K0 = _sample_macro_kernel(rng, n_macro_int)
    R = _sample_macro_kernel(rng, n_macro_int)
    K1 = (1.0 - s) * K0 + s * R

    hidden_type = np.zeros(n_micro, dtype=np.int64)
    for a, inds in enumerate(fibers):
        perm = rng.permutation(inds)
        n_type1 = int(np.round(split * len(inds)))
        hidden_type[perm[:n_type1]] = 1
        hidden_type[perm[n_type1:]] = 0

    P = np.zeros((n_micro, n_micro), dtype=float)
    for a, src_inds in enumerate(fibers):
        for i in src_inds:
            macro_vec = K1[a] if hidden_type[i] == 1 else K0[a]
            P[i] = _row_from_macro_masses(rng, macro_vec, fibers, n_micro)

    P = normalize_rows(P)
    P = make_ergodic(P, eps=1e-3)

    type_counts = [
        {
            "macro": a,
            "type0": int(np.sum(hidden_type[inds] == 0)),
            "type1": int(np.sum(hidden_type[inds] == 1)),
        }
        for a, inds in enumerate(fibers)
    ]

    meta = {
        "kind": "hidden_types",
        "seed": int(seed),
        "n_macro": n_macro_int,
        "fiber_sizes": sizes,
        "n_micro": n_micro,
        "type_split": split,
        "strength": s,
        "type_counts": type_counts,
    }
    return P, pi_map, meta
