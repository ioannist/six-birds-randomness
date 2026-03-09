"""Toy-safe hashing inversion-vs-budget experiment on truncated SHA-256 digests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _parse_int_list(raw: str, name: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one integer")
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    # Deduplicate while preserving first appearance.
    return list(dict.fromkeys(out))


def _validate_args(args: argparse.Namespace, n_bits_list: list[int], q_list: list[int]) -> None:
    if any(n < 4 or n > 20 for n in n_bits_list):
        raise ValueError("all n_bits must satisfy 4 <= n_bits <= 20")
    if any(q < 1 for q in q_list):
        raise ValueError("all q values must satisfy q >= 1")
    if args.trials < 1:
        raise ValueError("trials must be >= 1")
    if args.m_bits < max(n_bits_list):
        raise ValueError("m_bits must be >= max(n_bits)")
    if args.dictionary_size < 2:
        raise ValueError("dictionary_size must be >= 2")
    if not (0.0 < args.mix_dict_weight < 1.0):
        raise ValueError("mix_dict_weight must be in (0, 1)")
    if args.randtest_samples < 512:
        raise ValueError("randtest_samples must be >= 512")
    if args.randtest_rep_n_bits < 4 or args.randtest_rep_n_bits > 20:
        raise ValueError("randtest_rep_n_bits must satisfy 4 <= randtest_rep_n_bits <= 20")

    domain_size = 1 << int(args.m_bits)
    if args.dictionary_size >= domain_size:
        raise ValueError("dictionary_size must be < 2**m_bits to leave non-dictionary mass")

    max_q = max(q_list)
    if max_q > domain_size:
        raise ValueError("max(q_list) cannot exceed the input domain size for distinct querying")
    non_dict_size = domain_size - args.dictionary_size
    if max(0, max_q - args.dictionary_size) > non_dict_size:
        raise ValueError("not enough non-dictionary points for requested medium-mixture query budget")


def sha256_trunc_bits(data: bytes, n_bits: int) -> int:
    """Return first n_bits of SHA-256 digest as an integer."""
    if n_bits < 1 or n_bits > 256:
        raise ValueError("n_bits must be in [1, 256]")
    digest = hashlib.sha256(data).digest()
    n_bytes = (n_bits + 7) // 8
    prefix = int.from_bytes(digest[:n_bytes], byteorder="big", signed=False)
    extra = 8 * n_bytes - n_bits
    return prefix >> extra if extra > 0 else prefix


def _sha256_full_int(x: int, m_bytes: int) -> int:
    data = int(x).to_bytes(m_bytes, byteorder="big", signed=False)
    return int.from_bytes(hashlib.sha256(data).digest(), byteorder="big", signed=False)


def _trunc_from_full(full_digest: int, n_bits: int) -> int:
    return int(full_digest) >> (256 - int(n_bits))


def _sample_unique_domain(rng: np.random.Generator, count: int, domain_size: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    sample = rng.choice(domain_size, size=count, replace=False)
    return np.asarray(sample, dtype=np.int64)


def _sample_unique_outside(
    rng: np.random.Generator, count: int, domain_size: int, forbidden: set[int]
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)

    out: list[int] = []
    seen: set[int] = set()
    while len(out) < count:
        cand = int(rng.integers(0, domain_size))
        if cand in forbidden or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return np.asarray(out, dtype=np.int64)


def _first_hit_index(query_digests: np.ndarray, target_digest: int) -> int:
    hits = np.flatnonzero(query_digests == int(target_digest))
    return int(hits[0] + 1) if hits.size > 0 else int(query_digests.size + 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _sample_iid_uniform(rng: np.random.Generator, count: int, domain_size: int) -> np.ndarray:
    return np.asarray(rng.integers(0, domain_size, size=count), dtype=np.int64)


def _sample_iid_from_dictionary(
    rng: np.random.Generator, count: int, dictionary: np.ndarray
) -> np.ndarray:
    idx = np.asarray(rng.integers(0, dictionary.shape[0], size=count), dtype=np.int64)
    return np.asarray(dictionary[idx], dtype=np.int64)


def _sample_iid_non_dictionary(
    rng: np.random.Generator, count: int, domain_size: int, dictionary_set: set[int]
) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    i = 0
    while i < count:
        cand = int(rng.integers(0, domain_size))
        if cand in dictionary_set:
            continue
        out[i] = cand
        i += 1
    return out


def _hash_inputs_to_bytes(inputs: np.ndarray, m_bytes: int) -> np.ndarray:
    dig = np.empty((inputs.shape[0], 32), dtype=np.uint8)
    for i, x in enumerate(inputs):
        d = hashlib.sha256(int(x).to_bytes(m_bytes, byteorder="big", signed=False)).digest()
        dig[i] = np.frombuffer(d, dtype=np.uint8)
    return dig


def _trunc_values_from_digest_bytes(digest_bytes: np.ndarray, n_bits: int) -> np.ndarray:
    n_bytes = (n_bits + 7) // 8
    extra = 8 * n_bytes - n_bits
    out = np.empty(digest_bytes.shape[0], dtype=np.int64)
    for i in range(digest_bytes.shape[0]):
        prefix = int.from_bytes(bytes(digest_bytes[i, :n_bytes]), byteorder="big", signed=False)
        out[i] = prefix >> extra if extra > 0 else prefix
    return out


def _byte_chi2_stats(digest_bytes: np.ndarray) -> dict[str, float]:
    stream = digest_bytes.reshape(-1)
    counts = np.bincount(stream.astype(np.int64), minlength=256).astype(float)
    total = float(stream.size)
    expected = total / 256.0
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    df = 255.0
    z = float((chi2 - df) / np.sqrt(2.0 * df))
    max_rel_dev = float(np.max(np.abs(counts - expected) / expected))
    return {
        "byte_chi2": chi2,
        "byte_chi2_df": df,
        "byte_chi2_z": z,
        "byte_max_rel_dev": max_rel_dev,
    }


def _trunc_collision_stats(trunc_vals: np.ndarray, n_bits: int) -> dict[str, float]:
    _, counts = np.unique(trunc_vals, return_counts=True)
    pairs_obs = float(np.sum(counts * (counts - 1) // 2))
    s = float(trunc_vals.shape[0])
    pairs_exp = float((s * (s - 1) / 2.0) / float(2**n_bits))
    ratio = float(pairs_obs / pairs_exp) if pairs_exp > 0 else float("nan")
    distinct_count = int(counts.shape[0])
    distinct_frac = float(distinct_count / s) if s > 0 else float("nan")
    return {
        "trunc_collision_pairs_obs": pairs_obs,
        "trunc_collision_pairs_exp": pairs_exp,
        "trunc_collision_ratio": ratio,
        "trunc_distinct_count": distinct_count,
        "trunc_distinct_frac": distinct_frac,
    }


def _bitstream_stats(digest_bytes: np.ndarray) -> dict[str, float]:
    bits = np.unpackbits(digest_bytes.reshape(-1))
    n = int(bits.size)
    if n < 2:
        return {
            "bit_ones_frac": 0.0,
            "bit_runs": 0.0,
            "bit_runs_expected": 0.0,
            "bit_runs_z": 0.0,
            "bit_serial_corr": 0.0,
        }

    n1 = float(np.sum(bits))
    n0 = float(n - n1)
    ones_frac = float(n1 / n)
    runs = float(1 + np.sum(bits[1:] != bits[:-1]))
    expected_runs = float(1.0 + (2.0 * n1 * n0) / n)
    numer = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
    denom = (n**2) * (n - 1)
    var_runs = float(numer / denom) if denom > 0 else 0.0
    if var_runs > 0:
        runs_z = float((runs - expected_runs) / np.sqrt(var_runs))
    else:
        runs_z = 0.0

    x = bits[:-1].astype(float)
    y = bits[1:].astype(float)
    if np.std(x) <= 0 or np.std(y) <= 0:
        serial_corr = 0.0
    else:
        serial_corr = float(np.corrcoef(x, y)[0, 1])

    return {
        "bit_ones_frac": ones_frac,
        "bit_runs": runs,
        "bit_runs_expected": expected_runs,
        "bit_runs_z": runs_z,
        "bit_serial_corr": serial_corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy-safe hashing inversion budget experiments.")
    parser.add_argument("--n_bits", default="8,12,16,20")
    parser.add_argument("--q_list", default="1,4,16,64,256,1024")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--m_bits", type=int, default=32)
    parser.add_argument("--dictionary_size", type=int, default=256)
    parser.add_argument("--mix_dict_weight", type=float, default=0.5)
    parser.add_argument("--randtest_samples", type=int, default=5000)
    parser.add_argument("--randtest_rep_n_bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--outdir", default="results/hashing_toy")
    args = parser.parse_args()

    n_bits_list = _parse_int_list(args.n_bits, "n_bits")
    q_list = _parse_int_list(args.q_list, "q_list")
    _validate_args(args, n_bits_list, q_list)

    t0 = time.perf_counter()
    rng = np.random.default_rng(int(args.seed))
    domain_size = 1 << int(args.m_bits)
    max_q = max(q_list)
    m_bytes = (int(args.m_bits) + 7) // 8

    # Synthetic dictionary shared by low-entropy and medium-mixture distributions.
    dictionary = _sample_unique_domain(rng, int(args.dictionary_size), domain_size)
    dictionary_set = {int(x) for x in dictionary.tolist()}

    dict_full = np.asarray([_sha256_full_int(int(x), m_bytes) for x in dictionary], dtype=object)
    dict_trunc: dict[int, np.ndarray] = {
        n: np.asarray([_trunc_from_full(int(fd), n) for fd in dict_full], dtype=np.int64)
        for n in n_bits_list
    }

    dist_names = ["uniform", "low_entropy", "medium_mixture"]
    success_counts: dict[tuple[str, int, int], int] = {
        (dist, n, q): 0 for dist in dist_names for n in n_bits_list for q in q_list
    }

    for trial in range(int(args.trials)):
        rng_uniform = np.random.default_rng(int(args.seed) + 10_000 + trial)
        rng_low = np.random.default_rng(int(args.seed) + 20_000 + trial)
        rng_medium = np.random.default_rng(int(args.seed) + 30_000 + trial)

        # Uniform target and distinct random-domain queries.
        target_uniform = int(rng_uniform.integers(0, domain_size))
        target_uniform_full = _sha256_full_int(target_uniform, m_bytes)
        query_uniform = _sample_unique_domain(rng_uniform, max_q, domain_size)
        query_uniform_full = np.asarray(
            [_sha256_full_int(int(x), m_bytes) for x in query_uniform], dtype=object
        )

        # Low-entropy target with dictionary-prefix attacker.
        low_target_idx = int(rng_low.integers(0, int(args.dictionary_size)))
        target_low_full = int(dict_full[low_target_idx])

        # Medium-mixture target: dictionary with probability mix_dict_weight, else outside dictionary.
        if float(rng_medium.random()) < float(args.mix_dict_weight):
            med_target_idx = int(rng_medium.integers(0, int(args.dictionary_size)))
            target_med = int(dictionary[med_target_idx])
            target_med_full = int(dict_full[med_target_idx])
        else:
            target_med = int(_sample_unique_outside(rng_medium, 1, domain_size, dictionary_set)[0])
            target_med_full = _sha256_full_int(target_med, m_bytes)

        extra_needed = max(0, max_q - int(args.dictionary_size))
        extra_medium = _sample_unique_outside(rng_medium, extra_needed, domain_size, dictionary_set)
        extra_medium_full = np.asarray(
            [_sha256_full_int(int(x), m_bytes) for x in extra_medium], dtype=object
        )

        for n_bits in n_bits_list:
            q_arr = np.asarray(q_list, dtype=np.int64)

            uniform_trunc = np.asarray(
                [_trunc_from_full(int(fd), n_bits) for fd in query_uniform_full], dtype=np.int64
            )
            target_uniform_trunc = _trunc_from_full(target_uniform_full, n_bits)
            hit_uniform = _first_hit_index(uniform_trunc, target_uniform_trunc)
            success_uniform = q_arr >= hit_uniform
            for q, ok in zip(q_list, success_uniform, strict=True):
                success_counts[("uniform", n_bits, q)] += int(ok)

            low_trunc = dict_trunc[n_bits]
            target_low_trunc = _trunc_from_full(target_low_full, n_bits)
            hit_low = _first_hit_index(low_trunc, target_low_trunc)
            low_effective_budget = np.minimum(q_arr, int(low_trunc.size))
            success_low = low_effective_budget >= hit_low
            for q, ok in zip(q_list, success_low, strict=True):
                success_counts[("low_entropy", n_bits, q)] += int(ok)

            if extra_medium_full.size > 0:
                med_query_trunc = np.concatenate(
                    [
                        dict_trunc[n_bits],
                        np.asarray(
                            [_trunc_from_full(int(fd), n_bits) for fd in extra_medium_full],
                            dtype=np.int64,
                        ),
                    ]
                )
            else:
                med_query_trunc = dict_trunc[n_bits].copy()

            target_med_trunc = _trunc_from_full(target_med_full, n_bits)
            hit_med = _first_hit_index(med_query_trunc, target_med_trunc)
            med_effective_budget = np.minimum(q_arr, int(med_query_trunc.size))
            success_med = med_effective_budget >= hit_med
            for q, ok in zip(q_list, success_med, strict=True):
                success_counts[("medium_mixture", n_bits, q)] += int(ok)

    run_id = _run_id()
    run_dir = Path(args.outdir) / run_id
    figs_dir = run_dir / "figs"
    run_dir.mkdir(parents=True, exist_ok=False)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for distribution in dist_names:
        for n_bits in n_bits_list:
            two_pow_n = float(2**n_bits)
            for q in q_list:
                empirical = float(success_counts[(distribution, n_bits, q)] / float(args.trials))
                baseline_q_over = float(min(float(q) / two_pow_n, 1.0))
                baseline_exact = float(1.0 - (1.0 - (1.0 / two_pow_n)) ** int(q))
                abs_err = float(abs(empirical - baseline_q_over))
                success_se = float(math.sqrt(max(empirical * (1.0 - empirical), 0.0) / float(args.trials)))

                rows.append(
                    {
                        "run_id": run_id,
                        "distribution": distribution,
                        "n_bits": int(n_bits),
                        "q": int(q),
                        "trials": int(args.trials),
                        "m_bits": int(args.m_bits),
                        "dictionary_size": int(args.dictionary_size),
                        "mix_dict_weight": float(args.mix_dict_weight),
                        "empirical_success": empirical,
                        "baseline_q_over_2n": baseline_q_over,
                        "baseline_exact": baseline_exact,
                        "abs_err_to_q_over_2n": abs_err,
                        "success_se": success_se,
                    }
                )

    fieldnames = [
        "run_id",
        "distribution",
        "n_bits",
        "q",
        "trials",
        "m_bits",
        "dictionary_size",
        "mix_dict_weight",
        "empirical_success",
        "baseline_q_over_2n",
        "baseline_exact",
        "abs_err_to_q_over_2n",
        "success_se",
    ]
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    uniform_rows = [r for r in rows if r["distribution"] == "uniform"]
    uniform_subset = [r for r in uniform_rows if float(r["baseline_q_over_2n"]) <= 0.25]
    if uniform_subset:
        uniform_mean_abs_error = float(np.mean([r["abs_err_to_q_over_2n"] for r in uniform_subset]))
        uniform_max_abs_error = float(np.max([r["abs_err_to_q_over_2n"] for r in uniform_subset]))
    else:
        uniform_mean_abs_error = float("nan")
        uniform_max_abs_error = float("nan")

    representative_n = 16 if 16 in n_bits_list else n_bits_list[len(n_bits_list) // 2]

    def _success_at(distribution: str, n_bits: int, q: int) -> float:
        for r in rows:
            if (
                r["distribution"] == distribution
                and int(r["n_bits"]) == int(n_bits)
                and int(r["q"]) == int(q)
            ):
                return float(r["empirical_success"])
        return float("nan")

    nearest_n_for_dict = min(n_bits_list, key=lambda n: abs(n - 16))
    q_dict = int(args.dictionary_size)
    low_entropy_success_at_q_eq_dictsize = (
        _success_at("low_entropy", nearest_n_for_dict, q_dict)
        if q_dict in set(q_list)
        else float("nan")
    )
    medium_success_at_q_eq_dictsize = (
        _success_at("medium_mixture", nearest_n_for_dict, q_dict)
        if q_dict in set(q_list)
        else float("nan")
    )

    uniform_sorted = sorted(uniform_rows, key=lambda r: float(r["abs_err_to_q_over_2n"]))
    notable_rows = []
    if uniform_sorted:
        notable_rows.append(
            {
                "label": "uniform_best_error",
                "distribution": "uniform",
                "n_bits": int(uniform_sorted[0]["n_bits"]),
                "q": int(uniform_sorted[0]["q"]),
                "empirical_success": float(uniform_sorted[0]["empirical_success"]),
                "baseline_q_over_2n": float(uniform_sorted[0]["baseline_q_over_2n"]),
                "abs_err_to_q_over_2n": float(uniform_sorted[0]["abs_err_to_q_over_2n"]),
            }
        )
        worst = uniform_sorted[-1]
        notable_rows.append(
            {
                "label": "uniform_worst_error",
                "distribution": "uniform",
                "n_bits": int(worst["n_bits"]),
                "q": int(worst["q"]),
                "empirical_success": float(worst["empirical_success"]),
                "baseline_q_over_2n": float(worst["baseline_q_over_2n"]),
                "abs_err_to_q_over_2n": float(worst["abs_err_to_q_over_2n"]),
            }
        )

    # Plot 1: uniform success vs budget with q/2^n overlay.
    plt.figure(figsize=(8, 5))
    for n_bits in n_bits_list:
        sub = [r for r in rows if r["distribution"] == "uniform" and int(r["n_bits"]) == int(n_bits)]
        sub = sorted(sub, key=lambda r: int(r["q"]))
        x = np.asarray([int(r["q"]) for r in sub], dtype=float)
        y_emp = np.asarray([float(r["empirical_success"]) for r in sub], dtype=float)
        y_base = np.asarray([float(r["baseline_q_over_2n"]) for r in sub], dtype=float)

        plt.plot(x, y_emp, marker="o", label=f"empirical n={n_bits}")
        plt.plot(x, y_base, linestyle="--", label=f"q/2^n n={n_bits}")

    plt.xlabel("q (query budget)")
    plt.ylabel("Success probability")
    plt.title("Uniform targets: inversion success vs budget")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "uniform_success_vs_budget.png", dpi=150)
    plt.close()

    # Plot 2: success vs budget by distribution at representative n.
    plt.figure(figsize=(8, 5))
    for distribution in dist_names:
        sub = [
            r
            for r in rows
            if r["distribution"] == distribution and int(r["n_bits"]) == int(representative_n)
        ]
        sub = sorted(sub, key=lambda r: int(r["q"]))
        x = np.asarray([int(r["q"]) for r in sub], dtype=float)
        y = np.asarray([float(r["empirical_success"]) for r in sub], dtype=float)
        plt.plot(x, y, marker="o", label=distribution)

    plt.xlabel("q (query budget)")
    plt.ylabel("Success probability")
    plt.title(f"Success vs budget by distribution (n_bits={representative_n})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "success_vs_budget_by_dist_n16.png", dpi=150)
    plt.close()

    runtime = float(time.perf_counter() - t0)

    config = {
        "run_id": run_id,
        "n_bits": n_bits_list,
        "q_list": q_list,
        "trials": int(args.trials),
        "m_bits": int(args.m_bits),
        "dictionary_size": int(args.dictionary_size),
        "mix_dict_weight": float(args.mix_dict_weight),
        "randtest_samples": int(args.randtest_samples),
        "randtest_rep_n_bits": int(args.randtest_rep_n_bits),
        "seed": int(args.seed),
        "dictionary_generation": {
            "method": "uniform_without_replacement_over_m_bits_domain",
            "domain_size": int(domain_size),
            "fixed_for_entire_run": True,
        },
    }

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "runtime_seconds": runtime,
    }

    summary = {
        "uniform_error_subset_rule": "distribution==uniform and baseline_q_over_2n<=0.25",
        "uniform_mean_abs_error_to_q_over_2n": uniform_mean_abs_error,
        "uniform_max_abs_error_to_q_over_2n": uniform_max_abs_error,
        "low_entropy_success_at_q_eq_dictsize": low_entropy_success_at_q_eq_dictsize,
        "medium_success_at_q_eq_dictsize": medium_success_at_q_eq_dictsize,
        "n_bits_for_q_eq_dictsize_stat": int(nearest_n_for_dict),
        "q_eq_dictsize": int(q_dict),
        "notable_rows": notable_rows,
    }

    # Bounded-distinguisher randomness tests on digest sequences.
    rand_samples = int(args.randtest_samples)
    rep_n_bits = int(args.randtest_rep_n_bits)
    rand_rngs = {
        "uniform": np.random.default_rng(int(args.seed) + 100_000),
        "low_entropy": np.random.default_rng(int(args.seed) + 200_000),
        "medium_mixture": np.random.default_rng(int(args.seed) + 300_000),
    }

    rand_inputs: dict[str, np.ndarray] = {}
    rand_inputs["uniform"] = _sample_iid_uniform(rand_rngs["uniform"], rand_samples, domain_size)
    rand_inputs["low_entropy"] = _sample_iid_from_dictionary(
        rand_rngs["low_entropy"], rand_samples, dictionary
    )
    medium_mask = rand_rngs["medium_mixture"].random(rand_samples) < float(args.mix_dict_weight)
    medium_inputs = np.empty(rand_samples, dtype=np.int64)
    n_dict = int(np.sum(medium_mask))
    if n_dict > 0:
        medium_inputs[medium_mask] = _sample_iid_from_dictionary(
            rand_rngs["medium_mixture"], n_dict, dictionary
        )
    n_non_dict = int(rand_samples - n_dict)
    if n_non_dict > 0:
        medium_inputs[~medium_mask] = _sample_iid_non_dictionary(
            rand_rngs["medium_mixture"], n_non_dict, domain_size, dictionary_set
        )
    rand_inputs["medium_mixture"] = medium_inputs

    rand_rows: list[dict[str, Any]] = []
    rand_core: dict[str, dict[str, float]] = {}
    for distribution in dist_names:
        digest_bytes = _hash_inputs_to_bytes(rand_inputs[distribution], m_bytes)
        trunc_vals = _trunc_values_from_digest_bytes(digest_bytes, rep_n_bits)
        byte_stats = _byte_chi2_stats(digest_bytes)
        collision_stats = _trunc_collision_stats(trunc_vals, rep_n_bits)
        bit_stats = _bitstream_stats(digest_bytes)

        row = {
            "run_id": run_id,
            "distribution": distribution,
            "randtest_samples": rand_samples,
            "randtest_rep_n_bits": rep_n_bits,
            **byte_stats,
            **collision_stats,
            **bit_stats,
        }
        rand_rows.append(row)
        rand_core[distribution] = {
            "byte_chi2_z": float(row["byte_chi2_z"]),
            "trunc_collision_ratio": float(row["trunc_collision_ratio"]),
            "trunc_distinct_frac": float(row["trunc_distinct_frac"]),
            "bit_runs_z": float(row["bit_runs_z"]),
            "bit_serial_corr": float(row["bit_serial_corr"]),
        }

    rand_fieldnames = [
        "run_id",
        "distribution",
        "randtest_samples",
        "randtest_rep_n_bits",
        "byte_chi2",
        "byte_chi2_df",
        "byte_chi2_z",
        "byte_max_rel_dev",
        "trunc_collision_pairs_obs",
        "trunc_collision_pairs_exp",
        "trunc_collision_ratio",
        "trunc_distinct_count",
        "trunc_distinct_frac",
        "bit_ones_frac",
        "bit_runs",
        "bit_runs_expected",
        "bit_runs_z",
        "bit_serial_corr",
    ]
    randomness_tests_path = run_dir / "randomness_tests.csv"
    with randomness_tests_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rand_fieldnames)
        writer.writeheader()
        writer.writerows(rand_rows)

    # New Plot 1: byte chi-square z by distribution.
    labels = [r["distribution"] for r in rand_rows]
    byte_z = [float(r["byte_chi2_z"]) for r in rand_rows]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, byte_z)
    plt.xlabel("distribution")
    plt.ylabel("byte_chi2_z")
    plt.title("Byte-frequency chi-square z by distribution")
    plt.tight_layout()
    plt.savefig(figs_dir / "byte_chi2_z_by_dist.png", dpi=150)
    plt.close()

    # New Plot 2: collision ratio by distribution for representative truncation size.
    collision_ratio = [float(r["trunc_collision_ratio"]) for r in rand_rows]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, collision_ratio)
    plt.xlabel("distribution")
    plt.ylabel("trunc_collision_ratio")
    plt.title(f"Collision ratio by distribution (n_bits={rep_n_bits})")
    plt.tight_layout()
    plt.savefig(figs_dir / f"collision_ratio_by_dist_n{rep_n_bits}.png", dpi=150)
    plt.close()

    uniform_stats = rand_core["uniform"]
    low_stats = rand_core["low_entropy"]
    medium_stats = rand_core["medium_mixture"]

    uniform_not_grossly_flagged = bool(
        abs(uniform_stats["byte_chi2_z"]) <= 4.0
        and abs(uniform_stats["bit_runs_z"]) <= 4.0
        and abs(uniform_stats["bit_serial_corr"]) <= 0.01
    )
    low_entropy_detected = bool(
        low_stats["trunc_collision_ratio"] >= 20.0
        or low_stats["trunc_distinct_frac"] <= 0.25
        or abs(low_stats["bit_runs_z"]) >= 6.0
        or abs(low_stats["bit_serial_corr"]) >= 0.01
    )
    medium_markedly_differs_from_uniform = bool(
        medium_stats["trunc_collision_ratio"] >= uniform_stats["trunc_collision_ratio"] + 2.0
        or medium_stats["trunc_distinct_frac"] <= uniform_stats["trunc_distinct_frac"] - 0.05
    )

    summary["randomness_tests"] = {
        "randtest_samples": rand_samples,
        "randtest_rep_n_bits": rep_n_bits,
        "uniform_not_grossly_flagged": uniform_not_grossly_flagged,
        "low_entropy_detected": low_entropy_detected,
        "medium_markedly_differs_from_uniform": medium_markedly_differs_from_uniform,
        "uniform_stats": uniform_stats,
        "low_entropy_stats": low_stats,
        "medium_stats": medium_stats,
    }

    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)

    print(f"run_dir={run_dir}")
    print(f"uniform_mean_abs_error_to_q_over_2n={uniform_mean_abs_error:.6f}")
    print(f"low_entropy_success_at_q_eq_dictsize={low_entropy_success_at_q_eq_dictsize:.6f}")
    print(f"randomness_tests_csv={randomness_tests_path}")
    print(f"uniform_not_grossly_flagged={uniform_not_grossly_flagged}")
    print(f"low_entropy_detected={low_entropy_detected}")
    print(
        "uniform_stats:"
        f" byte_chi2_z={uniform_stats['byte_chi2_z']:.6f},"
        f" trunc_collision_ratio={uniform_stats['trunc_collision_ratio']:.6f},"
        f" bit_runs_z={uniform_stats['bit_runs_z']:.6f},"
        f" bit_serial_corr={uniform_stats['bit_serial_corr']:.6f}"
    )
    print(
        "low_entropy_stats:"
        f" byte_chi2_z={low_stats['byte_chi2_z']:.6f},"
        f" trunc_collision_ratio={low_stats['trunc_collision_ratio']:.6f},"
        f" bit_runs_z={low_stats['bit_runs_z']:.6f},"
        f" bit_serial_corr={low_stats['bit_serial_corr']:.6f}"
    )
    print(f"runtime_seconds={runtime:.3f}")


if __name__ == "__main__":
    main()
