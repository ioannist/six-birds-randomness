#!/usr/bin/env python3
"""Exact population quantities quoted in the manuscript.

Everything here is computed from known kernels (no sampling): the budget
chain's entropy staircase, its closure deficit and intrinsic floor, the three
finite counterexamples, and summary statistics of the tracked CSVs.
Output: paper/data/population_quantities.json.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "src"))

from randomness_ledger.generators import gen_hidden_types  # noqa: E402
from randomness_ledger.markov import stationary_dist  # noqa: E402
from randomness_ledger.metrics import (  # noqa: E402
    closure_deficit,
    intrinsic_term,
    macro_cond_entropy,
    packaged_history_entropy,
    route_mismatch,
)


def rows(name: str) -> list[dict[str, str]]:
    with (PAPER / "data" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chain_summary(seed: int, max_order: int) -> dict:
    P, labels, _ = gen_hidden_types(
        n_macro=3, fiber_sizes=[3, 2, 3], type_split=0.5, seed=seed, strength=0.9
    )
    pi = stationary_dist(P)
    return {
        "seed": seed,
        "cd": closure_deficit(P, labels, 1, pi),
        "intrinsic": intrinsic_term(P, labels, 1, pi),
        "h_next_given_current": macro_cond_entropy(P, labels, 1, pi),
        "rm_uniform": route_mismatch(P, labels, 1, "uniform", pi),
        "rm_stationary": route_mismatch(P, labels, 1, "stationary", pi),
        "history_entropy_by_order": [
            packaged_history_entropy(P, labels, 1, L, pi) for L in range(max_order + 1)
        ],
    }


def budget_test_segment(seed: int = 20260305) -> dict:
    """Score the exact population order-1 predictor on the budget run's test segment.

    Mirrors experiments/budget_curves/run_budget_curves.py (T=10000, burn-in 1000,
    train 1/2, validation 1/4, test 1/4, common evaluation start 5).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_budget_curves", ROOT / "experiments" / "budget_curves" / "run_budget_curves.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    P, labels, _, _ = rb._preset_build("hidden_types_strong", seed)
    y = rb._simulate_macro_sequence(P, labels, 1, 10000, 1000, seed + 77)
    test = y[7500:]
    pi = stationary_dist(P)
    k = int(labels.max()) + 1
    futures = np.zeros((len(labels), k))
    for x in range(len(labels)):
        futures[x] = np.bincount(labels, weights=P[x], minlength=k)
    package_mass = np.bincount(labels, weights=pi, minlength=k)
    rows_ = np.vstack([
        (pi[labels == a] / package_mass[a]) @ futures[labels == a] for a in range(k)
    ])
    losses = -np.log(rows_[test[4:-1], test[5:]])
    batches = np.array([b.mean() for b in np.array_split(losses, 25)])
    return {
        "n_targets": int(losses.size),
        "exact_order1_predictor_test_loss": float(losses.mean()),
        "batch_means_standard_error": float(batches.std(ddof=1) / np.sqrt(batches.size)),
    }


def counterexamples() -> dict:
    # Null-state example: zero deficit, yet the fiber is not closed.
    P0 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    lab0 = np.array([0, 1, 1])
    pi0 = np.array([0.0, 1.0, 0.0])
    null_cd = closure_deficit(P0, lab0, 1, pi0)

    # Uniform-lift example: Pinsker bound fails for the uniform lift.
    e = 1e-3
    P1 = np.array([[1 - e, e, 0], [0, 1 - e, e], [1, 0, 0]], dtype=float)
    lab1 = np.array([0, 1, 1])
    pi1 = stationary_dist(P1)
    cd1 = closure_deficit(P1, lab1, 1, pi1)
    rm_u = route_mismatch(P1, lab1, 1, "uniform", pi1)
    rm_s = route_mismatch(P1, lab1, 1, "stationary", pi1)
    exact_cd = ((1 + e) * math.log(1 + e) - e * math.log(e)) / (2 + e)

    # Deterministic four-cycle: deficit is not monotone in lag or refinement.
    C = np.roll(np.eye(4), 1, axis=1)
    pi_c = np.full(4, 0.25)
    coarse = np.zeros(4, dtype=int)
    fine = np.array([0, 0, 1, 1])
    return {
        "null_state": {"cd": null_cd, "stationary_law": pi0.tolist()},
        "uniform_lift": {
            "eps": e,
            "stationary_law": pi1.tolist(),
            "cd": cd1,
            "cd_exact_formula": exact_cd,
            "rm_uniform": rm_u,
            "rm_uniform_exact": (1 + e) / (2 + e),
            "rm_stationary": rm_s,
            "half_rm_uniform_sq": 0.5 * rm_u**2,
            "half_rm_stationary_sq": 0.5 * rm_s**2,
        },
        "four_cycle": {
            "cd_constant_packaging_lag1": closure_deficit(C, coarse, 1, pi_c),
            "cd_refined_by_lag": [closure_deficit(C, fine, tau, pi_c) for tau in (1, 2, 3)],
            "intrinsic_refined_by_lag": [intrinsic_term(C, fine, tau, pi_c) for tau in (1, 2, 3)],
            "log2": math.log(2.0),
        },
    }


def markov_summary() -> dict:
    data = rows("markov_metrics.csv")
    cd = np.array([float(r["cd"]) for r in data])
    rmu = np.array([float(r["rm_uniform"]) for r in data])
    rms = np.array([float(r["rm_stationary"]) for r in data])
    resid = np.array([float(r["resid"]) for r in data])
    lumpable = [r for r in data if r["family"] == "exactly_lumpable"]
    open_ = cd > 1e-12

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731 (no ties in these data)
        return float(np.corrcoef(rank(a), rank(b))[0, 1])

    return {
        "open_conditions": int(open_.sum()),
        "spearman_rm_uniform_cd_open": spearman(rmu[open_], cd[open_]),
        "spearman_rm_stationary_cd_open": spearman(rms[open_], cd[open_]),
        "conditions": len(data),
        "pearson_rm_uniform_cd": float(np.corrcoef(rmu, cd)[0, 1]),
        "pearson_rm_stationary_cd": float(np.corrcoef(rms, cd)[0, 1]),
        "max_abs_residual": float(np.max(np.abs(resid))),
        "min_pinsker_margin": float(np.min(cd - 0.5 * rms**2)),
        "exactly_lumpable_max_abs_cd": max(abs(float(r["cd"])) for r in lumpable),
        "exactly_lumpable_max_rm_uniform": max(float(r["rm_uniform"]) for r in lumpable),
    }


def hashing_summary() -> dict:
    data = [r for r in rows("hashing_metrics.csv") if r["distribution"] == "uniform"]
    low = [r for r in data if float(r["baseline_q_over_2n"]) <= 0.25]
    dev_simple = [abs(float(r["empirical_success"]) - float(r["baseline_q_over_2n"])) for r in low]
    dev_exact = [abs(float(r["empirical_success"]) - float(r["baseline_exact"])) for r in low]
    ref = [float(r["baseline_exact"]) for r in low]
    se = [math.sqrt(p * (1 - p) / float(r["trials"])) for p, r in zip(ref, low)]
    return {
        "uniform_low_success_conditions": len(low),
        "mean_reference_standard_error": float(np.mean(se)),
        "within_two_reference_standard_errors": int(sum(d <= 2 * e for d, e in zip(dev_exact, se))),
        "mad_to_q_over_2n": float(np.mean(dev_simple)),
        "mad_to_random_function_reference": float(np.mean(dev_exact)),
    }


def main() -> None:
    out = {
        "budget_chain": chain_summary(20260305, 9),
        "budget_test_segment": budget_test_segment(),
        "sweep_hidden_types_strength_0p9": chain_summary(104, 5),
        "counterexamples": counterexamples(),
        "markov": markov_summary(),
        "hashing": hashing_summary(),
    }
    path = PAPER / "data" / "population_quantities.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
