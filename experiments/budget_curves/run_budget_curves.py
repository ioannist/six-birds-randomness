"""Run budgeted predictor evaluations for macro-sequence NLL curves."""

from __future__ import annotations

import argparse
import csv
import json
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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomness_ledger.estimators import (  # noqa: E402
    fit_markov_order1,
    fit_markov_order2,
    nll_order1,
    nll_order2,
)
from randomness_ledger.generators import (  # noqa: E402
    gen_exactly_lumpable,
    gen_hidden_types,
    gen_metastable,
    gen_perturbed_lumpable,
)
from randomness_ledger.markov import simulate_chain, stationary_dist  # noqa: E402
from randomness_ledger.metrics import macro_cond_entropy, packaged_history_entropy  # noqa: E402


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _preset_build(preset: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if preset == "hidden_types_strong":
        params = {
            "n_macro": 3,
            "fiber_sizes": [3, 2, 3],
            "type_split": 0.5,
            "seed": int(seed),
            "strength": 0.9,
        }
        P, pi_map, meta = gen_hidden_types(**params)
        return P, pi_map, meta, params
    if preset == "perturbed_medium":
        params = {
            "n_macro": 3,
            "fiber_sizes": [2, 2, 2],
            "seed": int(seed),
            "aperiodic_eps": 1e-3,
            "heterogeneity_alpha": 0.5,
        }
        P, pi_map, meta = gen_perturbed_lumpable(**params)
        return P, pi_map, meta, params
    if preset == "exactly_lumpable":
        params = {
            "n_macro": 3,
            "fiber_sizes": [2, 2, 2],
            "seed": int(seed),
            "aperiodic_eps": 1e-3,
        }
        P, pi_map, meta = gen_exactly_lumpable(**params)
        return P, pi_map, meta, params
    if preset == "metastable":
        params = {
            "block_sizes": [3, 3, 2],
            "p_in": 1.0,
            "p_out": 0.08,
            "seed": int(seed),
        }
        P, pi_map, meta = gen_metastable(**params)
        return P, pi_map, meta, params
    raise ValueError(f"unsupported preset: {preset}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.tau < 1:
        raise ValueError("tau must be >= 1")
    if args.T < 20:
        raise ValueError("T must be >= 20")
    if args.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0.0 < args.validation_frac < 1.0):
        raise ValueError("validation_frac must be in (0, 1)")
    if args.max_order < 0 or args.max_order > 5:
        raise ValueError("max_order must satisfy 0 <= max_order <= 5")


def _fit_order0(train: np.ndarray, k: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.bincount(train, minlength=k).astype(float) + float(smoothing)
    probs = counts / counts.sum()
    return probs


def _nll_order0(test: np.ndarray, p0: np.ndarray, start: int = 1) -> float:
    if not 0 <= start < len(test):
        raise ValueError("invalid evaluation start")
    probs = p0[test[start:]]
    with np.errstate(divide="ignore"):
        return float(-np.mean(np.log(probs)))


def _context_ids(sequence: np.ndarray, L: int, k: int) -> np.ndarray:
    """Encode each length-L context window into base-k integer id."""
    if L == 0:
        return np.zeros(len(sequence), dtype=np.int64)
    if len(sequence) < L:
        return np.empty(0, dtype=np.int64)

    powers = (k ** np.arange(L - 1, -1, -1, dtype=np.int64)).astype(np.int64)
    n_ctx = len(sequence) - L + 1
    ids = np.empty(n_ctx, dtype=np.int64)
    for i in range(n_ctx):
        ids[i] = int(np.dot(sequence[i : i + L], powers))
    return ids


def _fit_orderL(train: np.ndarray, k: int, L: int, smoothing: float = 1e-6) -> np.ndarray:
    """Fit order-L conditional model with context encoding."""
    if L < 1:
        raise ValueError("L must be >= 1")
    if len(train) < L + 1:
        raise ValueError("train sequence too short for requested order")

    n_states = int(k**L)
    counts = np.full((n_states, k), float(smoothing), dtype=float)

    ctx_ids = _context_ids(train[:-1], L, k)
    targets = train[L:]
    np.add.at(counts, (ctx_ids, targets), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0.0)
    zero_rows = row_sums[:, 0] == 0.0
    if np.any(zero_rows):
        probs[zero_rows] = 1.0 / k
    return probs


def _nll_orderL(
    test: np.ndarray, model: np.ndarray, k: int, L: int, start: int | None = None
) -> float:
    """Evaluate average next-step NLL for order-L model on held-out test."""
    if L < 1:
        raise ValueError("L must be >= 1")
    if len(test) < L + 1:
        raise ValueError("test sequence too short for requested order")

    first = L if start is None else start
    if not L <= first < len(test):
        raise ValueError("invalid evaluation start")
    ctx_ids = _context_ids(test[:-1], L, k)[first - L :]
    probs = model[ctx_ids, test[first:]]
    with np.errstate(divide="ignore"):
        return float(-np.mean(np.log(probs)))


def _simulate_macro_sequence(
    P: np.ndarray, pi_map: np.ndarray, tau: int, T: int, burn_in: int, seed: int
) -> np.ndarray:
    """Simulate micro chain and subsample every tau steps to macro sequence."""
    rng = np.random.default_rng(int(seed))
    n = P.shape[0]
    micro_len = burn_in + (T - 1) * tau + 1
    x0 = int(rng.integers(n))
    x_chain = simulate_chain(P, T=micro_len, x0=x0, rng=rng)
    x_sub = x_chain[burn_in::tau]
    if len(x_sub) != T:
        raise RuntimeError(f"expected macro length {T}, got {len(x_sub)}")
    y_seq = np.asarray(pi_map[x_sub], dtype=np.int64)
    return y_seq


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run budgeted NLL curves for macro predictors.")
    parser.add_argument(
        "--preset",
        default="hidden_types_strong",
        choices=["hidden_types_strong", "perturbed_medium", "exactly_lumpable", "metastable"],
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--burn_in", type=int, default=1000)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--validation_frac", type=float, default=0.5,
                        help="Fraction of post-training data reserved for model selection.")
    parser.add_argument("--max_order", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--outdir", default="results/budget_curves")
    args = parser.parse_args()

    _validate_args(args)
    t0 = time.perf_counter()

    P, pi_map, meta, preset_params = _preset_build(args.preset, args.seed)
    pi_micro = stationary_dist(P)
    hyy_theory = float(macro_cond_entropy(P, pi_map, args.tau, pi_stationary=pi_micro))

    y_seq = _simulate_macro_sequence(P, pi_map, args.tau, args.T, args.burn_in, args.seed + 77)
    k = int(np.max(pi_map)) + 1
    T_train = int(args.train_frac * args.T)
    train = y_seq[:T_train]
    heldout = y_seq[T_train:]
    n_validation = int(args.validation_frac * len(heldout))
    validation, test = heldout[:n_validation], heldout[n_validation:]
    start = max(1, args.max_order)
    if len(train) <= args.max_order or len(validation) <= start:
        raise ValueError("train or validation split is too short for requested order")
    if len(test) < args.max_order + 5:
        raise ValueError(
            f"test length {len(test)} too short; need at least max_order+5={args.max_order + 5}"
        )

    run_id = _run_id()
    run_dir = Path(args.outdir) / run_id
    figs_dir = run_dir / "figs"
    run_dir.mkdir(parents=True, exist_ok=False)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    models: list[np.ndarray] = []
    for L in range(args.max_order + 1):
        if L == 0:
            model = _fit_order0(train, k)
        elif L == 1:
            model = fit_markov_order1(train, k)
        elif L == 2:
            model = fit_markov_order2(train, k)
        else:
            model = _fit_orderL(train, k, L)
        models.append(model)

    def score(sequence: np.ndarray, model: np.ndarray, L: int) -> float:
        if L == 0:
            return _nll_order0(sequence, model, start=start)
        if L == 1:
            return nll_order1(sequence, model, start=start)
        if L == 2:
            return nll_order2(sequence, model, start=start)
        return _nll_orderL(sequence, model, k, L, start=start)

    validation_losses = np.array([score(validation, m, L) for L, m in enumerate(models)])
    test_losses = np.array([score(test, m, L) for L, m in enumerate(models)])
    oracle_losses = np.minimum.accumulate(test_losses)
    for L in range(args.max_order + 1):
        selected = int(np.argmin(validation_losses[: L + 1]))
        rows.append({
            "run_id": run_id,
            "preset": args.preset,
            "seed": args.seed,
            "tau": args.tau,
            "T": args.T,
            "train_frac": args.train_frac,
            "burn_in": args.burn_in,
            "k_macro": k,
            "order": L,
            "budget_states": int(k**L),
            "budget_params": int((k**L) * (k - 1)),
            # Legacy names retained, with explicit semantics below and in config.
            "nll_exact": float(test_losses[L]),
            "nll": float(oracle_losses[L]),
            "nll_empirical_oracle": float(oracle_losses[L]),
            "validation_nll": float(validation_losses[L]),
            "selected_order": selected,
            "nll_selected": float(test_losses[selected]),
            "evaluation_start": start,
            "n_test_targets": len(test) - start,
            "n_validation_targets": len(validation) - start,
            "hyy_theory": hyy_theory,
            "history_entropy_theory": packaged_history_entropy(
                P, pi_map, args.tau, L, pi_stationary=pi_micro
            ),
            "delta_to_hyy": float(oracle_losses[L] - hyy_theory),
        })

    fieldnames = [
        "run_id",
        "preset",
        "seed",
        "tau",
        "T",
        "train_frac",
        "burn_in",
        "k_macro",
        "order",
        "budget_states",
        "budget_params",
        "nll",
        "nll_exact",
        "hyy_theory",
        "delta_to_hyy",
        "nll_empirical_oracle",
        "validation_nll",
        "selected_order",
        "nll_selected",
        "evaluation_start",
        "n_test_targets",
        "n_validation_targets",
        "history_entropy_theory",
    ]
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    orders = np.array([int(r["order"]) for r in rows], dtype=int)
    nlls = np.array([float(r["nll"]) for r in rows], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(orders, nlls, marker="o", label="test oracle envelope")
    plt.plot(orders, [r["nll_selected"] for r in rows], marker="s", label="validation selected")
    plt.plot(orders, [r["history_entropy_theory"] for r in rows], label="population entropy")
    plt.legend()
    plt.axhline(hyy_theory, linestyle="--", linewidth=1.0)
    plt.xlabel("order")
    plt.ylabel("nll")
    plt.title(f"Budget Curve (hyy_theory={hyy_theory:.6f})")
    plt.xticks(orders)
    plt.tight_layout()
    plt.savefig(figs_dir / "budget_curve.png", dpi=150)
    plt.close()

    increases = nlls[1:] - nlls[:-1] if len(nlls) >= 2 else np.array([], dtype=float)
    max_increase = float(np.max(np.maximum(increases, 0.0))) if increases.size else 0.0
    monotone = bool(np.all(increases <= 1e-3)) if increases.size else True

    summary = {
        "hyy_theory": hyy_theory,
        "oracle_monotonicity_is_by_construction": True,
        "evaluation_start": start,
        "n_test_targets": len(test) - start,
        "validation_selected": [{"order": r["order"], "selected_order": r["selected_order"],
                                  "nll": r["nll_selected"]} for r in rows],
        "history_entropies_theory": [r["history_entropy_theory"] for r in rows],
        "order_nll": [{"order": int(o), "nll": float(v)} for o, v in zip(orders, nlls)],
        "monotone_nonincreasing": monotone,
        "max_increase": max_increase,
    }
    _write_json(run_dir / "summary.json", summary)

    config_payload = {
        "run_id": run_id,
        "cli_args": vars(args),
        "nll_semantics": "nll is a test oracle minimum over fitted models; nll_exact is per-order test loss",
        "nll_selected_semantics": "model chosen on validation, scored on untouched common test targets",
        "population_semantics": "history_entropy_theory is enumerated from the known stationary chain",
        "preset": args.preset,
        "preset_params": preset_params,
        "generator_meta": {"kind": meta.get("kind"), **{k: v for k, v in meta.items() if k != "K"}},
    }
    _write_json(run_dir / "config.json", config_payload)

    runtime_seconds = float(time.perf_counter() - t0)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "runtime_seconds": runtime_seconds,
    }
    _write_json(run_dir / "manifest.json", manifest)

    print(str(run_dir))


if __name__ == "__main__":
    main()
