"""Create synthetic representation-packaging datasets from micro chains."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomness_ledger.generators import (  # noqa: E402
    gen_exactly_lumpable,
    gen_hidden_types,
    gen_metastable,
    gen_perturbed_lumpable,
)
from randomness_ledger.markov import simulate_chain  # noqa: E402
from randomness_ledger.obs_models import (  # noqa: E402
    make_gaussian_emission_model,
    make_mixed_emission_model,
)


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


def _sample_observation(obs_model: dict, x_t: int, rng: np.random.Generator) -> np.ndarray:
    means = np.asarray(obs_model["means"], dtype=float)
    noise_scale = float(obs_model["noise_scale"])
    obs = means[int(x_t)] + noise_scale * rng.normal(size=means.shape[1])
    if not np.all(np.isfinite(obs)):
        raise ValueError("non-finite observation generated")
    return obs.astype(np.float32)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic observation dataset.")
    parser.add_argument(
        "--preset",
        default="hidden_types_strong",
        choices=["hidden_types_strong", "perturbed_medium", "exactly_lumpable", "metastable"],
    )
    parser.add_argument("--obs", default="gaussian", choices=["gaussian", "mixed"])
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--burn_in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--outdir", default="results/rep_packaging")
    args = parser.parse_args()

    if args.T < 1:
        raise ValueError("T must be >= 1")
    if args.tau < 1:
        raise ValueError("tau must be >= 1")
    if args.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if args.d < 1:
        raise ValueError("d must be >= 1")

    t0 = time.perf_counter()
    P, pi_map, meta, preset_params = _preset_build(args.preset, args.seed)
    n_micro = int(P.shape[0])
    k_macro = int(np.max(pi_map)) + 1

    micro_len = int(args.burn_in + (args.T - 1) * args.tau + 1)
    rng_states = np.random.default_rng(int(args.seed) + 17)
    x0 = int(rng_states.integers(n_micro))
    x_chain = simulate_chain(P, micro_len, x0=x0, rng=rng_states)
    x = np.asarray(x_chain[args.burn_in :: args.tau], dtype=np.int64)
    if x.shape[0] != args.T:
        raise RuntimeError(f"expected x length {args.T}, got {x.shape[0]}")
    y_true = np.asarray(pi_map[x], dtype=np.int64)

    if args.obs == "gaussian":
        obs_model = make_gaussian_emission_model(
            n_states=n_micro, d=args.d, seed=args.seed + 101, group_map=pi_map
        )
    else:
        obs_model = make_mixed_emission_model(
            n_states=n_micro, d=args.d, seed=args.seed + 101, group_map=pi_map
        )

    rng_obs = np.random.default_rng(int(args.seed) + 23)
    o = np.empty((args.T, args.d), dtype=np.float32)
    for t in range(args.T):
        o[t] = _sample_observation(obs_model, int(x[t]), rng_obs)
    if not np.isfinite(o).all():
        raise ValueError("non-finite values found in observations")

    run_id = _run_id()
    run_dir = Path(args.outdir) / run_id
    figs_dir = run_dir / "figs"
    run_dir.mkdir(parents=True, exist_ok=False)
    figs_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        x=x.astype(np.int64),
        o=o.astype(np.float32),
        y_true=y_true.astype(np.int64),
        pi_map=np.asarray(pi_map, dtype=np.int64),
        P=np.asarray(P, dtype=np.float64),
    )

    obs_summary = {
        "type": obs_model["type"],
        "d": int(obs_model["d"]),
        "noise_scale": float(obs_model["noise_scale"]),
    }
    config_payload = {
        "run_id": run_id,
        "cli_args": vars(args),
        "preset": args.preset,
        "preset_params": preset_params,
        "generator_meta": {"kind": meta.get("kind"), **{k: v for k, v in meta.items() if k != "K"}},
        "obs_model": obs_summary,
    }
    _write_json(run_dir / "config.json", config_payload)

    runtime_seconds = float(time.perf_counter() - t0)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "runtime_seconds": runtime_seconds,
    }
    _write_json(run_dir / "manifest.json", manifest)

    summary = {
        "run_id": run_id,
        "x_shape": list(x.shape),
        "o_shape": list(o.shape),
        "y_true_shape": list(y_true.shape),
        "n_micro": n_micro,
        "k_macro": k_macro,
        "x_unique": int(np.unique(x).size),
        "y_unique": int(np.unique(y_true).size),
        "o_mean": float(np.mean(o)),
        "o_std": float(np.std(o)),
    }
    _write_json(run_dir / "summary.json", summary)

    rel_dataset = dataset_path.as_posix()
    print(f"dataset_path: {rel_dataset}")
    print(f"x.shape: {x.shape}")
    print(f"o.shape: {o.shape}")
    print(f"y_true.shape: {y_true.shape}")
    print(f"n_micro: {n_micro}")
    print(f"k_macro: {k_macro}")
    print(f"unique_micro_visited: {int(np.unique(x).size)}")
    print(f"unique_macro_visited: {int(np.unique(y_true).size)}")
    print(f"o.mean: {float(np.mean(o))}")
    print(f"o.std: {float(np.std(o))}")


if __name__ == "__main__":
    main()
