"""Run parameter sweeps for Markov coarse-graining benchmarks."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import shlex
import sys
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
from randomness_ledger.markov import stationary_dist  # noqa: E402
from randomness_ledger.metrics import (  # noqa: E402
    closure_deficit,
    decomposition_check,
    intrinsic_term,
    macro_cond_entropy,
    route_mismatch,
    step_entropy,
)
from randomness_ledger.packaging import macro_kernel  # noqa: E402


FAMILY_DISPATCH = {
    "exactly_lumpable": gen_exactly_lumpable,
    "perturbed_lumpable": gen_perturbed_lumpable,
    "metastable": gen_metastable,
    "hidden_types": gen_hidden_types,
}


def _ensure_int_list(name: str, values: Any) -> list[int]:
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty list")
    out: list[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must contain integers")
        out.append(int(v))
    return out


def _validate_family_entry(entry: dict[str, Any]) -> None:
    if "family" not in entry:
        raise ValueError("each family entry must include 'family'")
    family = entry["family"]
    if family not in FAMILY_DISPATCH:
        raise ValueError(f"unsupported family: {family}")

    fixed = entry.get("fixed", {})
    sweep = entry.get("sweep", {})
    taus = entry.get("taus")
    seeds = entry.get("seeds")

    if not isinstance(fixed, dict):
        raise ValueError(f"{family}.fixed must be an object")
    if not isinstance(sweep, dict):
        raise ValueError(f"{family}.sweep must be an object")
    for key, vals in sweep.items():
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"{family}.sweep.{key} must be a non-empty list")
    _ensure_int_list(f"{family}.taus", taus)
    _ensure_int_list(f"{family}.seeds", seeds)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be an object")

    if "run_name" not in cfg:
        raise ValueError("config must include run_name")
    if "replicates" not in cfg:
        raise ValueError("config must include replicates")
    if "families" not in cfg:
        raise ValueError("config must include families")

    if isinstance(cfg["replicates"], bool) or not isinstance(cfg["replicates"], int):
        raise ValueError("replicates must be an integer")
    if cfg["replicates"] < 1:
        raise ValueError("replicates must be >= 1")

    families = cfg["families"]
    if not isinstance(families, list) or len(families) == 0:
        raise ValueError("families must be a non-empty list")
    for entry in families:
        if not isinstance(entry, dict):
            raise ValueError("each family entry must be an object")
        _validate_family_entry(entry)
    return cfg


def _expand_sweep(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def _scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    for row in rows:
        fam = str(row["family"])
        families[fam] = families.get(fam, 0) + 1

    cd_values = np.asarray([float(r["cd"]) for r in rows], dtype=float)
    rm_values = np.asarray([float(r["rm_uniform"]) for r in rows], dtype=float)
    return {
        "counts_by_family": families,
        "cd": {
            "min": float(np.min(cd_values)),
            "mean": float(np.mean(cd_values)),
            "max": float(np.max(cd_values)),
        },
        "rm_uniform": {
            "min": float(np.min(rm_values)),
            "mean": float(np.mean(rm_values)),
            "max": float(np.max(rm_values)),
        },
    }


def run_sweep(config_path: Path, outdir: Path) -> Path:
    cfg = _load_config(config_path)
    run_id = _run_id()
    created_utc = datetime.now(timezone.utc).isoformat()

    run_dir = outdir / run_id
    figs_dir = run_dir / "figs"
    run_dir.mkdir(parents=True, exist_ok=False)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for fam_cfg in cfg["families"]:
        family = fam_cfg["family"]
        generator = FAMILY_DISPATCH[family]
        fixed = dict(fam_cfg.get("fixed", {}))
        sweep = dict(fam_cfg.get("sweep", {}))
        taus = _ensure_int_list(f"{family}.taus", fam_cfg["taus"])
        base_seeds = _ensure_int_list(f"{family}.seeds", fam_cfg["seeds"])

        combos = _expand_sweep(sweep)
        for combo in combos:
            for base_seed in base_seeds:
                for replicate in range(int(cfg["replicates"])):
                    gen_seed = int(base_seed + 10_000 * replicate)
                    gen_params = dict(fixed)
                    gen_params.update(combo)
                    gen_params["seed"] = gen_seed

                    P, pi_map, meta = generator(**gen_params)
                    pi_micro = stationary_dist(P)
                    k_macro = int(np.max(pi_map)) + 1
                    n_micro = int(P.shape[0])

                    for tau in taus:
                        rm_uniform = route_mismatch(
                            P,
                            pi_map,
                            tau=tau,
                            lift="uniform",
                            pi_stationary=pi_micro,
                            norm="l1",
                        )
                        rm_stationary = route_mismatch(
                            P,
                            pi_map,
                            tau=tau,
                            lift="stationary",
                            pi_stationary=pi_micro,
                            norm="l1",
                        )

                        macroP_uniform = macro_kernel(
                            P, pi_map, tau=tau, lift="uniform", pi_stationary=pi_micro
                        )
                        macroP_stationary = macro_kernel(
                            P, pi_map, tau=tau, lift="stationary", pi_stationary=pi_micro
                        )

                        row: dict[str, Any] = {
                            "run_id": run_id,
                            "run_name": cfg["run_name"],
                            "family": family,
                            "kind": meta.get("kind", family),
                            "seed": gen_seed,
                            "base_seed": base_seed,
                            "replicate": replicate,
                            "tau": tau,
                            "n_micro": n_micro,
                            "k_macro": k_macro,
                            "rm_uniform": float(rm_uniform),
                            "rm_stationary": float(rm_stationary),
                            "step_entropy_uniform": float(step_entropy(macroP_uniform)),
                            "step_entropy_stationary": float(step_entropy(macroP_stationary)),
                            "intrinsic": float(
                                intrinsic_term(P, pi_map, tau=tau, pi_stationary=pi_micro)
                            ),
                            "hyy": float(
                                macro_cond_entropy(P, pi_map, tau=tau, pi_stationary=pi_micro)
                            ),
                            "cd": float(
                                closure_deficit(P, pi_map, tau=tau, pi_stationary=pi_micro)
                            ),
                            "resid": float(
                                decomposition_check(P, pi_map, tau=tau, pi_stationary=pi_micro)
                            ),
                        }
                        for key, value in fixed.items():
                            row[key] = _scalar_for_csv(value)
                        for key, value in combo.items():
                            row[key] = _scalar_for_csv(value)
                        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("no rows produced; check config")

    base_fields = [
        "run_id",
        "run_name",
        "family",
        "kind",
        "seed",
        "base_seed",
        "replicate",
        "tau",
        "n_micro",
        "k_macro",
        "rm_uniform",
        "rm_stationary",
        "step_entropy_uniform",
        "step_entropy_stationary",
        "intrinsic",
        "hyy",
        "cd",
        "resid",
    ]
    dynamic_fields = sorted({k for row in rows for k in row.keys() if k not in base_fields})
    fieldnames = base_fields + dynamic_fields

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _scalar_for_csv(row.get(k, "")) for k in fieldnames})

    config_out = dict(cfg)
    config_out["run_id"] = run_id
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, sort_keys=True)

    manifest = {
        "run_id": run_id,
        "created_utc": created_utc,
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "row_count": len(rows),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_summary(rows), f, indent=2, sort_keys=True)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov coarse-graining sweep.")
    parser.add_argument("--config", required=True, help="Path to JSON sweep config.")
    parser.add_argument(
        "--outdir",
        default="results/markov_bench",
        help="Output base directory (default: results/markov_bench).",
    )
    args = parser.parse_args()

    run_dir = run_sweep(Path(args.config), Path(args.outdir))
    print(str(run_dir))


if __name__ == "__main__":
    main()
