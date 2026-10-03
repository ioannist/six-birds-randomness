"""Export review evidence without modifying the frozen publication inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-log", type=Path, required=True)
    parser.add_argument("--lean-log", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts/math-review")
    args = parser.parse_args()

    run_keys = {"markov_bench", "budget_curves", "hashing_toy",
                "rep_packaging_dataset", "rep_packaging_clustering"}
    runs = {}
    for line in args.reproduction_log.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in run_keys:
                runs[key] = ROOT / value
    if set(runs) != run_keys:
        raise ValueError("reproduction log does not identify all five completed runs")

    lean_log = args.lean_log.read_text()
    declarations = set(re.findall(r"^theorem (\w+)",
        (ROOT / "lean/RandomnessLedgerLean/KLBridge.lean").read_text(), re.MULTILINE))
    axioms = {name: [a.strip() for a in values.split(",") if a.strip()]
              for name, values in re.findall(
                  r"'RandomnessLedgerLean\.(\w+)' depends on axioms: \[(.*?)\]",
                  lean_log, re.DOTALL)}
    if set(axioms) != declarations or "Build completed successfully" not in lean_log:
        raise ValueError("Lean log does not cover all exported mathematical declarations")
    if any(set(values) - {"propext", "Classical.choice", "Quot.sound"} for values in axioms.values()):
        raise ValueError("unexpected transitive Lean axiom")
    passed = re.search(r"(\d+) passed", args.test_log.read_text())
    if passed is None or re.search(r"\d+ (failed|error)", args.test_log.read_text()):
        raise ValueError("test log does not report a passing suite")

    frozen = ROOT / "artifacts/final"
    checked = 0
    for entry in (frozen / "CHECKSUMS.sha256").read_text().splitlines():
        checksum, relative = entry.split(maxsplit=1)
        if digest(frozen / relative) != checksum:
            raise ValueError(f"frozen artifact changed: {relative}")
        checked += 1

    markov = read_rows(runs["markov_bench"] / "metrics.csv")
    old_markov = read_rows(frozen / "tables/markov_metrics.csv")
    if len(markov) != len(old_markov) or len(markov) != 17:
        raise ValueError("default Markov sweep does not cover all 17 conditions")
    identity_fields = ("family", "seed", "tau", "heterogeneity_alpha", "strength", "p_out")
    if any(any(a.get(k) != b.get(k) for k in identity_fields) for a, b in zip(markov, old_markov)):
        raise ValueError("Markov conditions differ from the frozen sweep")
    cd = np.array([float(r["cd"]) for r in markov])
    rm_s = np.array([float(r["rm_stationary"]) for r in markov])
    rm_u = np.array([float(r["rm_uniform"]) for r in markov])
    residuals = np.array([float(r["resid"]) for r in markov])
    if not np.all(np.isfinite(cd)) or np.min(cd - .5 * rm_s**2) < -1e-12:
        raise ValueError("Markov KL/Pinsker check failed")
    if np.max(np.abs(residuals)) > 1e-12:
        raise ValueError("Markov entropy decomposition check failed")

    budget = read_rows(runs["budget_curves"] / "metrics.csv")
    population = [float(r["history_entropy_theory"]) for r in budget]
    if np.any(np.diff(population) > 1e-12):
        raise ValueError("population history entropy increased")
    if len({r["n_test_targets"] for r in budget}) != 1:
        raise ValueError("budget models were evaluated on unequal target counts")
    for L, row in enumerate(budget):
        selected = int(np.argmin([float(r["validation_nll"]) for r in budget[:L+1]]))
        if int(row["selected_order"]) != selected:
            raise ValueError("budget selector does not follow validation losses")
        if float(row["nll_selected"]) != float(budget[selected]["nll_exact"]):
            raise ValueError("selected test loss does not match the selected model")

    hashing = read_rows(runs["hashing_toy"] / "metrics.csv")
    old_hashing = read_rows(frozen / "tables/hashing_metrics.csv")
    if len(hashing) != len(old_hashing) or len(hashing) != 72:
        raise ValueError("hashing sweep does not cover the frozen conditions")
    for a, b in zip(hashing, old_hashing):
        if any(a[k] != b[k] for k in ("distribution", "n_bits", "q", "empirical_success")):
            raise ValueError("hashing samples differ from the frozen experiment")

    sources = sorted(list((ROOT / "src/randomness_ledger").glob("*.py"))
                     + list((ROOT / "experiments").rglob("*.py"))
                     + list((ROOT / "tests").glob("*.py"))
                     + list((ROOT / "configs").glob("*.json"))
                     + list((ROOT / "lean").glob("*.lean"))
                     + list((ROOT / "lean/RandomnessLedgerLean").glob("*.lean"))
                     + [ROOT / "lean/lakefile.toml", ROOT / "lean/lake-manifest.json",
                        ROOT / "lean/lean-toolchain", Path(__file__)])
    evidence = {
        "baseline_commit": "a53368b",
        "scope": "mathematics and implementation review; publication inputs remain frozen",
        "review_method": "proof reconstruction and adversarial self-review; no independent reviewer",
        "test_count": int(passed[1]),
        "lean_axioms": axioms,
        "frozen_checksums_verified": checked,
        "runs": {k: str(v.relative_to(ROOT)) for k, v in runs.items()},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "markov": {
            "conditions": len(markov),
            "max_abs_decomposition_residual": float(np.max(np.abs(residuals))),
            "min_pinsker_margin": float(np.min(cd - .5 * rm_s**2)),
            "uniform_rm_cd_correlation": float(np.corrcoef(rm_u, cd)[0, 1]),
            "max_cd_change_from_frozen": max(abs(float(a["cd"]) - float(b["cd"]))
                                              for a, b in zip(markov, old_markov)),
        },
        "budget": {
            "history_entropies_population": population,
            "order1_to_order2_population_gain": population[1] - population[2],
            "order2_to_order3_population_gain": population[2] - population[3],
            "validation_selected_test_losses": [float(r["nll_selected"]) for r in budget],
            "test_oracle_envelope": [float(r["nll_empirical_oracle"]) for r in budget],
            "n_test_targets": int(budget[0]["n_test_targets"]),
            "oracle_monotonicity_is_by_construction": True,
        },
        "hashing": {"conditions": len(hashing), "empirical_successes_match_frozen": True,
                    "baseline": "uniform target under an ideal random function, including input overlap"},
        "numerical_scope": "floating finite-model evaluation; no interval certificates or universal sampled proof",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for kind, filename in (("markov_bench", "markov_metrics.csv"),
                           ("budget_curves", "budget_metrics.csv"),
                           ("hashing_toy", "hashing_metrics.csv"),
                           ("rep_packaging_clustering", "rep_clustering_metrics.csv")):
        shutil.copyfile(runs[kind] / "metrics.csv", args.outdir / filename)
    shutil.copyfile(runs["hashing_toy"] / "randomness_tests.csv", args.outdir / "hashing_randomness_tests.csv")
    for source, filename in ((args.lean_log, "lean-build.txt"),
                             (args.test_log, "tests.txt"),
                             (args.reproduction_log, "reproduction.txt")):
        shutil.copyfile(source, args.outdir / filename)
    (args.outdir / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.outdir), "tests": int(passed[1]),
                      "lean_declarations": len(axioms), "markov_conditions": len(markov)}, sort_keys=True))


if __name__ == "__main__":
    main()
