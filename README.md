# Six Birds: Randomness Instantiation

This repository contains the **randomness instantiation** for the paper:

> **To Cast a Stone with Six Birds: A Closure-Deficit Account of Randomness under Packaging and Budget**
>
> DOI: https://doi.org/10.5281/zenodo.18926433
>
> Companion repository: https://github.com/ioannist/six-birds-randomness

This paper develops the Six Birds account of randomness as the predictive residue of non-closure under a chosen packaging, staging scale, and maintenance budget. The repository provides the controlled experiments, metrics, figure-generation code, and Lean-side formal support used to produce the manuscript artifacts.

## What this repository provides

The randomness instantiation implements:

- **Controlled Markov benchmark**: exact closure-deficit evaluation on exactly lumpable, perturbed-lumpable, metastable, and hidden-type families, together with route-mismatch diagnostics
- **Budgeted prediction experiments**: held-out packaged log-loss curves under increasing memory order, exposing randomness as a resource-relative prediction ledger
- **Hashing toy experiments**: bounded-budget inversion and collision-style audits showing how random-lookingness depends on packaging and input entropy rather than entropy creation
- **Representation-packaging experiments**: learned observation packagings via clustering and optional discrete encoders, used to stress-test whether extra representational budget actually improves predictive closure
- **Lean anchors**: a small machine-checked development in `lean/` supporting the KL bridge discussed in the paper
- **Publication artifacts**: paper figures are generated from repository-visible experiment outputs and compiled into the LaTeX manuscript in `paper/`

## Scope and limitations

The paper and repository are explicit about what they do and do not establish:

- The main evidentiary claims come from controlled finite-state and toy constructions; they are diagnostics of the framework, not universal theorems about randomness
- Route mismatch, predictive-gap proxies, and budget curves are operational shadows of the exact closure-deficit quantity, not replacements for it
- The hashing study is deliberately toy-sized and is used to illustrate feasibility-limited one-wayness, not to make cryptographic security claims
- The representation-learning appendix is auxiliary evidence and includes misalignment failures on purpose; more bins or richer codes are not assumed to improve packaging
- The Lean development certifies a narrow algebraic bridge used by the manuscript; it is not a full formalization of the paper

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"
cd lean && lake build
```

If you want to run the discrete-encoder experiment, install the optional neural dependency as well:

```bash
pip install -e ".[nn]"
```

## Test

```bash
make test
make lint
bash scripts/smoke.sh
```

## Run experiments

Run the full reproduction pipeline:

```bash
bash scripts/reproduce_all.sh
```

Or run the main experiment families individually:

```bash
python experiments/markov_bench/run_sweep.py --config configs/markov_bench_default.json
python experiments/markov_bench/analyze.py --run_dir results/markov_bench/<run_id>
python experiments/budget_curves/run_budget_curves.py --preset hidden_types_strong --max_order 5 --T 10000 --tau 1 --seed 20260305
python experiments/hashing_toy/run_hashing_toy.py --n_bits 8,12,16,20 --q_list 1,4,16,64,256,1024 --trials 300 --randtest_samples 3000 --randtest_rep_n_bits 16 --seed 20260306
python experiments/rep_packaging/make_dataset.py --preset hidden_types_strong --obs gaussian --d 4 --T 3000 --tau 1 --burn_in 200 --seed 20260305
python experiments/rep_packaging/run_clustering_packaging.py --dataset results/rep_packaging/<run_id>/dataset.npz --k_list 2,4,8 --seed 20260306
```

Optional neural packaging run:

```bash
python experiments/rep_packaging/train_discrete_encoder.py --dataset results/rep_packaging/<run_id>/dataset.npz
```

## Build paper

```bash
make -C paper pdf
```

The output PDF is written to `paper/build/main.pdf`.

## Repository notes

- The active bibliography is `paper/references.bib`
- The Zenodo related-works CSV is `assets/zenodo_related_works.csv`
- The figure generator used by the paper is `paper/scripts/make_publication_figures.py`
