# 2026-03-06 Experiment Closeout

## Frozen Evidence Pointers
- `artifacts/final/core_registry.json`
- `artifacts/final/CHECKSUMS.sha256`
- Figures: `artifacts/final/figs/`
- Tables: `artifacts/final/tables/`

## Final Scope Decision
- **Core evidence**: markov bench, budget curves, hashing toy.
- **Auxiliary evidence**: rep packaging clustering.
- **Excluded exploratory evidence**: rep packaging NN (excluded due to code collapse / used codes below configured k in prior run).
- **Optional appendix artifact**: Lean KL bridge (`lean/RandomnessLedgerLean/KLBridge.lean`, `lean/Smoke.lean`).

## Numeric Findings
- Markov RM-CD correlation across all rows is `r=0.959494`; see `artifacts/final/figs/markov_rm_vs_cd_scatter.png` and `artifacts/final/tables/markov_summary.json`.
- Markov decomposition residual max absolute value is `2.869e-16` (numerical noise range); see `artifacts/final/figs/markov_entropy_decomposition_stacked.png` and `artifacts/final/tables/markov_metrics.csv`.
- Markov worst-case closure deficit is `cd_max=0.048158` with `rm_uniform_max=0.275729`; source `artifacts/final/tables/markov_metrics.csv`.
- Budget curve theory level is `hyy_theory=1.049535`; order-1 gap is `nll1-hyy=-0.005597` and order-2 gap is `nll2-hyy=-0.010982`; see `artifacts/final/figs/budget_curve_hidden_types.png` and `artifacts/final/tables/budget_metrics.csv`.
- Budget feasible-class monotonicity check is `monotone_nonincreasing=true` over orders `[0, 1, 2, 3, 4, 5]`; see `artifacts/final/tables/budget_summary.json`.
- Hashing inversion (uniform inputs) has mean small-probability tracking error `uniform_mean_abs_error_to_q_over_2n=0.003697`; see `artifacts/final/figs/hashing_uniform_success_vs_budget.png` and `artifacts/final/tables/hashing_summary.json`.
- Hashing low-entropy regime reaches `success=1.000000` at `q=dictionary_size`; medium mixture is `0.463333`; see `artifacts/final/figs/hashing_success_vs_budget_by_dist_n16.png` and `artifacts/final/tables/hashing_metrics.csv`.
- Random-lookingness gate reports `uniform_not_grossly_flagged=true` and `low_entropy_detected=true`; see `artifacts/final/tables/hashing_randomness_tests.csv` and `artifacts/final/figs/hashing_byte_chi2_z_by_dist.png`.
- At representative `n_bits=16`, collision-ratio separation is strong: uniform `0.815831` vs low-entropy `255.296847` vs medium `66.242568`; see `artifacts/final/figs/hashing_collision_ratio_by_dist_n16.png`.
- Auxiliary rep clustering shows `cd_emp` trend `0.078180 -> 0.157873` across `k=[2, 4, 8]`, while `nll1` changes `0.626210 -> 2.004682`; see `artifacts/final/figs/rep_cd_vs_k.png`, `artifacts/final/figs/rep_nll_vs_k.png`, and `artifacts/final/tables/rep_clustering_metrics.csv`.

## Known Caveats
- Rep NN results are excluded from frozen evidence because earlier runs showed code collapse (`used_codes < k`), so same-budget comparison claims are not treated as final evidence.
- Rep clustering is auxiliary context only and is not required for the core paper claims built on markov, budget, and hashing evidence.
- `results/` remains ignored; canonical writing inputs are the tracked exports under `artifacts/final/` plus `artifacts/final/CHECKSUMS.sha256`.
- Lean KL bridge is a formal appendix aid for KL-based statements, not a full information-theory formalization layer.

## Writing-Readiness Gate
- Are there any more experiments planned? **No**.
- Any more code planned before writing? **No**.
- Any more results to collect before writing? **No**.
- Frozen artifact location for writing: **`artifacts/final/`**.
- Optional appendix artifact: **Lean KL bridge** (`lean/RandomnessLedgerLean/KLBridge.lean`).
