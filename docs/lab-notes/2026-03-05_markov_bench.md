# Markov Bench Lab Note — 2026-03-05

## Header
- Date: 2026-03-05
- Run id: `20260305_123032_89c4cc93`
- Commands used to produce this run:
  - `python experiments/markov_bench/run_sweep.py --config configs/markov_bench_default.json`
  - `python experiments/markov_bench/analyze.py`
- Artifacts:
  - `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`
  - `results/markov_bench/20260305_123032_89c4cc93/figs/rm_vs_cd_scatter.png`
  - `results/markov_bench/20260305_123032_89c4cc93/figs/cd_vs_heterogeneity_by_tau.png`
  - `results/markov_bench/20260305_123032_89c4cc93/figs/entropy_decomposition_stacked.png`

## Families Run
- `exactly_lumpable`
- `perturbed_lumpable`
- `metastable`
- `hidden_types`

## Observations
- Across all finite rows, RM–CD Pearson correlation is `r_all = 0.959494`, showing a strong positive association (see `results/markov_bench/20260305_123032_89c4cc93/figs/rm_vs_cd_scatter.png`).
- Within `perturbed_lumpable`, RM–CD Pearson correlation is similarly high at `r_pert = 0.958938` (see `results/markov_bench/20260305_123032_89c4cc93/figs/rm_vs_cd_scatter.png` and `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- Global CD spans `3.148e-17` to `4.815848e-02`; the max occurs at `family=perturbed_lumpable`, `tau=1`, `heterogeneity_alpha=0.5` (see `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- Global RM (uniform lift) spans `5.287e-17` to `2.757292e-01`; the max is at the same condition (`perturbed_lumpable`, `tau=1`, `heterogeneity_alpha=0.5`) (see `results/markov_bench/20260305_123032_89c4cc93/metrics.csv` and `results/markov_bench/20260305_123032_89c4cc93/figs/rm_vs_cd_scatter.png`).
- `exactly_lumpable` is near closure: CD in `[3.148e-17, 4.345e-17]` and RM in `[5.287e-17, 1.211e-16]` (see `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- `perturbed_lumpable` shows the widest spread: CD in `[3.250e-17, 4.815848e-02]` and RM in `[1.405e-16, 2.757292e-01]`; this aligns with increasing heterogeneity settings (see `results/markov_bench/20260305_123032_89c4cc93/figs/cd_vs_heterogeneity_by_tau.png` and `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- `metastable` remains low but nonzero: CD in `[3.362e-05, 7.219e-04]` and RM in `[3.215e-03, 2.295e-02]` (see `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- `hidden_types` shows elevated mismatch/deficit compared to metastable: CD in `[4.117e-04, 3.095729e-02]` and RM in `[2.318534e-02, 2.124932e-01]` (see `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- Decomposition closure is numerically tight: `max(abs(resid)) = 2.868799e-16`, consistent with `HYY ≈ intrinsic + CD` in the stacked decomposition figure (see `results/markov_bench/20260305_123032_89c4cc93/figs/entropy_decomposition_stacked.png` and `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).
- Lift sensitivity is modest overall but nonzero: `max |rm_uniform - rm_stationary| = 1.189466e-02` at `family=hidden_types`, `tau=1`, `strength=0.9`, where RM goes from `0.212493` (uniform) to `0.200599` (stationary) (see `results/markov_bench/20260305_123032_89c4cc93/metrics.csv`).

## Anything Surprising
- No major surprises in this run.
- The strongest CD and RM both occur in the same `perturbed_lumpable` condition (`tau=1`, `heterogeneity_alpha=0.5`), reinforcing that this sweep axis is the dominant driver in this config.
- Lift choice had its largest effect in `hidden_types` (`|ΔRM| ≈ 0.011895`), which is noticeable but still much smaller than the full RM scale in that family.
