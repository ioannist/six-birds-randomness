Tracked inputs for the manuscript figures and quoted numbers (v2).

- `markov_metrics.csv`, `budget_metrics.csv`, `hashing_metrics.csv`,
  `hashing_randomness_tests.csv`, `rep_clustering_metrics.csv`: corrected
  experiment outputs, mirrored from `artifacts/math-review/`.
- `population_quantities.json`: exact population quantities computed from the
  known kernels by `scripts/derive_population_quantities.py`.
- `core_registry.json`: registry of the v1 publication freeze. The v1 inputs
  remain in `artifacts/final/` and are unchanged.

Rebuild: `python scripts/derive_population_quantities.py`, then
`python scripts/make_publication_figures.py`, then `make pdf`.
