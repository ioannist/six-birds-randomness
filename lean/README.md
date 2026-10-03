# RandomnessLedgerLean

Run `lake build` in this directory. Lean and mathlib are pinned to
`v4.29.0-rc4`; `lake-manifest.json` fixes the dependency commits. On a fresh
checkout, `lake exe cache get` can fetch mathlib's compiled dependencies.

`RandomnessLedgerLean/KLBridge.lean` exports ten mathematical declarations:

- The original totalized KL-to-log-ratio identity and nonnegativity statement,
  and the extended-valued zero-divergence characterization.
- An integrable probability KL bridge and a faithful zero-integral test.
- Automatic integrability on finite discrete probability spaces and a finite
  sum bridge, with absolute continuity as an explicit hypothesis.
- A faithful real-valued KL zero test under finiteness, and weighted zero tests
  identifying precisely the positive-weight rows. The finite version derives
  finiteness from absolute continuity rather than taking it as a hypothesis.

`Audit.lean` is a default build target and prints the transitive axioms of all
ten declarations. The reviewed build uses only `propext`, `Classical.choice`,
and `Quot.sound`, with no `sorryAx` or project-specific axioms.

The distinction between extended and real KL matters: `ENNReal.toReal` maps
infinity to zero, and Lean's Bochner integral is zero when a function is not
integrable. Absolute continuity alone therefore does not justify reading the
original general-space identity as a finite expected divergence. The added
finite-space bridge discharges integrability for the manuscript's setting.

This is a mechanization of the KL bridge and its weighted zero criterion. It
does not mechanize conditional mutual information, the entropy decomposition,
the construction of the stationary fiber mixture, Pinsker's inequality, or
the predictor-budget theorem. The weights and measures in the weighted
statements are explicit inputs; their connection to a packaged process is a
separate finite mathematical argument recorded in the review.

The smoke module checks imported interfaces; `Basic.lean` is unused mathematical
boilerplate. Neither is evidence for any additional theorem of the manuscript.
