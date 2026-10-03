# Paper v2 plan — 2026-10-03

Goal: bring the manuscript in line with the mathematical review of 2026-10-03
(`2026-10-03_mathematical_review.md`, commit `e22734f`), rewrite it for
readability, and release it as **v2**. The title stays unchanged. The v1 date
(March 9, 2026) stays on the cover, with v2 dated October 3, 2026.

## Ground rules

- The body reads as a self-contained paper with no revision language. A short
  version-history appendix records what changed relative to v1.
- Numbers come from the corrected outputs in `artifacts/math-review/` and from
  population quantities computed by a tracked script. The v1 freeze in
  `artifacts/final/` is untouched.
- Every revised text is reviewed by codex (gpt-6.1-sol, high effort) in a single
  persistent thread (`paper/.codex_thread_id`). Codex checks facts and
  mathematics; style is Claude's responsibility. The text is not final until
  codex signs off.

## Steps

1. **Data.** Mirror the corrected CSVs into `paper/data/`. Add
   `paper/scripts/derive_population_quantities.py`, which writes the exact
   population numbers (budget-chain CD, intrinsic term, history entropies, and
   the counterexample values) to `paper/data/population_quantities.json`.
2. **Mathematics in the text.**
   - Prop. 2 is restricted to occupied microstates, with an explicit
     zero-weight convention.
   - Prop. 3 keeps full support, plus a remark with the null-state
     counterexample.
   - Prop. 4 keeps the stationary lift, plus the uniform-lift counterexample.
   - New proposition (memory sandwich): for a Markov substrate,
     H(Z|X) ≤ H(Z|S_L) ≤ H(Z|Y), the history gain is at most CD, and entropy
     decreases monotonically in L. The population predictive gap is therefore
     bounded by CD.
   - Rand_B monotonicity is stated only for nested feasible classes.
   - "Intrinsic" is qualified by sufficiency of the microstate.
   - The lag/refinement non-monotonicity counterexample (four-cycle).
3. **Results text.**
   - Budget: population staircase, validation-selected test loss, and the test
     oracle envelope. Saturation is not described as exhaustion. The budget
     chain is a different seed from the sweep chain.
   - Markov: remove cost claims and the general staging claim.
   - Hashing: random-function reference, success meaning any preimage, a
     heuristic reading of the test thresholds, and which digests feed which test.
   - Clustering: descriptive plug-in CMI with its caveats.
   - Lean appendix: integrability, the finite bridge, weighted zero tests, and
     coverage limits.
4. **Figures.**
   - New two-panel concept figure (closed vs. non-closed fiber).
   - Markov: RM vs CD on log axes with the Pinsker curve, plus a family-knob
     panel.
   - Budget: population staircase with the intrinsic floor and fitted losses.
   - Hashing: success vs the exact reference with error bars, plus input-regime
     panels.
   - Clustering: merged two-panel figure.
5. **Prose.** New abstract, plus heavy edits throughout for plain, direct
   language.
6. **Cover.** "v1: March 9, 2026 · v2: October 3, 2026", plus a version-history
   appendix.
7. **Build and read.** Build the PDF, read every page, and fix layout.
8. **Codex review.** Bootstrap the review thread, then review in passes:
   (a) Secs. 1–4 and the Lean appendix, (b) Secs. 5–8, the abstract, and the
   remaining appendices, (c) a final whole-paper sign-off. Fix and iterate until
   codex signs off.
9. **Commit.**
