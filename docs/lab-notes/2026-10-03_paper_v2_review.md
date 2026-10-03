# Paper v2 — codex review log, 2026-10-03

Reviewer: codex CLI, model `gpt-6.1-sol`, reasoning effort high, read-only
sandbox, one persistent thread (id in `paper/.codex_thread_id`, untracked).
Scope: factual correctness and mathematics only. Claude wrote the text and applied
every fix; codex re-checked each fix on the next pass.

## Pass A — Sections 1–4, Appendix A (counterexamples), Appendix C (Lean), Figure 1

Six findings, all accepted:

- **Blocker.** In the Lean appendix, the finiteness hypothesis was attached to the
  real-valued (`toReal`) readout. It must be attached to the extended-valued
  divergence, which is what `klDiv ≠ ⊤` states.
- **Major.**
  - "A small mismatch does not force a small deficit" was replaced. Pinsker alone
    gives no upper bound; entropy continuity gives one that depends on the
    alphabet size.
  - Fitted held-out losses estimate the risks of the fitted predictors. They do
    not estimate the population gap or `Rand_B` directly.
  - "The question" in the setup paragraph now compares occupied microstates
    only, since unoccupied ones can differ without creating randomness.
  - The version history no longer calls the stationary solver "certified".
- **Minor.** The v1 reproduction bound now applies to the CD values only.

Codex verified every proposition, all three counterexamples exactly, the Lean
paraphrases, the axiom audit, and the Figure 1 mixtures.

## Pass B — abstract, Sections 5–8, Appendices B and D, Figures 2–6, Tables 2–3

Eleven findings, all accepted:

- **Blocker.** Not every test loss lies below the population curve. Orders 4 and
  5 lie above it. The claim is now restricted to the validation-selected losses
  at budgets 1–5.
- **Major.**
  - The finite-window recovery claim was over-extended. The text now says "up to
    eight packages", adds that a ninth package gains about 1e-8 nats, and says
    microstate access would attain the floor (not that only microstate access
    could close the gap).
  - The hashing regimes also differ in the attacker's query strategy, not just
    the input source.
  - "Only by search" became "our prescribed attacker does so by search".
- **Minor.**
  - Seventeen *conditions*, made from ten chains.
  - More than three orders of magnitude, not four.
  - Floating-point arithmetic, not exact arithmetic.
  - Bounds 7.6e-17 and 2.5e-17.
  - `p_out` is an escape *weight*.
  - The lag-two decrease claim is restricted to open conditions.
  - "Only when" became "when" for accurate fits.

## Pass C — whole paper, end to end

All fixes were confirmed, and no findings remained.
**VERDICT: SIGN-OFF.**

The only edit after sign-off was cosmetic: a pinned title height in
`fig_markov_knobs.pdf`, with no change to the plotted data.
