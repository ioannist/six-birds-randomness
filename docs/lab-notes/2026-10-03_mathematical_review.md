# Mathematical review — 2026-10-03

The four mathematical propositions in the manuscript are correct in their stated
finite setting. Their strength and the title can be retained. The review repaired
numerical support errors, an unsafe stationary-law fallback, prediction evaluation
errors, and an incomplete interpretation of Lean's real KL readout. It also added
population memory entropies to support the budget claim without relying on a
minimum chosen after seeing the test losses.

This was proof reconstruction followed by adversarial **self-review**, not an
independent review. No manuscript, publication figure, frozen publication table,
or external corpus file was changed. The user's initial `paper/Makefile` change
was checkpointed in `a53368b` before implementation work. The tracked corrected
outputs and fresh build/test receipts are in `artifacts/math-review/`; the frozen
publication inputs remain in `artifacts/final/` and `paper/data/`.

## Claim coverage and proof reconstruction

Let `Z = Y_(t+tau)`, `Y = Pi(X_t)`, and write `p_x` for the conditional future
law, `q_y` for its stationary fiber mixture. All entropies and KL divergences in
this section use natural logarithms on finite alphabets.

| Manuscript object | Judgment | Mathematical inputs and scope |
| --- | --- | --- |
| Micro-closure deficit, Definition 1 | Correct | `CD = I(X_t; Z | Y)`; a population quantity for the specified law and packaging. |
| Exact decomposition, Proposition 1 | Correct, unchanged | Conditional mutual information chain rule and deterministic `Y = Pi(X_t)`. Stationarity and the Markov property are not needed for the identity itself. |
| Expected KL form, Proposition 2 | Correct, unchanged | The conditional law given `(X_t,Y)` equals `p_x`. Average KL against `q_Pi(x)` only over occupied microstates, or explicitly declare zero-weight terms irrelevant. |
| Tau-closed packaging, Definition 2 | Correct | Equality of destination laws on every entire fiber. At lag one in a Markov chain this is strong lumpability. At general lag it is strong lumpability of `P^tau`. |
| Closure, sufficiency, zero deficit, Proposition 3 | Correct, unchanged | Strong closure implies zero CD. The converse on every microstate requires full stationary support, as already stated. Without full support, zero CD characterizes equality of the occupied rows only. |
| Stationary route-mismatch bound, Proposition 4 | Correct, unchanged | Pinsker in nats gives `KL >= (1/2)*L1^2`; averaging under a probability law and Jensen gives `CD >= (1/2)*RM_stat^2`. The corresponding TV constant is `2`. |
| Ideal order-one versus order-two gap | Correct at population level | With exact conditional predictors on the same stationary law, `H(Z|Y_t)-H(Z|Y_(t-1),Y_t)` is conditional mutual information. Finite fitted held-out differences can be negative and need not equal it. |
| Budget infimum and entropy optimum | Correct with compatible nested rules | Log loss equals conditional entropy plus expected KL. The infimum equals entropy if all conditional laws on the accessible summary are feasible. Monotonicity requires that each earlier predictor can still be used with the later information and budget. |
| Deterministic hashing does not create entropy | Correct | `H(X,H(X))=H(X)=H(H(X))+H(X|H(X))`, hence `H(H(X)) <= H(X)`. This does not imply computational one-wayness or uniform outputs for every deterministic map. |
| Learned clustering CMI | Correct as a descriptive plug-in statistic | The evaluated statistic is `I_hat(X_t;Y_next|Y_t)` from counts. Labels come from noisy observations, so they are not a deterministic map of the original finite hidden state. Population and causal interpretations need additional evidence. |

For an occupied microstate, set `w_x = pi(x)/pi_Y(Pi(x)) > 0`. Then
`q_Pi(x)(z) >= w_x*p_x(z)`. Thus its KL is finite, nonnegative, and bounded
above by `-log(w_x)`. In particular, there is no infinite divergence on any
positive-weight row in this finite mixture. The average also satisfies
`0 <= CD <= H(X_t|Y_t)` and `CD <= H(Z|Y_t)`. KL's equality criterion gives
`CD=0` precisely when all occupied rows agree with their mixture. Full support
then extends this to every row of every fiber. Statements about arbitrary
conditionals on null events are unnecessary for this argument.

The decomposition's intrinsic term vanishes when the future is a deterministic
function of the current complete microstate. For a general stationary process,
`H(Z|X_t)` measures uncertainty conditional on that state; calling it physically
intrinsic does not prove that further unrecorded history cannot improve prediction.
That stronger interpretation uses a sufficient microstate, in particular the
Markov dynamics used in the experiments.

For a stationary Markov substrate and a staged history
`S_L=(Y_(t-(L-1)tau),...,Y_t)`, `L>=1`, conditional independence gives
`H(Z|X_t,S_L)=H(Z|X_t)`. Consequently,

```
H(Z|X_t) <= H(Z|S_L) <= H(Z|Y_t),
0 <= H(Z|Y_t)-H(Z|S_L) <= CD,
H(Z|S_(L+1)) <= H(Z|S_L).
```

These facts justify the memory-budget interpretation at population level. They
do not identify the best of a handful of fitted models with the population
infimum, and they do not guarantee monotonic test loss for models selected on
separate validation data.

The referenced PICA corpus paper defines uniform lifts and a uniform per-fiber
route score. This repository uses the same destination-row construction but
averages distances under the stationary micro law. That weighting is part of
this paper's definition, not an identity with the older per-fiber score. The
Notch corpus paper uses an order-one/order-two predictive gap, sometimes clamped
for an operational defect. This repository correctly retains the signed
held-out gap. Neither citation supplies a theorem equating either diagnostic
with CD.

## Numerical repairs

1. **Null states and support.** A valid chain
   `P=[[1,0,0],[0,1,0],[1,0,0]]`, `pi=[0,1,0]`, `Pi=[0,1,1]` previously returned
   `NaN` CD: an unoccupied row had infinite KL and the sum evaluated `0*infinity`.
   Null microstates are now excluded from the KL calculation. The mixture is
   evaluated in log space so underflow of a positive mixture component does not
   manufacture disjoint support and infinite KL.
2. **Positive fibers below a tolerance.** Stationary lifts and entropy mixtures
   previously replaced any fiber mass at most `1e-15` by a uniform conditional.
   For `pi=[1-5e-16,4e-16,1e-16]` and `Pi=[0,1,1]`, the correct conditional on
   the second fiber is `[.8,.2]`, not `[.5,.5]`. Only exactly zero fibers now use
   the arbitrary uniform fallback. Explicit raw-weight normalization likewise
   preserves ratios of tiny positive weights and handles overflowing row sums.
3. **Stationary fallback.** The former eigenvector fallback clipped a negative
   eigenvector before correcting its sign and could then return uniform mass.
   A positive four-state counterexample generated with seed 5 and `max_iter=1`
   produced L1 stationarity error `0.4422255942`. The fallback now solves the
   normalized linear system, checks nonnegativity, normalizes, and checks the
   invariance residual; it raises if a valid result cannot be certified to the
   stated numerical tolerance. Reducible and periodic kernels are tested.
4. **Declared input laws.** Scientific metric and simulation entry points now
   reject negative probabilities, empty kernels, invalid mass, and supplied
   nonstationary laws. Accepted row-sum roundoff is normalized explicitly;
   arbitrary invalid dynamics are no longer silently replaced by another chain.
   `normalize_rows` remains an explicit construction from raw weights. Row
   entropy permits a supplied probability weighting, documented as such.
5. **Generator contracts.** Out-of-range or nonfinite heterogeneity, type split,
   and strength parameters are rejected instead of clipped. The exactly
   lumpable generator now exports the actual post-mixing macro kernel in `K`:
   `(1-eps)*K_base + eps*fiber_sizes/n_micro`. Its old metadata exported the
   pre-mixing kernel. Mixing fails explicitly if floating underflow prevents
   its promised strict positivity.
6. **Log loss and target alignment.** Predictors are checked as probability
   laws; zero probability for an observed event costs infinity. Losses are no
   longer silently capped at approximately 690. Order-one/order-two gaps now
   use common targets starting at index two. The budget runner uses a single
   common start for every order. Tests include identical order-one and
   history-independent order-two predictors that must have equal losses.
7. **Budget selection.** The training half is retained. The remaining sequence
   is split into validation and test halves by default. For each budget, the
   fitted order is selected from permitted orders on validation alone and
   scored on common untouched test targets. `nll_selected` reports that loss.
   The old CSV names remain explicit legacy fields: `nll_exact` is per-order
   fitted test loss; `nll`/`nll_empirical_oracle` is the test oracle envelope.
   Its monotonicity is by construction. Neither is an exact population infimum.
   `history_entropy_theory` now enumerates the population finite-model optimum.
8. **Noisy labels and cache reuse.** The clustering estimator now documents its
   actual second entropy, `H_hat(Y_next|X_t,Y_t)`, and validates the declared
   alphabets. Neural comparison baselines must match the corrected evaluation
   start, train fraction, and packaging seed. Old cached baselines cannot be
   mixed with corrected target alignment. Neural training remains exploratory
   and excluded from core evidence; it was reviewed statically, not retrained.
9. **Hash reference probability.** For a uniform target in a domain of size
   `N=2^m`, `q` distinct independent input queries, and an ideal random function
   with `M=2^n` possible outputs, the exact averaged success reference is
   `1-(1-q/N)*(1-1/M)^q`. The old `baseline_exact` omitted the possible query of
   the original target. The corrected expression uses stable logarithms and
   is verified against exhaustive enumeration of every map on a two-input,
   two-output domain. Metadata explicitly describes this random-function
   reference and success as finding **any** matching preimage.

The new `packaged_history_entropy` routine groups all staged histories while
retaining their unnormalized hidden-state distributions. It is checked against
an independently implemented complete enumeration of micro paths on a small
chain. It is exponential in history length and uses floating arithmetic; it
is an offline benchmark with access to the known kernel, not a claim that a
budget-limited observer can produce the same optimum from data.

## Lean coverage and repair

The original three exported theorems already kernel-checked. However,
`ENNReal.toReal(top)=0` and an undefined Bochner integral is also zero in Lean.
Thus the original general-space identity under absolute continuity alone does
not establish finiteness or a faithful real-valued divergence. For example,
on positive integer atoms choose probabilities proportional to `1/n^2` and to
`exp(-n)`. The first law is absolutely continuous with respect to the second,
but its expected log ratio diverges, since the positive tail behaves like
`sum 1/n`. Its extended KL is infinite; the totalized Lean identity reads `0=0`.
This is an interpretation problem, not an inconsistency in the original proof.

Seven declarations were added without changing the original three statements:

- `probability_klDiv_eq_ofReal_integral_llr` retains absolute continuity and
  explicitly assumes integrability. It proves equality in extended values,
  so infinity cannot be hidden by the real projection.
- `probability_integral_llr_eq_zero_iff` uses those finite hypotheses to give
  the faithful zero-integral characterization.
- `finite_probability_llr_integrable` derives integrability on finite discrete
  probability spaces; it does not take the needed integrability as an input.
- `finite_probability_klDiv_eq_sum_llr` gives the genuine finite sum bridge
  under absolute continuity, with integrability discharged internally.
- `probability_toReal_klDiv_eq_zero_iff` makes finiteness explicit.
- `weighted_probability_klDiv_eq_zero_iff` characterizes equality precisely on
  positive-weight rows, with nonnegative weights and finiteness on those rows.
- `finite_weighted_probability_klDiv_eq_zero_iff` replaces the finiteness input
  by absolute continuity and derives it in the finite setting.

The weighted declarations do not construct conditional laws or prove the
entropy/CMI identity. Their weights and measures are supplied inputs. Their
application to the paper requires the mixture/support argument above; that
argument is proved mathematically here, not claimed to be kernel-checked.
There is still no formalization of conditional mutual information, Pinsker,
the stationary mixture construction, or budget-class optimization.

`Audit.lean` is a default build target. The fresh pinned build checks all ten
mathematical declarations, and their transitive axioms are exactly the standard
`propext`, `Classical.choice`, and `Quot.sound`. There are no project axioms,
`admit`, `sorry`, or `sorryAx` dependencies. The smoke checks and the unused
`hello` boilerplate provide no additional mathematical evidence.

## Evidence and counterexamples

The corrected 17-condition Markov sweep reproduces the frozen quantities within
floating roundoff; the largest CD difference is below `2e-16`. The RM/CD
correlation remains approximately `0.959494`. Every row passes the entropy
decomposition and stationary-lift Pinsker checks to `1e-12`. These are numerical
checks alongside the mathematical proofs, not substitutes for them.

The population entropies for the budget generator at seed `20260305` are:

| Memory order | Population entropy, nats |
| --- | --- |
| 0 | 1.0969667731 |
| 1 | 1.0495346571 |
| 2 | 1.0444254893 |
| 3 | 1.0420696657 |
| 4 | 1.0417563032 |
| 5 | 1.0417204958 |

The order-one to order-two population improvement is `0.0051091678` nats.
There is a further `0.0023558236` gain from order two to three. This strengthens
the memory-budget demonstration while showing that the frozen empirical plateau
is not exhaustion of the underlying predictive structure. The budget generator
and the Markov sweep's hidden-type chain use the same family parameters but
**different seeds** (`20260305` versus `104`); they are different realized kernels.
Their exact CD values must not be interchanged.

The 72 corrected hashing success fractions exactly reproduce the frozen values.
Only the ideal reference probability changes. The clustering raw CMIs likewise
reproduce unchanged; aligned order-one losses shift by at most approximately
`0.000275` nats. Corrected outputs were exported separately, not substituted
into the publication freeze.

Three exact finite counterexamples are retained as regression tests:

- **Stationary support matters.** In the three-state null-state example above,
  CD is zero although an unoccupied state in an occupied fiber has a different
  destination law. This defeats the converse without full support.
- **A uniform lift does not inherit the bound.** With `e=.001`,
  `P=[[1-e,e,0],[0,1-e,e],[1,0,0]]`, `Pi=[0,1,1]`, the stationary law has full
  support. CD is about `.00395215`, uniform RM about `.50024987`, and stationary
  RM about `.00199700`. The proposed uniform inequality would require CD at
  least `.12512497`, which fails. The stated stationary inequality holds.
- **Lag and refinement are not universally monotone.** On the deterministic
  four-cycle with uniform stationary mass, constant packaging has CD zero,
  while the refinement `Pi=[0,0,1,1]` has lag-one CD `log(2)`. For that same
  refinement the deficits at lags one, two, three are `log(2),0,log(2)`.
  At odd lags, each current fiber has two equally likely microstates with
  different packaged futures. At lag two, the future package is the deterministic
  opposite of the current one. The intrinsic term is zero at every lag.
  Refining also changes the future target; it is not a fixed-target conditioning
  comparison. A lag-two closed packaging need not be lag-three closed.

For the uniform-lift example the stationary law is exactly `(1,1,e)/(2+e)`,
uniform RM is `(1+e)/(2+e)`, and CD is
`((1+e)*log(1+e)-e*log(e))/(2+e)`. Since `log(1+e)<=e` and
`log(1000)<10`, CD is below `.006`, whereas half the squared uniform RM is
at least `.125`. The failed bound therefore does not depend on a floating
comparison.

The counterexamples are finite rational constructions with elementary exact
proofs; their implementations use floats with explicit roundoff tolerances.
The separate sampled invariant checks on positive kernels are stress tests,
not exhaustive proofs over all kernels.

## Corrections reserved for the later paper phase

No mathematical proposition or title needs a material downgrade. The following
interpretation corrections should be applied when manuscript edits are authorized:

1. Preserve the existing full-support and stationary-lift hypotheses. Restrict
   statements involving null conditional laws to occupied events, or give the
   zero-weight convention explicitly.
2. Describe the budget figure as a finite fitted test oracle envelope, not a
   direct measurement of `Rand_B` or a deployable model selected before test.
   Monotonicity of that envelope is not experimental evidence of improved
   prediction. The newly computed population entropies supply direct finite-
   model evidence for the substantive memory claim.
3. Interpret saturation as failure of the tested finite-data fits to demonstrate
   further gain. It does not establish that all accessible predictive distinctions
   have been exhausted. Held-out averages below a population entropy line are
   possible sampling fluctuations, not violations of the entropy optimum.
4. Treat the clustering trend as descriptive raw empirical CMI. Its plug-in
   bias grows with alphabet/context complexity, its current labels are noisy,
   and its future alphabet changes with `k`. Larger raw NLL for a larger future
   alphabet alone does not establish worse predictive closure or a causal
   effect of misalignment. The exact refinement counterexample proves the
   useful claim that automatic improvement has no general guarantee.
5. Keep hashing as a chosen toy search experiment. The exact ideal law is a
   random-function reference, not a theorem about SHA-256 or every adversary.
   Success means any preimage, not original-input identification. The budget
   counts queries in prescribed candidate prefixes; precomputing a full trial
   schedule amortizes simulation and does not establish a wall-time bound.
   Byte/bit statistics use full 256-bit digests; only the collision test uses
   truncated digests. The flag thresholds are heuristic diagnostics, not
   calibrated security or indistinguishability guarantees. High Shannon entropy
   alone does not ensure uniform-looking output or hard inversion.
6. State integrability for the general finite-valued KL bridge, or use the added
   finite discrete theorem. Do not identify the totalized real projection with
   a finite divergence under absolute continuity alone.
7. Statements that route mismatch is substantially cheaper need a computational
   cost model: in this exact implementation both calculations already have the
   destination rows and stationary law. The sweep establishes association,
   not a general cost advantage or a general monotone staging law.

These are scope and interpretation repairs; the finite decomposition, closure
criterion, stated route bound, and population budget mechanism remain intact.

## Verification and reproducibility

With the installed `.[dev,viz]` environment, run from the repository root:

```
PATH="$PWD/.venv/bin:$PATH" make test
PATH="$PWD/.venv/bin:$PATH" bash scripts/reproduce_all.sh
ruff check src tests experiments scripts
cd lean && lake build
```

The reviewed final suite has **54 passing tests**. The full reproduction pipeline
completed all five runs. Lean's pinned build and the default axiom-audit target
passed. All **21** frozen artifact checksums passed, and the manuscript tree has
no review changes relative to the baseline checkpoint.

The evidence exporter can be rerun using the captured logs:

```
.venv/bin/python scripts/audit_mathematics.py \
  --reproduction-log artifacts/math-review/reproduction.txt \
  --lean-log artifacts/math-review/lean-build.txt \
  --test-log artifacts/math-review/tests.txt \
  --outdir /tmp/randomness-math-audit
```

It validates all default sweep conditions, the declared Lean theorem coverage
and transitive axioms, the population entropy ordering, validation selection,
common target counts, and frozen checksums, then records source hashes and
exports the corrected tables. Source hashes identify the implementation
reviewed; the run directories remain ignored and the exported evidence is tracked.

The optional neural trainer was not run; its mathematical role remains excluded
from the paper's evidence. A repository-wide lint run retains an existing unused
`math` import in `paper/scripts/make_publication_figures.py`; it was not changed
because this review leaves `paper/` untouched. Ruff's newer expanded default
style rules also produce baseline style diagnostics; the existing E/F rule
families pass for the mathematical sources, experiments, tests, and scripts.
