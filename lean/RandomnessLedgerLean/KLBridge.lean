import Mathlib.InformationTheory.KullbackLeibler.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Integral.IntegrableOn

namespace RandomnessLedgerLean

open MeasureTheory

/-- A totalized identity. Without integrability, KL may be infinite: both
`ENNReal.toReal ∞` and the undefined Bochner integral are zero in Lean.
Use `probability_klDiv_eq_ofReal_integral_llr` for a finite divergence reading. -/
theorem probability_toReal_klDiv_eq_integral_llr
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμν : μ ≪ ν) :
    (InformationTheory.klDiv μ ν).toReal = ∫ x, MeasureTheory.llr μ ν x ∂μ := by
  simpa using
    (InformationTheory.toReal_klDiv_of_measure_eq
      (μ := μ) (ν := ν) hμν (by simp))

/-- The totalized probability log-ratio integral is nonnegative. Its usual
expectation interpretation also requires integrability. -/
theorem probability_integral_llr_nonneg
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμν : μ ≪ ν) :
    0 ≤ ∫ x, MeasureTheory.llr μ ν x ∂μ := by
  rw [← probability_toReal_klDiv_eq_integral_llr (μ := μ) (ν := ν) hμν]
  exact ENNReal.toReal_nonneg

/-- KL divergence on probability measures vanishes exactly when the measures are equal. -/
theorem probability_klDiv_eq_zero_iff
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    InformationTheory.klDiv μ ν = 0 ↔ μ = ν := by
  simpa using (InformationTheory.klDiv_eq_zero_iff (μ := μ) (ν := ν))

/-- Integrability and absolute continuity certify that probability KL is finite. -/
theorem probability_klDiv_eq_ofReal_integral_llr
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμν : μ ≪ ν) (h_int : Integrable (llr μ ν) μ) :
    InformationTheory.klDiv μ ν = ENNReal.ofReal (∫ x, llr μ ν x ∂μ) := by
  simpa using InformationTheory.klDiv_of_ac_of_integrable hμν h_int

/-- Zero expected log ratio characterizes equality when the divergence is finite. -/
theorem probability_integral_llr_eq_zero_iff
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμν : μ ≪ ν) (h_int : Integrable (llr μ ν) μ) :
    (∫ x, llr μ ν x ∂μ) = 0 ↔ μ = ν := by
  rw [← probability_toReal_klDiv_eq_integral_llr hμν,
    ENNReal.toReal_eq_zero_iff]
  simp [InformationTheory.klDiv_ne_top hμν h_int, probability_klDiv_eq_zero_iff]

/-- On a finite discrete probability space, integrability is proved rather than assumed. -/
theorem finite_probability_llr_integrable
    {α : Type*} [Finite α] [MeasurableSpace α] [MeasurableSingletonClass α]
    {μ ν : Measure α} [IsProbabilityMeasure μ] :
    Integrable (llr μ ν) μ := by
  exact integrableOn_univ.mp (IntegrableOn.of_finite (Set.toFinite (Set.univ : Set α))
    (f := llr μ ν) (μ := μ))

/-- The finite discrete KL bridge, with no `toReal ∞` ambiguity. -/
theorem finite_probability_klDiv_eq_sum_llr
    {α : Type*} [Fintype α] [MeasurableSpace α] [MeasurableSingletonClass α]
    {μ ν : Measure α} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμν : μ ≪ ν) :
    InformationTheory.klDiv μ ν =
      ENNReal.ofReal (∑ x, μ.real {x} * llr μ ν x) := by
  rw [probability_klDiv_eq_ofReal_integral_llr hμν finite_probability_llr_integrable,
    integral_fintype _ finite_probability_llr_integrable]
  rfl

/-- The real KL readout has a faithful zero test only after excluding infinity. -/
theorem probability_toReal_klDiv_eq_zero_iff
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h_fin : InformationTheory.klDiv μ ν ≠ ⊤) :
    (InformationTheory.klDiv μ ν).toReal = 0 ↔ μ = ν := by
  simp [ENNReal.toReal_eq_zero_iff, h_fin, probability_klDiv_eq_zero_iff]

/-- A finite expected KL vanishes exactly on equality of its positive-weight rows.
Null rows impose no condition. Finiteness must be established on positive rows;
it cannot be inferred from the real readout alone. -/
theorem weighted_probability_klDiv_eq_zero_iff
    {ι α : Type*} [Fintype ι] [MeasurableSpace α]
    (w : ι → ℝ) (μ ν : ι → Measure α)
    [∀ i, IsProbabilityMeasure (μ i)] [∀ i, IsProbabilityMeasure (ν i)]
    (hw : ∀ i, 0 ≤ w i)
    (h_fin : ∀ i, 0 < w i → InformationTheory.klDiv (μ i) (ν i) ≠ ⊤) :
    (∑ i, w i * (InformationTheory.klDiv (μ i) (ν i)).toReal) = 0 ↔
      ∀ i, 0 < w i → μ i = ν i := by
  classical
  rw [Finset.sum_eq_zero_iff_of_nonneg (fun i _ ↦ mul_nonneg (hw i) ENNReal.toReal_nonneg)]
  constructor
  · intro h i hi
    apply (probability_toReal_klDiv_eq_zero_iff (h_fin i hi)).mp
    exact (mul_eq_zero.mp (h i (Finset.mem_univ i))).resolve_left (ne_of_gt hi)
  · intro h i _
    by_cases hi : 0 < w i
    · rw [h i hi]
      simp
    · have h_zero : w i = 0 := le_antisymm (le_of_not_gt hi) (hw i)
      simp [h_zero]

/-- In the finite discrete case absolute continuity on occupied rows suffices:
integrability and finiteness are discharged internally. -/
theorem finite_weighted_probability_klDiv_eq_zero_iff
    {ι α : Type*} [Fintype ι] [Finite α] [MeasurableSpace α] [MeasurableSingletonClass α]
    (w : ι → ℝ) (μ ν : ι → Measure α)
    [∀ i, IsProbabilityMeasure (μ i)] [∀ i, IsProbabilityMeasure (ν i)]
    (hw : ∀ i, 0 ≤ w i) (h_ac : ∀ i, 0 < w i → μ i ≪ ν i) :
    (∑ i, w i * (InformationTheory.klDiv (μ i) (ν i)).toReal) = 0 ↔
      ∀ i, 0 < w i → μ i = ν i :=
  weighted_probability_klDiv_eq_zero_iff w μ ν hw fun i hi ↦
    InformationTheory.klDiv_ne_top (h_ac i hi) finite_probability_llr_integrable

end RandomnessLedgerLean
