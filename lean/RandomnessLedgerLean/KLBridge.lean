import Mathlib.InformationTheory.KullbackLeibler.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability

namespace RandomnessLedgerLean

open MeasureTheory

/-- For probability measures, `toReal` of KL divergence equals expected log-likelihood ratio. -/
theorem probability_toReal_klDiv_eq_integral_llr
    {α : Type*} [MeasurableSpace α]
    {μ ν : Measure α}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμν : μ ≪ ν) :
    (InformationTheory.klDiv μ ν).toReal = ∫ x, MeasureTheory.llr μ ν x ∂μ := by
  simpa using
    (InformationTheory.toReal_klDiv_of_measure_eq
      (μ := μ) (ν := ν) hμν (by simp))

/-- For probability measures, expected log-likelihood ratio is nonnegative. -/
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

end RandomnessLedgerLean
