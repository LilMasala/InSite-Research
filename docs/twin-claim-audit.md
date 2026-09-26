# Evaluation evidence

This page records what supports each research result in this snapshot.

| Evaluation | Evidence | Interpretation |
| --- | --- | --- |
| Numerical implementation | Two virtual subjects, six-hour matched-input check; maximum difference 1.34 mg/dL and pooled RMSE 0.53 mg/dL against simglucose. | Agreement between implementations under the checked inputs. |
| Synthetic fitting | Six-day fixed-seed case: one initialization day, four fitted days, one held-out day; RMSE 24.26 → 17.58 mg/dL. | Fitting and replay on a reproducible artificial history. |
| Research-data comparison | Saved aggregate reports for T1DSim_AI and ReplayBG; 27 aggregate values independently checked, report and source hashes recorded. | Comparable glucose replay errors; stronger HUPA low-event discrimination in this exploratory evaluation. |
| Input integrity | Focused tests preserve actual-delivery gaps, explicit zeros, and continuous synthetic delivery across midnight. | Data-contract behavior in the tested paths. |
| Alternate-dose effects | Synthetic direction checks and an included paired-experiment implementation. | Controlled simulator behavior; validation against observed intervention outcomes remains a research goal. |

[Full benchmark methods](twin-benchmark.md) describe splitting, sequence selection, future-input conditioning, initialization differences, noise draws, and statistical analysis. Confidence intervals resample participants; the reported Wilcoxon tests use paired sequences. Saved reports have limited fit/data provenance, and learning-rate selection was exploratory.

The benchmark fits were completed before this portfolio pass. This pass reviewed aggregate artifacts and scoring source, checked numerical transcriptions, and recreated the comparison figure. Fresh fits and statistical resampling were not run. The [verification record](../verification.json) lists the checks executed here.

The app-v1 fitting service and the AID experiments are maintained separately from the selected physiological library in this snapshot. Their deployment and control-performance evidence have their own evaluation requirements.
