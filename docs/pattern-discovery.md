# Pattern discovery

The bundled engine builds a retrospective brief of associations and similar situations in one person's recorded history. The participant app keeps pattern cards disabled in this release (`patternsAvailable == false`). The implementation is in [`personal_context_engine.py`](../src/insite_analytics/personal_context_engine.py); numeric rule proposals are in [`numeric_search.py`](../src/insite_analytics/numeric_search.py).

## As-of features and outcomes

At each event or daily anchor, the engine uses measurements recorded and available by that time. Historical rows and current context share the same lag-feature builder. Features include preceding glucose summaries, recorded carbohydrates and delivered boluses, activity and steps, completed-sleep measures, mood, temperature, heart rate and HRV, and explicitly recorded cycle and infusion-site history. Some features compare recent seven-day observations with the earlier 28-day history. Missing, late, or incomplete values remain unknown.

Meal and delivered-bolus episodes have 0–3-hour and 3–6-hour glucose-response targets; daily rows include local three-hour windows, overnight and 24-hour responses, and variance targets. Each outcome window must be complete by the analysis cutoff. Bolus findings describe observed responses with later meals and boluses retained as co-exposures. Production inputs currently lack explicit fasting provenance and reliable correction-only intent.

## Search and evidence

The default full configuration uses `pysubgroup==0.9.0` `BeamSearch` with depth 4, six quantile bins fitted on discovery rows, beam width 256, and a 128-rule shortlist per direction. Clock, event, and daily proposals compete for a shared six-candidate confirmation family. The search is heuristic, so it can miss rules outside the beam.

Discovery thresholds and the nuisance ridge fit use earlier chronological rows; outcome windows that cross the confirmation boundary are purged. Matched comparison, minimum support, effect thresholds, bootstrap intervals, and multiplicity adjustment gate release. The defaults require six units per arm and use 10,000 block-bootstrap draws. Event candidates use Holm adjustment; the frozen cross-family confirmation uses Bonferroni-adjusted block intervals. Ridge adjustment depends on the target and observed inputs: it uses recorded carbohydrates when available and includes delivered bolus for meal-response targets, while bolus-response targets keep dose as the exposure. Later meals and boluses remain co-exposures.

Current-context matching applies confirmed daily rules to current as-of features. Similar-history retrieval separately finds comparable earlier situations and summarizes their observed later glucose. Both summarize recorded history.

## Wording and app status

The saved synthetic example preserves a sentence accepted in a previous offline local-model pass, together with its engine evidence. The model runtime, weights, and wording pipeline are outside this bundle. Fresh demo runs create engine findings without a new language-model pass.
