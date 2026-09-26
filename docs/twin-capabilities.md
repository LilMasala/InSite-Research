# Physiologic coverage of the bundled twin

The twin advances a selected virtual-adult physiology on a two-minute UTC grid and carries state across days. Its fitted parameters are model-conditional estimates, not causal conclusions.

| Recorded input | Model mechanism | Fitted quantity | Limit / interpretation |
| --- | --- | --- | --- |
| CGM, delivered insulin, body mass | A selected simglucose adult ODE converts delivered insulin and glucose into a trajectory; recorded mass replaces base mass. | Insulin sensitivity (`log_si`, Vmx), glucose production (`log_egp`, kp1), insulin absorption (`log_insulin_speed`, ka1/ka2/kd), action onset (`log_insulin_action_speed`, p2u/ki), hypoglycemia uptake (`log_hypo_uptake`). | Multipliers are relative to the adult and priors. Pump CR/ISF/scheduled basal are controller settings; scheduled basal never fills unknown delivery. Fit windows need observed delivery, six-hour burn-in, and ≥70% CGM coverage; blank days from an established meal logger are skipped. |
| Logged meal grams/times; food-photo or unpaired-bolus anchors | Logged meals and candidate meals receive latent amount/time adjustments; intake is spread over ten minutes. | Carb absorption/availability (`log_carb_speed`, `log_carb_effect`), event sizes and timing. | Time shifts have 15-minute (logged) and 20-minute (candidate) prior SDs, softly bounded at ±25 minutes. Candidate count is heuristic and amounts can shrink near zero. With default flux fitting, unexplained CGM rises do not create meal anchors. |
| Cycle-onset days and cycle status | Recorded onset days map to phase on a fixed 28-day cycle. If a cycle is reported without onset data, the fit can estimate a free-phase 28-day sensitivity harmonic. | Luteal/menstrual terms or inferred sine/cosine terms. | Inferred phase is a rhythm, not an observation; effects are disabled when no cycle is reported. |
| Exercise minutes, heart rate, resting heart rate | Heart-rate reserve estimates effort; an immediate uptake effect and sensitivity load decay over 16 hours. | `exercise_uptake`, `exercise_si`. | Hourly-mean heart rate dilutes brief activity; elevated heart rate can mark exercise even without a workout log. |
| Sleep stages or nightly duration | Stages collapse to asleep duration. Hours below 8.5 affect sensitivity and endogenous glucose production. | `sleep_si_per_h`. | Duration is the signal; REM, deep sleep, efficiency, and stage transitions have no separate effects. |
| Local clock time | A 03:00–08:00 dawn ramp changes glucose production; a 24-hour sine/cosine basis varies sensitivity. | `dawn_egp`, `circadian_cos_si`, `circadian_sin_si`. | Time-of-day terms do not identify biological cause. |
| Logged site changes or pump suspensions | Site age after day one enters sensitivity, capped at six excess days. Inference requires zero delivery ≥15 minutes with glucose ≥120 mg/dL and candidates ≥36 hours apart. | `site_age_si_per_day`. | Inferred changes are flagged and may be wrong. |
| Optional 0–100 stress stream or mood event | Stress scales to 0–1; mood arousal/valence maps to 0/0.5/1 held for six hours. | `stress_egp`, `stress_si`. | Mood is a short-term proxy, not a fit to raw affect. |
| Repeated CGM, meal and insulin history | Daily sensitivity/glucose-production drift; ten 30-minute carb and bolus response corrections over five hours; signed disturbance flux every 30 minutes. | Drift spreads, response kernels, disturbance blocks, and CGM noise scale (`log_cgm_sd`) plus proportional noise. | Kernels are empirical corrections, not absorption/action estimates. Flexible terms can absorb unrecorded influences and weaken attribution. |

Temperature and HRV are not fields in the twin timeline. Sleep stages collapse to duration, heart rate contributes to exercise effort, and mood events become the stress proxy above.

The fit screens a base adult, runs multi-start MAP, then variational inference (full covariance for global parameters; mean-field for daily and meal latents). The default `holdout_days` is zero, so a fit alone is not prospective validation.

## Source code

- [Timeline inputs, missingness, meal anchors, site inference](../twin/t1d_twin/data.py) and [context features](../twin/t1d_twin/context.py)
- [Parameters and priors](../twin/t1d_twin/params.py); [fit and inference](../twin/t1d_twin/fit.py)
- [Meal latents, response corrections, and rollout](../twin/t1d_twin/model.py); [base physiology](../twin/t1d_twin/ode.py)
- [Controller experiments](../twin/t1d_twin/experiment.py)
