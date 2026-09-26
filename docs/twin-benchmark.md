# Existing digital-twin comparison

The saved benchmark compares the fitted physiological twin with T1DSim_AI and ReplayBG under held-out, recorded-input replay. Glucose errors are comparable to the personalized baselines. HUPA-UCM shows stronger low-event discrimination for the twin, with positive participant-cluster intervals for the AUROC differences. The tables below give the cohort sizes, metrics, and evaluation conditions.

The aggregate values below were checked against saved aggregate report files and the scoring/reporting source in the separate `t1d-twin` development repository. I did not rerun the fits or open participant-level score files. The aggregate-only transcription, including report and evaluation-source hashes, is in [twin-benchmark-summary.json](../examples/twin-benchmark-summary.json). The source report JSON files themselves contain a participant-identifier list and should not be copied into a shareable repo.

## Held-out five-hour results

The report pools per-sequence RMSE across the held-out windows. The twin version shown is `twin_v2`; T1DSim_AI results include its default published training recipe and a separate learning-rate-1e-3 run. Values are mg/dL.

| Cohort | People / sequences | t1d-twin v2, warm | T1DSim_AI personal, tuned | Paired difference: twin − tuned (95% participant-cluster CI) | Sequence-level Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| HUPA-UCM | 17 / 217 | 47.48 | 48.41 | −0.93 [−7.06, 3.59] | 0.682 |
| T1D-UOM | 4 / 56 | 41.79 | 45.29 | −3.50 [−14.26, 6.51] | 0.607 |

Neither paired RMSE result is statistically distinguishable from the tuned personal T1DSim_AI result in the saved report. The twin is lower-RMSE than T1DSim_AI's unpersonalized population model in both cohorts: HUPA-UCM difference −5.95 mg/dL (95% participant-cluster CI [−13.60, −0.33], sequence-level p=0.0095), and T1D-UOM difference −15.38 mg/dL (CI [−19.80, −10.64], p=0.0005). With only four T1D-UOM participants and a sequence-level p-value, that latter inference is especially fragile. This population-model contrast is not the same as beating another personalized twin.

On HUPA-UCM, the low-event task has 50 events across 217 windows. The event is at least three consecutive five-minute readings below 70 mg/dL during the five-hour window. The twin's warm ensemble reports AUROC 0.739 and Brier 0.156. T1DSim_AI's tuned point prediction, after adding the same fitted CGM-noise model to make probabilities, reports AUROC 0.597 and Brier 0.252. The paired difference is +0.142 AUROC (95% participant-cluster CI [0.058, 0.222]) and −0.096 Brier (CI [−0.143, −0.048]).

The saved HUPA report also compares the twin with ReplayBG, a second physiological simulator. Against ReplayBG's cold start, the warm twin has RMSE 47.48 versus 48.99 mg/dL; the paired difference is −1.51 mg/dL (95% participant-cluster CI [−5.86, 4.09], sequence-level Wilcoxon p=0.287). For low events, the noise-matched ReplayBG cold baseline has AUROC 0.674 and Brier 0.184; twin-minus-ReplayBG differences are +0.065 AUROC (CI [0.016, 0.126]) and −0.028 Brier (CI [−0.048, −0.011]). These are retrospective aggregate results, not external or prospective validation.

T1D-UOM has only seven low events in 56 windows. Its v2 report gives twin AUROC 0.499 and Brier 0.110 versus 0.541 and 0.124 for the noise-matched tuned T1DSim_AI result; both paired confidence intervals include zero. The small event count does not support a useful low-event comparison. The available UOM ReplayBG aggregate uses `twin_final`, not `twin_v2`, so it is excluded from the v2 comparison above.

## Evaluation protocol

The exporter maps `twin_final.json` fitted days to the training split and takes usable days after the last fitted day as held out. The neural model uses T1DSim_AI's sequence selection on the exported test split, and the scoring script evaluates the physiological fit named by the report (`twin_v2.json`) on those same saved test windows. The aggregate report does not include day-boundary or input hashes, so I cannot independently confirm that the v2 fit used exactly the exporter's `twin_final` fit-day set. T1DSim_AI's published recipe is configured as a 128-128-64-32 network with learning rate 1e-4, batch size 32, 150 epochs, 90% overlap, and five-hour sequences. The report's tuned comparator uses learning rate 1e-3. I found no documented nested validation procedure for selecting that tuned rate; treat that comparison as exploratory because selection may have used the same held-out windows.

This is a conditional replay with observed future inputs. During the scored five hours, the twin receives recorded insulin delivery and logged meals. Its warm arm also replays six hours of preceding recorded inputs before resetting glucose to the first CGM value. The comparators use their own initialization procedures. The warm-up history is therefore an important difference in the information available to each model at the origin. The export also supplies available heart-rate and sleep features to T1DSim_AI; sleep efficiency is represented as 0.85 while asleep for T1D-UOM, while this benchmark's HUPA-UCM export sets its sleep input to zero. The public HUPA-UCM dataset does include sleep variables, but this export path does not pass them through. Recorded future context is not equivalent to information available at forecast time.

RMSE is computed over the 60 five-minute steps after the anchored origin; the origin itself is excluded. The reported paired confidence intervals resample participants 4,000 times and retain each selected participant's full set of sequences. The RMSE Wilcoxon p-values are calculated over flattened per-sequence differences, without participant clustering, so repeated windows can make those p-values look more certain than the cohort size supports. The low-event probabilities use 32 noise draws per sequence; their confidence intervals also resample participants.

The current scoring source shadows its assimilation flag with a local list. I therefore leave the saved assimilation rows out of this summary and report only the warm, non-assimilated arm. The `twin_v2` report names the fit file but does not contain model-weight hashes, input-data hashes, dataset revision, or the exact population-fit set. The fitting source supports leave-one-person-out empirical priors, but the saved aggregate cannot prove which fit inputs generated these particular files.

## Further validation

Counterfactual validity remains an open question. The benchmark replays observed treatment delivery; it does not observe the same person under alternate insulin settings. The repository's separate settings-effect robustness analysis compares simulated arms across fitted variants. That checks whether a model conclusion survives parameter uncertainty, but it cannot establish that the simulated alternate-dose outcome would match a real outcome.

## Other physiology-simulator work

`PhysiologyT1DSimulator` contains a separate adapter-equivalence check against stock simglucose: a fabricated three-meal day, a virtual patient, and effectors disabled, with a stated 1e-6 mg/dL tolerance. Its additional effector sweeps test expected behavior inside the same simulator. These checks are not patient-twin comparisons. In that repository, `specs/digital_twin_shadow_contract.md` labels itself a design contract and says implementation and patient-data validation are not established.

## Sources

- [HUPA-UCM Diabetes Dataset (DOI)](https://doi.org/10.17632/3hbcscwz44.1)
- [T1D-UOM dataset (Zenodo DOI)](https://doi.org/10.5281/zenodo.15806142)
- [T1DSim_AI source repository](https://github.com/mosqueralopez/T1DSim_AI) and [paper](https://doi.org/10.1007/s00521-026-12018-x)
- In the original `t1d-twin` repository: `scripts/benchmark_t1dsimai_export.py`, `scripts/benchmark_t1dsimai_train.py`, `scripts/benchmark_t1dsimai_score.py`, `scripts/benchmark_t1dsimai_report.py`, and `scripts/benchmark_replaybg_run.py`. These paths are relative to that source repository. The raw benchmark inputs and per-person score files are not included in this portfolio.
