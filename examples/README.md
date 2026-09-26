# Synthetic example outputs

[`saved-synthetic-brief.json`](saved-synthetic-brief.json) and [`saved-synthetic-brief.html`](saved-synthetic-brief.html) are the checked-in, local-only seed-131 `full` / `null_clock` result accepted for this portfolio. The HTML is a static view of the synthetic brief and its linked evidence. This is a saved output from an earlier local run; it does not mean the quick demo reruns language-model narration.

Run the copied source snapshot against fresh fictional data from the portfolio root:

```sh
python scripts/run_demo.py quick
```

The quick command uses a 14-day seed-131 `full` fixture and reduced support, search, and bootstrap settings. It exercises synthetic generation, observation normalization, feature construction, and evidence packaging. It is a smoke run, not a benchmark and not a reproduction of the checked-in output.

The full command reproduces the engine run shape with the default `BriefConfig`: seed 131, 120 days each for `full` and `null_clock`.

```sh
python scripts/run_demo.py full
```

It can take tens of minutes. Both modes write only synthetic engine evidence JSON under `outputs/` (or a caller-selected local `--output-dir`). The CLI reports counts and paths; it does not print record rows, evidence text, or invoke a language model.

To run the synthetic response-outcome contract tests from the portfolio root:

```sh
python -m pytest tests
```

The separate twin demo has a synthetic numerical check and a small test suite:

```sh
PYTHONPATH=src:twin python -m pytest twin/tests/test_twin.py
python demo/server.py
```

The twin package has additional optional dependencies in [`twin/requirements.txt`](../twin/requirements.txt). The numerical check JSON and its figure are linked separately from the participant-context example.

## Synthetic calibration

[`synthetic-calibration.json`](synthetic-calibration.json) contains a six-day, fixed-seed example: one initialization day, four fitted days, and one held-out day. Held-out replay uses recorded meals and delivered insulin. Its 288 glucose observations are reserved for scoring. RMSE changes from 24.26 to 17.58 mg/dL after 30 MAP and 30 variational iterations.

```sh
python scripts/build_calibration_example.py
```

The interactive demo displays the saved observations, starting-parameter curve, fitted curve, and run settings. The result describes this single synthetic example.
