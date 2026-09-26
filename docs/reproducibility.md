# Reproducibility

This is a selected source snapshot prepared on September 26, 2026. It supports local synthetic runs. Dataset downloads, cloud accounts, and app signing are outside the portable example.

## What can be inspected immediately

- `assets/`: actual iOS simulator screenshots from the synthetic preview.
- `examples/saved-synthetic-brief.html`: a self-contained saved brief and supporting evidence, viewable locally in a browser.
- `examples/saved-synthetic-brief.json`: the same visible findings and referenced engine evidence in machine-readable form.
- `src/insite_analytics/`: the included pattern implementation.
- `twin/t1d_twin/`: the included physiological twin implementation.

The saved brief is a **subset** of a previous synthetic run: the visible item and its referenced evidence, plus the quiet control. Other archived findings and deployment metadata are omitted. The sentence was generated once by an offline Qwen3-4B-Instruct-2507-4bit pass and accepted after claim validation. The saved example preserves it. Running the pattern demo produces fresh engine findings; it does not run or redistribute that language model.

## Previous full synthetic run

Two 120-day artificial histories (`full` and `null_clock`), seed 131, were processed using the then-current default brief configuration. The completed cloud run took about 44 minutes. The later local wording pass retained one current-context finding and no finding for the quiet control. Response findings with lower means but no qualifying sustained low/high episodes remained in the evidence archive.

These outcomes characterize one fixed demonstration pair. Estimating sensitivity, false-positive rates, or clinical utility requires a broader, prespecified evaluation across seeds and independent histories.

## Verification in this snapshot

See `verification.json` for the commands actually run and their outcomes. Quick checks establish package wiring and specific implementation contracts. The full benchmark is a separate, longer run. Source manifests identify the files copied from the working implementation, including any packaging changes.

## Interactive model and calibration example

The interactive comparison uses the full twin rollout and its existing context feature/multiplier functions. Its coefficients are model priors. The calibration example is generated separately by `scripts/build_calibration_example.py`; its JSON records the split, optimization budget, available replay inputs, and withheld observations. The UI displays the resulting held-out curves and errors.

## Twin evaluation levels

1. **Numerical agreement:** compare the differentiable ODE implementation with the installed simglucose reference under the same synthetic inputs.
2. **Data contracts:** preserve delivered-insulin gaps, explicit zero delivery, timestamps, and missingness.
3. **Synthetic fitting:** verify fitting, serialization, and held-out diagnostic generation with known artificial inputs.
4. **Intervention recovery:** compare fitted-twin effects with known synthetic generating effects; the full recovery script is included.
5. **Research-data evaluation:** perform held-out comparisons with disclosed splitting, conditioning, and uncertainty procedures. This portfolio's preparation did not rerun participant-level research benchmarks.

Only the levels and individual checks recorded in `verification.json` should be described as rerun for this snapshot. The [claim audit](twin-claim-audit.md) records the evidence still needed for broader performance claims.

## Existing research-data comparison

The saved benchmark summary includes cohort-level metrics checked against three aggregate report artifacts. The source reports and seven evaluation-source files are identified by SHA-256 in [`twin-benchmark-summary.json`](../examples/twin-benchmark-summary.json). Individual records, model weights, and per-person results stay in the private working environment.

To render the checked-in aggregate results:

```sh
pip install -r requirements-figures.txt
python scripts/plot_twin_benchmark.py
```

This recreates the figure from the saved summary. The original fits and bootstrap resampling are separate computations described in [the benchmark methods](twin-benchmark.md).
