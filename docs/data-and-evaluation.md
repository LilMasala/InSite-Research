# Data and evaluation

The bundled pattern examples use generated histories. The saved HTML and JSON preserve a reviewed seed-131 example and its linked engine evidence; the sentence is output from a prior local-model pass. This example demonstrates how a finding links to its evidence.

## Reproducible checks in this snapshot

The tested commands and their results are recorded in [`verification.json`](../verification.json). The quick smoke run is:

```sh
python scripts/run_demo.py quick
```

It uses a short synthetic history and reduced search and support settings to exercise fixture generation, normalization, feature construction, and evidence packaging. It creates fresh engine evidence without regenerating the saved example's language-model wording. The full default-configuration run is also available:

```sh
python scripts/run_demo.py full
```

It processes the two 120-day synthetic scenarios and can take tens of minutes; it was not rerun for this snapshot. The included [example guide](../examples/README.md) documents both modes and the synthetic contract-test command. Broader evaluation across independent histories, seeds, missing-input cases, and repeated refreshes remains future work.

## External data

The demo uses synthetic inputs; no external dataset is packaged. The separate HUPA-UCM dataset is outside this snapshot. Its [Mendeley page](https://data.mendeley.com/datasets/3hbcscwz44/1) lists CC BY 4.0, and the related [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11214197/) provides study context. This portfolio includes no HUPA records or derived rows. Separate use should follow the source attribution and license terms.
