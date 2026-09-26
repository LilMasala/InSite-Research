# Digital-twin demo

The interactive demo uses the included twin model to compare a reference day with a modified scenario. Both trajectories retain physiological state through a preceding day of simulation and the displayed 24 hours.

## Run locally

Use Python 3.11 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r twin/requirements.txt
python demo/server.py --port 8874
```

Open **http://127.0.0.1:8874**. Stop the server with Ctrl-C.

## Controls

| Group | Inputs | Model path |
| --- | --- | --- |
| Daily context | Sleep duration, exercise duration and start time, cycle day, site age, stress | Existing context features and sensitivity, glucose-production, and glucose-uptake multipliers |
| Lunch | Carbohydrates, delivered bolus amount and timing | Logged meal and insulin inputs to the physiological model |
| Physiology | Insulin sensitivity, insulin absorption, carbohydrate absorption | Existing core parameter transforms |

The two base profiles are published virtual adults supplied by simglucose. They differ in body mass, steady-state basal input, glucose production, insulin response, and absorption parameters. The scenario metadata exposes the model values used in each run.

Interactive comparisons use prior context coefficients from the included library. The model's cycle mapping uses a fixed 28-day representation; its site-age and stress terms are effective model parameters. [The capability table](../docs/twin-capabilities.md) explains each input's meaning and limitations.

## Calibration example

The page also loads `examples/synthetic-calibration.json`, produced by `scripts/build_calibration_example.py`. This is a separate fitting run on generated records with a trailing day reserved for evaluation. Its metadata identifies the generation seed, optimization budget, inputs available to replay, and withheld observations. Displayed metrics belong to that single run.

To regenerate it, run `python scripts/build_calibration_example.py`. The recorded run used 30 MAP and 30 variational iterations and took about 3.3 minutes.

## Implementation

- `scenarios.py`: deterministic scenario construction and full-context rollout.
- `server.py`: bounded local requests and static example serving.
- `index.html`: controls, glucose comparisons, and calibration view.

The service binds to localhost and accepts the defined synthetic inputs. Public hosting requires a deployment configuration with resource limits and dependency pinning. [CAPSML](https://capsml.com/) is a related virtual glucose-control interface; this implementation uses the twin code bundled here.
