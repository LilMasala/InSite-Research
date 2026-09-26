from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "twin")]

from demo.scenarios import (  # noqa: E402
    DEFAULTS,
    _raw_globals,
    build_timeline,
    context_only,
    normalize_request,
    simulate_scenario,
)
from t1d_twin.context import build_features  # noqa: E402
from t1d_twin.model import transform_globals  # noqa: E402
from t1d_twin.params import twin_priors  # noqa: E402


def _different(left, right):
    return not np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=1e-8)


def test_default_reference_and_scenario_are_identical_finite_and_continuous():
    result = simulate_scenario({})
    explicit_defaults = simulate_scenario(DEFAULTS)

    assert result["baseline"] == result["scenario"]
    assert result["baseline"] == explicit_defaults["baseline"]
    assert result["minute"] == list(range(0, 1441, 2))
    assert len(result["baseline"]) == len(result["scenario"]) == 721
    assert len(result["summary"]) == 2
    assert "above180Minutes" in result["summary"][0]
    assert np.isfinite(result["baseline"]).all()
    assert np.isfinite(result["scenario"]).all()
    assert result["metadata"]["simulation"]["burnInHours"] == 24
    # Lunch, dinner, and exercise from the prior synthetic day have already
    # affected the state shown at the start of the displayed day.
    assert not math.isclose(result["baseline"][0], 120.0, abs_tol=1e-3)


def test_each_context_input_changes_its_source_model_feature_or_factor():
    neutral = context_only(normalize_request({}))

    short_sleep = context_only(normalize_request({"sleepHours": 6.0}))
    assert _different(neutral["factors"]["insulinSensitivity"], short_sleep["factors"]["insulinSensitivity"])
    assert _different(neutral["factors"]["endogenousGlucoseProduction"], short_sleep["factors"]["endogenousGlucoseProduction"])

    exercise = context_only(normalize_request({"exerciseMinutes": 60.0, "exerciseHour": 9}))
    later_exercise = context_only(normalize_request({"exerciseMinutes": 60.0, "exerciseHour": 16}))
    assert _different(neutral["factors"]["insulinIndependentUptake"], exercise["factors"]["insulinIndependentUptake"])
    assert _different(exercise["factors"]["insulinSensitivity"], later_exercise["factors"]["insulinSensitivity"])

    cycle = context_only(normalize_request({"cycleDay": 20}))
    assert cycle["features"]["cycle"]["known"]
    assert cycle["features"]["cycle"]["luteal"]
    assert _different(neutral["factors"]["insulinSensitivity"], cycle["factors"]["insulinSensitivity"])

    older_site = context_only(normalize_request({"siteAgeDays": 4.0}))
    assert older_site["features"]["site"]["excessDaysAtDisplayStart"] == pytest.approx(3.0)
    assert _different(neutral["factors"]["insulinSensitivity"], older_site["factors"]["insulinSensitivity"])

    stress = context_only(normalize_request({"stress": 0.8}))
    assert _different(neutral["factors"]["insulinSensitivity"], stress["factors"]["insulinSensitivity"])
    assert _different(neutral["factors"]["endogenousGlucoseProduction"], stress["factors"]["endogenousGlucoseProduction"])


def test_unknown_cycle_is_neutral_and_does_not_enable_inferred_rhythm():
    unknown = context_only(normalize_request({"siteAgeDays": 0.0}))
    cycle = unknown["features"]["cycle"]
    assert cycle["known"] is False
    assert cycle["luteal"] is False
    assert cycle["menstrual"] is False
    assert cycle["dayCosineAmplitude"] == 0.0
    assert cycle["daySineAmplitude"] == 0.0
    # With all other controls at their identity defaults, the only time-varying
    # context multiplier is the source model's natural dawn effect on EGP.
    assert np.allclose(unknown["factors"]["insulinSensitivity"], 1.0)
    assert np.allclose(unknown["factors"]["insulinIndependentUptake"], 1.0)
    assert np.min(unknown["factors"]["endogenousGlucoseProduction"]) == pytest.approx(1.0)
    assert np.max(unknown["factors"]["endogenousGlucoseProduction"]) > 1.0


def test_meal_and_physiology_controls_are_applied_as_one_factor_overrides():
    default = normalize_request({})
    custom = normalize_request({
        "carbs": 75,
        "bolus": 7,
        "offset": -10,
        "sensitivity": 0.8,
        "insulinAbsorption": 1.2,
        "carbAbsorption": 0.7,
    })
    _, _, schedule = build_timeline(custom)
    lunch = next(row for row in schedule["schedule"] if row["day"] == 1 and row["meal"] == "lunch")
    assert lunch["carbs"] == 75
    assert lunch["bolus"] == 7
    assert lunch["bolusMinute"] == 710

    priors = twin_priors(cycle_observed=True)
    reference_raw = _raw_globals(priors, default)
    custom_raw = _raw_globals(priors, custom)
    reference_values = transform_globals(reference_raw[None, :], priors)
    custom_values = transform_globals(custom_raw[None, :], priors)
    assert custom_values["log_si"][0] == pytest.approx(0.8)
    assert custom_values["log_insulin_speed"][0] == pytest.approx(1.2)
    assert custom_values["log_carb_speed"][0] == pytest.approx(0.7)
    assert reference_values["log_si"][0] == pytest.approx(1.0)
    assert reference_values["log_insulin_speed"][0] == pytest.approx(1.0)
    assert reference_values["log_carb_speed"][0] == pytest.approx(1.0)


@pytest.mark.parametrize("payload", [
    [],
    {"unknown": 1},
    {"subject": "not-listed"},
    {"carbs": True},
    {"bolus": float("nan")},
    {"offset": 61},
    {"sleepHours": 3.9},
    {"exerciseMinutes": 91},
    {"exerciseHour": 5},
    {"exerciseHour": 9.5},
    {"cycleDay": True},
    {"cycleDay": 28},
    {"siteAgeDays": -0.1},
    {"stress": 1.1},
    {"sensitivity": 0.49},
    {"insulinAbsorption": 1.51},
    {"carbAbsorption": "fast"},
])
def test_invalid_scenarios_are_rejected(payload):
    with pytest.raises(ValueError):
        normalize_request(payload)
