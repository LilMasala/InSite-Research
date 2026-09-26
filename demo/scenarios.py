"""Deterministic synthetic scenarios run through the packaged T1D twin."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import threading
from datetime import datetime, timezone

os.environ["TWIN_COMPILE"] = "0"

import numpy as np
import torch

from t1d_twin import context, model, ode, params
from t1d_twin.context import build_features, multipliers
from t1d_twin.data import Meal, PersonTimeline
from t1d_twin.model import DTYPE, FEATURE_KEYS, transform_globals
from t1d_twin.params import TwinPriors, twin_priors

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

STEPS_PER_DAY = int(24 * 60 / ode.DT_MIN)
SIM_DAYS = 2
TOTAL_STEPS = STEPS_PER_DAY * SIM_DAYS
DISPLAY_START = STEPS_PER_DAY
DISPLAY_END = TOTAL_STEPS
DISPLAY_STEPS = STEPS_PER_DAY
DISPLAY_MINUTES = list(range(0, 24 * 60 + int(ode.DT_MIN), int(ode.DT_MIN)))
SUBJECTS = ("adult#001", "adult#006")
SIMULATION_LOCK = threading.Lock()

DEFAULTS = {
    "subject": "adult#001",
    "carbs": 60.0,
    "bolus": 6.0,
    "offset": 0.0,
    "sleepHours": 8.5,
    "exerciseMinutes": 0.0,
    "exerciseHour": 16,
    "cycleDay": None,
    "siteAgeDays": 1.0,
    "stress": 0.0,
    "sensitivity": 1.0,
    "insulinAbsorption": 1.0,
    "carbAbsorption": 1.0,
}

RANGES = {
    "carbs": (0.0, 100.0),
    "bolus": (0.0, 10.0),
    "offset": (-30.0, 60.0),
    "sleepHours": (4.0, 10.0),
    "exerciseMinutes": (0.0, 90.0),
    "siteAgeDays": (0.0, 7.0),
    "stress": (0.0, 1.0),
    "sensitivity": (0.5, 1.5),
    "insulinAbsorption": (0.5, 1.5),
    "carbAbsorption": (0.5, 1.5),
}

PARAMETER_UNITS = {
    "log_si": "dimensionless insulin-sensitivity multiplier",
    "log_egp": "dimensionless endogenous-glucose-production multiplier",
    "log_insulin_speed": "dimensionless subcutaneous insulin-absorption multiplier",
    "log_insulin_action_speed": "dimensionless insulin-action-speed multiplier",
    "log_carb_speed": "dimensionless carbohydrate-absorption multiplier",
    "log_carb_effect": "dimensionless carbohydrate-bioavailability multiplier",
    "log_hypo_uptake": "dimensionless hypoglycaemia uptake coefficient",
    "cycle_luteal_si": "fractional insulin-sensitivity effect in luteal phase",
    "cycle_menstrual_si": "fractional insulin-sensitivity effect in menstrual phase",
    "exercise_si": "fractional insulin-sensitivity effect at full exercise load",
    "exercise_uptake": "fractional insulin-independent uptake effect at full intensity",
    "sleep_si_per_h": "fractional insulin-sensitivity change per sleep-deficit hour",
    "dawn_egp": "fractional endogenous-glucose-production effect at full dawn ramp",
    "site_age_si_per_day": "fractional insulin-sensitivity change per excess site-age day",
    "stress_egp": "fractional endogenous-glucose-production effect at stress 1",
    "stress_si": "fractional insulin-sensitivity effect at stress 1",
    "cycle_cos_si": "fractional insulin-sensitivity amplitude (cycle cosine)",
    "cycle_sin_si": "fractional insulin-sensitivity amplitude (cycle sine)",
    "circadian_cos_si": "fractional insulin-sensitivity amplitude (daily cosine)",
    "circadian_sin_si": "fractional insulin-sensitivity amplitude (daily sine)",
    "log_cgm_sd": "mg/dL CGM residual standard deviation",
    "log_day_si_sd": "standard deviation of daily log insulin sensitivity",
    "log_day_egp_sd": "standard deviation of daily log glucose production",
}


@dataclass(frozen=True)
class Scenario:
    subject: str
    carbs: float
    bolus: float
    offset: float
    sleepHours: float
    exerciseMinutes: float
    exerciseHour: int
    cycleDay: int | None
    siteAgeDays: float
    stress: float
    sensitivity: float
    insulinAbsorption: float
    carbAbsorption: float

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in DEFAULTS}


def _number(value, *, field: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{field} is outside demo bounds")
    return result


def normalize_request(request: dict) -> Scenario:
    """Validate the small, fixed synthetic-input schema."""
    if not isinstance(request, dict):
        raise ValueError("Scenario must be an object")
    if set(request) - set(DEFAULTS):
        raise ValueError("Unknown scenario fields")

    subject = request.get("subject", DEFAULTS["subject"])
    if subject not in SUBJECTS:
        raise ValueError("Choose a listed virtual subject")

    values = {}
    for name, (low, high) in RANGES.items():
        values[name] = _number(request.get(name, DEFAULTS[name]), field=name, low=low, high=high)

    hour = request.get("exerciseHour", DEFAULTS["exerciseHour"])
    if isinstance(hour, bool) or not isinstance(hour, (int, float)) or not math.isfinite(float(hour)) or int(hour) != hour or not 6 <= int(hour) <= 21:
        raise ValueError("exerciseHour must be an integer from 6 to 21")

    cycle_day = request.get("cycleDay", DEFAULTS["cycleDay"])
    if cycle_day is not None and (
        isinstance(cycle_day, bool) or not isinstance(cycle_day, (int, float))
        or not math.isfinite(float(cycle_day)) or int(cycle_day) != cycle_day
        or not 0 <= int(cycle_day) <= 27
    ):
        raise ValueError("cycleDay must be null or an integer from 0 to 27")

    return Scenario(
        subject=subject,
        carbs=values["carbs"],
        bolus=values["bolus"],
        offset=values["offset"],
        sleepHours=values["sleepHours"],
        exerciseMinutes=values["exerciseMinutes"],
        exerciseHour=int(hour),
        cycleDay=None if cycle_day is None else int(cycle_day),
        siteAgeDays=values["siteAgeDays"],
        stress=values["stress"],
        sensitivity=values["sensitivity"],
        insulinAbsorption=values["insulinAbsorption"],
        carbAbsorption=values["carbAbsorption"],
    )


def _raw_globals(priors: TwinPriors, scenario: Scenario) -> torch.Tensor:
    """Prior-mean parameters, with the three explicit demo controls applied."""
    names = priors.names()
    raw = priors.means().to(dtype=DTYPE).clone()
    for name, multiplier in (
        ("log_si", scenario.sensitivity),
        ("log_insulin_speed", scenario.insulinAbsorption),
        ("log_carb_speed", scenario.carbAbsorption),
    ):
        raw[names.index(name)] = math.log(multiplier)
    return raw


def _scheduled_exercise(scenario: Scenario, day: int) -> tuple[np.ndarray, int, int]:
    values = np.zeros(TOTAL_STEPS, dtype=np.float32)
    if scenario.exerciseMinutes <= 0:
        return values, 0, 0
    start = day * STEPS_PER_DAY + scenario.exerciseHour * 60 // int(ode.DT_MIN)
    remaining = scenario.exerciseMinutes
    step = start
    while remaining > 0 and step < (day + 1) * STEPS_PER_DAY:
        minutes = min(float(ode.DT_MIN), remaining)
        # ContextFeatures expects exercise minutes represented per five-minute bin.
        values[step] = minutes * 5.0 / float(ode.DT_MIN)
        remaining -= minutes
        step += 1
    return values, start, step - start


def build_timeline(scenario: Scenario, *, reference: bool = False) -> tuple[PersonTimeline, list[tuple[int, float]], dict]:
    """Create a two-day synthetic timeline; day one is a continuous-state burn-in."""
    base = ode.base_patient(scenario.subject)
    basal = base.basal_u_per_hr / 60.0
    n = TOTAL_STEPS
    steps = np.arange(n)
    day_index = steps // STEPS_PER_DAY
    local_hour = (steps % STEPS_PER_DAY) * float(ode.DT_MIN) / 60.0
    exercise, _, _ = _scheduled_exercise(scenario, 0)
    exercise_second, _, _ = _scheduled_exercise(scenario, 1)
    exercise = np.maximum(exercise, exercise_second)

    cycle = np.full(SIM_DAYS, np.nan, dtype=np.float64)
    if scenario.cycleDay is not None:
        cycle[:] = [(scenario.cycleDay - 1) % 28, scenario.cycleDay]

    # The selected age is anchored at the start of the displayed day (day two).
    site_change_step = DISPLAY_START - int(round(scenario.siteAgeDays * STEPS_PER_DAY))
    site_changes = [site_change_step]

    basal_upm = np.full(n, basal, dtype=np.float32)
    bolus_upm = np.zeros(n, dtype=np.float32)
    meal_rows: list[tuple[int, float]] = []
    meals: list[Meal] = []
    schedule = []

    fixed_meals = ((8 * 60, 40.0, 4.0, "breakfast"), (18 * 60, 50.0, 5.0, "dinner"))
    lunch_carbs = scenario.carbs
    lunch_bolus = scenario.bolus
    lunch_offset = scenario.offset

    for day in range(SIM_DAYS):
        for minute, grams, dose, name in fixed_meals:
            meal_step = day * STEPS_PER_DAY + int(minute / float(ode.DT_MIN))
            meal_rows.append((meal_step, grams))
            meals.append(Meal(meal_step, grams, logged=True))
            bolus_step = meal_step
            schedule.append({"day": day, "meal": name, "minute": minute, "carbs": grams, "bolus": dose, "bolusMinute": minute})
            bolus_upm[bolus_step] += dose / float(ode.DT_MIN)
        meal_step = day * STEPS_PER_DAY + int(12 * 60 / float(ode.DT_MIN))
        grams = lunch_carbs
        dose = lunch_bolus
        bolus_minute = 12 * 60 + lunch_offset
        bolus_step = day * STEPS_PER_DAY + int(round(bolus_minute / float(ode.DT_MIN)))
        if grams > 0:
            meal_rows.append((meal_step, grams))
            meals.append(Meal(meal_step, grams, logged=True))
        bolus_upm[bolus_step] += dose / float(ode.DT_MIN)
        schedule.append({
            "day": day,
            "meal": "lunch",
            "minute": 12 * 60,
            "carbs": grams,
            "bolus": dose,
            "bolusMinute": int(round(bolus_step * float(ode.DT_MIN) - day * 24 * 60)),
        })

    insulin_upm = basal_upm + bolus_upm
    tl = PersonTimeline(
        person_id="synthetic-virtual-subject",
        t0_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        tz="UTC",
        n_steps=n,
        cgm=np.full(n, np.nan, dtype=np.float32),
        basal_upm=basal_upm,
        bolus_upm=bolus_upm,
        meals=meals,
        hr=np.full(n, np.nan, dtype=np.float32),
        exercise_min=exercise,
        asleep=np.full(n, np.nan, dtype=np.float32),
        stress=np.full(n, scenario.stress, dtype=np.float32),
        sleep_hours=np.full(SIM_DAYS, scenario.sleepHours, dtype=np.float32),
        local_hour=local_hour,
        day_index=day_index,
        day_dates=["2026-01-01", "2026-01-02"],
        days_since_period=cycle,
        resting_hr=np.full(SIM_DAYS, np.nan, dtype=np.float32),
        site_change_steps=site_changes,
        food_photo_steps=[],
        body_mass_kg=None,
        cr=np.full(n, np.nan, dtype=np.float32),
        isf=np.full(n, np.nan, dtype=np.float32),
        basal_setting_uph=np.full(n, np.nan, dtype=np.float32),
        insulin_observed=np.ones(SIM_DAYS, dtype=bool),
        cgm_coverage=np.zeros(SIM_DAYS, dtype=np.float32),
        notes=["Synthetic, repeated fixed meal schedule; no participant data."],
        sex=None,
        site_changes_inferred=False,
        meals_observed=np.ones(SIM_DAYS, dtype=bool),
    )
    details = {
        "schedule": schedule,
        "basalUnitsPerHour": float(base.basal_u_per_hr),
        "siteChangeStep": site_change_step,
        "exerciseStartMinute": scenario.exerciseHour * 60,
        "exerciseMinutesPerDay": scenario.exerciseMinutes,
        "insulinUpm": torch.tensor(insulin_upm, dtype=DTYPE)[None, :],
        "bolusUpm": torch.tensor(bolus_upm, dtype=DTYPE)[None, :],
        "mealGrams": [(step, amount) for step, amount in meal_rows],
    }
    return tl, meal_rows, details


def _unit_for(name: str) -> str:
    if name in PARAMETER_UNITS:
        return PARAMETER_UNITS[name]
    if name.startswith("carb_resp_k"):
        return "mg/kg/min per gram of logged carbohydrate"
    if name.startswith("ins_resp_k"):
        return "mg/kg/min per unit of bolus insulin"
    raise KeyError(f"No unit metadata for twin parameter {name}")


def _parameter_metadata(priors: TwinPriors, raw_reference: torch.Tensor, raw_scenario: torch.Tensor) -> list[dict]:
    reference_values = transform_globals(raw_reference[None, :], priors)
    scenario_values = transform_globals(raw_scenario[None, :], priors)
    result = []
    for spec in priors.specs:
        result.append({
            "name": spec.name,
            "description": spec.description,
            "unit": _unit_for(spec.name),
            "priorMeanRaw": round(float(spec.prior_mean), 8),
            "priorSDRaw": round(float(spec.prior_sd), 8),
            "priorSpace": "unconstrained log space" if spec.name.startswith("log_") else "model parameter space",
            "reference": round(float(reference_values[spec.name][0]), 8),
            "scenario": round(float(scenario_values[spec.name][0]), 8),
        })
    return result


def _factor_curves(tl: PersonTimeline, priors: TwinPriors, raw: torch.Tensor) -> tuple[dict, dict]:
    features = build_features(tl)
    sl = slice(DISPLAY_START, DISPLAY_END)
    feature_tensors = {key: torch.tensor(getattr(features, key)[sl], dtype=DTYPE)[None, :] for key in FEATURE_KEYS}
    globals_tensors = transform_globals(raw[None, :], priors)
    zero_drift = torch.zeros(1, 1, DISPLAY_STEPS, dtype=DTYPE)
    si, egp, vm0 = multipliers(feature_tensors, globals_tensors, zero_drift)
    curves = {
        "insulinSensitivity": [round(float(v), 6) for v in si[0, 0]],
        "endogenousGlucoseProduction": [round(float(v), 6) for v in egp[0, 0]],
        "insulinIndependentUptake": [round(float(v), 6) for v in vm0[0, 0]],
    }
    stats = {
        "cycle": {
            "known": bool(features.known["cycle"]),
            "luteal": bool(np.any(features.luteal[sl] > 0)),
            "menstrual": bool(np.any(features.menstrual[sl] > 0)),
            "dayCosineAmplitude": float(globals_tensors["cycle_cos_si"][0]),
            "daySineAmplitude": float(globals_tensors["cycle_sin_si"][0]),
        },
        "sleep": {
            "sleepHours": float(tl.sleep_hours[1]),
            "deficitHours": float(features.sleep_deficit_h[DISPLAY_START]),
            "insulinSensitivityChangePerDeficitHour": float(globals_tensors["sleep_si_per_h"][0]),
        },
        "exercise": {
            "minutes": float(np.sum(tl.exercise_min[sl]) * float(ode.DT_MIN) / 5.0),
            "startMinute": int(np.flatnonzero(tl.exercise_min[sl] > 0)[0] * float(ode.DT_MIN)) if np.any(tl.exercise_min[sl] > 0) else None,
            "loadPeak": float(np.max(features.exercise_load[sl])),
            "intensityPeak": float(np.max(features.exercise_now[sl])),
            "sensitivityEffectAtFullLoad": float(globals_tensors["exercise_si"][0]),
            "uptakeEffectAtFullIntensity": float(globals_tensors["exercise_uptake"][0]),
        },
        "site": {
            "ageAtDisplayStartDays": float((DISPLAY_START - tl.site_change_steps[-1]) * float(ode.DT_MIN) / 1440.0),
            "excessDaysAtDisplayStart": float(features.site_age_excess_d[DISPLAY_START]),
            "sensitivityChangePerExcessDay": float(globals_tensors["site_age_si_per_day"][0]),
        },
        "stress": {
            "input": float(tl.stress[DISPLAY_START]),
            "sensitivityEffectAtStressOne": float(globals_tensors["stress_si"][0]),
            "glucoseProductionEffectAtStressOne": float(globals_tensors["stress_egp"][0]),
        },
        "dawn": {
            "naturalModelContext": True,
            "peakRamp": float(np.max(features.dawn_ramp[sl])),
            "glucoseProductionEffectAtFullRamp": float(globals_tensors["dawn_egp"][0]),
        },
    }
    return curves, stats


def _source_fingerprints() -> dict:
    modules = {"ode": ode, "model": model, "context": context, "params": params}
    out = {}
    for name, module in modules.items():
        source = Path(module.__file__)
        out[f"t1d_twin.{name}"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return out


def _simulate_one(
    scenario: Scenario,
    priors: TwinPriors,
    raw: torch.Tensor,
    *,
    reference: bool = False,
) -> tuple[np.ndarray, PersonTimeline, dict, dict, dict]:
    tl, meals, schedule = build_timeline(scenario, reference=reference)
    feats = build_features(tl)
    base = ode.base_patient(scenario.subject)
    meal_inputs = [(step, torch.tensor([grams], dtype=DTYPE)) for step, grams in meals]
    with SIMULATION_LOCK, torch.no_grad():
        trajectory = model.rollout(
            tl,
            scenario.subject,
            priors,
            raw[None, :],
            0,
            TOTAL_STEPS,
            meal_grams=meal_inputs,
            insulin_upm=schedule["insulinUpm"],
            feats=feats,
            bolus_upm=schedule["bolusUpm"],
        )[0].cpu().numpy()
    # State at the display boundary follows the full 24-hour burn-in. Subsequent
    # values are the next day's two-minute rollout; minute zero is not reset.
    displayed = np.concatenate(([float(trajectory[DISPLAY_START - 1])], trajectory[DISPLAY_START:DISPLAY_END]))
    factor_curves, stats = _factor_curves(tl, priors, raw)
    return displayed, tl, schedule, factor_curves, stats


def _summary(glucose: np.ndarray) -> dict:
    # The first point is the inherited state at minute zero, not a two-minute
    # interval, so interval-weighted duration metrics start at index one.
    intervals = glucose[1:]
    return {
        "timeInRangePercent": round(float(np.mean((intervals >= 70.0) & (intervals <= 180.0)) * 100.0), 1),
        "below70Minutes": int(np.sum(intervals < 70.0) * ode.DT_MIN),
        "above180Minutes": int(np.sum(intervals > 180.0) * ode.DT_MIN),
        "meanMgDl": round(float(np.mean(intervals)), 1),
    }


@lru_cache(maxsize=2)
def _cached_reference(subject: str) -> tuple[np.ndarray, list, dict, dict]:
    priors = twin_priors(cycle_observed=True)
    scenario = normalize_request({"subject": subject})
    raw = _raw_globals(priors, scenario)
    curve, _, schedule, factors, stats = _simulate_one(scenario, priors, raw, reference=True)
    return curve, schedule["schedule"], factors, stats


def _simulate_uncached(scenario: Scenario) -> dict:
    # Use the population prior means consistently in both arms. With no cycle
    # day, the cycle feature masks remain empty and inferred rhythms are pinned
    # to zero; the known phase effect parameters do not act on unknown days.
    priors = twin_priors(cycle_observed=True)
    default_for_subject = normalize_request({"subject": scenario.subject})
    raw_reference = _raw_globals(priors, default_for_subject)
    raw_scenario = _raw_globals(priors, scenario)
    reference, ref_schedule, ref_curve_factors, ref_stats = _cached_reference(scenario.subject)
    result, sc_tl, sc_schedule, scenario_curve_factors, sc_stats = _simulate_one(scenario, priors, raw_scenario)
    base = ode.base_patient(scenario.subject)
    parameter_meta = _parameter_metadata(priors, raw_reference, raw_scenario)
    displayed_schedule = [row for row in sc_schedule["schedule"] if row["day"] == 1]
    reference_schedule = [row for row in ref_schedule if row["day"] == 1]
    actual_bolus_minute = next(row["bolusMinute"] for row in displayed_schedule if row["meal"] == "lunch")

    if not np.isfinite(reference).all() or not np.isfinite(result).all():
        raise ValueError("Scenario produced a nonfinite result")

    def clean(values):
        return [round(float(v), 3) for v in values]

    return {
        "syntheticOnly": True,
        "model": "t1d_twin.model.rollout + t1d_twin.context.build_features/multipliers",
        "subject": scenario.subject,
        "patient": {
            "weightKg": round(float(base.params["BW"]), 1),
            "basalUnitsPerHour": round(float(base.basal_u_per_hr), 3),
        },
        "scope": "Unfitted virtual-subject prior scenario; fabricated two-day schedule with continuous 24-hour burn-in.",
        "minute": DISPLAY_MINUTES,
        "baseline": clean(reference),
        "scenario": clean(result),
        "actualBolusMinute": actual_bolus_minute,
        "summary": [_summary(reference), _summary(result)],
        "metadata": {
            "fitted": False,
            "parameterProvenance": "Population prior means from twin/t1d_twin/params.py; no fit or participant data used.",
            "simulation": {
                "engine": "t1d_twin.model.rollout",
                "contextFeatures": "t1d_twin.context.build_features",
                "contextMultipliers": "t1d_twin.context.multipliers",
                "solver": "two-minute RK4",
                "burnInHours": 24,
                "displayHours": 24,
                "initialState": "simglucose virtual-subject basal state at 120 mg/dL, propagated through the burn-in",
                "displayPointCount": len(DISPLAY_MINUTES),
                "displayStepMinutes": float(ode.DT_MIN),
            },
            "controls": {
                "defaults": DEFAULTS.copy(),
                "scenario": scenario.as_dict(),
                "units": {
                    "carbs": "g at lunch",
                    "bolus": "U at lunch",
                    "offset": "minutes after lunch; negative values pre-bolus",
                    "sleepHours": "hours slept per modeled night",
                    "exerciseMinutes": "minutes, starting at exerciseHour",
                    "exerciseHour": "local synthetic clock hour",
                    "cycleDay": "days since period start, 0-27; null means unknown",
                    "siteAgeDays": "days at displayed-day midnight",
                    "stress": "unitless intensity from 0 to 1",
                    "sensitivity": "multiplier on prior insulin sensitivity",
                    "insulinAbsorption": "multiplier on prior subcutaneous insulin absorption rates",
                    "carbAbsorption": "multiplier on prior carbohydrate absorption rates",
                },
            },
            "schedule": {
                "timezone": "UTC synthetic clock",
                "repeatedOnBurnInAndDisplayDay": True,
                "fixedBreakfastAndDinner": [
                    {"minute": 480, "carbs": 40.0, "bolus": 4.0},
                    {"minute": 1080, "carbs": 50.0, "bolus": 5.0},
                ],
                "referenceDisplayedDay": reference_schedule,
                "scenarioDisplayedDay": displayed_schedule,
                "siteAgeAnchoredAtDisplayedDayStart": scenario.siteAgeDays,
            },
            "parameters": parameter_meta,
            "context": {
                "reference": {"factors": ref_curve_factors, "features": ref_stats},
                "scenario": {"factors": scenario_curve_factors, "features": sc_stats},
                "factorUnits": {
                    "insulinSensitivity": "relative to selected virtual-subject prior",
                    "endogenousGlucoseProduction": "relative to selected virtual-subject prior",
                    "insulinIndependentUptake": "relative to selected virtual-subject prior",
                },
            },
            "sourceFingerprintsSha256": _source_fingerprints(),
        },
    }


@lru_cache(maxsize=16)
def _cached_simulation(scenario: Scenario) -> dict:
    return _simulate_uncached(scenario)


def simulate_scenario(request: dict) -> dict:
    """Return a fresh JSON-serializable response for a validated synthetic request."""
    scenario = normalize_request(request)
    return deepcopy(_cached_simulation(scenario))


def context_only(scenario: Scenario, *, reference: bool = False) -> dict:
    """Build context features and factor curves without integrating the ODE (test helper)."""
    priors = twin_priors(cycle_observed=True)
    used = normalize_request({}) if reference else scenario
    raw = _raw_globals(priors, used)
    tl, _, _ = build_timeline(used)
    curves, stats = _factor_curves(tl, priors, raw)
    return {"factors": curves, "features": stats}
