#!/usr/bin/env python3
"""Build one fixed synthetic training/held-out calibration example.

This is an illustrative, bounded fit/replay. It is not a convergence study or
a clinical validation. The final day's CGM is removed before fitting and is
used only to score and plot the replay.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ["TWIN_COMPILE"] = "0"

PORTFOLIO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORTFOLIO_ROOT / "twin"))
sys.path.insert(0, str(PORTFOLIO_ROOT / "src"))

import numpy as np
import torch

torch.set_num_threads(1)

from t1d_twin import ode  # noqa: E402
from t1d_twin.data import build_timeline  # noqa: E402
from t1d_twin.fit import FitConfig, fit_twin  # noqa: E402
from t1d_twin.model import BURN_IN_H, DTYPE, TwinProblem  # noqa: E402
from t1d_twin.params import twin_priors  # noqa: E402
from t1d_twin.synthetic import generate_person  # noqa: E402
from t1d_twin.fit import prior_latents  # noqa: E402


SEED = 2026
PERSON_LABEL = "synthetic-calibration-demo"
BASE = "adult#001"
N_DAYS = 6
TRAIN_DAYS = N_DAYS - 1
TZ = "America/Detroit"
MAP_ITERS = 30
SVI_ITERS = 30
PARTICLES = 2
HOLDOUT_SAMPLE_INTERVAL_MIN = 5


def _finite_round(value: float, digits: int = 3) -> float | None:
    number = float(value)
    return round(number, digits) if np.isfinite(number) else None


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> tuple[float | None, int]:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(mask):
        return None, 0
    return float(np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2))), int(mask.sum())


def _replay_curves(full_tl, fit, holdout_day: int) -> dict:
    priors = fit.priors()
    features_days = [holdout_day]
    # There is no CGM-rise meal inference on the held-out day. This generator
    # configuration logs every meal; meals come only from its recorded carb
    # events. TwinProblem reads the final day's CGM for scoring, while its
    # initial state is taken from the known six-hour training burn-in.
    prob = TwinProblem(
        full_tl,
        fit.base,
        features_days,
        priors,
        unlogged=[],
        min_cgm_coverage=0.7,
        flux=False,
    )
    if len(prob.windows) != 1 or prob.windows[0].day != holdout_day:
        raise RuntimeError("held-out replay window was not constructed as requested")

    fitted_globals = torch.tensor(fit.loc, dtype=DTYPE)[None, :]
    zero_drift = torch.zeros((1, prob.n_days), dtype=DTYPE)
    fitted_meal_scale = torch.full((1, prob.n_logged), float(fit.carb_count_bias), dtype=DTYPE)
    no_unlogged_meals = torch.empty((1, 0), dtype=DTYPE)
    with torch.no_grad():
        fitted = prob.simulate(
            fitted_globals,
            zero_drift,
            fitted_meal_scale,
            no_unlogged_meals,
            shift_logged=None,
            shift_unlogged=None,
            flux_blocks=None,
            day_log_egp=zero_drift,
        )[0, 0].cpu().numpy()

        prior_values = prior_latents(prob, K=1)
        prior = prob.simulate(*prior_values)[0, 0].cpu().numpy()

    window = prob.windows[0]
    day_start, day_end = full_tl.day_steps(holdout_day)
    burn_steps = int(BURN_IN_H * 60 / ode.DT_MIN)
    if window.start != day_start - burn_steps or window.loss_from != burn_steps:
        raise RuntimeError("replay burn-in does not match the six-hour training-only boundary")

    # The raw generator records one CGM point at each five-minute bin centre;
    # the timeline places it on its nearest two-minute engine step.
    n_bins = int(round((day_end - day_start) * ode.DT_MIN / HOLDOUT_SAMPLE_INTERVAL_MIN))
    local_zone = ZoneInfo(full_tl.tz)
    times, observed, prior_curve, fitted_curve = [], [], [], []
    for bin_index in range(n_bins):
        absolute_step = day_start + int(round(
            (HOLDOUT_SAMPLE_INTERVAL_MIN * bin_index + HOLDOUT_SAMPLE_INTERVAL_MIN / 2) / ode.DT_MIN
        ))
        local_time = (full_tl.t0_utc + timedelta(minutes=ode.DT_MIN * absolute_step)).astimezone(local_zone)
        relative_step = absolute_step - window.start
        times.append(local_time.isoformat())
        actual_value = full_tl.cgm[absolute_step]
        observed.append(_finite_round(actual_value, 1) if np.isfinite(actual_value) else None)
        prior_curve.append(_finite_round(prior[relative_step], 1))
        fitted_curve.append(_finite_round(fitted[relative_step], 1))

    observed_arr = np.asarray([np.nan if value is None else value for value in observed], dtype=float)
    prior_arr = np.asarray(prior_curve, dtype=float)
    fitted_arr = np.asarray(fitted_curve, dtype=float)
    prior_rmse, prior_count = _rmse(observed_arr, prior_arr)
    fitted_rmse, fitted_count = _rmse(observed_arr, fitted_arr)
    if prior_count != fitted_count:
        raise RuntimeError("prior and fitted replay scored different held-out observations")
    expected_observations = int(np.isfinite(full_tl.cgm[day_start:day_end]).sum())
    if fitted_count != expected_observations:
        raise RuntimeError("five-minute plot grid does not preserve every observed held-out CGM point")

    return {
        "day": full_tl.day_dates[holdout_day],
        "plot_interval_minutes": HOLDOUT_SAMPLE_INTERVAL_MIN,
        "observed_readings": fitted_count,
        "candidate_readings": n_bins,
        "times_local": times,
        "observed_mgdl": observed,
        "prior_mean_mgdl": prior_curve,
        "fitted_mean_mgdl": fitted_curve,
        "rmse_mgdl": {
            "prior_mean": _finite_round(prior_rmse, 2),
            "fitted_posterior_mean": _finite_round(fitted_rmse, 2),
            "fitted_minus_prior": _finite_round(fitted_rmse - prior_rmse, 2),
            "scored_observations": fitted_count,
            "definition": "RMSE against observed synthetic CGM at recorded five-minute bin centres; missing CGM points are excluded.",
        },
        "input_audit": {
            "insulin": "actual delivered basal plus delivered bolus from the synthetic day timeline",
            "scheduled_basal_fill": False,
            "heldout_meals": "recorded logged carbs only; carb log scale uses the training fitted global mean bias",
            "heldout_meal_time_shift_min": 0.0,
            "heldout_unlogged_meal_candidates": 0,
            "heldout_si_drift": 0.0,
            "heldout_egp_drift": 0.0,
            "heldout_flux": 0.0,
            "initial_state": "first observed CGM in the six-hour burn-in, which lies on the preceding training day",
            "heldout_cgm_use": "scoring and observed plot series only; not used for initialization, feature construction, event detection, or parameter fitting",
        },
    }


def build_example() -> dict:
    started = time.monotonic()
    generation_started = started
    priors = twin_priors(has_cycle=False, cycle_observed=False, sex="male")
    records, _ = generate_person(
        PERSON_LABEL,
        n_days=N_DAYS,
        seed=SEED,
        base=BASE,
        priors=priors,
        is_female=False,
        aid=False,
        unlogged_fraction=0.0,
        carb_count_sd=0.0,
        meal_log_time_sd_min=0.0,
        app_fidelity=False,
        cgm_dropout=0.01,
        truth_sd_scale=0.25,
    )
    generation_seconds = time.monotonic() - generation_started

    normalization_started = time.monotonic()
    full_tl = build_timeline(records, PERSON_LABEL, sex="male", infer_sites=False)
    if full_tl.n_days != N_DAYS:
        raise RuntimeError(f"expected {N_DAYS} synthetic days, found {full_tl.n_days}")
    holdout_day = full_tl.n_days - 1
    train_dates = full_tl.day_dates[:TRAIN_DAYS]
    holdout_date = full_tl.day_dates[holdout_day]

    # Rebuild from records ending at the training boundary. The fitter never
    # receives a timeline containing held-out observations.
    training_records = records[:TRAIN_DAYS]
    training_tl = build_timeline(training_records, PERSON_LABEL, sex="male", infer_sites=False)
    if training_tl.day_dates != train_dates:
        raise RuntimeError("training timeline does not end at the declared split")
    training_seconds = time.monotonic() - normalization_started

    burn_steps = int(BURN_IN_H * 60 / ode.DT_MIN)
    holdout_start, holdout_end = full_tl.day_steps(holdout_day)
    replay_start = holdout_start - burn_steps
    if replay_start < 0:
        raise RuntimeError("held-out replay has no training burn-in")
    if not np.all(np.isfinite(training_tl.basal_upm)):
        raise RuntimeError("training delivered-basal input has a gap; scheduled basal will not be substituted")
    if not np.all(np.isfinite(full_tl.basal_upm[replay_start:holdout_end])):
        raise RuntimeError("held-out replay delivered-basal input has a gap; scheduled basal will not be substituted")
    if not bool(np.all(training_tl.insulin_observed)) or not bool(full_tl.insulin_observed[holdout_day]):
        raise RuntimeError("training or held-out delivered-insulin history is incomplete")
    if full_tl.cgm_coverage[holdout_day] < 0.7:
        raise RuntimeError("held-out synthetic CGM coverage is below the configured scoring threshold")

    train_cgm_count = int(np.isfinite(training_tl.cgm).sum())
    train_meal_count = sum(1 for meal in training_tl.meals if meal.logged)
    holdout_meal_count = sum(
        1 for meal in full_tl.meals
        if meal.logged and holdout_start <= meal.step < holdout_end
    )
    holdout_basal_coverage = float(np.isfinite(full_tl.basal_upm[replay_start:holdout_end]).mean())

    config = FitConfig(
        map_iters=MAP_ITERS,
        iters=SVI_ITERS,
        particles=PARTICLES,
        seed=SEED,
        holdout_days=0,
        base=BASE,
        clock_offsets=(),
        flux=False,
        response_kernels=False,
        dosing_prior=False,
    )
    fit_started = time.monotonic()
    fit = fit_twin(training_tl, priors=priors, config=config, verbose=False)
    fit_seconds = time.monotonic() - fit_started

    # The first training day supplies the six-hour burn-in for the next day.
    # TwinProblem correctly skips the first calendar day because it lacks an
    # earlier burn-in; only the following four days contribute fit losses.
    expected_fit_dates = train_dates[1:]
    if fit.fitted_days != expected_fit_dates:
        raise RuntimeError("fit did not use the expected four loss days after the burn-in day")
    if holdout_date in fit.fitted_days:
        raise RuntimeError("held-out day entered the fit result")

    replay_started = time.monotonic()
    replay = _replay_curves(full_tl, fit, holdout_day)
    replay_seconds = time.monotonic() - replay_started

    key_parameters = {}
    for name in ("log_si", "log_egp"):
        entry = fit.summary.get("params", {}).get(name)
        if entry:
            key_parameters[name] = {
                field: entry[field]
                for field in ("median", "p05", "p95", "posterior_sd_over_prior_sd", "identified")
                if field in entry
            }

    return {
        "schema": "insite.twin.synthetic-calibration.v1",
        "synthetic_only": True,
        "purpose": "One bounded fit/held-out replay example showing what the fitter learns from a small synthetic history.",
        "limitations": [
            "One fixed synthetic person and one fixed seed; this does not measure generalization or parameter recovery.",
            "The 30-iteration MAP and 30-iteration SVI fit is a limited-budget illustration; convergence is not established.",
            "The UVA/Padova adult base is fixed to adult#001, so base-patient selection is not evaluated.",
            "Held-out predictions replay recorded insulin and logged meals; they are not a prospective forecast or treatment recommendation.",
        ],
        "reproducibility": {
            "generator": "t1d_twin.synthetic.generate_person",
            "seed": SEED,
            "python_randomness": "NumPy generator seeded by generate_person; PyTorch seed set by FitConfig",
            "torch_threads": 1,
            "torch_compile": False,
            "days_total": N_DAYS,
            "training_days": train_dates,
            "heldout_day": holdout_date,
            "generation_settings": {
                "base": BASE,
                "aid": False,
                "app_fidelity": False,
                "truth_sd_scale": 0.25,
                "all_meals_logged": True,
                "unlogged_fraction": 0.0,
                "meal_log_time_sd_min": 0.0,
                "carb_count_sd": 0.0,
                "cgm_dropout": 0.01,
                "timezone": TZ,
            },
            "fit_config": {
                "map_iters": MAP_ITERS,
                "svi_iters": SVI_ITERS,
                "particles": PARTICLES,
                "holdout_days": 0,
                "base": BASE,
                "clock_offsets": [],
                "flux": False,
                "response_kernels": False,
                "dosing_prior": False,
            "training_scope": "fit_twin received a separately rebuilt five-day timeline; day 1 supplies the six-hour burn-in, days 2-5 contribute fit losses, and the final-day record was not supplied to the fit",
            },
        },
        "training_summary": {
            "calendar_days_available": len(train_dates),
            "input_day_labels": train_dates,
            "burn_in_only_day": train_dates[0],
            "fit_loss_days": fit.fitted_days,
            "fit_loss_day_count": len(fit.fitted_days),
            "observed_cgm_points_available": train_cgm_count,
            "logged_meals": train_meal_count,
            "fit_unlogged_candidates": len(fit.unlogged_meal_steps),
            "fit_rmse_mgdl": _finite_round(fit.diagnostics.get("train_rmse_mgdl"), 2),
            "prior_rmse_mgdl": _finite_round(fit.diagnostics.get("train_rmse_uncalibrated_mgdl"), 2),
            "fit_observations_used": int(fit.diagnostics.get("n_obs", 0)),
            "global_parameter_source": "fit_twin posterior mean `fit.loc`, learned from the five training days only",
            "key_posterior_intervals_transformed_units": key_parameters,
        },
        "heldout_replay": replay,
        "timing_seconds": {
            "generation": _finite_round(generation_seconds, 2),
            "timeline_build": _finite_round(training_seconds, 2),
            "fit": _finite_round(fit_seconds, 2),
            "heldout_replay": _finite_round(replay_seconds, 2),
            "total": _finite_round(time.monotonic() - started, 2),
        },
    }


def main() -> int:
    result = build_example()
    output = PORTFOLIO_ROOT / "examples" / "synthetic-calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    replay = result["heldout_replay"]
    print(
        "synthetic calibration example written; "
        f"fit_loss_days={result['training_summary']['fit_loss_day_count']} "
        f"heldout_points={replay['rmse_mgdl']['scored_observations']} "
        f"prior_rmse={replay['rmse_mgdl']['prior_mean']} "
        f"fitted_rmse={replay['rmse_mgdl']['fitted_posterior_mean']} "
        f"fit_seconds={result['timing_seconds']['fit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
