from copy import deepcopy

import numpy as np

from t1d_twin.data import build_timeline
from t1d_twin.params import twin_priors
from t1d_twin.synthetic import generate_person


def _records():
    priors = twin_priors(has_cycle=False, cycle_observed=False, sex="male")
    records, _ = generate_person(
        "synthetic-delivery-test",
        n_days=4,
        seed=24,
        base="adult#001",
        priors=priors,
        is_female=False,
        aid=False,
        unlogged_fraction=0.0,
        carb_count_sd=0.0,
        meal_log_time_sd_min=0.0,
        app_fidelity=False,
        truth_sd_scale=0.25,
    )
    return records


def test_non_aid_synthetic_delivered_basal_intervals_cover_every_day_boundary():
    records = _records()

    for record in records:
        events = record["events"]["temp_basal"]["events"]
        assert events
        assert events[0]["timestamp"] == record["utc_anchor"]
        assert all(float(event["duration"]) > 0.0 for event in events)
        assert sum(float(event["duration"]) for event in events) == 24.0 * 60.0

    timeline = build_timeline(
        records,
        "synthetic-delivery-test",
        sex="male",
        infer_sites=False,
    )
    assert np.all(np.isfinite(timeline.basal_upm))
    assert np.all(timeline.insulin_observed)


def test_scheduled_basal_settings_do_not_fill_missing_delivery_events():
    records = deepcopy(_records())
    for record in records:
        record["events"]["temp_basal"]["events"] = []

    timeline = build_timeline(
        records,
        "synthetic-delivery-test",
        sex="male",
        infer_sites=False,
    )
    assert np.isnan(timeline.basal_upm).all()
    assert not np.any(timeline.insulin_observed)
