"""Synthetic-only timelines for the personal-context v3 acceptance benchmark.

The truth manifest returned by :func:`build_personal_context_fixture` is kept
separate from the canonical records. The engine never receives it. The
generator models serially correlated CGM, event jitter, competing context,
missingness, and deliberately invalid insulin lookalikes.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

TIMEZONE = "America/New_York"
ZONE = ZoneInfo(TIMEZONE)
DATASET_ID = "personal-context-v3-synthetic"
SUPPORTED_SCENARIOS = frozenset({"full", "delayed", "null", "null_clock", "missing", "ood"})


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _record(
    identifier: str, metric: str, value: float, unit: str, start: datetime, *,
    end: datetime | None = None, metadata: dict[str, Any] | None = None,
    available_at: datetime | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "schema": "insite.observations.v2", "id": identifier, "metric": metric,
        "value": float(value), "unit": unit, "start": _iso(start), "timezone": TIMEZONE,
        "source": "synthetic_v3", "kind": "measured", "quality": "synthetic_demo",
        "syntheticDemo": True, "datasetId": DATASET_ID,
    }
    if end is not None:
        item["end"] = _iso(end)
    if available_at is not None:
        item["available_at"] = _iso(available_at)
    if metadata:
        item["metadata"] = dict(metadata)
    return item


def _circular_minutes(left: int, right: int) -> int:
    delta = abs(left - right) % 1440
    return min(delta, 1440 - delta)


def _meal_wave(minute: int, meal_minute: int, amplitude: float, duration: int = 240) -> float:
    elapsed = minute - meal_minute
    if elapsed < 0 or elapsed >= duration:
        return 0.0
    return amplitude * math.sin(math.pi * elapsed / duration)


def build_personal_context_fixture(
    *, days: int = 128, seed: int = 0, scenario: str = "full",
    include_non_delivery_decoys: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return fictional canonical observations and a separate truth manifest.

    Histories as short as 14 days are valid generator outputs. Statistical
    acceptance is evaluated later at several frozen as-of points. ``full``
    plants a clock interval, a two-predicate event response, and two daily
    context responses. ``delayed`` plants only the two-predicate 3--6 hour
    response. ``null`` preserves meal rhythms and context without planted
    context effects, ``null_clock`` removes those meal waves to provide a
    flat-clock autocorrelated null, ``missing`` increases source gaps, and
    ``ood`` makes the last query novel.
    """
    if days < 14:
        raise ValueError("personal-context fixture requires at least 14 days")
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"unsupported synthetic scenario: {scenario}")

    rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=ZONE)
    clock_width = (2, 3, 4)[seed % 3]
    # Keep the planted interval away from the generated meal peaks and the
    # overnight daily-context outcome while still varying it by seed.
    clock_start = (10, 12, 14)[(seed // 3) % 3]
    scheduled_amount = 900.0 + (seed % 37) + 0.125
    ambiguous_amount = 600.0 + (seed % 29) + 0.375

    # Precompute context so clean/contaminated fixtures differ only by decoys.
    contexts: list[dict[str, Any]] = []
    site_change_day = 0
    next_site_change = rng.randint(6, 9)
    next_cycle_day = rng.randint(3, 12)
    for day_index in range(days):
        if day_index == next_site_change:
            site_change_day = day_index
            next_site_change += rng.randint(6, 9)
        cycle_onset = day_index == next_cycle_day
        if cycle_onset:
            next_cycle_day += rng.randint(29, 37)
        stratum = (day_index * 3 + seed) % 4
        active = stratum in {0, 1}
        short_sleep = stratum in {0, 2}
        regime_high = ((day_index + seed * 2) // 12) % 2 == 1
        activity_regime_high = ((day_index + seed * 3) // 8) % 2 == 1
        activity_minutes = (
            ((55.0 if activity_regime_high else 38.0) + rng.uniform(-7.0, 8.0))
            if active else ((18.0 if activity_regime_high else 4.0) + rng.uniform(-2.0, 4.0))
        )
        temp_shift = 0.20 if regime_high else -0.04
        contexts.append({
            "active": active, "short_sleep": short_sleep, "regime_high": regime_high,
            "activity_regime_high": activity_regime_high,
            "contradictory": (day_index * 17 + seed * 11) % 19 == 0,
            "site_change": day_index == site_change_day, "site_age": day_index - site_change_day,
            "cycle_onset": cycle_onset,
            "morning_minute": 7 * 60 + 20 + rng.randint(-38, 42),
            "evening_minute": 18 * 60 + 35 + rng.randint(-55, 55),
            "sleep_hours": (5.15 + rng.uniform(-0.22, 0.25)) if short_sleep else (6.85 + rng.uniform(-0.28, 0.34)),
            "activity_minutes": activity_minutes,
            "body_temperature": 36.58 + temp_shift + 0.035 * math.sin(day_index / 8.0) + rng.gauss(0.0, 0.025),
            "wrist_temperature": 36.42 + temp_shift * 0.8 + rng.gauss(0.0, 0.03),
            "morning_carbs": 43.0 + rng.uniform(-9.0, 11.0),
            "evening_carbs": 55.0 + rng.uniform(-13.0, 14.0),
        })

    # Define the planted long-context condition from the same observable
    # summaries available at a noon anchor. No engine helper or latent regime
    # flag participates in this truth definition.
    for day_index, context in enumerate(contexts):
        recent = contexts[max(0, day_index - 6):day_index + 1]
        prior = contexts[max(0, day_index - 34):max(0, day_index - 6)]
        activity_7d = sum(float(item["activity_minutes"]) for item in recent) / 7.0 if len(recent) == 7 else None
        temp_7d = sum(float(item["body_temperature"]) for item in recent) / 7.0 if len(recent) == 7 else None
        temp_prior28 = sum(float(item["body_temperature"]) for item in prior) / 28.0 if len(prior) == 28 else None
        temp_delta = temp_7d - temp_prior28 if temp_7d is not None and temp_prior28 is not None else None
        wrist_7d = sum(float(item["wrist_temperature"]) for item in recent) / 7.0 if len(recent) == 7 else None
        wrist_prior28 = sum(float(item["wrist_temperature"]) for item in prior) / 28.0 if len(prior) == 28 else None
        wrist_delta = wrist_7d - wrist_prior28 if wrist_7d is not None and wrist_prior28 is not None else None
        context["activity_7d_observed"] = activity_7d
        context["body_temperature_7d_vs_prior28_observed"] = temp_delta
        context["observable_body_pair"] = bool(
            activity_7d is not None and activity_7d >= 29.0
            and temp_delta is not None and temp_delta >= 0.055
        )
        context["observable_wrist_pair"] = bool(
            activity_7d is not None and activity_7d >= 29.0
            and wrist_delta is not None and wrist_delta >= 0.045
        )

    milestones = tuple(day for day in (21, 42, 84, 120) if day <= days)
    if days not in milestones:
        milestones += (days,)
    as_of = _iso(start + timedelta(days=days - 1, hours=12))
    truth: dict[str, Any] = {
        "datasetId": DATASET_ID, "seed": seed, "days": days, "scenario": scenario,
        # Keep both spellings while callers migrate from the JSON-facing
        # camelCase manifest to the Python-facing snake_case contract.
        "asOf": as_of, "as_of": as_of,
        "asOfPoints": {str(day): _iso(start + timedelta(days=day - 1, hours=12)) for day in sorted(set(milestones))},
        "expectedFindings": {
            "clock": {
                "applicable": scenario == "full", "analysisKind": "clock_window_recurrence",
                "startHour": clock_start, "widthHours": clock_width,
                "outcome": "low_rate", "effectSign": "positive",
                "outcomeAlternatives": [
                    {"outcome": "low_rate", "effectSign": "positive"},
                    {"outcome": "tir", "effectSign": "negative"},
                ],
            },
            "event": {"applicable": scenario == "full", "analysisKind": "event_response", "predicates": ["after_activity", "short_sleep_before"], "outcome": "firstResponse", "horizon": "0_3h", "effectSign": "negative"},
            "delayedEvent": {"applicable": scenario == "delayed", "analysisKind": "event_response", "predicates": ["after_activity", "short_sleep_before"], "outcome": "lateResponse", "horizon": "3_6h", "effectSign": "negative"},
            "siteNight": {"applicable": scenario == "full", "analysisKind": "daily_context_response", "predicates": ["site_age_late"], "outcome": "overnightResponse", "horizon": "overnight", "effectSign": "negative"},
            "temperatureActivityNight": {
                "applicable": scenario == "full", "analysisKind": "daily_context_response",
                "predicates": ["recent_activity_high", "body_temperature_high"],
                "predicateAlternatives": [
                    ["recent_activity_high", "body_temperature_high"],
                    ["recent_activity_high", "wrist_temperature_high"],
                ],
                "outcome": "overnightResponse", "horizon": "overnight", "effectSign": "positive",
                "observableDefinition": {
                    "activity7dAtLeastMinutesPerDay": 29.0,
                    "bodyTemperature7dVsPrior28AtLeastC": 0.055,
                    "wristTemperature7dVsPrior28AtLeastC": 0.045,
                    "combination": "separate additive activity-by-temperature conditions",
                },
            },
        },
        "negativeControls": {"scheduledAmount": scheduled_amount, "ambiguousAmount": ambiguous_amount, "metrics": ["delivered_bolus"]},
    }

    records: list[dict[str, Any]] = [_record(
        "profile-0000", "therapy_profile_observed", 1.0, "event", start,
        metadata={"profile_fingerprint": f"synthetic-profile-{seed % 5}"},
    )]
    ar_state = rng.gauss(0.0, 3.0)
    slow_state = rng.gauss(0.0, 2.0)
    for day_index, context in enumerate(contexts):
        day = start + timedelta(days=day_index)
        sleep_end = day.replace(hour=6, minute=25) + timedelta(minutes=rng.randint(-18, 22))
        sleep_start = sleep_end - timedelta(hours=float(context["sleep_hours"]))
        sleep_missing = scenario == "missing" and day_index % 17 in {4, 5, 6}
        if not sleep_missing:
            records.append(_record(
                f"sleep-{day_index:04d}", "sleep_hours", float(context["sleep_hours"]), "hours",
                sleep_start, end=sleep_end, available_at=sleep_end + timedelta(minutes=rng.randint(1, 14)),
                metadata={"completed": True},
            ))

        activity_time = day.replace(hour=5, minute=15) + timedelta(minutes=rng.randint(-24, 24))
        context_missing = scenario == "missing" and day_index % 23 in {8, 9, 10, 11}
        if not context_missing:
            activity = float(context["activity_minutes"])
            records.append(_record(
                f"activity-{day_index:04d}", "exercise_minutes", activity, "minutes", activity_time,
                end=activity_time + timedelta(minutes=max(1.0, activity)), metadata={"observed": True},
            ))
            records.append(_record(
                f"steps-{day_index:04d}", "steps", max(300.0, 5600.0 + activity * 72.0 + rng.gauss(0.0, 950.0)),
                "count", day.replace(hour=11, minute=20), end=day.replace(hour=11, minute=55), metadata={"observed": True},
            ))
            records.append(_record(
                f"hr-{day_index:04d}", "heart_rate", 65.0 + activity * 0.08 + rng.gauss(0.0, 3.2),
                "bpm", day.replace(hour=10, minute=40), metadata={"observed": True},
            ))

        temp_missing = scenario == "missing" and day_index % 29 in {12, 13, 14, 15, 16}
        if not temp_missing:
            records.append(_record(
                f"body-{day_index:04d}", "body_temperature", float(context["body_temperature"]),
                "C", day.replace(hour=6, minute=50),
            ))
            records.append(_record(
                f"wrist-{day_index:04d}", "sleeping_wrist_temperature", float(context["wrist_temperature"]),
                "C", day.replace(hour=6, minute=55),
            ))
        if context["cycle_onset"] and not (scenario == "missing" and day_index % 2):
            records.append(_record(f"cycle-{day_index:04d}", "period_onset", 1.0, "event", day.replace(hour=7), metadata={"explicitCycleStart": True}))
        if context["site_change"]:
            records.append(_record(f"site-{day_index:04d}", "site_change_event", 1.0, "event", day.replace(hour=8), metadata={"observed": True}))

        for meal_name, minute_key, carb_key in (
            ("morning", "morning_minute", "morning_carbs"), ("evening", "evening_minute", "evening_carbs"),
        ):
            meal_time = day + timedelta(minutes=int(context[minute_key]))
            carbs = float(context[carb_key])
            event_id = f"meal-{meal_name}-{day_index:04d}"
            records.append(_record(event_id, "meal_carbs", carbs, "grams", meal_time, metadata={"event_id": event_id}))
            records.append(_record(
                f"bolus-{meal_name}-{day_index:04d}", "delivered_bolus", carbs / 10.5 + rng.uniform(-0.32, 0.33), "U",
                meal_time + timedelta(minutes=rng.randint(-6, 9)), metadata={"event_id": event_id, "delivery": "actual", "actualDelivery": True},
            ))

        if include_non_delivery_decoys:
            if day_index % 3 == 0:
                records.append(_record(
                    f"scheduled-{day_index:04d}", "delivered_bolus", scheduled_amount, "U", day.replace(hour=6, minute=45),
                    metadata={"delivery": "scheduled", "scheduleOnly": True},
                ))
            if day_index % 4 == 0:
                records.append(_record(
                    f"ambiguous-{day_index:04d}", "delivered_bolus", ambiguous_amount, "U", day.replace(hour=7, minute=5),
                    metadata={"delivery": "requested", "actualDelivery": False},
                ))

        morning_minute = int(context["morning_minute"])
        evening_minute = int(context["evening_minute"])
        pair_condition = bool(context["active"] and context["short_sleep"])
        for sample in range(288):
            minute = sample * 5
            sample_time = day + timedelta(minutes=minute)
            slow_state = 0.995 * slow_state + rng.gauss(0.0, 0.18)
            ar_state = 0.82 * ar_state + rng.gauss(0.0, 2.4)
            value = 118.0 + 5.5 * math.sin(2.0 * math.pi * (minute - 260) / 1440.0) + slow_state + ar_state
            if scenario != "null_clock":
                value += _meal_wave(minute, morning_minute, 26.0 + rng.uniform(-1.3, 1.3))
                value += _meal_wave(minute, evening_minute, 31.0 + rng.uniform(-1.7, 1.7))

            if scenario == "full":
                center = (clock_start * 60 + clock_width * 30) % 1440
                distance = _circular_minutes(minute, center)
                if distance < clock_width * 30:
                    clock_effect = -62.0 * (0.9 + 0.1 * math.cos(math.pi * distance / max(1, clock_width * 30)))
                    value += -0.35 * clock_effect if context["contradictory"] else clock_effect
                elapsed = minute - morning_minute
                if 0 <= elapsed < 180:
                    if pair_condition:
                        value += 18.0 if context["contradictory"] else -46.0
                    elif bool(context["active"]) != bool(context["short_sleep"]):
                        value += 11.0
                night_context = context if minute >= 22 * 60 else (contexts[day_index - 1] if day_index > 0 and minute < 6 * 60 else None)
                if night_context is not None:
                    night_effect = (-42.0 if int(night_context["site_age"]) >= 3 else 0.0)
                    night_effect += 28.0 if night_context["observable_body_pair"] else 0.0
                    night_effect += 28.0 if night_context["observable_wrist_pair"] else 0.0
                    value += -0.45 * night_effect if night_context["contradictory"] else night_effect

            if scenario == "delayed":
                elapsed = minute - morning_minute
                if 180 <= elapsed < 360:
                    if pair_condition:
                        value += 15.0 if context["contradictory"] else -43.0
                    elif bool(context["active"]) != bool(context["short_sleep"]):
                        value += 9.0
            if scenario == "ood" and day_index == days - 1 and minute <= 12 * 60:
                value += 105.0 + 0.05 * minute

            missing_probability = 0.012 + (0.035 if pair_condition and scenario in {"full", "delayed"} else 0.0)
            if scenario == "missing" and day_index % 21 in {6, 7, 8}:
                missing_probability += 0.24
            if rng.random() < missing_probability:
                continue
            available_delay = 2 if rng.random() > 0.025 else rng.randint(30, 180)
            records.append(_record(
                f"glucose-{day_index:04d}-{sample:03d}", "glucose", max(35.0, min(360.0, value)), "mg/dL",
                sample_time, end=sample_time + timedelta(minutes=5), available_at=sample_time + timedelta(minutes=available_delay),
            ))

    return records, truth


__all__ = ["DATASET_ID", "SUPPORTED_SCENARIOS", "TIMEZONE", "build_personal_context_fixture"]
