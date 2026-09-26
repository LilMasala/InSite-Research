"""Personal context engine v3.

This module is the backend counterpart to the daily brief surface.  It keeps
the wire document deliberately small while doing the work in separate,
testable stages:

* localized clock-window recurrence discovery;
* event-aligned response comparisons with a discovery-only ridge baseline; and
* current-context analogue retrieval using only features known at each anchor.

The engine is observational.  It never estimates a dose effect, recommends a
setting, fills a missing stream with zero, or selects a historical episode by
its future outcome.  ``build_daily_brief`` remains the public entrypoint; its
legacy implementation is used only when ``BriefConfig.engine_mode`` is set to
``"legacy"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from .daily_brief import (
    BRIEF_SCHEMA,
    BriefConfig,
    BriefRecord,
    _attr,
    _day_bounds,
    _finite,
    _is_actual_delivery,
    _iso,
    _local_date,
    _local_datetime,
    _outcome,
    _parse_datetime,
    _safe_zone,
    _series,
    _RecordIndex,
    build_daily_brief,
    normalize_brief_records,
)
from .numeric_search import (
    DAILY_FEATURES,
    EVENT_FEATURES,
    FEATURES as NUMERIC_FEATURES,
    describe_rule,
    discover_numeric_rules,
    evaluate_rule,
    prepare_numeric_search_space,
)
from .brief_presentation import present_brief

ENGINE_REVISION = "personal-context-v7-support-aware-recovery-presentation"
_OUTCOME_ORDER = ("0_3h", "3_6h", "overnight", "24h")
OUTCOME_DESCRIPTORS: dict[str, dict[str, Any]] = {
    "firstResponse": {"key": "firstResponse", "episodeKind": "meal", "window": "0–3 hours after recorded meal", "unit": "mg/dL", "label": "0–3 hour glucose change", "minimumEffect": 6.0, "family": "event"},
    "lateResponse": {"key": "lateResponse", "episodeKind": "meal", "window": "3–6 hours after recorded meal", "unit": "mg/dL", "label": "3–6 hour glucose change", "minimumEffect": 6.0, "family": "event"},
    "mealVariance3h": {"key": "mealVariance3h", "episodeKind": "meal", "window": "0–3 hours after recorded meal", "unit": "(mg/dL)^2", "label": "0–3 hour glucose variance", "minimumEffect": 36.0, "family": "event"},
    "mealVariance24h": {"key": "mealVariance24h", "episodeKind": "meal", "window": "0–24 hours after recorded meal", "unit": "(mg/dL)^2", "label": "24-hour glucose variance", "minimumEffect": 36.0, "family": "event"},
    "bolusResponse0_3h": {"key": "bolusResponse0_3h", "episodeKind": "delivered_bolus", "window": "0–3 hours following recorded delivered bolus", "unit": "mg/dL", "label": "0–3 hour glucose change following recorded delivered bolus", "minimumEffect": 6.0, "family": "event", "adjustBolus": False},
    "bolusResponse3_6h": {"key": "bolusResponse3_6h", "episodeKind": "delivered_bolus", "window": "3–6 hours following recorded delivered bolus", "unit": "mg/dL", "label": "3–6 hour glucose change following recorded delivered bolus", "minimumEffect": 6.0, "family": "event", "adjustBolus": False},
    "correctionOnlyResponse0_3h": {"key": "correctionOnlyResponse0_3h", "episodeKind": "delivered_bolus", "classification": "correction_only", "window": "0–3 hours following explicitly classified correction-only delivered bolus", "unit": "mg/dL", "label": "0–3 hour glucose change following recorded correction-only delivered bolus", "minimumEffect": 6.0, "family": "event", "adjustBolus": False},
    "correctionOnlyResponse3_6h": {"key": "correctionOnlyResponse3_6h", "episodeKind": "delivered_bolus", "classification": "correction_only", "window": "3–6 hours following explicitly classified correction-only delivered bolus", "unit": "mg/dL", "label": "3–6 hour glucose change following recorded correction-only delivered bolus", "minimumEffect": 6.0, "family": "event", "adjustBolus": False},
    "preMealResponse": {"key": "preMealResponse", "episodeKind": "pre_meal", "window": "3 hours before recorded meal", "unit": "mg/dL", "label": "retrospective 3-hour pre-meal glucose change", "minimumEffect": 6.0, "family": "event"},
    "fastingResponse0_3h": {"key": "fastingResponse0_3h", "episodeKind": "explicit_fasting", "window": "first 3 hours within an explicitly recorded fasting interval", "unit": "mg/dL", "label": "retrospective 0–3 hour glucose change during explicitly recorded fasting", "minimumEffect": 6.0, "family": "event"},
    "overnightResponse": {"key": "overnightResponse", "episodeKind": "daily_context", "window": "following overnight window", "unit": "mg/dL", "label": "following overnight mean glucose change", "minimumEffect": 12.0, "family": "daily"},
    "dayResponse": {"key": "dayResponse", "episodeKind": "daily_context", "window": "following 24 hours", "unit": "mg/dL", "label": "following 24-hour mean glucose change", "minimumEffect": 12.0, "family": "daily"},
    "dayVarianceResponse": {"key": "dayVarianceResponse", "episodeKind": "daily_context", "window": "following 24 hours", "unit": "(mg/dL)^2", "label": "following 24-hour glucose variance", "minimumEffect": 36.0, "family": "daily"},
}
for _clock_hour in range(0, 24, 3):
    _clock_interval = f"{_clock_hour:02d}:00–{(_clock_hour + 3) % 24:02d}:00"
    for _target_base, _unit, _label, _threshold in (
        ("clockBinMeanChange", "mg/dL", "3-hour local-bin glucose mean change", 6.0),
        ("clockBinVariance", "(mg/dL)^2", "3-hour local-bin glucose variance", 36.0),
    ):
        _target_key = f"{_target_base}{_clock_hour:02d}"
        OUTCOME_DESCRIPTORS[_target_key] = {
            "key": _target_key, "episodeKind": f"clock_bin_{_clock_hour:02d}",
            "window": _clock_interval, "unit": _unit,
            "label": f"{_label} ({_clock_interval} local)", "minimumEffect": _threshold,
            "family": "event",
        }
_FAMILY_NAMES = ("glucose", "insulin", "carbohydrate", "activity", "physiology", "sleep", "temperature", "cycle", "site", "mood")
_SLEEP_STAGE_METRICS = {
    "sleep_in_bed_interval": "in_bed",
    "sleep_awake_interval": "awake",
    "sleep_core_interval": "core",
    "sleep_deep_interval": "deep",
    "sleep_rem_interval": "rem",
    "sleep_unspecified_interval": "unspecified",
}
_EVENT_PREDICATE_SETS = (
    ("after_activity",), ("short_sleep_before",), ("recorded_cycle_early",), ("site_age_late",),
    ("starting_glucose_high",), ("bolus_large",), ("carbs_large",), ("morning_event",),
    ("afternoon_event",), ("evening_event",), ("actual_bolus_present",),
    ("sleep_rem_low",), ("sleep_fragmentation_high",), ("hrv_sdnn_low",),
    ("heart_rate_deviation_high",), ("mood_valence_low",), ("evening_workout",), ("workout_long",),
    ("daytime_steps_high",),
    ("after_activity", "short_sleep_before"), ("recorded_cycle_early", "short_sleep_before"),
    ("recorded_cycle_early", "sleep_fragmentation_high"), ("hrv_sdnn_low", "after_activity"),
    ("mood_valence_low", "carbs_large"), ("sleep_rem_low", "carbs_large"),
    ("daytime_steps_high", "carbs_large"),
)
_DAILY_PREDICATE_SETS = (
    ("site_age_late",), ("recent_activity_high",), ("body_temperature_high",),
    ("wrist_temperature_high",), ("short_sleep_before",), ("recorded_cycle_early",),
    ("sleep_rem_low",), ("sleep_fragmentation_high",), ("hrv_sdnn_low",),
    ("heart_rate_deviation_high",), ("mood_valence_low",), ("evening_workout",), ("workout_long",),
    ("recent_activity_high", "body_temperature_high"),
    ("recent_activity_high", "wrist_temperature_high"),
    ("recorded_cycle_early", "short_sleep_before"),
    ("recorded_cycle_early", "sleep_fragmentation_high"),
    ("hrv_sdnn_low", "evening_workout"), ("mood_valence_low", "short_sleep_before"),
    ("site_age_late", "evening_workout"),
)


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if _finite(value) is not None]
    return sum(clean) / len(clean) if clean else None


def _median(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if _finite(value) is not None]
    return statistics.median(clean) if clean else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if _finite(value) is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    low = int(math.floor(position)); high = int(math.ceil(position))
    if low == high:
        return clean[low]
    return clean[low] + (clean[high] - clean[low]) * (position - low)


def _week(value: date) -> str:
    return (value - timedelta(days=value.weekday())).isoformat()


def _elapsed_hours(start: datetime, end: datetime) -> float:
    """Physical elapsed hours; same-ZoneInfo subtraction is wall-clock time."""
    return (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() / 3600.0


def _add_elapsed(anchor: datetime, *, hours: float) -> datetime:
    return (anchor.astimezone(UTC) + timedelta(hours=hours)).astimezone(anchor.tzinfo or UTC)


def _safe_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:16]


def _clock_label(start_hour: int, width: int) -> str:
    def label(hour: int) -> str:
        return f"{hour % 24:02d}:00"
    return f"{label(start_hour)}–{label(start_hour + width)}"


def _metric(records: Sequence[BriefRecord], name: str) -> list[BriefRecord]:
    return [record for record in records if record.metric == name]


def _known_before(record: BriefRecord, anchor: datetime) -> bool:
    return record.known_at <= anchor and record.end_or_start <= anchor


def _availability_view(records: Sequence[BriefRecord], anchor: datetime, cutoff: datetime, mode: str) -> list[BriefRecord]:
    if mode == "strict_as_of_anchor":
        return list(records)
    if mode != "retrospective_reconstruction":
        raise ValueError("unsupported availability mode")
    # Reconstruction can use a later import only when the observation itself
    # completed before the historical anchor and was known by report time.
    return [
        replace(record, available_at=record.end_or_start)
        for record in records
        if record.end_or_start <= anchor and record.known_at <= cutoff
    ]


def _event_records(
    records: Sequence[BriefRecord], metric: str, start: datetime, end: datetime, *, anchor: datetime | None = None,
) -> list[BriefRecord]:
    return [
        record for record in records
        if record.metric == metric and start <= record.start < end
        and (anchor is None or _known_before(record, anchor))
    ]


def _glucose_outcome(
    records: Sequence[BriefRecord], start: datetime, end: datetime, *, cutoff: datetime,
    min_coverage: float, index: _RecordIndex | None = None,
) -> dict[str, Any] | None:
    physical_start = start.astimezone(UTC); physical_end = end.astimezone(UTC)
    if physical_end > cutoff.astimezone(UTC):
        return None
    series = _series(records, physical_start, physical_end, cutoff=cutoff, min_coverage=min_coverage, index=index)
    return _outcome(series, physical_start, physical_end)


def _observed_runs(points: Sequence[Mapping[str, Any]], *, start: datetime, kind: str) -> list[dict[str, Any]]:
    """Count only contiguous observed five-minute CGM bins, without filling gaps."""
    threshold = 70.0 if kind == "low" else 180.0
    qualifies = (lambda value: value < threshold) if kind == "low" else (lambda value: value > threshold)
    runs: list[dict[str, Any]] = []
    first: float | None = None
    prior: float | None = None
    bins = 0

    def finish() -> None:
        if first is not None and bins >= 3:
            runs.append({
                "start": _iso(start.astimezone(UTC) + timedelta(minutes=first)),
                "end": _iso(start.astimezone(UTC) + timedelta(minutes=first + bins * 5)),
                "durationMinutes": bins * 5,
            })

    for point in sorted(points, key=lambda point: float(point["minute"])):
        minute = float(point["minute"])
        if not qualifies(float(point["glucose"])):
            finish(); first = prior = None; bins = 0
            continue
        if prior is None or minute != prior + 5:
            finish(); first = minute; bins = 0
        bins += 1; prior = minute
    finish()
    return runs


def _metric_outcome(outcome: Mapping[str, Any], target: str, width_hours: float) -> float | None:
    if target == "low_rate":
        return float(outcome["lowMinutes"]) / max(float(outcome.get("observedBins", 0)) * 5.0, 1.0)
    if target == "high_rate":
        return float(outcome["highMinutes"]) / max(float(outcome.get("observedBins", 0)) * 5.0, 1.0)
    if target == "mean":
        return float(outcome["mean"])
    if target == "tir":
        return float(outcome["tir"])
    return None


def _target_label(target: str) -> tuple[str, str]:
    return {
        "low_rate": ("time below 70", "fraction of observed minutes"),
        "high_rate": ("time above 180", "fraction of observed minutes"),
        "mean": ("mean glucose", "mg/dL"),
        "tir": ("time in range", "fraction of observed readings"),
    }[target]


def _target_threshold(target: str) -> float:
    return {"low_rate": 0.08, "high_rate": 0.10, "mean": 12.0, "tir": 0.08}[target]


def _discovery_score(effect: float, target: str) -> float:
    """Dimensionless discovery ranking across outcome units."""
    threshold = _target_threshold(target) if target in {"low_rate", "high_rate", "mean", "tir"} else _response_threshold(target)
    return abs(float(effect)) / max(threshold, 1e-9)


def _response_threshold(target: str) -> float:
    descriptor = OUTCOME_DESCRIPTORS.get(str(target), {})
    return float(descriptor.get("minimumEffect", 12.0 if target in {"overnightResponse", "dayResponse"} else 6.0))


def _normalized_discovery_score(subgroup_quality: float, residual_scale_population_sd: float) -> float | None:
    """Scale native numeric subgroup quality by discovery-only target spread."""
    quality = _finite(subgroup_quality)
    scale = _finite(residual_scale_population_sd)
    if quality is None or scale is None or quality <= 0.0 or scale <= 0.0:
        return None
    score = quality / scale
    return score if math.isfinite(score) else None


def _day_candidates(records: Sequence[BriefRecord], zone: ZoneInfo, current_day: date) -> list[date]:
    days = sorted({_local_date(record.start, zone) for record in records if record.metric == "glucose"})
    return [day for day in days if day < current_day]


def _block_bootstrap(
    differences: Sequence[float], dates: Sequence[date], *, draws: int, block_days: int, seed: int,
    family_size: int = 1, alpha: float = 0.05,
) -> tuple[float, float, float, bool]:
    """Approximate whole-day-block interval and sign test.

    The result is intentionally labelled approximate in evidence.  Blocks are
    contiguous in the sorted date order and are never individual CGM points.
    """
    # Bonferroni-adjusted percentile intervals cover the single frozen family
    # across both discovery engines. Twenty expected draws in each adjusted
    # tail is a numerical-resolution floor, not an error-rate guarantee.
    family_size = max(1, int(family_size))
    tail = float(alpha) / (2.0 * family_size)
    if not differences or int(draws) * tail < 20:
        return 1.0, 0.0, 0.0, False
    # One independent value per calendar day. Gaps terminate a block rather
    # than turning two distant observed dates into a contiguous block.
    by_day: dict[date, list[float]] = defaultdict(list)
    for day, value in zip(dates, differences):
        by_day[day].append(float(value))
    paired = sorted((day, float(_mean(values))) for day, values in by_day.items())
    values = [value for _, value in paired]
    size = max(1, int(block_days))
    blocks: list[list[float]] = []
    current: list[float] = []
    prior_day: date | None = None
    for day, value in paired:
        if prior_day is None or ((day - prior_day).days == 1 and len(current) < size):
            current.append(value)
        else:
            if current:
                blocks.append(current)
            current = [value]
        prior_day = day
    if current:
        blocks.append(current)
    if len(values) < 3 or len(blocks) < 3:
        return 1.0, 0.0, 0.0, False
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(max(40, int(draws))):
        sampled: list[float] = []
        while len(sampled) < len(values):
            sampled.extend(rng.choice(blocks))
        sampled = sampled[:len(values)]
        bootstrap.append(sum(sampled) / max(1, len(sampled)))
    low = _quantile(bootstrap, tail) or 0.0
    high = _quantile(bootstrap, 1.0 - tail) or 0.0
    # Release is based on the adjusted interval. Do not manufacture a
    # sign-flip p-value from a heuristic block scheme.
    p_value = 0.0 if (low > 0 or high < 0) else 1.0
    stable = _leave_one_out_stable(values)
    return p_value, low, high, stable


def _leave_one_out_stable(values: Sequence[float]) -> bool:
    if len(values) < 3:
        return False
    full = sum(values) / len(values)
    if full == 0:
        return False
    same = 0
    for index in range(len(values)):
        left = list(values[:index]) + list(values[index + 1:])
        if left and (sum(left) / len(left)) * full > 0:
            same += 1
    # Every leave-one-day omission must preserve the direction.  An 80% rule
    # can allow one influential reversal in a small confirmation arm.
    return same == len(values)


def _holm(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (total - rank)))
        adjusted[index] = running
    return adjusted


def _same_window(start_hour: int, width: int, day: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    # Keep local wall-clock bounds until the physical-duration guard below.
    # _local_datetime returns UTC, so adding a width there would silently
    # bypass the DST crossing check.
    start = datetime.combine(day, time(start_hour, 0), tzinfo=zone)
    return start, start + timedelta(hours=width)


def _clock_window_rows(
    records: Sequence[BriefRecord], days: Sequence[date], *, zone: ZoneInfo, cutoff: datetime,
    start_hour: int, width: int, target: str, min_coverage: float, index: _RecordIndex | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width_hours = float(width)
    for day in days:
        start, end = _same_window(start_hour, width, day, zone)
        if end > cutoff:
            continue
        # The adjacent interval has the same duration and stays within the
        # same local day.  This avoids comparing an overnight window with an
        # unequal remainder of the day.
        day_start, day_end = _day_bounds(day, zone)
        if end + timedelta(hours=width) <= day_end:
            comparison_start, comparison_end = end, end + timedelta(hours=width)
        elif start - timedelta(hours=width) >= day_start:
            comparison_start, comparison_end = start - timedelta(hours=width), start
        else:
            continue
        if abs(_elapsed_hours(start, end) - width_hours) > 1e-6 or abs(_elapsed_hours(comparison_start, comparison_end) - width_hours) > 1e-6:
            # Local clock windows crossing a DST transition do not have the
            # declared physical duration; omit that day from like-duration inference.
            continue
        observed = _glucose_outcome(records, start, end, cutoff=cutoff, min_coverage=min_coverage, index=index)
        comparison = _glucose_outcome(records, comparison_start, comparison_end, cutoff=cutoff, min_coverage=min_coverage, index=index)
        if observed is None or comparison is None:
            continue
        left = _metric_outcome(observed, target, width_hours)
        right = _metric_outcome(comparison, target, width_hours)
        if left is None or right is None:
            continue
        rows.append({
            "date": day, "start": start, "end": end, "comparisonStart": comparison_start,
            "comparisonEnd": comparison_end, "value": left, "comparison": right,
            "difference": left - right, "observed": observed, "comparisonOutcome": comparison,
        })
    return rows


def _dedupe_windows(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (-_discovery_score(item["discoveryEffect"], item["target"]), item["startHour"], item["width"], item["target"])):
        overlap = False
        for prior in kept:
            if prior["target"] != candidate["target"]:
                continue
            left = max(prior["startHour"], candidate["startHour"])
            right = min(prior["startHour"] + prior["width"], candidate["startHour"] + candidate["width"])
            shared = max(0, right - left)
            smaller = min(prior["width"], candidate["width"])
            if smaller and shared / smaller >= 0.60:
                overlap = True
                break
        if not overlap:
            kept.append(candidate)
    return kept


def _episode_events(records: Sequence[BriefRecord], start: datetime, end: datetime, anchor: datetime, *, cutoff: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    labels = {
        "delivered_bolus": ("recorded delivered bolus insulin", "U"), "meal_carbs": ("recorded carbohydrate", "g"),
        "exercise_minutes": ("recorded activity", "minutes"), "sleep_hours": ("completed sleep", "hours"),
        "sleep_rem_fraction": ("recorded REM fraction", "fraction"),
        "sleep_deep_fraction": ("recorded deep-sleep fraction", "fraction"),
        "sleep_fragmentation_per_hour": ("recorded awake bouts per asleep hour", "bouts/hour"),
        "hrv_sdnn": ("recorded HRV SDNN", "ms"), "workout_minutes": ("recorded workout", "minutes"),
        "workout_average_hr": ("recorded workout average heart rate", "bpm"),
        "mood_valence": ("recorded mood valence", "score"),
        "steps": ("recorded steps", "count"),
        "body_temperature": ("body temperature", "C"), "sleeping_wrist_temperature": ("sleeping-wrist temperature", "C"),
        "period_onset": ("recorded cycle onset", "event"), "site_age_days": ("recorded site age", "days"),
        "site_change": ("recorded site change", "event"), "site_change_event": ("recorded site change", "event"),
    }
    for record in records:
        if record.metric == "delivered_bolus" and not _is_actual_delivery(record):
            continue
        # Evidence may show what happened after an historical anchor.  The
        # availability cutoff is the snapshot boundary; requiring the event
        # itself to end before the anchor would erase every post-anchor event.
        if record.metric not in labels or not (start <= record.start < end) or record.known_at > cutoff:
            continue
        label, unit = labels[record.metric]
        events.append({
            "minute": round((record.start.astimezone(UTC) - anchor.astimezone(UTC)).total_seconds() / 60.0, 2),
            "kind": record.metric, "label": label, "value": float(record.value), "unit": unit,
        })
    return sorted(events, key=lambda item: (item["minute"], item["kind"]))[:48]


def _episode_trace(records: Sequence[BriefRecord], start: datetime, end: datetime, cutoff: datetime, *, anchor: datetime, index: _RecordIndex | None = None) -> tuple[list[dict[str, Any]], float]:
    physical_start = start.astimezone(UTC); physical_end = end.astimezone(UTC)
    series = _series(records, physical_start, physical_end, cutoff=cutoff, min_coverage=0.0, index=index)
    points = list(series.get("points", []))
    for point in points:
        instant = physical_start + timedelta(minutes=float(point["minute"]))
        point["minute"] = round((instant - anchor.astimezone(UTC)).total_seconds() / 60.0, 2)
    return points[:288], float(series.get("coverage", 0.0))


def _current_sleep(records: Sequence[BriefRecord], anchor: datetime) -> float | None:
    candidates = [
        record for record in records if record.metric == "sleep_hours" and record.end is not None and _known_before(record, anchor)
        and anchor - timedelta(hours=36) <= record.end_or_start <= anchor
    ]
    return float(max(candidates, key=lambda item: item.end_or_start).value) if candidates else None


def _latest_value(records: Sequence[BriefRecord], metric: str, anchor: datetime, lookback_days: int = 56) -> float | None:
    candidates = [
        record for record in records if record.metric == metric and _known_before(record, anchor)
        and anchor - timedelta(days=lookback_days) <= record.end_or_start <= anchor
    ]
    return float(max(candidates, key=lambda item: item.end_or_start).value) if candidates else None


def _temp_delta_at(records: Sequence[BriefRecord], metric: str, anchor: datetime) -> float | None:
    current = [record.value for record in records if record.metric == metric and _known_before(record, anchor) and anchor - timedelta(hours=24) <= record.start <= anchor]
    prior_by_day: dict[date, list[float]] = defaultdict(list)
    zone = _safe_zone(records[0].timezone if records else "UTC")
    current_day = _local_date(anchor, zone)
    for record in records:
        if record.metric != metric or not _known_before(record, anchor):
            continue
        day = _local_date(record.start, zone)
        if current_day - timedelta(days=14) <= day < current_day:
            prior_by_day[day].append(float(record.value))
    prior = [_median(values) for values in prior_by_day.values()]
    prior = [value for value in prior if value is not None]
    return (_median(current) - _median(prior)) if current and len(prior) >= 5 else None


def _days_since_onset(records: Sequence[BriefRecord], anchor: datetime) -> float | None:
    onsets = [record.start for record in records if record.metric == "period_onset" and _known_before(record, anchor) and record.start <= anchor]
    return (anchor.date() - max(onsets).date()).days if onsets else None


def _site_age_at(records: Sequence[BriefRecord], anchor: datetime) -> float | None:
    """Days since the latest explicit site-change event, if one exists."""
    changes = [
        record.start for record in records
        if record.metric in {"site_change", "site_change_event"}
        and _known_before(record, anchor) and record.start <= anchor
    ]
    if not changes:
        return None
    return max(0.0, (anchor - max(changes)).total_seconds() / 86400.0)


def _baseline_deviation(
    records: Sequence[BriefRecord], metric: str, anchor: datetime, zone: ZoneInfo, *,
    recent_hours: float = 24.0, baseline_days: int = 14,
) -> float | None:
    """Compare a recent sample average with earlier observed daily medians."""
    recent_start = _add_elapsed(anchor, hours=-recent_hours)
    baseline_start = _add_elapsed(anchor, hours=-24.0 * baseline_days)
    daily: dict[date, list[float]] = defaultdict(list)
    recent = [
        float(record.value) for record in records
        if record.metric == metric and recent_start <= record.start < anchor and _known_before(record, anchor)
    ]
    for record in records:
        if record.metric == metric and baseline_start <= record.start < recent_start and _known_before(record, anchor):
            daily[_local_date(record.start, zone)].append(float(record.value))
    required_days = max(3, math.ceil((baseline_days - recent_hours / 24.0) * 0.40))
    if not recent or len(daily) < required_days:
        return None
    baseline = _mean(_median(values) for values in daily.values())
    return float(_mean(recent) - baseline) if baseline is not None else None


def _recent_mean(records: Sequence[BriefRecord], metric: str, anchor: datetime, *, hours: float) -> float | None:
    values = [
        float(record.value) for record in records
        if record.metric == metric and _add_elapsed(anchor, hours=-hours) <= record.start < anchor
        and _known_before(record, anchor)
    ]
    return _mean(values)


def _recent_workout_context(records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo) -> dict[str, Any]:
    prior_start = _add_elapsed(anchor, hours=-24.0)
    workouts = [
        record for record in records if record.metric == "workout_minutes" and prior_start <= record.start < anchor
        and _known_before(record, anchor)
    ]
    recent = [record for record in workouts if record.start >= _add_elapsed(anchor, hours=-6.0)]
    local_day = anchor.astimezone(zone).date()
    evening = [
        record for record in workouts
        if record.start.astimezone(zone).date() == local_day
        and 17 <= record.start.astimezone(zone).hour < 22
    ]
    return {
        "workout_minutes_6h": sum(float(record.value) for record in recent) if recent else None,
        "workout_minutes_24h": sum(float(record.value) for record in workouts) if workouts else None,
        "workout_count_24h": len(workouts) if workouts else None,
        "workout_average_hr_24h": _recent_mean(records, "workout_average_hr", anchor, hours=24.0),
        # No workout event is not evidence of no exercise. Only observed
        # workout days receive a positive or negative timing flag.
        "evening_workout": bool(evening) if workouts else None,
    }


def _granular_context_at(records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo) -> dict[str, Any]:
    sleep_hours = _current_sleep(records, anchor)
    if sleep_hours is None:
        sleep_hours = _latest_value(records, "sleep_stage_asleep_hours", anchor, lookback_days=2)
    return {
        "sleep_hours": sleep_hours,
        "sleep_rem_fraction": _latest_value(records, "sleep_rem_fraction", anchor, lookback_days=2),
        "sleep_deep_fraction": _latest_value(records, "sleep_deep_fraction", anchor, lookback_days=2),
        "sleep_fragmentation_per_hour": _latest_value(records, "sleep_fragmentation_per_hour", anchor, lookback_days=2),
        "sleep_stage_efficiency": _latest_value(records, "sleep_stage_efficiency", anchor, lookback_days=2),
        "heart_rate_deviation_24h": _baseline_deviation(records, "heart_rate", anchor, zone),
        "hrv_sdnn_deviation_24h": _baseline_deviation(records, "hrv_sdnn", anchor, zone),
        "mood_valence_24h": _recent_mean(records, "mood_valence", anchor, hours=24.0),
        "mood_arousal_24h": _recent_mean(records, "mood_arousal", anchor, hours=24.0),
        **_recent_workout_context(records, anchor, zone),
    }


def _steps_by_local_part(records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo, *, daytime: bool) -> float | None:
    start = _add_elapsed(anchor, hours=-24.0)
    values = []
    for record in records:
        if record.metric != "steps" or not (start <= record.start < anchor) or not _known_before(record, anchor):
            continue
        local_hour = record.start.astimezone(zone).hour
        if (6 <= local_hour < 18) == daytime:
            values.append(float(record.value))
    return sum(values) if values else None


def _therapy_profile_at(records: Sequence[BriefRecord], anchor: datetime) -> str | None:
    candidates = [record for record in records if record.metric == "therapy_profile_observed" and _known_before(record, anchor)]
    if not candidates:
        return None
    fingerprint = max(candidates, key=lambda record: record.end_or_start).attributes.get("profile_fingerprint")
    return str(fingerprint) if isinstance(fingerprint, str) and fingerprint else None


def _union_seconds(pieces: Sequence[tuple[datetime, datetime]]) -> float:
    ordered = sorted(pieces)
    if not ordered:
        return 0.0
    total = 0.0; left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += max(0.0, (right - left).total_seconds())
            left, right = start, end
    return total + max(0.0, (right - left).total_seconds())


def _sleep_stage_summaries(intervals: Sequence[BriefRecord]) -> list[BriefRecord]:
    """Derive per-night stage summaries from the app's labeled intervals.

    Each time slice contributes to at most one stage. Duplicate intervals of
    one category are unioned; overlapping different stages are left unknown,
    apart from an in-bed interval overlapped by a more specific stage.
    """
    ordered = sorted(
        (record for record in intervals if record.end is not None and record.end > record.start),
        key=lambda record: (record.start, record.end_or_start, record.metric),
    )
    if not ordered:
        return []
    groups: list[list[BriefRecord]] = []; current: list[BriefRecord] = []; current_end: datetime | None = None
    for record in ordered:
        if not current or current_end is None or record.start <= current_end + timedelta(minutes=90):
            current.append(record); current_end = max(current_end or record.end_or_start, record.end_or_start)
        else:
            groups.append(current); current = [record]; current_end = record.end_or_start
    if current:
        groups.append(current)

    summaries: list[BriefRecord] = []
    feature_units = {
        "sleep_stage_asleep_hours": "hours", "sleep_stage_in_bed_hours": "hours",
        "sleep_rem_fraction": "fraction", "sleep_deep_fraction": "fraction",
        "sleep_core_fraction": "fraction", "sleep_unspecified_fraction": "fraction",
        "sleep_stage_awake_minutes": "minutes", "sleep_stage_efficiency": "fraction",
        "sleep_fragmentation_per_hour": "bouts/hour",
    }
    for group in groups:
        boundaries = sorted({value for record in group for value in (record.start, record.end_or_start)})
        segments: list[tuple[datetime, datetime, str | None]] = []
        stage_seconds: dict[str, float] = defaultdict(float)
        for left, right in zip(boundaries, boundaries[1:]):
            if right <= left:
                continue
            active = {
                _SLEEP_STAGE_METRICS[record.metric]
                for record in group if record.start < right and record.end_or_start > left
            }
            if len(active) > 1 and "in_bed" in active:
                active.remove("in_bed")
            label = next(iter(active)) if len(active) == 1 else None
            segments.append((left, right, label))
            if label is not None:
                stage_seconds[label] += (right - left).total_seconds()

        asleep_hours = sum(stage_seconds[name] for name in ("core", "deep", "rem", "unspecified")) / 3600.0
        in_bed_hours = _union_seconds([
            (record.start, record.end_or_start) for record in group
            if record.metric == "sleep_in_bed_interval"
        ]) / 3600.0
        if asleep_hours <= 0:
            continue
        awake_bouts = 0; in_awake = False; awake_start: datetime | None = None; awake_end: datetime | None = None
        for left, right, stage in segments:
            if stage == "awake":
                if not in_awake:
                    in_awake = True; awake_start = left
                awake_end = right
            elif in_awake:
                duration = (awake_end - awake_start).total_seconds() if awake_start and awake_end else 0.0
                previous_asleep = any(label in {"core", "deep", "rem", "unspecified"} and end <= awake_start for _, end, label in segments)
                next_asleep = any(label in {"core", "deep", "rem", "unspecified"} and start >= awake_end for start, _, label in segments)
                if duration >= 300 and previous_asleep and next_asleep:
                    awake_bouts += 1
                in_awake = False; awake_start = awake_end = None
        denominator = sum(stage_seconds[name] for name in ("core", "deep", "rem", "unspecified"))
        values = {
            "sleep_stage_asleep_hours": asleep_hours,
            "sleep_stage_in_bed_hours": in_bed_hours if in_bed_hours > 0 else None,
            "sleep_rem_fraction": stage_seconds["rem"] / denominator if denominator > 0 and stage_seconds["rem"] > 0 else None,
            "sleep_deep_fraction": stage_seconds["deep"] / denominator if denominator > 0 and stage_seconds["deep"] > 0 else None,
            "sleep_core_fraction": stage_seconds["core"] / denominator if denominator > 0 and stage_seconds["core"] > 0 else None,
            "sleep_unspecified_fraction": stage_seconds["unspecified"] / denominator if denominator > 0 and stage_seconds["unspecified"] > 0 else None,
            "sleep_stage_awake_minutes": stage_seconds["awake"] / 60.0 if stage_seconds["awake"] > 0 else None,
            "sleep_stage_efficiency": asleep_hours / in_bed_hours if in_bed_hours > 0 else None,
            "sleep_fragmentation_per_hour": awake_bouts / asleep_hours,
        }
        available = [record.known_at for record in group]
        end = max(record.end_or_start for record in group)
        for metric, value in values.items():
            if value is None:
                continue
            summaries.append(BriefRecord(
                id="sleep-stage-" + _safe_id(metric, group[0].start, end, len(group)),
                metric=metric, value=float(value), unit=feature_units[metric],
                start=min(record.start for record in group), end=end, timezone=group[0].timezone,
                source="derived_recorded_sleep_stage", kind="measured",
                available_at=max(available) if available else None,
                attributes={"sleepDerived": "nonoverlapping_stage_union", "intervalCount": len(group)},
            ))
    return summaries


def _collapse_sleep_intervals(records: Sequence[BriefRecord]) -> list[BriefRecord]:
    """Union completed asleep intervals and preserve stage-level summaries."""
    intervals = sorted(
        (record for record in records if record.metric == "sleep_asleep_interval" and record.end is not None),
        key=lambda record: record.start,
    )
    derived: list[BriefRecord] = []
    if intervals:
        groups: list[list[BriefRecord]] = []; current: list[BriefRecord] = []; current_end: datetime | None = None
        for record in intervals:
            if not current or current_end is None or record.start <= current_end + timedelta(minutes=90):
                current.append(record); current_end = max(current_end or record.end_or_start, record.end_or_start)
            else:
                groups.append(current); current = [record]; current_end = record.end_or_start
        if current:
            groups.append(current)
        for group in groups:
            pieces = [(record.start, record.end_or_start) for record in group if record.end is not None]
            seconds = _union_seconds(pieces)
            if seconds <= 0:
                continue
            available = [record.known_at for record in group]
            end = max(record.end_or_start for record in group)
            derived.append(BriefRecord(
                id="sleep-union-" + _safe_id(group[0].start, end, len(group)),
                metric="sleep_hours", value=seconds / 3600.0, unit="hours",
                start=min(record.start for record in group), end=end, timezone=group[0].timezone,
                source="derived_recorded_sleep", kind="measured",
                available_at=max(available) if available else None,
                attributes={"sleepDerived": "asleep_union", "intervalCount": len(group)},
            ))
    stages = [record for record in records if record.metric in _SLEEP_STAGE_METRICS and record.end is not None]
    stage_summaries = _sleep_stage_summaries(stages)
    base = [record for record in records if record.metric != "sleep_asleep_interval"]
    return base + derived + stage_summaries


def _feature_value(records: Sequence[BriefRecord], metric: str, start: datetime, end: datetime, anchor: datetime, *, aggregation: str = "sum") -> tuple[float | None, bool]:
    candidates = _event_records(records, metric, start, end, anchor=anchor)
    if metric == "delivered_bolus":
        candidates = [record for record in candidates if _is_actual_delivery(record)]
    values = [float(record.value) for record in candidates]
    # Event-like quantities are additive over a window.  Averaging a bolus or
    # carbohydrate total silently changes the question from "what was
    # recorded before this anchor?" to "what was a typical event?".
    if not values:
        return None, False
    return ((_mean(values) if aggregation == "mean" else sum(values)), True)


def _glucose_window(records: Sequence[BriefRecord], anchor: datetime, hours: float, min_coverage: float, index: _RecordIndex | None) -> dict[str, Any]:
    start = _add_elapsed(anchor, hours=-hours)
    series = _series(records, start.astimezone(UTC), anchor.astimezone(UTC), cutoff=anchor, min_coverage=min_coverage, index=index)
    points = series.get("points", [])
    values = [float(point["glucose"]) for point in points]
    if len(values) < 2:
        return {"mean": None, "slope": None, "variability": None, "low_rate": None, "high_rate": None, "valid": False}
    slope = (values[-1] - values[0]) / max(1.0, (points[-1]["minute"] - points[0]["minute"]) / 60.0)
    return {
        "mean": _mean(values), "slope": slope,
        "variability": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "low_rate": sum(value < 70 for value in values) / len(values),
        "high_rate": sum(value > 180 for value in values) / len(values),
        "valid": bool(series.get("valid")),
    }


def _trailing_mean(records: Sequence[BriefRecord], metric: str, anchor: datetime, start: datetime, *, aggregation: str = "sum") -> float | None:
    values = [float(record.value) for record in records if record.metric == metric and start <= record.start < anchor and _known_before(record, anchor)]
    if metric == "delivered_bolus":
        values = [float(record.value) for record in records if record.metric == metric and start <= record.start < anchor and _known_before(record, anchor) and _is_actual_delivery(record)]
    if not values:
        return None
    return float(_mean(values) if aggregation == "mean" else sum(values))


def _period_daily_rate(records: Sequence[BriefRecord], metric: str, anchor: datetime, start: datetime, *, actual_delivery: bool = False) -> float | None:
    eligible = [record for record in records if record.metric == metric and start <= record.start < anchor and _known_before(record, anchor)]
    if actual_delivery:
        eligible = [record for record in eligible if _is_actual_delivery(record)]
    expected_days = max(1, int(round(_elapsed_hours(start, anchor) / 24.0)))
    by_day: dict[date, float] = defaultdict(float)
    for record in eligible:
        by_day[record.start.date()] += float(record.value)
    if len(by_day) < max(3, math.ceil(expected_days * 0.70)):
        return None
    return _mean(by_day.values())


def _completed_sleep_mean(records: Sequence[BriefRecord], start: datetime, anchor: datetime) -> float | None:
    values = [float(record.value) for record in records if record.metric == "sleep_hours" and record.end is not None and start <= record.end_or_start < anchor and _known_before(record, anchor)]
    expected_days = max(1, int(round(_elapsed_hours(start, anchor) / 24.0)))
    return _mean(values) if len(values) >= max(3, math.ceil(expected_days * 0.50)) else None


def _period_mean(records: Sequence[BriefRecord], metric: str, start: datetime, end: datetime, *, known_by: datetime) -> float | None:
    by_day: dict[date, list[float]] = defaultdict(list)
    for record in records:
        if record.metric == metric and start <= record.start < end and record.end_or_start <= known_by and record.known_at <= known_by:
            by_day[record.start.date()].append(float(record.value))
    expected_days = max(1, int(round(_elapsed_hours(start, end) / 24.0)))
    if len(by_day) < max(3, math.ceil(expected_days * 0.40)):
        return None
    return _mean(float(_median(values)) for values in by_day.values())


def _period_daily_total(records: Sequence[BriefRecord], metric: str, start: datetime, end: datetime, *, known_by: datetime) -> float | None:
    eligible = [
        record for record in records
        if record.metric == metric and start <= record.start < end and record.end_or_start <= known_by and record.known_at <= known_by
    ]
    expected_days = max(1, int(round(_elapsed_hours(start, end) / 24.0)))
    by_day: dict[date, float] = defaultdict(float)
    for record in eligible:
        by_day[record.start.date()] += float(record.value)
    if len(by_day) < max(3, math.ceil(expected_days * 0.70)):
        return None
    return _mean(by_day.values())


def _anchor_features(
    records: Sequence[BriefRecord], anchor: datetime, *, cutoff: datetime, zone: ZoneInfo,
    min_coverage: float, index: _RecordIndex | None = None, availability_mode: str = "strict_as_of_anchor",
) -> dict[str, Any]:
    records = _availability_view(records, anchor, cutoff, availability_mode)
    index = _RecordIndex.build(records)
    glucose_windows = {hours: _glucose_window(records, anchor, hours, min_coverage, index) for hours in (1.0, 3.0, 6.0, 24.0)}
    prior_7_start = _add_elapsed(anchor, hours=-24.0 * 7)
    prior_35_start = _add_elapsed(anchor, hours=-24.0 * 35)
    prior_7_glucose = _glucose_window(records, anchor, 24.0 * 7, min_coverage, index)
    preceding_28_end = prior_7_start
    preceding_28 = _series(records, prior_35_start.astimezone(UTC), preceding_28_end.astimezone(UTC), cutoff=preceding_28_end, min_coverage=min_coverage, index=index)
    preceding_values = [float(point["glucose"]) for point in preceding_28.get("points", [])]
    prior_28_mean = _mean(preceding_values) if preceding_28.get("valid") else None
    bolus3, bolus3_valid = _feature_value(records, "delivered_bolus", _add_elapsed(anchor, hours=-3), anchor, anchor)
    bolus6, bolus_valid = _feature_value(records, "delivered_bolus", _add_elapsed(anchor, hours=-6), anchor, anchor)
    bolus24, bolus24_valid = _feature_value(records, "delivered_bolus", _add_elapsed(anchor, hours=-24), anchor, anchor)
    carbs3, _ = _feature_value(records, "meal_carbs", _add_elapsed(anchor, hours=-3), anchor, anchor)
    carbs6, carbs_valid = _feature_value(records, "meal_carbs", _add_elapsed(anchor, hours=-6), anchor, anchor)
    carbs24, _ = _feature_value(records, "meal_carbs", _add_elapsed(anchor, hours=-24), anchor, anchor)
    activity, activity_valid = _feature_value(records, "exercise_minutes", _add_elapsed(anchor, hours=-6), anchor, anchor)
    activity24, activity24_valid = _feature_value(records, "exercise_minutes", _add_elapsed(anchor, hours=-24), anchor, anchor)
    steps, steps_valid = _feature_value(records, "steps", _add_elapsed(anchor, hours=-6), anchor, anchor)
    steps24, steps24_valid = _feature_value(records, "steps", _add_elapsed(anchor, hours=-24), anchor, anchor)
    heart_rate, heart_rate_valid = _feature_value(records, "heart_rate", _add_elapsed(anchor, hours=-6), anchor, anchor, aggregation="mean")
    heart_rate24, _ = _feature_value(records, "heart_rate", _add_elapsed(anchor, hours=-24), anchor, anchor, aggregation="mean")
    hrv24, hrv_valid = _feature_value(records, "hrv_sdnn", _add_elapsed(anchor, hours=-24), anchor, anchor, aggregation="mean")
    workout6, workout6_valid = _feature_value(records, "workout_minutes", _add_elapsed(anchor, hours=-6), anchor, anchor)
    workout24, workout24_valid = _feature_value(records, "workout_minutes", _add_elapsed(anchor, hours=-24), anchor, anchor)
    workout_hr24, _ = _feature_value(records, "workout_average_hr", _add_elapsed(anchor, hours=-24), anchor, anchor, aggregation="mean")
    sleep = _current_sleep(records, anchor)
    granular = _granular_context_at(records, anchor, zone)
    if sleep is None:
        sleep = granular["sleep_hours"]
    body_delta = _temp_delta_at(records, "body_temperature", anchor)
    wrist_delta = _temp_delta_at(records, "sleeping_wrist_temperature", anchor)
    cycle = _days_since_onset(records, anchor)
    site = _latest_value(records, "site_age_days", anchor)
    if site is None:
        site = _site_age_at(records, anchor)
    therapy_profile = _therapy_profile_at(records, anchor)
    g1, g3, g6, g24 = (glucose_windows[value] for value in (1.0, 3.0, 6.0, 24.0))
    glucose_long_delta = (float(prior_7_glucose["mean"]) - float(prior_28_mean)) if prior_7_glucose.get("mean") is not None and prior_28_mean is not None else None
    activity_7 = _period_daily_rate(records, "exercise_minutes", anchor, prior_7_start)
    # The preceding comparison ends seven days before the anchor.
    activity_prior28 = _period_daily_rate(records, "exercise_minutes", prior_7_start, prior_35_start)
    bolus_7 = _period_daily_rate(records, "delivered_bolus", anchor, prior_7_start, actual_delivery=True)
    bolus_prior28 = _period_daily_rate(records, "delivered_bolus", prior_7_start, prior_35_start, actual_delivery=True)
    sleep_7 = _completed_sleep_mean(records, prior_7_start, anchor)
    sleep_prior28 = _completed_sleep_mean(records, prior_35_start, prior_7_start)
    carbs_7 = _period_daily_total(records, "meal_carbs", prior_7_start, anchor, known_by=anchor)
    carbs_prior28 = _period_daily_total(records, "meal_carbs", prior_35_start, prior_7_start, known_by=prior_7_start)
    steps_7 = _period_daily_total(records, "steps", prior_7_start, anchor, known_by=anchor)
    steps_prior28 = _period_daily_total(records, "steps", prior_35_start, prior_7_start, known_by=prior_7_start)
    heart_rate_7 = _period_mean(records, "heart_rate", prior_7_start, anchor, known_by=anchor)
    heart_rate_prior28 = _period_mean(records, "heart_rate", prior_35_start, prior_7_start, known_by=prior_7_start)
    body_temp_7 = _period_mean(records, "body_temperature", prior_7_start, anchor, known_by=anchor)
    body_temp_prior28 = _period_mean(records, "body_temperature", prior_35_start, prior_7_start, known_by=prior_7_start)
    wrist_temp_7 = _period_mean(records, "sleeping_wrist_temperature", prior_7_start, anchor, known_by=anchor)
    wrist_temp_prior28 = _period_mean(records, "sleeping_wrist_temperature", prior_35_start, prior_7_start, known_by=prior_7_start)
    mood_history = _mood_lagged_summaries(records, anchor, zone)
    cycle_site_history = _cycle_site_history(records, anchor, zone)
    symptom_history = _menstrual_symptom_summaries(records, anchor, zone)
    return {
        "glucose_1h_mean": g1["mean"], "glucose_1h_slope": g1["slope"],
        "glucose_3h_mean": g3["mean"], "glucose_3h_slope": g3["slope"], "glucose_3h_variability": g3["variability"],
        "glucose_3h_low_rate": g3["low_rate"], "glucose_3h_high_rate": g3["high_rate"],
        "glucose_6h_mean": g6["mean"], "glucose_24h_mean": g24["mean"], "glucose_7d_vs_prior28_mean": glucose_long_delta,
        "glucose_valid": g3["valid"],
        "delivered_bolus_3h": bolus3, "delivered_bolus_6h": bolus6, "delivered_bolus_24h": bolus24,
        "delivered_bolus_7d_daily": bolus_7, "delivered_bolus_prior28_daily": bolus_prior28,
        "meal_carbs_3h": carbs3, "meal_carbs_6h": carbs6, "meal_carbs_24h": carbs24,
        "meal_carbs_7d_daily": carbs_7, "meal_carbs_prior28_daily": carbs_prior28,
        "exercise_minutes_6h": activity, "exercise_minutes_24h": activity24,
        "exercise_minutes_7d_daily": activity_7, "exercise_minutes_prior28_daily": activity_prior28,
        "steps_6h": steps, "steps_24h": steps24, "steps_7d_daily": steps_7, "steps_prior28_daily": steps_prior28,
        "steps_daytime_24h": _steps_by_local_part(records, anchor, zone, daytime=True),
        "steps_overnight_24h": _steps_by_local_part(records, anchor, zone, daytime=False),
        "heart_rate_6h_mean": heart_rate, "heart_rate_24h_mean": heart_rate24,
        "heart_rate_7d_mean": heart_rate_7, "heart_rate_prior28_mean": heart_rate_prior28,
        "heart_rate_deviation_24h": granular["heart_rate_deviation_24h"],
        "hrv_sdnn_24h_mean": hrv24, "hrv_sdnn_deviation_24h": granular["hrv_sdnn_deviation_24h"],
        "workout_minutes_6h": workout6, "workout_minutes_24h": workout24,
        "workout_average_hr_24h": workout_hr24, "workout_count_24h": granular["workout_count_24h"],
        "evening_workout": granular["evening_workout"],
        "sleep_hours": sleep, "body_temperature_delta": body_delta, "wrist_temperature_delta": wrist_delta,
        **_sleep_lagged_summaries(records, anchor, zone),
        "sleep_rem_fraction": granular["sleep_rem_fraction"],
        "sleep_deep_fraction": granular["sleep_deep_fraction"],
        "sleep_fragmentation_per_hour": granular["sleep_fragmentation_per_hour"],
        "sleep_stage_efficiency": granular["sleep_stage_efficiency"],
        "mood_valence_24h": mood_history["mood_valence_24h_mean"], "mood_arousal_24h": mood_history["mood_arousal_24h_mean"],
        **mood_history, **cycle_site_history, **symptom_history,
        "body_temperature_7d_vs_prior28": (body_temp_7 - body_temp_prior28) if body_temp_7 is not None and body_temp_prior28 is not None else None,
        "wrist_temperature_7d_vs_prior28": (wrist_temp_7 - wrist_temp_prior28) if wrist_temp_7 is not None and wrist_temp_prior28 is not None else None,
        "sleep_7d_mean": sleep_7, "sleep_prior28_mean": sleep_prior28,
        "days_since_period": cycle, "site_age_days": site,
        "therapy_profile_fingerprint": therapy_profile,
        "bolus_valid": bolus_valid, "bolus3_valid": bolus3_valid, "bolus24_valid": bolus24_valid,
        "carbs_valid": carbs_valid, "activity_valid": activity_valid or activity24_valid,
        "families": {
            "glucose": bool(g3["valid"]),
            "insulin": bolus_valid or bolus3_valid or bolus24_valid, "carbohydrate": carbs_valid,
            "activity": activity_valid or activity24_valid or steps_valid or steps24_valid,
            "physiology": heart_rate_valid or hrv_valid,
            "mood": mood_history["mood_valence_24h_mean"] is not None or mood_history["mood_arousal_24h_mean"] is not None,
            "sleep": sleep is not None or granular["sleep_rem_fraction"] is not None,
            "temperature": body_delta is not None or wrist_delta is not None,
            "cycle": cycle is not None, "site": site is not None,
        },
        "anchor": anchor, "anchorHour": anchor.astimezone(zone).hour,
    }


def _outcome_summary(rows: Sequence[Mapping[str, Any]], key: str, *, label: str, horizon: str, unit: str) -> dict[str, Any] | None:
    values = [float(row[key]) for row in rows if _finite(row.get(key)) is not None]
    if not values:
        return None
    return {
        "label": label, "horizon": horizon, "unit": unit, "median": float(_median(values)),
        "lower": float(_quantile(values, 0.25)), "upper": float(_quantile(values, 0.75)),
        "rangeKind": "observed_interquartile", "count": len(values),
    }


def _make_evidence(
    *, title: str, source_label: str, window_label: str, summary: str, limitations: list[str],
    facts: list[dict[str, Any]], analysis_kind: str, evidence_level: str, method_label: str,
    availability_mode: str, support: dict[str, Any], similarity: dict[str, Any] | None = None,
    outcome_summary: list[dict[str, Any]] | None = None, episodes: list[dict[str, Any]] | None = None,
    comparison: dict[str, Any] | None = None, context_predicates: list[str] | None = None,
    matched_on: list[str] | None = None, context_rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "title": title, "sourceLabel": source_label, "windowLabel": window_label, "summary": summary,
        "limitations": limitations, "facts": facts, "episodes": episodes or [], "contextPredicates": context_predicates or [],
        "matchedOn": matched_on or [], "analysisKind": analysis_kind, "evidenceLevel": evidence_level,
        "methodLabel": method_label, "availabilityMode": availability_mode, "support": support,
    }
    if similarity is not None:
        evidence["similarity"] = similarity
    if outcome_summary:
        evidence["outcomeSummary"] = outcome_summary
    if comparison is not None:
        evidence["comparison"] = comparison
    if context_rules is not None:
        evidence["contextRules"] = [dict(rule) for rule in context_rules]
    return evidence


def _condition_descriptions(item: Mapping[str, Any], *, event: bool) -> list[str]:
    rules = item.get("ruleDescriptors")
    if rules:
        return [describe_rule(rule) for rule in rules]
    labels = _EVENT_LABELS if event else _PARTICIPANT_CONTEXT_LABELS
    return [labels.get(name, str(name).replace("_", " ")) for name in item.get("predicates", ())]


def _response_trace_points(row: Mapping[str, Any]) -> list[dict[str, float]]:
    """Keep distinct, finite observed five-minute bins in the first three hours."""
    trace = row.get("trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        return []
    bins: dict[int, list[float]] = defaultdict(list)
    for point in trace:
        if not isinstance(point, Mapping):
            continue
        minute = _finite(point.get("minute"))
        if minute is None or minute < 0.0 or minute >= 180.0:
            continue
        bin_index = int(round(minute / 5.0))
        if bin_index < 0 or bin_index >= 36 or abs(minute - bin_index * 5.0) > 0.011:
            continue
        glucose = _finite(point.get("glucose"))
        if glucose is None:
            continue
        bins[bin_index].append(glucose)
    return [
        {"minute": float(bin_index * 5), "glucose": float(_median(values))}
        for bin_index, values in sorted(bins.items())
    ]


def _response_outcome_group(rows: Sequence[Mapping[str, Any] | None], *, min_coverage: float) -> dict[str, Any]:
    total = len(rows)
    required_coverage = _finite(min_coverage)
    eligible_points: list[list[dict[str, float]]] = []
    low_minutes: list[float] = []
    high_minutes: list[float] = []
    low_episodes = 0
    high_episodes = 0
    fully_in_range = 0
    anchor_hours: list[float] = []
    all_hours_known = total > 0
    profile_known = 0
    eligible_profiles: list[str | None] = []

    for row in rows:
        context = row.get("context") if isinstance(row, Mapping) else None
        context = context if isinstance(context, Mapping) else {}
        hour = _finite(context.get("hour"))
        if hour is None or hour < 0.0 or hour >= 24.0:
            all_hours_known = False
        else:
            anchor_hours.append(hour)

        fingerprint = context.get("therapy_profile_fingerprint")
        fingerprint = fingerprint.strip() if isinstance(fingerprint, str) and fingerprint.strip() else None
        if fingerprint is not None:
            profile_known += 1

        points = _response_trace_points(row) if isinstance(row, Mapping) else []
        coverage = len(points) / 36.0
        eligible = bool(points) and required_coverage is not None and coverage >= required_coverage
        if not eligible:
            continue
        eligible_points.append(points)
        eligible_profiles.append(fingerprint)
        low_values = [point["glucose"] for point in points if point["glucose"] < 70.0]
        high_values = [point["glucose"] for point in points if point["glucose"] > 180.0]
        low_minutes.append(float(len(low_values) * 5))
        high_minutes.append(float(len(high_values) * 5))
        if _observed_runs(points, start=datetime(1970, 1, 1, tzinfo=UTC), kind="low"):
            low_episodes += 1
        if _observed_runs(points, start=datetime(1970, 1, 1, tzinfo=UTC), kind="high"):
            high_episodes += 1
        if len(points) == 36 and all(70.0 <= point["glucose"] <= 180.0 for point in points):
            fully_in_range += 1

    eligible_count = len(eligible_points)
    same_profile = (
        eligible_count > 0
        and all(profile is not None for profile in eligible_profiles)
        and len(set(eligible_profiles)) == 1
    )
    return {
        "totalEpisodes": total,
        "eligibleEpisodes": eligible_count,
        "complete": total > 0 and eligible_count == total,
        "lowEpisodes": low_episodes,
        "highEpisodes": high_episodes,
        "fullyInRangeEpisodes": fully_in_range,
        "lowMinutesMedian": float(_median(low_minutes)) if low_minutes else None,
        "highMinutesMedian": float(_median(high_minutes)) if high_minutes else None,
        "localAnchorHourMin": float(min(anchor_hours)) if all_hours_known else None,
        "localAnchorHourMax": float(max(anchor_hours)) if all_hours_known else None,
        "profileKnownEpisodes": profile_known,
        "singleObservedProfile": bool(same_profile),
    }


def _response_outcome_context(
    item: Mapping[str, Any], *, target: str, config: BriefConfig, episode_kind: str,
) -> dict[str, Any] | None:
    expected_kind = {"firstResponse": "meal", "bolusResponse0_3h": "delivered_bolus"}.get(target)
    if expected_kind is None or episode_kind != expected_kind:
        return None
    pairs = item.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        pairs = []
    groups: dict[str, list[Mapping[str, Any] | None]] = {"context": [], "comparison": []}
    for pair in pairs:
        pair = pair if isinstance(pair, Mapping) else {}
        groups["context"].append(pair.get("left") if isinstance(pair.get("left"), Mapping) else None)
        groups["comparison"].append(pair.get("right") if isinstance(pair.get("right"), Mapping) else None)
    return {
        "version": 1,
        "target": target,
        "windowMinutes": 180,
        "lowThresholdMgDl": 70,
        "highThresholdMgDl": 180,
        "minimumRunMinutes": 15,
        "groups": {
            group: _response_outcome_group(rows, min_coverage=config.min_coverage)
            for group, rows in groups.items()
        },
    }


def _event_response_outcome(
    records: Sequence[BriefRecord], anchor: datetime, start: datetime, end: datetime, *, cutoff: datetime, min_coverage: float,
    baseline: float | None, index: _RecordIndex | None = None,
) -> tuple[float | None, dict[str, Any] | None]:
    outcome = _glucose_outcome(records, start, end, cutoff=cutoff, min_coverage=min_coverage, index=index)
    if outcome is None or baseline is None:
        return None, outcome
    return float(outcome["mean"] - baseline), outcome


def _sleep_lagged_summaries(
    records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo,
) -> dict[str, float | int | None]:
    """Build as-of-anchor sleep summaries from completed, available nights."""
    recent_start = _add_elapsed(anchor, hours=-7 * 24)
    baseline_start = _add_elapsed(anchor, hours=-35 * 24)
    by_window: dict[str, dict[date, list[float]]] = {"recent": defaultdict(list), "baseline": defaultdict(list)}
    for record in records:
        if record.metric != "sleep_hours" or record.end is None or not _known_before(record, anchor):
            continue
        end = record.end_or_start
        local_day = end.astimezone(zone).date()
        if recent_start <= end < anchor:
            by_window["recent"][local_day].append(float(record.value))
        elif baseline_start <= end < recent_start:
            by_window["baseline"][local_day].append(float(record.value))
    recent = [_median(values) for _, values in sorted(by_window["recent"].items())]
    baseline = [_median(values) for _, values in sorted(by_window["baseline"].items())]
    recent = [float(value) for value in recent if value is not None]
    baseline = [float(value) for value in baseline if value is not None]
    earlier_mean = _mean(baseline) if len(baseline) >= 12 else None
    variability = float(np.std(recent, ddof=0)) if len(recent) >= 3 else None
    shortfall = (
        float(sum(max(0.0, float(earlier_mean) - value) for value in recent))
        if len(recent) >= 3 and earlier_mean is not None else None
    )
    return {
        "sleep_variability_7d": variability,
        "sleep_shortfall_7d_hours": shortfall,
        "sleep_baseline_28d_hours": earlier_mean,
        "sleep_observed_nights_7d": len(recent),
        "sleep_baseline_nights_28d": len(baseline),
    }


def _mood_lagged_summaries(records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo) -> dict[str, Any]:
    """Observed mood summaries; last values always carry their measured age."""
    recent_start = _add_elapsed(anchor, hours=-7 * 24)
    baseline_start = _add_elapsed(anchor, hours=-35 * 24)
    result: dict[str, Any] = {}
    for domain in ("valence", "arousal"):
        metric = "mood_" + domain
        observed = [record for record in records if record.metric == metric
                    and _known_before(record, anchor) and record.end_or_start <= anchor]
        recent24 = [float(record.value) for record in observed if _add_elapsed(anchor, hours=-24) <= record.end_or_start < anchor]
        recent_days: dict[date, list[float]] = defaultdict(list)
        baseline_days: dict[date, list[float]] = defaultdict(list)
        for record in observed:
            day = _local_date(record.end_or_start, zone)
            if recent_start <= record.end_or_start < anchor:
                recent_days[day].append(float(record.value))
            elif baseline_start <= record.end_or_start < recent_start:
                baseline_days[day].append(float(record.value))
        recent_daily = [float(_median(values)) for _, values in sorted(recent_days.items())]
        baseline_daily = [float(_median(values)) for _, values in sorted(baseline_days.items())]
        last = max(observed, key=lambda record: record.end_or_start, default=None)
        recent_mean = _mean(recent_daily) if recent_daily else None
        prior_mean = _mean(baseline_daily) if baseline_daily else None
        result.update({
            f"mood_{domain}_last": float(last.value) if last else None,
            f"mood_{domain}_last_age_hours": _elapsed_hours(last.end_or_start, anchor) if last else None,
            f"mood_{domain}_24h_mean": _mean(recent24),
            f"mood_{domain}_24h_std": float(statistics.pstdev(recent24)) if len(recent24) >= 2 else None,
            f"mood_{domain}_7d_mean": recent_mean,
            f"mood_{domain}_7d_std": float(statistics.pstdev(recent_daily)) if len(recent_daily) >= 2 else None,
            f"mood_{domain}_7d_vs_prior28": (recent_mean - prior_mean) if recent_mean is not None and prior_mean is not None else None,
            f"mood_{domain}_observed_days_7d": len(recent_days),
            f"mood_{domain}_observed_days_prior28": len(baseline_days),
        })
    return result


def _cycle_site_history(records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo) -> dict[str, Any]:
    starts_by_day: dict[date, datetime] = {}
    for record in records:
        if record.metric == "period_onset" and _known_before(record, anchor) and record.start <= anchor:
            day = _local_date(record.start, zone)
            starts_by_day[day] = min(starts_by_day.get(day, record.start), record.start)
    cycle_starts = sorted(starts_by_day.values())
    cycle_intervals = [_elapsed_hours(left, right) / 24.0 for left, right in zip(cycle_starts, cycle_starts[1:])]
    elapsed = _elapsed_hours(cycle_starts[-1], anchor) / 24.0 if cycle_starts else None
    cycle_median = statistics.median(cycle_intervals[-12:]) if len(cycle_intervals) >= 3 else None

    site_events: dict[datetime, datetime] = {}
    location: str | None = None
    allowed_locations = {"Left Arm", "Right Arm", "Abdomen", "Butt", "Thigh"}
    known_site_records = [record for record in records if record.metric in {"site_change", "site_change_event"} and _known_before(record, anchor) and record.start <= anchor]
    for record in known_site_records:
        # Exact timestamp dedupes mirrored daily/event records while retaining
        # genuinely distinct replacements even if they occurred on one day.
        site_events[record.start.astimezone(UTC)] = record.start
    latest_site = max(known_site_records, key=lambda record: record.start, default=None)
    raw_location = _attr(latest_site, "location") if latest_site else None
    if isinstance(raw_location, str) and raw_location in allowed_locations:
        location = raw_location
    changes = sorted(site_events.values())
    dwell = [_elapsed_hours(left, right) / 24.0 for left, right in zip(changes, changes[1:])]
    site_age = _elapsed_hours(changes[-1], anchor) / 24.0 if changes else None
    site_median = statistics.median(dwell[-12:]) if len(dwell) >= 3 else None
    return {
        "cycle_days": elapsed,
        "cycle_last_interval_days": cycle_intervals[-1] if cycle_intervals else None,
        "cycle_median_interval_days": cycle_median,
        "cycle_interval_std_days": float(statistics.pstdev(cycle_intervals[-12:])) if len(cycle_intervals) >= 3 else None,
        "cycle_elapsed_vs_prior_median_days": elapsed - cycle_median if elapsed is not None and cycle_median is not None else None,
        "cycle_onset_count": len(cycle_starts), "cycle_completed_interval_count": len(cycle_intervals),
        "site_age_days": site_age,
        "site_previous_dwell_days": dwell[-1] if dwell else None,
        "site_median_dwell_days": site_median,
        "site_dwell_std_days": float(statistics.pstdev(dwell[-12:])) if len(dwell) >= 3 else None,
        "site_age_vs_prior_median_days": site_age - site_median if site_age is not None and site_median is not None else None,
        "site_location": location, "site_change_count": len(changes), "site_completed_dwell_count": len(dwell),
    }


def _menstrual_symptom_summaries(records: Sequence[BriefRecord], anchor: datetime, zone: ZoneInfo) -> dict[str, Any]:
    week_start = _add_elapsed(anchor, hours=-7 * 24)
    day_start = _add_elapsed(anchor, hours=-24)
    flow = [record for record in records if record.metric == "menstrual_flow_severity" and _known_before(record, anchor) and week_start <= record.end_or_start < anchor]
    cramps = [record for record in records if record.metric == "menstrual_cramp_severity" and _known_before(record, anchor) and week_start <= record.end_or_start < anchor]
    flow_last = max(flow, key=lambda record: record.end_or_start, default=None)
    cramp_last = max(cramps, key=lambda record: record.end_or_start, default=None)
    cramp24 = [float(record.value) for record in cramps if day_start <= record.end_or_start < anchor]
    return {
        "flow_severity_7d_mean": _mean(float(record.value) for record in flow),
        "flow_severity_last_7d": float(flow_last.value) if flow_last else None,
        "flow_last_age_hours": _elapsed_hours(flow_last.end_or_start, anchor) if flow_last else None,
        "cramp_severity_24h_mean": _mean(cramp24),
        "cramp_severity_last_7d": float(cramp_last.value) if cramp_last else None,
        "cramp_last_age_hours": _elapsed_hours(cramp_last.end_or_start, anchor) if cramp_last else None,
        "flow_observation_days_7d": len({record.start.date() for record in flow}),
        "cramp_observations_7d": len(cramps),
    }


def _daily_rule_context(
    features: Mapping[str, Any], *, anchor: datetime, zone: ZoneInfo,
    sleep_summaries: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared lag-feature builder for historical daily rows and current state."""
    context = {key: features.get(key) for key in NUMERIC_FEATURES if key in features}
    context.update({
        "start_glucose": features.get("glucose_1h_mean"),
        "pre_glucose_slope": features.get("glucose_1h_slope"),
        "carbs": features.get("meal_carbs_6h"),
        "bolus": features.get("delivered_bolus_6h"),
        "hour": anchor.astimezone(zone).hour + anchor.astimezone(zone).minute / 60.0,
        "site_age_days": features.get("site_age_days"),
        "cycle_days": features.get("days_since_period"),
        "activity_7d_daily": features.get("exercise_minutes_7d_daily"),
        "active_6h": features.get("exercise_minutes_6h"),
        "body_temperature_context": features.get("body_temperature_7d_vs_prior28"),
        "wrist_temperature_context": features.get("wrist_temperature_7d_vs_prior28"),
    })
    for key in (
        "glucose_valid", "carbs_valid", "bolus_valid", "activity_valid", "therapy_profile_fingerprint", "families",
        "sleep_baseline_28d_hours", "sleep_observed_nights_7d", "sleep_baseline_nights_28d",
        "cycle_onset_count", "cycle_completed_interval_count", "site_change_count", "site_completed_dwell_count",
        "flow_observation_days_7d", "cramp_observations_7d",
        "mood_valence_observed_days_7d", "mood_valence_observed_days_prior28",
        "mood_arousal_observed_days_7d", "mood_arousal_observed_days_prior28",
    ):
        if key in features:
            context[key] = features[key]
    if sleep_summaries is None:
        sleep_summaries = {
            key: features.get(key)
            for key in ("sleep_variability_7d", "sleep_shortfall_7d_hours", "sleep_baseline_28d_hours", "sleep_observed_nights_7d", "sleep_baseline_nights_28d")
        }
    context.update(sleep_summaries)
    return context


def _build_event_episodes(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, config: BriefConfig, index: _RecordIndex | None = None) -> list[dict[str, Any]]:
    meals = [record for record in records if record.metric == "meal_carbs" and record.value > 0 and record.known_at <= cutoff]
    boluses = [record for record in records if record.metric == "delivered_bolus" and record.value > 0 and _is_actual_delivery(record) and record.known_at <= cutoff]
    output: list[dict[str, Any]] = []
    for meal in sorted(meals, key=lambda item: item.start):
        nearby = [bolus for bolus in boluses if abs((bolus.start - meal.start).total_seconds()) <= 15 * 60]
        bolus = min(nearby, key=lambda item: abs((item.start - meal.start).total_seconds()), default=None)
        anchor = meal.start
        pre = _glucose_outcome(records, anchor - timedelta(minutes=30), anchor, cutoff=min(cutoff, anchor), min_coverage=0.0, index=index)
        if pre is None or pre.get("mean") is None:
            continue
        baseline = float(pre["mean"])
        pre_trend = _glucose_window(records, anchor, 1.0, config.min_coverage, index).get("slope")
        if pre_trend is None:
            continue
        h3 = _add_elapsed(anchor, hours=3); h6 = _add_elapsed(anchor, hours=6)
        first, first_outcome = _event_response_outcome(records, anchor, anchor, h3, cutoff=cutoff, min_coverage=config.min_coverage, baseline=baseline, index=index)
        late, late_outcome = _event_response_outcome(records, anchor, h3, h6, cutoff=cutoff, min_coverage=config.min_coverage, baseline=baseline, index=index)
        h24 = _add_elapsed(anchor, hours=24)
        variance24_outcome = _glucose_outcome(records, anchor, h24, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        if first is None and late is None:
            continue
        exercise_values = [record.value for record in records if record.metric == "exercise_minutes" and anchor - timedelta(hours=6) <= record.start < anchor and _known_before(record, anchor)]
        exercise_valid = bool(exercise_values)
        sleep = _current_sleep(records, anchor)
        local = anchor.astimezone(zone)
        granular = _granular_context_at(records, anchor, zone)
        if sleep is None:
            sleep = granular["sleep_hours"]
        anchor_features = _anchor_features(
            records, anchor, cutoff=anchor, zone=zone, min_coverage=config.min_coverage,
            index=index, availability_mode="strict_as_of_anchor",
        )
        context = _daily_rule_context(
            anchor_features, anchor=anchor, zone=zone,
        )
        site = _latest_value(records, "site_age_days", anchor)
        if site is None:
            site = _site_age_at(records, anchor)
        cycle = _days_since_onset(records, anchor)
        context.update({
            "active_6h": sum(float(value) for value in exercise_values) if exercise_values else None, "active_valid": exercise_valid,
            "sleep_hours": sleep, "cycle_days": cycle, "site_age_days": site,
            "start_glucose": baseline, "pre_glucose_slope": float(pre_trend),
            "carbs": float(meal.value), "carbs_valid": True, "bolus": float(bolus.value) if bolus else None,
            "bolus_valid": bolus is not None, "hour": local.hour + local.minute / 60.0,
            "body_temperature_delta": _temp_delta_at(records, "body_temperature", anchor),
            "wrist_temperature_delta": _temp_delta_at(records, "sleeping_wrist_temperature", anchor),
            "steps_daytime_24h": _steps_by_local_part(records, anchor, zone, daytime=True),
            "steps_overnight_24h": _steps_by_local_part(records, anchor, zone, daytime=False),
            "therapy_profile_fingerprint": _therapy_profile_at(records, anchor),
            **granular,
        })
        end = h24
        trace, coverage = _episode_trace(records, anchor, end, cutoff, anchor=anchor, index=index)
        output.append({
            "id": _safe_id("meal", meal.start, meal.value), "date": _local_date(anchor, zone), "anchor": anchor,
            "episodeKind": "meal", "anchorType": "meal_event",
            "outcomeEnds": {"firstResponse": h3, "lateResponse": h6, "mealVariance3h": h3, "mealVariance24h": h24},
            "context": context, "firstResponse": first, "lateResponse": late,
            "mealVariance3h": first_outcome.get("variance") if first_outcome else None,
            "mealVariance24h": variance24_outcome.get("variance") if variance24_outcome else None,
            "firstOutcome": first_outcome, "lateOutcome": late_outcome, "trace": trace, "coverage": coverage,
            "events": _episode_events(records, anchor - timedelta(minutes=30), end, anchor, cutoff=cutoff),
            "sourceMeal": meal, "sourceBolus": bolus,
        })
    return output


def _episode_context_at(records: Sequence[BriefRecord], anchor: datetime, *, cutoff: datetime, zone: ZoneInfo,
                        config: BriefConfig, index: _RecordIndex, carbs: float | None = None,
                        bolus: float | None = None, carbs_known: bool = False) -> tuple[dict[str, Any], float] | None:
    prior = _glucose_outcome(records, _add_elapsed(anchor, hours=-0.5), anchor, cutoff=anchor,
                             min_coverage=0.0, index=index)
    trend = _glucose_window(records, anchor, 1.0, config.min_coverage, index).get("slope")
    if prior is None or prior.get("mean") is None or trend is None:
        return None
    features = _anchor_features(records, anchor, cutoff=anchor, zone=zone,
                                min_coverage=config.min_coverage, index=index,
                                availability_mode="strict_as_of_anchor")
    context = _daily_rule_context(features, anchor=anchor, zone=zone)
    context.update({
        "start_glucose": float(prior["mean"]), "pre_glucose_slope": float(trend),
        "carbs": carbs, "carbs_valid": bool(carbs_known),
        "bolus": bolus, "bolus_valid": bolus is not None,
        "therapy_profile_fingerprint": _therapy_profile_at(records, anchor),
    })
    return context, float(prior["mean"])


def _build_pre_meal_episodes(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo,
                             config: BriefConfig, index: _RecordIndex) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for meal in sorted((record for record in records if record.metric == "meal_carbs" and record.value > 0 and record.known_at <= cutoff), key=lambda row: row.start):
        anchor = _add_elapsed(meal.start, hours=-3)
        built = _episode_context_at(records, anchor, cutoff=cutoff, zone=zone, config=config, index=index)
        if built is None:
            continue
        context, baseline = built
        outcome = _glucose_outcome(records, anchor, meal.start, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        if outcome is None or outcome.get("mean") is None:
            continue
        value = float(outcome["mean"]) - baseline
        trace, coverage = _episode_trace(records, anchor, meal.start, cutoff, anchor=anchor, index=index)
        output.append({
            "id": _safe_id("pre-meal", meal.id, anchor), "episodeKind": "pre_meal", "anchorType": "retrospective_pre_meal",
            "date": _local_date(anchor, zone), "anchor": anchor,
            "outcomeEnds": {"preMealResponse": meal.start}, "preMealResponse": value,
            "context": context, "trace": trace, "coverage": coverage,
            "events": _episode_events(records, anchor, meal.start + timedelta(minutes=1), anchor, cutoff=cutoff),
            "endpointMealRecorded": True,
        })
    return output


def _explicit_correction_class(record: BriefRecord) -> str:
    value = str(_attr(record, "correction_classification") or "").strip().lower()
    if value in {"correction_only", "mixed_meal_correction"}:
        return value
    if _attr(record, "correction_only") is True:
        return "correction_only"
    return "unknown"


def _build_bolus_episodes(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo,
                          config: BriefConfig, index: _RecordIndex) -> tuple[list[dict[str, Any]], dict[str, int]]:
    meals = [record for record in records if record.metric == "meal_carbs" and record.value > 0]
    boluses = [record for record in records if record.metric == "delivered_bolus" and record.value > 0 and _is_actual_delivery(record) and record.known_at <= cutoff]
    output: list[dict[str, Any]] = []
    counts = {"actualDelivered": len(boluses), "correctionOnly": 0, "mixedMealCorrection": 0, "intentUnknown": 0}
    for bolus in sorted(boluses, key=lambda row: row.end_or_start):
        anchor = bolus.end_or_start
        meal = min((candidate for candidate in meals if candidate.known_at <= anchor and abs(_elapsed_hours(candidate.start, anchor)) <= 0.25),
                   key=lambda candidate: abs(_elapsed_hours(candidate.start, anchor)), default=None)
        classification = _explicit_correction_class(bolus)
        if classification == "correction_only": counts["correctionOnly"] += 1
        elif classification == "mixed_meal_correction": counts["mixedMealCorrection"] += 1
        else: counts["intentUnknown"] += 1
        built = _episode_context_at(records, anchor, cutoff=cutoff, zone=zone, config=config, index=index,
                                    carbs=float(meal.value) if meal else None, bolus=float(bolus.value), carbs_known=meal is not None)
        if built is None:
            continue
        context, baseline = built
        h3 = _add_elapsed(anchor, hours=3); h6 = _add_elapsed(anchor, hours=6)
        first, first_outcome = _event_response_outcome(records, anchor, anchor, h3, cutoff=cutoff,
                            min_coverage=config.min_coverage, baseline=baseline, index=index)
        late, _ = _event_response_outcome(records, anchor, h3, h6, cutoff=cutoff,
                            min_coverage=config.min_coverage, baseline=baseline, index=index)
        if first is None and late is None:
            continue
        trace, coverage = _episode_trace(records, anchor, h6, cutoff, anchor=anchor, index=index)
        events = _episode_events(records, anchor, h6, anchor, cutoff=cutoff)
        row = {
            "id": _safe_id("bolus", bolus.id, anchor), "episodeKind": "delivered_bolus", "anchorType": "recorded_bolus_response",
            "date": _local_date(anchor, zone), "anchor": anchor, "episodeKnownAt": bolus.known_at,
            "outcomeEnds": {"bolusResponse0_3h": h3, "bolusResponse3_6h": h6},
            "bolusResponse0_3h": first, "bolusResponse3_6h": late,
            "context": context, "trace": trace, "coverage": coverage, "events": events,
            "sourceBolus": bolus, "mealAssociation": "recorded_carbs_within_15_minutes" if meal else "unknown",
            "correctionClassification": classification,
            "coexposures": [event for event in events if event["kind"] in {"meal_carbs", "delivered_bolus"}],
            "bolusOutcomeCoverage": first_outcome.get("coverage") if first_outcome else None,
        }
        output.append(row)
        if classification == "correction_only":
            row["correctionOnlyResponse0_3h"] = first
            row["correctionOnlyResponse3_6h"] = late
            row["outcomeEnds"].update({"correctionOnlyResponse0_3h": h3, "correctionOnlyResponse3_6h": h6})
    return output, counts


def _build_fasting_episodes(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo,
                            config: BriefConfig, index: _RecordIndex) -> tuple[list[dict[str, Any]], dict[str, int]]:
    intervals = [record for record in records if record.metric == "fasting_interval" and record.end is not None
                 and _known_before(record, cutoff) and str(_attr(record, "fasting_status") or "").lower() == "fasting"
                 and isinstance(_attr(record, "fasting_provenance"), str) and str(_attr(record, "fasting_provenance")).strip()]
    output = []
    for interval in intervals:
        anchor = interval.start
        if _elapsed_hours(anchor, interval.end_or_start) < 3.0:
            continue
        built = _episode_context_at(records, anchor, cutoff=cutoff, zone=zone, config=config, index=index)
        if built is None:
            continue
        context, baseline = built
        end = _add_elapsed(anchor, hours=3)
        if end.astimezone(UTC) > interval.end_or_start.astimezone(UTC):
            continue
        outcome = _glucose_outcome(records, anchor, end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        if outcome is None or outcome.get("mean") is None:
            continue
        trace, coverage = _episode_trace(records, anchor, end, cutoff, anchor=anchor, index=index)
        output.append({
            "id": _safe_id("fasting", interval.id, anchor), "episodeKind": "explicit_fasting", "anchorType": "retrospective_explicit_fasting_interval",
            "date": _local_date(anchor, zone), "anchor": anchor, "outcomeEnds": {"fastingResponse0_3h": end},
            "fastingResponse0_3h": float(outcome["mean"]) - baseline, "context": context,
            "trace": trace, "coverage": coverage, "events": [],
            "fastingDurationHours": _elapsed_hours(anchor, interval.end_or_start),
            "fastingProvenancePresent": True,
        })
    return output, {"explicitIntervals": len(intervals), "usableEpisodes": len(output)}


def _build_clock_bin_episodes(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo,
                              config: BriefConfig, index: _RecordIndex) -> tuple[list[dict[str, Any]], dict[str, int]]:
    current_day = _local_date(cutoff, zone)
    days = _day_candidates(records, zone, current_day)
    output: list[dict[str, Any]] = []
    diagnostics = {"candidateBins": 0, "dstLengthSkipped": 0, "usableBins": 0}
    for day in days:
        for hour in range(0, 24, 3):
            local_start = datetime.combine(day, time(hour, 0), tzinfo=zone)
            local_end = datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=zone) if hour == 21 else datetime.combine(day, time(hour + 3, 0), tzinfo=zone)
            start = local_start.astimezone(UTC); end = local_end.astimezone(UTC)
            diagnostics["candidateBins"] += 1
            # Skip bins whose physical duration changes at DST boundaries;
            # they do not represent three elapsed hours.
            if abs(_elapsed_hours(start, end) - 3.0) > 1e-9:
                diagnostics["dstLengthSkipped"] += 1
                continue
            anchor = local_start
            built = _episode_context_at(records, anchor, cutoff=cutoff, zone=zone, config=config, index=index)
            if built is None:
                continue
            context, baseline = built
            outcome = _glucose_outcome(records, start, end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
            if outcome is None or outcome.get("mean") is None:
                continue
            target_mean = f"clockBinMeanChange{hour:02d}"
            target_var = f"clockBinVariance{hour:02d}"
            trace, coverage = _episode_trace(records, start, end, cutoff, anchor=anchor, index=index)
            output.append({
                "id": _safe_id("clock-bin", day, hour), "episodeKind": f"clock_bin_{hour:02d}", "anchorType": "local_clock_bin",
                "date": day, "anchor": anchor, "clockHour": hour,
                "outcomeEnds": {target_mean: end, target_var: end},
                target_mean: float(outcome["mean"]) - baseline, target_var: float(outcome["variance"]),
                "context": context, "trace": trace, "coverage": coverage,
                "events": _episode_events(records, start, end, anchor, cutoff=cutoff),
                "actualElapsedHours": _elapsed_hours(start, end),
            })
            diagnostics["usableBins"] += 1
    return output, diagnostics


def _build_daily_context_rows(
    records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, config: BriefConfig,
) -> list[dict[str, Any]]:
    index = _RecordIndex.build(records)
    query_local = cutoff.astimezone(zone)
    rows: list[dict[str, Any]] = []
    for day in _day_candidates(records, zone, query_local.date()):
        anchor = _local_datetime(day, time(query_local.hour, query_local.minute), zone)
        if anchor >= cutoff:
            continue
        features = _anchor_features(
            records, anchor, cutoff=cutoff, zone=zone, min_coverage=config.min_coverage,
            index=index, availability_mode="strict_as_of_anchor",
        )
        start_glucose = _finite(features.get("glucose_1h_mean"))
        if start_glucose is None or not features.get("glucose_valid"):
            continue
        outcomes = _analogue_outcomes(records, anchor, cutoff=cutoff, zone=zone, config=config, index=index)
        response: dict[str, float | None] = {"overnightResponse": None, "dayResponse": None, "dayVarianceResponse": None}
        if "overnight" in outcomes:
            response["overnightResponse"] = float(outcomes["overnight"]["mean"]) - start_glucose
        if "24h" in outcomes:
            response["dayResponse"] = float(outcomes["24h"]["mean"]) - start_glucose
            response["dayVarianceResponse"] = float(outcomes["24h"]["variance"])
        if all(value is None for value in response.values()):
            continue
        context = _daily_rule_context(
            features, anchor=anchor, zone=zone,
        )
        context.update({
            "carbs_valid": features.get("meal_carbs_6h") is not None,
            "bolus_valid": features.get("delivered_bolus_6h") is not None,
            "active_valid": features.get("exercise_minutes_6h") is not None,
            "evening_workout": features.get("evening_workout"),
            "therapy_profile_fingerprint": features.get("therapy_profile_fingerprint"),
        })
        end = _add_elapsed(anchor, hours=24)
        trace, coverage = _episode_trace(records, anchor, end, cutoff, anchor=anchor, index=index)
        rows.append({
            "id": _safe_id("daily", anchor), "anchorType": "daily_context", "date": day, "anchor": anchor,
            "outcomeEnds": {
                "overnightResponse": outcomes.get("overnight", {}).get("end"),
                "dayResponse": outcomes.get("24h", {}).get("end"),
                "dayVarianceResponse": outcomes.get("24h", {}).get("end"),
            },
            "context": context, **response, "outcomes": outcomes, "trace": trace, "coverage": coverage,
            "events": _episode_events(records, anchor, end, anchor, cutoff=cutoff),
        })
    return rows


def _ridge_vector(row: Mapping[str, Any], *, include_bolus: bool, include_carbs: bool,
                  adjust_for_bolus: bool = True) -> list[float]:
    context = row["context"]
    vector = [1.0, float(context["start_glucose"]), float(context["pre_glucose_slope"])]
    if include_carbs:
        vector.append(float(context["carbs"]))
    if include_bolus and adjust_for_bolus:
        vector.append(float(context["bolus"]))
    hour = float(context["hour"])
    return vector + [math.sin(hour / 24.0 * 2 * math.pi), math.cos(hour / 24.0 * 2 * math.pi)]


def _ridge_fit(rows: Sequence[Mapping[str, Any]], target: str, alpha: float,
               *, adjust_for_bolus: bool = True) -> dict[tuple[bool, bool], np.ndarray]:
    models: dict[tuple[bool, bool], np.ndarray] = {}
    for include_bolus in (False, True):
        for include_carbs in (False, True):
            usable = [
                row for row in rows if _finite(row.get(target)) is not None
                and bool(row["context"].get("bolus_valid")) == include_bolus
                and bool(row["context"].get("carbs_valid")) == include_carbs
                and _finite(row["context"].get("start_glucose")) is not None
                and _finite(row["context"].get("pre_glucose_slope")) is not None
                and (not include_carbs or _finite(row["context"].get("carbs")) is not None)
                and (not include_bolus or _finite(row["context"].get("bolus")) is not None)
            ]
            if len(usable) < 3:
                continue
            x = np.asarray([_ridge_vector(row, include_bolus=include_bolus, include_carbs=include_carbs,
                                          adjust_for_bolus=adjust_for_bolus) for row in usable], dtype=float)
            y = np.asarray([float(row[target]) for row in usable], dtype=float)
            penalty = np.eye(x.shape[1], dtype=float) * max(0.01, float(alpha)); penalty[0, 0] = 0.0
            try:
                models[(include_bolus, include_carbs)] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            except np.linalg.LinAlgError:
                models[(include_bolus, include_carbs)] = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return models


def _ridge_residuals(rows: Sequence[Mapping[str, Any]], target: str, alpha: float,
                     model: Mapping[tuple[bool, bool], np.ndarray] | None = None,
                     *, adjust_for_bolus: bool = True) -> dict[str, float]:
    fitted = model if model is not None else _ridge_fit(rows, target, alpha, adjust_for_bolus=adjust_for_bolus)
    if not fitted:
        return {}
    residuals: dict[str, float] = {}
    for row in rows:
        if _finite(row.get(target)) is None:
            continue
        include_bolus = bool(row["context"].get("bolus_valid"))
        include_carbs = bool(row["context"].get("carbs_valid"))
        coefficients = fitted.get((include_bolus, include_carbs))
        if coefficients is None:
            continue
        vector = np.asarray(_ridge_vector(row, include_bolus=include_bolus, include_carbs=include_carbs,
                                          adjust_for_bolus=adjust_for_bolus), dtype=float)
        residuals[str(row["id"])] = float(row[target]) - float(vector @ coefficients)
    return residuals


def _event_predicates(rows: Sequence[Mapping[str, Any]], *, discovery: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in discovery:
        context = row["context"]
        for key in (
            "start_glucose", "carbs", "bolus", "active_6h", "activity_7d_daily",
            "body_temperature_context", "wrist_temperature_context", "sleep_rem_fraction",
            "sleep_fragmentation_per_hour", "hrv_sdnn_deviation_24h",
            "heart_rate_deviation_24h", "mood_valence_24h", "workout_minutes_24h",
            "steps_daytime_24h", "steps_overnight_24h",
        ):
            if _finite(context.get(key)) is not None:
                values[key].append(float(context[key]))
    thresholds = {key: _quantile(series, 0.60) for key, series in values.items()}
    for key, field in (
        ("sleep_rem_fraction_low", "sleep_rem_fraction"),
        ("hrv_sdnn_deviation_low", "hrv_sdnn_deviation_24h"),
        ("mood_valence_low", "mood_valence_24h"),
    ):
        thresholds[key] = _quantile(values.get(field, ()), 0.40)
    for key, field in (
        ("sleep_fragmentation_high", "sleep_fragmentation_per_hour"),
        ("heart_rate_deviation_high", "heart_rate_deviation_24h"),
        ("workout_minutes_high", "workout_minutes_24h"),
        ("daytime_steps_high", "steps_daytime_24h"),
    ):
        thresholds[key] = _quantile(values.get(field, ()), 0.60)
    return thresholds


def _event_flag(row: Mapping[str, Any], name: str, thresholds: Mapping[str, float | None]) -> bool | None:
    context = row["context"]
    if name == "after_activity":
        if not context.get("active_valid") or context.get("active_6h") is None:
            return None
        return float(context["active_6h"]) >= float(thresholds.get("active_6h") or 30.0)
    if name == "short_sleep_before":
        if context.get("sleep_hours") is None:
            return None
        return float(context["sleep_hours"]) < 6.0
    if name == "recorded_cycle_early":
        if context.get("cycle_days") is None:
            return None
        return 0 <= float(context["cycle_days"]) <= 7
    if name == "site_age_late":
        if context.get("site_age_days") is None:
            return None
        return float(context["site_age_days"]) >= 3.0
    if name == "recent_activity_high":
        if context.get("activity_7d_daily") is None:
            return None
        return float(context["activity_7d_daily"]) >= float(thresholds.get("activity_7d_daily") or 30.0)
    if name == "body_temperature_high":
        if context.get("body_temperature_context") is None:
            return None
        delta = float(context["body_temperature_context"])
        return delta > 0.0 and delta >= float(thresholds.get("body_temperature_context") or 0.0)
    if name == "wrist_temperature_high":
        if context.get("wrist_temperature_context") is None:
            return None
        delta = float(context["wrist_temperature_context"])
        return delta > 0.0 and delta >= float(thresholds.get("wrist_temperature_context") or 0.0)
    if name == "starting_glucose_high":
        if context.get("start_glucose") is None:
            return None
        return float(context["start_glucose"]) >= float(thresholds.get("start_glucose") or 140.0)
    if name == "bolus_large":
        if context.get("bolus") is None:
            return None
        return float(context["bolus"]) >= float(thresholds.get("bolus") or 3.0)
    if name == "carbs_large":
        if context.get("carbs") is None:
            return None
        return float(context["carbs"]) >= float(thresholds.get("carbs") or 45.0)
    if name == "morning_event":
        return float(context.get("hour") or 0.0) < 11.0
    if name == "afternoon_event":
        return 11.0 <= float(context.get("hour") or 0.0) < 17.0
    if name == "evening_event":
        return float(context.get("hour") or 0.0) >= 17.0
    if name == "actual_bolus_present":
        return True if context.get("bolus_valid") else None
    threshold_map = {
        "sleep_rem_low": ("sleep_rem_fraction", "sleep_rem_fraction_low", "low"),
        "sleep_fragmentation_high": ("sleep_fragmentation_per_hour", "sleep_fragmentation_high", "high"),
        "hrv_sdnn_low": ("hrv_sdnn_deviation_24h", "hrv_sdnn_deviation_low", "low"),
        "heart_rate_deviation_high": ("heart_rate_deviation_24h", "heart_rate_deviation_high", "high"),
        "mood_valence_low": ("mood_valence_24h", "mood_valence_low", "low"),
        "workout_long": ("workout_minutes_24h", "workout_minutes_high", "high"),
    }
    if name in threshold_map:
        feature, threshold_key, direction = threshold_map[name]
        value = _finite(context.get(feature)); threshold = _finite(thresholds.get(threshold_key))
        if value is None or threshold is None:
            return None
        return value <= threshold if direction == "low" else value >= threshold
    if name == "evening_workout":
        value = context.get("evening_workout")
        return bool(value) if isinstance(value, bool) else None
    if name == "daytime_steps_high":
        value = _finite(context.get("steps_daytime_24h"))
        threshold = _finite(thresholds.get("daytime_steps_high"))
        if value is None or threshold is None:
            return None
        return value >= threshold
    return None


_EVENT_LABELS = {
    "after_activity": "a recorded activity period preceded the event",
    "short_sleep_before": "shorter completed sleep preceded the event",
    "recorded_cycle_early": "an explicitly recorded early-cycle day",
    "site_age_late": "a recorded later site-age day",
    "recent_activity_high": "recent recorded activity was above its discovery reference",
    "body_temperature_high": "recent body temperature was above its prior recorded baseline",
    "wrist_temperature_high": "recent sleeping-wrist temperature was above its prior recorded baseline",
    "starting_glucose_high": "starting glucose was above the discovery reference",
    "bolus_large": "the recorded bolus was above the discovery reference",
    "carbs_large": "recorded carbohydrates were above the discovery reference",
    "morning_event": "the event occurred in the morning",
    "afternoon_event": "the event occurred in the afternoon",
    "evening_event": "the event occurred in the evening",
    "actual_bolus_present": "a delivered bolus was recorded",
    "sleep_rem_low": "the latest completed sleep had a lower recorded REM fraction than its discovery reference",
    "sleep_fragmentation_high": "the latest completed sleep had more recorded awake bouts per asleep hour than its discovery reference",
    "hrv_sdnn_low": "recent recorded HRV was below its earlier personal baseline reference",
    "heart_rate_deviation_high": "recent recorded heart rate was above its earlier personal baseline reference",
    "mood_valence_low": "recorded mood valence was below its discovery reference",
    "evening_workout": "a recorded workout occurred during the evening",
    "workout_long": "recorded workout duration was above its discovery reference",
    "daytime_steps_high": "recorded daytime steps were above their discovery reference",
}

_PARTICIPANT_CONTEXT_LABELS = {
    "after_activity": "earlier recorded activity",
    "short_sleep_before": "shorter recorded sleep",
    "recorded_cycle_early": "an early recorded cycle day",
    "site_age_late": "more days since a recorded site change",
    "recent_activity_high": "more recorded activity in the preceding week",
    "body_temperature_high": "higher recent body temperature than its earlier baseline",
    "wrist_temperature_high": "higher recent wrist temperature than its earlier baseline",
    "starting_glucose_high": "higher starting glucose",
    "bolus_large": "a larger recorded delivered bolus",
    "carbs_large": "more recorded carbohydrates",
    "morning_event": "a morning event",
    "afternoon_event": "an afternoon event",
    "evening_event": "an evening event",
    "actual_bolus_present": "a recorded delivered bolus",
    "sleep_rem_low": "a lower recorded REM fraction during the latest completed sleep",
    "sleep_fragmentation_high": "more recorded awake bouts during the latest completed sleep",
    "hrv_sdnn_low": "lower recent recorded HRV than the earlier personal baseline",
    "heart_rate_deviation_high": "higher recent recorded heart rate than the earlier personal baseline",
    "mood_valence_low": "lower recorded mood valence",
    "evening_workout": "a recorded evening workout",
    "workout_long": "a longer recorded workout",
    "daytime_steps_high": "more recorded daytime steps",
}


def _match_event_rows(rows: Sequence[Mapping[str, Any]], names: Sequence[str], thresholds: Mapping[str, float | None]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []; negative: list[dict[str, Any]] = []
    for row in rows:
        flags = [_event_flag(row, name, thresholds) for name in names]
        if any(flag is None for flag in flags):
            continue
        (positive if all(flags) else negative).append(dict(row))
    return positive, negative


def _match_numeric_rule_rows(
    rows: Sequence[Mapping[str, Any]], rules: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for row in rows:
        flags = [evaluate_rule(row.get("context", {}), rule) for rule in rules]
        # A row missing any feature required by this rule is unknown for both
        # arms; it is never treated as a negative comparison.
        if not flags or any(flag is None for flag in flags):
            continue
        (positive if all(flags) else negative).append(dict(row))
    return positive, negative


def _matched_event_pairs(positive: Sequence[Mapping[str, Any]], negative: Sequence[Mapping[str, Any]],
                         residuals: Mapping[str, float], *, match_bolus: bool = True) -> list[dict[str, Any]]:
    available = {str(row["id"]): row for row in negative}
    pairs: list[dict[str, Any]] = []
    used_days: set[date] = set()
    for left in sorted(positive, key=lambda row: (row["date"], str(row["id"]))):
        if left["date"] in used_days:
            continue
        left_context = left["context"]
        options: list[tuple[float, Mapping[str, Any]]] = []
        for right in available.values():
            if right["date"] == left["date"] or right["date"] in used_days:
                continue
            if left.get("episodeKind") != right.get("episodeKind"):
                continue
            right_context = right["context"]
            left_profile = left_context.get("therapy_profile_fingerprint")
            right_profile = right_context.get("therapy_profile_fingerprint")
            if left_profile is not None and right_profile is not None and left_profile != right_profile:
                continue
            if match_bolus and bool(left_context.get("bolus_valid")) != bool(right_context.get("bolus_valid")):
                continue
            if bool(left_context.get("carbs_valid")) != bool(right_context.get("carbs_valid")):
                continue
            distance = 0.0; valid = True
            match_fields = [("start_glucose", 25.0), ("pre_glucose_slope", 20.0), ("carbs", 15.0)]
            if match_bolus:
                match_fields.append(("bolus", 1.5))
            for key, caliper in match_fields:
                lv = _finite(left_context.get(key)); rv = _finite(right_context.get(key))
                if lv is None or rv is None:
                    if key == "bolus" and not left_context.get("bolus_valid") and not right_context.get("bolus_valid"):
                        continue
                    if key == "carbs" and not left_context.get("carbs_valid") and not right_context.get("carbs_valid"):
                        continue
                    valid = False; break
                difference = abs(lv - rv)
                if difference > caliper:
                    valid = False; break
                distance += difference / caliper
            if valid:
                options.append((distance, right))
        if not options:
            continue
        _, right = min(options, key=lambda item: (item[0], item[1]["date"], str(item[1]["id"])))
        available.pop(str(right["id"]), None)
        if str(left["id"]) not in residuals or str(right["id"]) not in residuals:
            continue
        pairs.append({"left": left, "right": right, "difference": float(residuals[left["id"]] - residuals[right["id"]])})
        used_days.update((left["date"], right["date"]))
    return pairs


def _response_candidate(
    rows: Sequence[Mapping[str, Any]], discovery: Sequence[Mapping[str, Any]], confirmation: Sequence[Mapping[str, Any]],
    names: tuple[str, ...], target: str, config: BriefConfig, thresholds: Mapping[str, float | None], *,
    discovery_only: bool = False, family_size: int | None = None,
    rule_descriptors: Sequence[Mapping[str, Any]] | None = None, candidate_id: str | None = None,
    diagnostic_counts: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    descriptor = OUTCOME_DESCRIPTORS.get(str(target), {})
    adjust_for_bolus = bool(descriptor.get("adjustBolus", True))
    model = _ridge_fit(discovery, target, config.v3_ridge_alpha, adjust_for_bolus=adjust_for_bolus)
    residuals = _ridge_residuals(discovery, target, config.v3_ridge_alpha, model=model, adjust_for_bolus=adjust_for_bolus)
    matcher = _match_numeric_rule_rows if rule_descriptors is not None else _match_event_rows
    if rule_descriptors is not None:
        positive_d, negative_d = matcher(discovery, rule_descriptors)
    else:
        positive_d, negative_d = matcher(discovery, names, thresholds)
    pairs_d = _matched_event_pairs(positive_d, negative_d, residuals, match_bolus=adjust_for_bolus)
    if len(pairs_d) < max(3, config.v3_min_units):
        if diagnostic_counts is not None:
            diagnostic_counts["discoverySupportRejected"] = int(diagnostic_counts.get("discoverySupportRejected", 0)) + 1
        return None
    discovery_effect = _mean(pair["difference"] for pair in pairs_d)
    if discovery_effect is None or abs(discovery_effect) < _response_threshold(target):
        if diagnostic_counts is not None:
            diagnostic_counts["discoveryEffectRejected"] = int(diagnostic_counts.get("discoveryEffectRejected", 0)) + 1
        return None
    if discovery_only:
        item = {
            "id": candidate_id or ("event-" + "__".join(names) + "-" + target), "predicates": names, "target": target,
            "discoveryEffect": float(discovery_effect), "discoveryPairs": pairs_d, "predicateThresholds": dict(thresholds),
        }
        if rule_descriptors is not None:
            item["ruleDescriptors"] = [dict(rule) for rule in rule_descriptors]
        item["outcomeDescriptor"] = dict(descriptor)
        return item
    # Fit the nuisance adjustment on discovery only, then apply that frozen
    # model to confirmation rows.  Re-fitting here would leak confirmation
    # outcomes into the comparison.
    confirmation_residuals = _ridge_residuals(confirmation, target, config.v3_ridge_alpha, model=model, adjust_for_bolus=adjust_for_bolus)
    if rule_descriptors is not None:
        positive_c, negative_c = _match_numeric_rule_rows(confirmation, rule_descriptors)
    else:
        positive_c, negative_c = _match_event_rows(confirmation, names, thresholds)
    pairs_c = _matched_event_pairs(positive_c, negative_c, confirmation_residuals, match_bolus=adjust_for_bolus)
    if len(pairs_c) < max(3, config.v3_min_units):
        if diagnostic_counts is not None:
            diagnostic_counts["confirmationSupportRejected"] = int(diagnostic_counts.get("confirmationSupportRejected", 0)) + 1
        return None
    differences = [pair["difference"] for pair in pairs_c]
    effect = _mean(differences)
    if effect is None:
        if diagnostic_counts is not None:
            diagnostic_counts["confirmationEffectRejected"] = int(diagnostic_counts.get("confirmationEffectRejected", 0)) + 1
        return None
    dates = [pair["left"]["date"] for pair in pairs_c]
    p_value, low, high, stable = _block_bootstrap(
        differences, dates, draws=config.v3_bootstrap_draws, block_days=config.v3_block_days,
        seed=config.seed + len(names) + len(target), family_size=max(1, family_size or config.v3_max_candidates), alpha=config.alpha,
    )
    item = {
        "id": candidate_id or ("event-" + "__".join(names) + "-" + target), "predicates": names, "target": target,
        "discoveryEffect": float(discovery_effect), "effect": float(effect), "low": float(low), "high": float(high),
        "p": p_value, "stable": stable, "pairs": pairs_c, "discoveryPairs": pairs_d, "predicateThresholds": dict(thresholds),
    }
    if rule_descriptors is not None:
        item["ruleDescriptors"] = [dict(rule) for rule in rule_descriptors]
    item["outcomeDescriptor"] = dict(descriptor)
    return item


def _event_evidence(item: Mapping[str, Any], *, target: str, config: BriefConfig) -> dict[str, Any]:
    effect = float(item["effect"]); low = float(item["low"]); high = float(item["high"])
    direction = "higher" if effect >= 0 else "lower"
    descriptor = dict(item.get("outcomeDescriptor") or OUTCOME_DESCRIPTORS.get(target, {}))
    target_label = str(descriptor.get("label", target))
    unit = str(descriptor.get("unit", "mg/dL"))
    descriptions = _condition_descriptions(item, event=True)
    condition = " and ".join(descriptions)
    response_key = str(descriptor.get("key", target))
    observed_responses = [{response_key: pair["left"].get(response_key)} for pair in item["pairs"]]
    observed_summary = _outcome_summary(
        observed_responses, response_key, label=target_label,
        horizon=target, unit=unit,
    )
    episode_kind = str(descriptor.get("episodeKind", "event"))
    episodes: list[dict[str, Any]] = []
    for pair in item["pairs"][: max(2, config.max_evidence_episodes // 2)]:
        for group, row in (("context", pair["left"]), ("comparison", pair["right"])):
            episode = {
                "id": f"{item['id']}-{group}-{row['date'].isoformat()}",
                "label": f"{group.title()} · {row['date'].isoformat()}", "group": group,
                "start": _iso(row["anchor"]), "outcomeEnd": _iso(row["outcomeEnds"][target]),
                "points": row.get("trace", []), "events": row.get("events", []),
                "episodeKind": episode_kind,
            }
            if episode_kind == "delivered_bolus":
                source_bolus = row.get("sourceBolus")
                known_at = row.get("episodeKnownAt") or (source_bolus.known_at if isinstance(source_bolus, BriefRecord) else None)
                episode.update({
                    "availabilityMode": "retrospective_event_exposure_with_strict_pre_anchor_context",
                    "episodeKnownAt": _iso(known_at) if isinstance(known_at, datetime) else None,
                    "correctionClassification": row.get("correctionClassification", "unknown"),
                    "eventExposure": {
                        "metric": "delivered_bolus", "value": _finite(row.get("context", {}).get("bolus")),
                        "unit": "U", "knownAt": _iso(known_at) if isinstance(known_at, datetime) else None,
                    },
                })
            episodes.append(episode)
    title = {
        "delivered_bolus": "Responses following recorded delivered boluses",
        "pre_meal": "Retrospective pre-meal glucose trajectories",
        "explicit_fasting": "Glucose during explicitly recorded fasting intervals",
    }.get(episode_kind, "Recorded event responses")
    display = f"{abs(effect):.1f} {unit}"
    limitations = [
        "This is a retrospective observational comparison, not a cause, forecast, or treatment recommendation.",
        "The nuisance baseline was fit on earlier discovery episodes only; matching and the frozen independent confirmation set preserve the existing evidence gate.",
        "Intervals use approximate contiguous-day blocks and a Bonferroni adjustment for the frozen global confirmation family; repeated future snapshots are not anytime-valid.",
    ]
    if episode_kind == "delivered_bolus":
        limitations.extend([
            "This describes observed glucose following recorded delivered insulin; it does not isolate a causal insulin effect. Later meals and additional delivered boluses are retained as visible coexposures.",
            "The numeric dose is an event exposure, not a pre-dose context value. Its source-known timestamp is retained for retrospective interpretation; unknown correction intent is not inferred from dose size or missing carbohydrate records.",
        ])
    elif episode_kind == "pre_meal":
        limitations.append("This trajectory is identified retrospectively by a later recorded meal and is not a currently predictable episode.")
    elif episode_kind == "explicit_fasting":
        limitations.append("Only explicitly recorded fasting intervals with positive provenance qualify; absence of meal records is not treated as fasting.")
    elif episode_kind.startswith("clock_bin_"):
        limitations.append("Bins are anchored to local 00:00, 03:00, …, 21:00; bins whose physical duration changes at daylight-saving transitions are omitted.")
    evidence = _make_evidence(
        title=title, source_label="Recorded glucose and time-aligned observed context",
        window_label=f"{target_label} · later comparable episodes", summary=(
            f"When {condition}, the recorded {target_label.lower()} was {display} {direction} in the matched comparison after a discovery-only baseline adjustment."
        ), limitations=[
            *limitations,
        ], facts=[
            {"label": "Recorded difference", "value": f"{effect:+.1f} {unit}"},
            {"label": "Matched event pairs", "value": str(len(item["pairs"]))},
            {"label": "Approximate adjusted interval", "value": f"[{low:+.1f}, {high:+.1f}] {unit}"},
        ], analysis_kind="event_response", evidence_level="retrospectively_replicated", method_label="discovery-only ridge residuals + independent-day matched comparison",
        availability_mode=("strict_pre_event_context_with_retrospective_event_exposure"
                           if episode_kind == "delivered_bolus" else "strict_pre_event_context_retrospective_outcomes"),
        support={"unit": "independent event pairs", "count": len(item["pairs"]), "pairCount": len(item["pairs"])},
        outcome_summary=[observed_summary] if observed_summary else None,
        episodes=episodes, comparison={"outcome": target, "descriptor": descriptor, "unit": unit, "effect": effect, "low": low, "high": high, "rangeKind": "approximate_adjusted_block_interval", "contextN": len(item["pairs"]), "comparisonN": len(item["pairs"]), "pairCount": len(item["pairs"])},
        context_predicates=descriptions, context_rules=item.get("ruleDescriptors"),
        matched_on=["starting glucose", "pre-anchor glucose trend", "recorded carbohydrate amount/availability"] + (["recorded delivered-bolus availability and amount"] if descriptor.get("adjustBolus", True) else []),
    )
    # Narration needs observed outcomes for BOTH arms. The adjusted matched
    # effect above is a different quantity and cannot stand in for either arm.
    evidence["observedOutcomesByGroup"] = {
        group: _outcome_summary(
            [pair[side] for pair in item["pairs"]], response_key,
            label=target_label, horizon=target, unit=unit,
        )
        for group, side in (("context", "left"), ("comparison", "right"))
    }
    positive_episode_ids = [str(pair["left"].get("id")) for pair in item["pairs"] if pair.get("left", {}).get("id")]
    evidence["presentationCoverage"] = {
        "kind": "numeric_association", "complete": bool(positive_episode_ids)
        and len(positive_episode_ids) == len(item["pairs"]) and len(set(positive_episode_ids)) == len(positive_episode_ids),
        "outcome": target, "episodeKind": episode_kind, "direction": direction,
        "positiveEpisodeIds": positive_episode_ids,
        "positiveEpisodeCount": len(positive_episode_ids),
    }
    response_outcome_context = _response_outcome_context(
        item, target=target, config=config, episode_kind=episode_kind,
    )
    if response_outcome_context is not None:
        evidence["responseOutcomeContext"] = response_outcome_context
    return evidence


def _daily_evidence(item: Mapping[str, Any], *, config: BriefConfig) -> dict[str, Any]:
    target = str(item["target"])
    descriptor = dict(item.get("outcomeDescriptor") or OUTCOME_DESCRIPTORS.get(target, {}))
    target_label = str(descriptor.get("label", target))
    unit = str(descriptor.get("unit", "mg/dL"))
    horizon = str(descriptor.get("window", "following 24 hours"))
    effect = float(item["effect"]); low = float(item["low"]); high = float(item["high"])
    names = tuple(item["predicates"])
    descriptions = _condition_descriptions(item, event=False)
    condition = " and ".join(descriptions)
    observed = [{target: pair["left"].get(target)} for pair in item["pairs"]]
    observed_summary = _outcome_summary(observed, target, label=target_label, horizon=horizon, unit=unit)
    episodes: list[dict[str, Any]] = []
    for pair in item["pairs"][: max(2, config.max_evidence_episodes // 2)]:
        for group, row in (("context", pair["left"]), ("comparison", pair["right"])):
            episodes.append({
                "id": f"{item['id']}-{group}-{row['date'].isoformat()}",
                "label": f"{group.title()} · {row['date'].isoformat()}", "group": group,
                "start": _iso(row["anchor"]), "outcomeEnd": _iso(row["outcomeEnds"][item["target"]]),
                "points": row.get("trace", []), "events": row.get("events", []),
            })
    direction = "higher" if effect > 0 else "lower"
    evidence = _make_evidence(
        title="Recorded daily-context responses", source_label="Recorded pre-anchor context and subsequent CGM observations",
        window_label=f"Same local clock time · {horizon}", summary=(
            f"When {condition}, {target_label} was {abs(effect):.1f} {unit} {direction} on later independent days after a discovery-only baseline adjustment."
        ),
        limitations=[
            "This is a retrospective observational comparison, not a cause, forecast, or treatment recommendation.",
            "The nuisance baseline used starting glucose, pre-anchor glucose trend, and recorded carbohydrate and delivered-bolus history within the same availability strata; unmeasured context may remain.",
            "Insulin quantities are recorded delivered boluses only; measured basal delivery was unavailable.",
            "The approximate interval uses contiguous-day blocks and the frozen global-family adjustment; repeated future snapshots are not anytime-valid.",
        ],
        facts=[
            {"label": "Recorded difference", "value": f"{effect:+.1f} {unit}"},
            {"label": "Independent day pairs", "value": str(len(item["pairs"]))},
            {"label": "Approximate adjusted interval", "value": f"[{low:+.1f}, {high:+.1f}] {unit}"},
        ],
        analysis_kind="daily_context_response", evidence_level="retrospectively_replicated",
        method_label="same-clock daily anchors + discovery-only availability-stratified ridge + independent-day matching",
        availability_mode="strict_context_retrospective_outcomes",
        support={"unit": "independent day pairs", "count": len(item["pairs"]), "pairCount": len(item["pairs"])},
        outcome_summary=[observed_summary] if observed_summary else None, episodes=episodes,
        comparison={"outcome": target, "descriptor": descriptor, "unit": unit, "effect": effect, "low": low, "high": high, "rangeKind": "approximate_adjusted_block_interval", "contextN": len(item["pairs"]), "comparisonN": len(item["pairs"]), "pairCount": len(item["pairs"])},
        context_predicates=descriptions, context_rules=item.get("ruleDescriptors"),
        matched_on=["same local clock time", "starting glucose", "pre-anchor glucose trend", "recorded carbohydrate amount/availability", "recorded delivered-bolus amount/availability"],
    )
    positive_episode_ids = [str(pair["left"].get("id")) for pair in item["pairs"] if pair.get("left", {}).get("id")]
    evidence["presentationCoverage"] = {
        "kind": "numeric_association", "complete": bool(positive_episode_ids)
        and len(positive_episode_ids) == len(item["pairs"]) and len(set(positive_episode_ids)) == len(positive_episode_ids),
        "outcome": target, "episodeKind": str(descriptor.get("episodeKind", "daily_context")),
        "direction": direction, "positiveEpisodeIds": positive_episode_ids,
        "positiveEpisodeCount": len(positive_episode_ids),
    }
    return evidence


def _discover_event_responses(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, config: BriefConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _RecordIndex.build(records)
    rows = _build_event_episodes(records, cutoff=cutoff, zone=zone, config=config, index=index)
    rows.sort(key=lambda row: (row["date"], row["anchor"]))
    dates = sorted({row["date"] for row in rows})
    if len(dates) < max(2 * config.v3_min_units, 12):
        return [], {"status": "insufficient_event_days", "eventCount": len(rows), "eligibleDays": len(dates), "tested": 0}
    split = max(1, int(len(dates) * 0.60))
    discovery_dates = set(dates[:split]); confirmation_dates = set(dates[split:])
    discovery = [row for row in rows if row["date"] in discovery_dates]
    confirmation = [row for row in rows if row["date"] in confirmation_dates]
    thresholds = _event_predicates(rows, discovery=discovery)
    # Discovery ranks a predeclared family.  Confirmation is run only after
    # this ranking so confirmation outcomes cannot decide which candidates are
    # tested.
    candidates: list[dict[str, Any]] = []
    predicate_sets = [predicates for predicates in _EVENT_PREDICATE_SETS if len(predicates) == 1 or config.v3_max_candidates > 1]
    for predicates in predicate_sets:
        for target in ("firstResponse", "lateResponse"):
            item = _response_candidate(rows, discovery, confirmation, predicates, target, config, thresholds, discovery_only=True)
            if item is not None:
                candidates.append(item)
    candidates.sort(key=lambda item: (-_discovery_score(item["discoveryEffect"], item["target"]), item["id"]))
    candidates = candidates[: max(1, config.v3_max_candidates)]
    confirmed: list[dict[str, Any]] = []
    for selected in candidates:
        item = _response_candidate(rows, discovery, confirmation, tuple(selected["predicates"]), selected["target"], config, thresholds)
        if item is None:
            # Failed confirmation remains part of the frozen family and gets
            # a conservative p=1 for the multiplicity adjustment.
            item = {**selected, "effect": 0.0, "low": 0.0, "high": 0.0, "p": 1.0, "stable": False, "pairs": [], "confirmationPairs": 0}
        confirmed.append(item)
    adjusted = _holm([item["p"] for item in confirmed])
    findings: list[dict[str, Any]] = []
    for item, q_value in zip(confirmed, adjusted):
        item = dict(item); item["adjustedP"] = q_value
        item["supported"] = bool(q_value <= config.alpha and (item["low"] > 0 or item["high"] < 0) and item["stable"] and abs(item["effect"]) >= _target_threshold("mean") / 2)
        if item["supported"]:
            findings.append(item)
    return findings, {
        "status": "ready", "eventCount": len(rows), "discoveryDays": len(discovery_dates), "confirmationDays": len(confirmation_dates),
        "tested": len(confirmed), "supported": len(findings), "thresholdsFitOnDiscovery": True,
        "baseline": "ridge_discovery_only", "familyAdjustment": "holm", "minimumUnitRule": f"at least {config.v3_min_units} units and {max(2, config.v3_block_days)}-day blocks",
        "candidateFamily": "bounded event predicates × response horizons; ranked on discovery only",
    }


def _discover_numeric_family(
    rows: Sequence[Mapping[str, Any]], by_target: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *, engine: str, targets: Sequence[str], feature_names: Sequence[str], config: BriefConfig,
    thresholds: Mapping[str, float | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propose numeric rules on discovery residuals; downstream gates stay shared."""
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    search_runs: list[dict[str, Any]] = []
    prepared_spaces: dict[tuple[Any, ...], Any] = {}
    prepared_space_cache_hits = 0
    target_diagnostics: dict[str, dict[str, Any]] = {}
    for target in targets:
        target_split = by_target[target]
        discovery = target_split["discovery"]
        target_diag: dict[str, Any] = {
            "eligibleDiscoveryRows": len(discovery),
            "eligibleConfirmationRows": len(target_split.get("confirmation", ())),
            "discoveryResidualScalePopulationSD": None,
            "rawNativeProposals": 0,
            "discoverySupportRejected": 0,
            "discoveryEffectRejected": 0,
            "discoveryDirectionRejected": 0,
            "matchedDiscoveryCandidates": 0,
            "nominated": 0,
            "selectedForConfirmation": 0,
            "confirmationSupportRejected": 0,
            "confirmationEffectRejected": 0,
            "confirmationDirectionRejected": 0,
            "confirmationInferenceRejected": 0,
            "confirmedAndReleased": 0,
        }
        target_diagnostics[str(target)] = target_diag
        if not discovery:
            continue
        descriptor = OUTCOME_DESCRIPTORS.get(str(target), {})
        adjust_for_bolus = bool(descriptor.get("adjustBolus", True))
        model = _ridge_fit(discovery, target, config.v3_ridge_alpha, adjust_for_bolus=adjust_for_bolus)
        residuals = _ridge_residuals(discovery, target, config.v3_ridge_alpha, model=model, adjust_for_bolus=adjust_for_bolus)
        residual_values = [float(value) for value in residuals.values() if _finite(value) is not None]
        residual_scale = float(np.std(np.asarray(residual_values, dtype=float), ddof=0)) if residual_values else 0.0
        target_diag["discoveryResidualScalePopulationSD"] = residual_scale
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            target_diag["rankingAbstainedZeroOrNonfiniteScale"] = True
            continue
        target_diag["rankingAbstainedZeroOrNonfiniteScale"] = False
        aligned_ids = tuple(str(row.get("id")) for row in discovery if str(row.get("id")) in residuals)
        search_space_key = (tuple(feature_names), config.v3_search_bins, aligned_ids)
        prepared_space = prepared_spaces.get(search_space_key)
        if prepared_space is None:
            prepared_space = prepare_numeric_search_space(
                discovery, residuals, features=feature_names, bins=config.v3_search_bins,
            )
            prepared_spaces[search_space_key] = prepared_space
        else:
            prepared_space_cache_hits += 1
        for direction in (1, -1):
            proposals, search_diagnostics = discover_numeric_rules(
                discovery, residuals, features=feature_names, engine=engine, target=target,
                direction=direction, depth=config.v3_search_depth, bins=config.v3_search_bins,
                beam_width=config.v3_search_beam_width, result_size=config.v3_search_shortlist,
                min_support=max(3, config.v3_min_units // 2),
                prepared_space=prepared_space,
            )
            search_runs.append({"target": target, "direction": "higher" if direction > 0 else "lower", **search_diagnostics})
            target_diag["rawNativeProposals"] += len(proposals)
            for proposal in proposals:
                rules = proposal["ruleDescriptors"]
                item = _response_candidate(
                    rows, discovery, target_split["confirmation"], tuple(proposal["predicates"]), target,
                    config, thresholds, discovery_only=True, rule_descriptors=rules,
                    candidate_id=str(proposal["id"]),
                    diagnostic_counts=target_diag,
                )
                if item is None:
                    continue
                if float(item["discoveryEffect"]) * direction <= 0:
                    target_diag["discoveryDirectionRejected"] += 1
                    continue
                target_diag["matchedDiscoveryCandidates"] += 1
                normalized_score = _normalized_discovery_score(float(proposal["subgroupScore"]), residual_scale)
                if normalized_score is None:
                    continue
                candidate = {
                    **proposal, **item, "engine": engine,
                    "anchorType": "daily_context" if engine == "daily" else str(descriptor.get("episodeKind", "event_response")),
                    "outcomeDescriptor": dict(descriptor),
                    "discoveryScore": normalized_score,
                    "rankingMethod": "native_positive_subgroup_quality_divided_by_discovery_residual_population_sd",
                    "discoveryResidualScalePopulationSD": residual_scale,
                }
                positive, negative = _match_numeric_rule_rows(discovery, rules)
                eligible_ids = tuple(sorted(str(row["id"]) for row in (*positive, *negative)))
                positive_ids = tuple(sorted(str(row["id"]) for row in positive))
                # Missingness is part of the candidate's discovery semantics:
                # identical positive masks with different eligible masks are
                # distinct and must remain separately confirmable.
                coverage_signature = (engine, target, eligible_ids, positive_ids)
                def representative_rank(value: Mapping[str, Any]) -> tuple[Any, ...]:
                    descriptors = value.get("ruleDescriptors", ())
                    equality_count = sum(rule.get("kind") == "equal" for rule in descriptors)
                    canonical = json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
                    return (len(descriptors), equality_count, -float(value.get("subgroupScore", 0.0)), canonical)
                previous = candidates.get(coverage_signature)
                if previous is None or representative_rank(candidate) < representative_rank(previous):
                    candidates[coverage_signature] = candidate
    ordered = sorted(
        candidates.values(),
        key=lambda item: (-item["discoveryScore"], len(item["ruleDescriptors"]), -item["subgroupScore"], item["id"]),
    )
    for candidate in ordered:
        target_diagnostics[str(candidate["target"])]["nominated"] += 1
    feature_coverage = search_runs[0].get("featureCoverage", {}) if search_runs else {}
    canonicalized_aliases: dict[str, str] = {}
    for run in search_runs:
        canonicalized_aliases.update(run.get("canonicalizedFeatureAliases", {}))
    diagnostics = {
        "searchEngine": "pysubgroup.BeamSearch", "searchHeuristic": True,
        "searchVocabularyCanonicalization": (
            "explicit semantic aliases removed only when discovery values and missingness match exactly; "
            "this can change heuristic beam traversal"
        ),
        "canonicalizedFeatureAliases": canonicalized_aliases,
        "configuredDepth": config.v3_search_depth, "configuredBins": config.v3_search_bins,
        "configuredBeamWidth": config.v3_search_beam_width, "shortlistPerDirection": config.v3_search_shortlist,
        "searchTasks": len(search_runs), "evaluatedSubgroups": sum(int(run.get("evaluated", 0)) for run in search_runs),
        "maxDepthReached": max((int(run.get("maxDepthReached", 0)) for run in search_runs), default=0),
        "selectorCount": max((int(run.get("selectors", 0)) for run in search_runs), default=0),
        "proposalCount": sum(int(run.get("proposals", 0)) for run in search_runs),
        "matchedCandidateCount": len(ordered), "featureCoverage": feature_coverage,
        "preparedSearchSpaceCacheHits": prepared_space_cache_hits,
        "rankingMethod": "native_positive_subgroup_quality_divided_by_discovery_residual_population_sd",
        "targetDiagnostics": target_diagnostics,
        "runs": search_runs,
    }
    return ordered, diagnostics


def _discover_clock_windows(records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, current_day: date, config: BriefConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _RecordIndex.build(records)
    days = _day_candidates(records, zone, current_day)
    if len(days) < max(2 * config.v3_min_units, 12):
        return [], {"status": "insufficient_days", "eligibleDays": len(days), "tested": 0, "selected": 0}
    split = max(1, int(len(days) * 0.60))
    discovery_days = days[:split]; confirmation_days = days[split:]
    candidates: list[dict[str, Any]] = []
    for width in config.v3_window_widths:
        for start_hour in range(0, 24 - width + 1):
            for target in ("low_rate", "high_rate", "mean", "tir"):
                drows = _clock_window_rows(records, discovery_days, zone=zone, cutoff=cutoff, start_hour=start_hour, width=width, target=target, min_coverage=config.min_coverage, index=index)
                if len(drows) < config.v3_min_units:
                    continue
                d_effect = _mean(row["difference"] for row in drows)
                if d_effect is None or abs(d_effect) < _target_threshold(target):
                    continue
                candidates.append({
                    "id": f"clock-{start_hour:02d}-{width}-{target}", "startHour": start_hour, "width": width, "target": target,
                    "discoveryEffect": float(d_effect), "discovery": drows,
                })
    candidates = _dedupe_windows(candidates)
    candidates.sort(key=lambda item: (-_discovery_score(item["discoveryEffect"], "mean"), item["id"]))
    candidates = candidates[: max(1, config.v3_max_candidates)]
    confirmed: list[dict[str, Any]] = []
    for selected in candidates:
        crows = _clock_window_rows(records, confirmation_days, zone=zone, cutoff=cutoff, start_hour=int(selected["startHour"]), width=int(selected["width"]), target=selected["target"], min_coverage=config.min_coverage, index=index)
        if len(crows) < config.v3_min_units:
            confirmed.append({**selected, "confirmation": crows, "effect": 0.0, "low": 0.0, "high": 0.0, "p": 1.0, "stable": False})
            continue
        c_effect = _mean(row["difference"] for row in crows)
        if c_effect is None or selected["discoveryEffect"] * c_effect <= 0:
            confirmed.append({**selected, "confirmation": crows, "effect": float(c_effect or 0.0), "low": 0.0, "high": 0.0, "p": 1.0, "stable": False})
            continue
        differences = [row["difference"] for row in crows]
        p_value, low, high, stable = _block_bootstrap(differences, [row["date"] for row in crows], draws=config.v3_bootstrap_draws, block_days=config.v3_block_days, seed=config.seed + int(selected["startHour"]) * 31 + int(selected["width"]) * 7, family_size=max(1, config.v3_max_candidates))
        confirmed.append({**selected, "confirmation": crows, "effect": float(c_effect), "low": float(low), "high": float(high), "p": p_value, "stable": stable})
    adjusted = _holm([item["p"] for item in confirmed])
    findings: list[dict[str, Any]] = []
    for item, q_value in zip(confirmed, adjusted):
        item = dict(item); item["adjustedP"] = q_value
        item["supported"] = bool(q_value <= config.alpha and (item["low"] > 0 or item["high"] < 0) and item["stable"] and abs(item["effect"]) >= _target_threshold(item["target"]))
        if item["supported"]:
            findings.append(item)
    return findings, {
        "status": "ready", "eligibleDays": len(days), "discoveryDays": len(discovery_days), "confirmationDays": len(confirmation_days),
        "tested": len(confirmed), "selected": len(findings), "windowWidths": list(config.v3_window_widths),
        "blockDays": config.v3_block_days, "minimumUnitRule": f"at least {config.v3_min_units} complete days per arm",
        "familyAdjustment": "holm", "intervalMethod": "approximate_contiguous_day_block_bootstrap",
        "candidateFamily": "bounded 2/3/4-hour clock windows × outcome targets; ranked on discovery only",
    }


def _target_boundary_purge_count(
    raw_discovery: Sequence[Mapping[str, Any]], retained_discovery: Sequence[Mapping[str, Any]],
    *, eligible_for_target: Any,
) -> int:
    eligible_count = sum(1 for row in raw_discovery if eligible_for_target(row))
    return max(0, eligible_count - len(retained_discovery))


def _prepare_event_family(
    records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, config: BriefConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    index = _RecordIndex.build(records)
    meal_rows = _build_event_episodes(records, cutoff=cutoff, zone=zone, config=config, index=index)
    pre_meal_rows = _build_pre_meal_episodes(records, cutoff=cutoff, zone=zone, config=config, index=index)
    bolus_rows, bolus_diagnostics = _build_bolus_episodes(records, cutoff=cutoff, zone=zone, config=config, index=index)
    fasting_rows, fasting_diagnostics = _build_fasting_episodes(records, cutoff=cutoff, zone=zone, config=config, index=index)
    clock_bin_rows, clock_bin_diagnostics = _build_clock_bin_episodes(records, cutoff=cutoff, zone=zone, config=config, index=index)
    rows = [*meal_rows, *pre_meal_rows, *bolus_rows, *fasting_rows, *clock_bin_rows]
    rows.sort(key=lambda row: (row["date"], row["anchor"]))
    dates = sorted({row["date"] for row in rows})
    base = {
        "eventCount": len(rows), "eligibleDays": len(dates), "discoverySearchCount": 0,
        "episodeCounts": {"meal": len(meal_rows), "preMeal": len(pre_meal_rows), "deliveredBolus": len(bolus_rows),
                          "explicitFasting": len(fasting_rows), "localClockBin": len(clock_bin_rows)},
        "deliveredBolusClassification": bolus_diagnostics,
        "fastingSource": {**fasting_diagnostics, "productionSourceAvailable": False,
                          "requirement": "explicit fasting_interval with duration and positive fasting_provenance"},
        "clockBinTiming": clock_bin_diagnostics,
        "targetEpisodeCounts": {
            target: sum(1 for row in rows if row.get("episodeKind") == descriptor.get("episodeKind")
                        and _finite(row.get(target)) is not None
                        and (descriptor.get("classification") is None
                             or row.get("correctionClassification") == descriptor["classification"]))
            for target, descriptor in OUTCOME_DESCRIPTORS.items() if descriptor.get("family") == "event"
        },
        "targetDiagnostics": {
            target: {
                "eligibleEpisodeRows": sum(
                    1 for row in rows
                    if row.get("episodeKind") == descriptor.get("episodeKind")
                    and _finite(row.get(target)) is not None
                    and (descriptor.get("classification") is None
                         or row.get("correctionClassification") == descriptor["classification"])
                ),
                "status": "not_searched",
            }
            for target, descriptor in OUTCOME_DESCRIPTORS.items() if descriptor.get("family") == "event"
        },
    }
    if len(dates) < max(2 * config.v3_min_units, 12):
        return [], {}, {**base, "status": "insufficient_event_days"}
    split = max(1, int(len(dates) * 0.60))
    discovery_dates = set(dates[:split]); confirmation_dates = set(dates[split:])
    raw_discovery = [row for row in rows if row["date"] in discovery_dates]
    confirmation = [row for row in rows if row["date"] in confirmation_dates]
    boundary = min((row["anchor"] for row in confirmation), default=cutoff)
    by_target: dict[str, dict[str, Any]] = {}
    boundary_purged_by_target: dict[str, int] = {}
    targets: list[str] = []
    for target, descriptor in OUTCOME_DESCRIPTORS.items():
        if descriptor.get("family") != "event":
            continue
        kind = descriptor["episodeKind"]
        classification = descriptor.get("classification")
        def has_target(row: Mapping[str, Any]) -> bool:
            return (row.get("episodeKind") == kind and _finite(row.get(target)) is not None
                    and (classification is None or row.get("correctionClassification") == classification))
        target_eligible_discovery = [row for row in raw_discovery if has_target(row)]
        target_discovery = [row for row in target_eligible_discovery
                            if row.get("outcomeEnds", {}).get(target) is not None
                            and row["outcomeEnds"][target].astimezone(UTC) <= boundary.astimezone(UTC)]
        boundary_purged_by_target[target] = _target_boundary_purge_count(
            raw_discovery, target_discovery, eligible_for_target=has_target,
        )
        target_confirmation = [row for row in confirmation if has_target(row)]
        by_target[target] = {"discovery": target_discovery, "confirmation": target_confirmation,
                             "descriptor": dict(descriptor)}
        targets.append(target)
    threshold_discovery = next((split["discovery"] for split in by_target.values() if split["discovery"]), [])
    thresholds = _event_predicates(rows, discovery=threshold_discovery)
    candidates, search_diagnostics = _discover_numeric_family(
        rows, by_target, engine="event", targets=targets,
        feature_names=EVENT_FEATURES, config=config, thresholds=thresholds,
    )
    context = {"rows": rows, "byTarget": by_target, "thresholds": thresholds, "search": search_diagnostics,
               "outcomeDescriptors": {target: dict(OUTCOME_DESCRIPTORS[target]) for target in targets}}
    diagnostics = {
        **base, "status": "prepared", "discoveryDays": len(discovery_dates), "confirmationDays": len(confirmation_dates),
        "splitBoundary": _iso(boundary),
        "episodeCounts": {"meal": len(meal_rows), "preMeal": len(pre_meal_rows), "deliveredBolus": len(bolus_rows),
                          "explicitFasting": len(fasting_rows), "localClockBin": len(clock_bin_rows)},
        "deliveredBolusClassification": bolus_diagnostics,
        "fastingSource": {**fasting_diagnostics, "productionSourceAvailable": False,
                          "requirement": "explicit fasting_interval with duration and positive fasting_provenance"},
        "clockBinTiming": clock_bin_diagnostics,
        "boundaryPurgedRows": boundary_purged_by_target,
        "discoverySearchCount": search_diagnostics["evaluatedSubgroups"],
        "discoveryEligibleCount": len(candidates), "thresholdsFitOnDiscovery": True,
        "baseline": "discovery-only availability-stratified ridge residuals",
        **search_diagnostics,
    }
    return candidates, context, diagnostics


def _prepare_daily_family(
    records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, config: BriefConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = _build_daily_context_rows(records, cutoff=cutoff, zone=zone, config=config)
    rows.sort(key=lambda row: (row["date"], row["anchor"]))
    dates = sorted({row["date"] for row in rows})
    daily_targets = ("overnightResponse", "dayResponse", "dayVarianceResponse")
    base = {
        "anchorCount": len(rows), "eligibleDays": len(dates), "discoverySearchCount": 0,
        "targetDiagnostics": {
            target: {"eligibleOutcomeRows": sum(1 for row in rows if _finite(row.get(target)) is not None), "status": "not_searched"}
            for target in daily_targets
        },
    }
    if len(dates) < max(2 * config.v3_min_units, 12):
        return [], {}, {**base, "status": "insufficient_daily_context_days"}
    split = max(1, int(len(dates) * 0.60))
    discovery_dates = set(dates[:split]); confirmation_dates = set(dates[split:])
    raw_discovery = [row for row in rows if row["date"] in discovery_dates]
    confirmation = [row for row in rows if row["date"] in confirmation_dates]
    boundary = min((row["anchor"] for row in confirmation), default=cutoff)
    by_target: dict[str, dict[str, Any]] = {}
    boundary_purged_by_target: dict[str, int] = {}
    for target in daily_targets:
        eligible_discovery = [row for row in raw_discovery if _finite(row.get(target)) is not None]
        discovery = [
            row for row in eligible_discovery
            if row.get("outcomeEnds", {}).get(target) is not None
            and row["outcomeEnds"][target].astimezone(UTC) <= boundary.astimezone(UTC)
        ]
        boundary_purged_by_target[target] = _target_boundary_purge_count(
            raw_discovery, discovery,
            eligible_for_target=lambda row, key=target: _finite(row.get(key)) is not None,
        )
        by_target[target] = {"discovery": discovery, "confirmation": confirmation}
    threshold_discovery = next((split["discovery"] for split in by_target.values() if split["discovery"]), [])
    thresholds = _event_predicates(rows, discovery=threshold_discovery)
    candidates, search_diagnostics = _discover_numeric_family(
        rows, by_target, engine="daily", targets=("overnightResponse", "dayResponse", "dayVarianceResponse"),
        feature_names=DAILY_FEATURES, config=config, thresholds=thresholds,
    )
    return candidates, {"rows": rows, "byTarget": by_target, "thresholds": thresholds, "search": search_diagnostics}, {
        **base, "status": "prepared", "discoveryDays": len(discovery_dates), "confirmationDays": len(confirmation_dates),
        "splitBoundary": _iso(boundary), "boundaryPurgedRows": boundary_purged_by_target,
        "discoverySearchCount": search_diagnostics["evaluatedSubgroups"],
        "discoveryEligibleCount": len(candidates),
        "baseline": "discovery-only availability-stratified daily-context ridge residuals",
        **search_diagnostics,
    }


def _prepare_clock_family(
    records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, current_day: date, config: BriefConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    index = _RecordIndex.build(records)
    days = _day_candidates(records, zone, current_day)
    targets = ("low_rate", "high_rate", "mean", "tir")
    base = {
        "eligibleDays": len(days), "discoverySearchCount": 0,
        "targetDiagnostics": {
            target: {"discoveryWindowsTested": 0, "nominated": 0, "selectedForConfirmation": 0,
                     "status": "not_searched"}
            for target in targets
        },
    }
    if len(days) < max(2 * config.v3_min_units, 12):
        return [], {}, {**base, "status": "insufficient_days"}
    split = max(1, int(len(days) * 0.60))
    discovery_days, confirmation_days = days[:split], days[split:]
    candidates: list[dict[str, Any]] = []
    target_diagnostics = {
        target: {
            "discoveryWindowsTested": 0, "discoveryEligibleUnits": 0,
            "discoverySupportRejected": 0, "discoveryEffectRejected": 0,
            "matchedDiscoveryCandidates": 0, "nominated": 0,
            "selectedForConfirmation": 0, "confirmationSupportRejected": 0,
            "confirmationDirectionRejected": 0, "confirmationInferenceRejected": 0,
            "confirmedAndReleased": 0,
        }
        for target in targets
    }
    searched = 0
    for width in config.v3_window_widths:
        for start_hour in range(0, 24 - width + 1):
            for target in targets:
                searched += 1
                target_diag = target_diagnostics[target]
                target_diag["discoveryWindowsTested"] += 1
                drows = _clock_window_rows(records, discovery_days, zone=zone, cutoff=cutoff, start_hour=start_hour, width=width, target=target, min_coverage=config.min_coverage, index=index)
                if len(drows) < config.v3_min_units:
                    target_diag["discoverySupportRejected"] += 1
                    continue
                target_diag["discoveryEligibleUnits"] += len(drows)
                effect = _mean(row["difference"] for row in drows)
                if effect is None or abs(effect) < _target_threshold(target):
                    target_diag["discoveryEffectRejected"] += 1
                    continue
                target_diag["matchedDiscoveryCandidates"] += 1
                candidates.append({
                    "id": f"clock-{start_hour:02d}-{width}-{target}", "engine": "clock", "startHour": start_hour,
                    "width": width, "target": target, "discoveryEffect": float(effect), "discovery": drows,
                    "discoveryScore": _discovery_score(float(effect), target),
                })
    candidates = _dedupe_windows(candidates)
    candidates.sort(key=lambda item: (-item["discoveryScore"], item["id"]))
    for candidate in candidates:
        target_diagnostics[str(candidate["target"])]["nominated"] += 1
    return candidates, {"records": records, "confirmationDays": confirmation_days, "index": index,
                        "targetDiagnostics": target_diagnostics}, {
        **base, "status": "prepared", "discoveryDays": len(discovery_days), "confirmationDays": len(confirmation_days),
        "discoverySearchCount": searched, "discoveryEligibleCount": len(candidates), "windowWidths": list(config.v3_window_widths),
        "targetDiagnostics": target_diagnostics,
    }


def _bounded_global_family(
    clock_candidates: Sequence[dict[str, Any]], event_candidates: Sequence[dict[str, Any]],
    daily_candidates: Sequence[dict[str, Any]] = (), *, maximum: int,
) -> list[dict[str, Any]]:
    maximum = max(1, int(maximum))
    # Multiple clock outcomes for the identical interval are especially
    # redundant under a small global budget. Keep the strongest discovery
    # endpoint per interval before allocating ranks across engines.
    clock_diverse: list[dict[str, Any]] = []
    seen_clock_windows: set[tuple[int, int]] = set()
    for candidate in clock_candidates:
        window = (int(candidate["startHour"]), int(candidate["width"]))
        if window not in seen_clock_windows:
            clock_diverse.append(candidate)
            seen_clock_windows.add(window)
    daily_diverse: list[dict[str, Any]] = []
    seen_daily_rules: set[tuple[tuple[str, ...], str]] = set()
    for candidate in daily_candidates:
        # Body and sleeping-wrist temperature are distinct measurements but
        # alternate sensors for the same declared temperature-context rule.
        # Confirm the stronger discovery version rather than spend two global
        # slots on sensor variants of one condition.
        if candidate.get("ruleDescriptors"):
            predicates = tuple(sorted(
                json.dumps(rule, sort_keys=True, separators=(",", ":"))
                for rule in candidate["ruleDescriptors"]
            ))
        else:
            predicates = tuple(sorted(
                "temperature_high" if name in {"body_temperature_high", "wrist_temperature_high"} else str(name)
                for name in candidate.get("predicates", ())
            ))
        signature = (predicates, str(candidate["target"]))
        if signature not in seen_daily_rules:
            daily_diverse.append(candidate)
            seen_daily_rules.add(signature)

    def target_diverse_first(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        # Inputs are already ranked using discovery-only scores. Let the
        # strongest candidate for each outcome compete before a target can
        # consume another family slot with a second selector variant.
        first_by_target: list[dict[str, Any]] = []
        seen_targets: set[str] = set()
        remaining: list[dict[str, Any]] = []
        for candidate in candidates:
            target = str(candidate.get("target", ""))
            if target not in seen_targets:
                first_by_target.append(candidate)
                seen_targets.add(target)
            else:
                remaining.append(candidate)
        return [*first_by_target, *remaining]

    event_diverse = target_diverse_first(event_candidates)
    daily_diverse = target_diverse_first(daily_diverse)
    families = [list(candidates) for candidates in (clock_diverse, event_diverse, daily_diverse) if candidates]
    chosen: list[dict[str, Any]] = []
    # Allocate the bounded confirmation budget by rank across discovery
    # engines. This prevents a dense family of overlapping clock endpoints
    # from consuming every slot while still using only discovery information.
    rank = 0
    while len(chosen) < maximum and any(rank < len(family) for family in families):
        for family in families:
            if rank < len(family) and len(chosen) < maximum:
                chosen.append(dict(family[rank]))
        rank += 1
    return chosen[:maximum]


def _confirm_global_family(
    family: Sequence[dict[str, Any]], *, event_context: Mapping[str, Any], clock_context: Mapping[str, Any],
    daily_context: Mapping[str, Any], cutoff: datetime, zone: ZoneInfo, config: BriefConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    family_size = max(1, len(family))
    event_findings: list[dict[str, Any]] = []
    daily_findings: list[dict[str, Any]] = []
    clock_findings: list[dict[str, Any]] = []
    for selected in family:
        if selected["engine"] in {"event", "daily"}:
            selected_context = event_context if selected["engine"] == "event" else daily_context
            target_split = selected_context["byTarget"][selected["target"]]
            target_diag = selected_context.get("search", {}).get("targetDiagnostics", {}).get(str(selected["target"]))
            item = _response_candidate(
                selected_context["rows"], target_split["discovery"], target_split["confirmation"],
                tuple(selected["predicates"]), selected["target"], config, selected_context["thresholds"], family_size=family_size,
                rule_descriptors=selected.get("ruleDescriptors"), candidate_id=str(selected["id"]),
                diagnostic_counts=target_diag if isinstance(target_diag, dict) else None,
            )
            if item is None:
                continue
            direction_agrees = float(selected["discoveryEffect"]) * float(item["effect"]) > 0
            if not direction_agrees and isinstance(target_diag, dict):
                target_diag["confirmationDirectionRejected"] += 1
            item = {**item, "id": selected["id"], "engine": selected["engine"],
                    "anchorType": selected.get("anchorType"), "familySize": family_size,
                    "outcomeDescriptor": dict(selected.get("outcomeDescriptor", OUTCOME_DESCRIPTORS.get(str(selected["target"]), {})))}
            item["supported"] = bool(direction_agrees and item["stable"] and (item["low"] > 0 or item["high"] < 0) and abs(item["effect"]) >= _response_threshold(item["target"]))
            if item["supported"]:
                if isinstance(target_diag, dict):
                    target_diag["confirmedAndReleased"] += 1
                (event_findings if selected["engine"] == "event" else daily_findings).append(item)
            elif isinstance(target_diag, dict):
                target_diag["confirmationInferenceRejected"] += 1
            continue
        crows = _clock_window_rows(
            clock_context["records"], clock_context["confirmationDays"], zone=zone, cutoff=cutoff,
            start_hour=int(selected["startHour"]), width=int(selected["width"]), target=selected["target"],
            min_coverage=config.min_coverage, index=clock_context["index"],
        )
        if len(crows) < config.v3_min_units:
            target_diag = clock_context.get("targetDiagnostics", {}).get(str(selected["target"]))
            if isinstance(target_diag, dict):
                target_diag["confirmationSupportRejected"] += 1
            continue
        effect = _mean(row["difference"] for row in crows)
        if effect is None or float(selected["discoveryEffect"]) * effect <= 0:
            target_diag = clock_context.get("targetDiagnostics", {}).get(str(selected["target"]))
            if isinstance(target_diag, dict):
                target_diag["confirmationDirectionRejected"] += 1
            continue
        p_value, low, high, stable = _block_bootstrap(
            [row["difference"] for row in crows], [row["date"] for row in crows], draws=config.v3_bootstrap_draws,
            block_days=config.v3_block_days, seed=config.seed + int(selected["startHour"]) * 31 + int(selected["width"]) * 7,
            family_size=family_size, alpha=config.alpha,
        )
        item = {**selected, "confirmation": crows, "effect": float(effect), "low": float(low), "high": float(high), "p": p_value, "stable": stable, "familySize": family_size}
        item["supported"] = bool(stable and (low > 0 or high < 0) and abs(effect) >= _target_threshold(item["target"]))
        if item["supported"]:
            target_diag = clock_context.get("targetDiagnostics", {}).get(str(selected["target"]))
            if isinstance(target_diag, dict):
                target_diag["confirmedAndReleased"] += 1
            clock_findings.append(item)
        else:
            target_diag = clock_context.get("targetDiagnostics", {}).get(str(selected["target"]))
            if isinstance(target_diag, dict):
                target_diag["confirmationInferenceRejected"] += 1
    return clock_findings, event_findings, daily_findings


def _dedupe_association_findings(findings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def rule_keys(item: Mapping[str, Any]) -> set[str]:
        rules = item.get("ruleDescriptors")
        if rules:
            return {json.dumps(rule, sort_keys=True, separators=(",", ":")) for rule in rules}
        return {str(name) for name in item.get("predicates", ())}

    kept: list[dict[str, Any]] = []
    ordered = sorted(findings, key=lambda item: (len(item.get("predicates", ())), -abs(float(item["effect"])), item["id"]))
    for item in ordered:
        predicates = rule_keys(item)
        threshold = _response_threshold(str(item["target"]))
        duplicate = any(
            prior["engine"] == item["engine"] and prior["target"] == item["target"]
            and float(prior["effect"]) * float(item["effect"]) > 0
            and rule_keys(prior).issubset(predicates)
            and abs(float(prior["effect"]) - float(item["effect"])) < threshold
            for prior in kept
        )
        if not duplicate:
            kept.append(item)
    return kept


def _clock_evidence(item: Mapping[str, Any], *, config: BriefConfig) -> dict[str, Any]:
    target_label, unit = _target_label(item["target"])
    direction = "higher" if item["effect"] > 0 else "lower"
    width_label = _clock_label(int(item["startHour"]), int(item["width"]))
    confirmation = item["confirmation"]
    positives = sum(1 for row in confirmation if row["difference"] > 0)
    negatives = len(confirmation) - positives
    effect_display = item["effect"] * 100.0 if unit.startswith("fraction") else item["effect"]
    low_display = item["low"] * 100.0 if unit.startswith("fraction") else item["low"]
    high_display = item["high"] * 100.0 if unit.startswith("fraction") else item["high"]
    comparison_unit = "percentage points" if unit.startswith("fraction") else unit
    episodes: list[dict[str, Any]] = []
    for row in confirmation[: max(1, config.max_evidence_episodes // 2)]:
        episodes.append({
            "id": f"{item['id']}-selected-{row['date'].isoformat()}", "label": f"Selected interval · {row['date'].isoformat()}",
            "group": "clock_window", "start": _iso(row["start"]),
            "outcomeEnd": _iso(_add_elapsed(row["start"], hours=int(item["width"]))),
            "points": row["observed"].get("points", []), "events": [],
        })
        episodes.append({
            "id": f"{item['id']}-comparison-{row['date'].isoformat()}", "label": f"Adjacent comparison · {row['date'].isoformat()}",
            "group": "comparison", "start": _iso(row["comparisonStart"]),
            "outcomeEnd": _iso(_add_elapsed(row["comparisonStart"], hours=int(item["width"]))),
            "points": row["comparisonOutcome"].get("points", []), "events": [],
        })
    observed_summary = _outcome_summary(
        confirmation, "value", label=target_label,
        horizon=f"clock_{item['startHour']:02d}_{item['width']}h", unit=unit,
    )
    if observed_summary and unit.startswith("fraction"):
        observed_summary = {
            **observed_summary,
            "unit": "%",
            "median": observed_summary["median"] * 100.0,
            "lower": observed_summary["lower"] * 100.0,
            "upper": observed_summary["upper"] * 100.0,
        }
    evidence = _make_evidence(
        title="Recurring clock interval", source_label="Recorded CGM observations",
        window_label=f"{width_label} local · later complete days", summary=(
            f"Between {width_label}, recorded {target_label.lower()} was {abs(effect_display):.1f} {comparison_unit} {direction} than the adjacent same-duration interval on later comparable days."
        ), limitations=[
            "This is a retrospective recurring observation, not a cause, forecast, or treatment recommendation.",
            "The comparison uses equal-duration adjacent windows and observed bins only; missing bins remain unknown.",
            "The interval is selected from a frozen global confirmation family. Uncertainty is approximate contiguous-day resampling with a Bonferroni-adjusted interval; future daily snapshots are not anytime-valid.",
        ], facts=[
            {"label": "Observed interval", "value": width_label},
            {"label": "Difference", "value": f"{effect_display:+.1f} {comparison_unit}"},
            {"label": "Days in later comparison", "value": str(len(confirmation))},
            {"label": "Approximate adjusted interval", "value": f"[{low_display:+.1f}, {high_display:+.1f}] {comparison_unit}"},
            {"label": "Direction agreement", "value": f"{max(positives, negatives)} of {len(confirmation)} days"},
        ], analysis_kind="clock_window_recurrence", evidence_level="retrospectively_replicated", method_label="frozen 2/3/4-hour clock family + adjacent-window comparison",
        availability_mode="retrospective_reconstruction", support={"unit": "complete days", "count": len(confirmation), "comparisonCount": len(confirmation), "pairCount": len(confirmation)},
        outcome_summary=[observed_summary] if observed_summary else None,
        episodes=episodes, comparison={"outcome": item["target"], "unit": comparison_unit, "effect": effect_display, "low": low_display, "high": high_display, "rangeKind": "approximate_adjusted_block_interval", "contextN": len(confirmation), "comparisonN": len(confirmation), "pairCount": len(confirmation)},
        matched_on=["same local day", "same-duration adjacent interval"],
    )
    if item["effect"] > 0:
        higher_start = float(item["startHour"])
    elif confirmation:
        comparator = confirmation[0]["comparisonStart"]
        higher_start = comparator.hour + comparator.minute / 60.0
    else:
        higher_start = float(item["startHour"])
    support_dates = [row["date"].isoformat() for row in confirmation]
    evidence["presentationCoverage"] = {
        "kind": "clock_window", "complete": bool(support_dates)
        and len(support_dates) == len(set(support_dates)),
        "outcome": str(item["target"]), "orientation": "higher_outcome",
        "higherWindow": {"startHour": higher_start, "widthHours": float(item["width"])},
        "confirmationSupportDates": support_dates,
    }
    return evidence


def _overnight_bounds(anchor: datetime, zone: ZoneInfo) -> tuple[datetime, datetime]:
    local = anchor.astimezone(zone)
    night_day = local.date() - timedelta(days=1) if local.hour < 6 else local.date()
    start = _local_datetime(night_day, time(22, 0), zone)
    end = _local_datetime(night_day + timedelta(days=1), time(6, 0), zone)
    if end <= anchor:
        start = _local_datetime(night_day + timedelta(days=1), time(22, 0), zone)
        end = _local_datetime(night_day + timedelta(days=2), time(6, 0), zone)
    if start < anchor:
        start = anchor
    return start, end


def _analogue_outcomes(
    records: Sequence[BriefRecord], anchor: datetime, *, cutoff: datetime, zone: ZoneInfo,
    config: BriefConfig, index: _RecordIndex | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    h3 = _add_elapsed(anchor, hours=3); h6 = _add_elapsed(anchor, hours=6); h24 = _add_elapsed(anchor, hours=24)
    overnight_start, overnight_end = _overnight_bounds(anchor, zone)
    for name, start, end in (("0_3h", anchor, h3), ("3_6h", h3, h6), ("overnight", overnight_start, overnight_end), ("24h", anchor, h24)):
        if end > cutoff:
            continue
        outcome = _glucose_outcome(records, start, end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        if outcome is None:
            continue
        observed_minutes = max(1.0, float(outcome.get("observedBins", 0)) * 5.0)
        output[name] = {
            "mean": float(outcome["mean"]), "lowRate": float(outcome["lowMinutes"]) / observed_minutes,
            "tir": float(outcome["tir"]), "variance": float(outcome["variance"]),
            "points": outcome["points"], "start": start, "end": end,
            "coverage": float(outcome["coverage"]), "observedBins": int(outcome["observedBins"]),
            "expectedBins": int(outcome["expectedBins"]),
            "runs": {kind: _observed_runs(outcome["points"], start=start, kind=kind) for kind in ("low", "high")},
        }
    return output


def _analogue_rows(
    records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, query_anchor: datetime,
    config: BriefConfig, availability_mode: str = "strict_as_of_anchor",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = _RecordIndex.build(records)
    current = _anchor_features(records, query_anchor, cutoff=cutoff, zone=zone, min_coverage=config.min_coverage, index=index, availability_mode=availability_mode)
    days = _day_candidates(records, zone, _local_date(query_anchor, zone))
    rows: list[dict[str, Any]] = []
    local_query = query_anchor.astimezone(zone)
    for day in days:
        anchor = _local_datetime(day, time(local_query.hour, local_query.minute), zone)
        if anchor >= query_anchor:
            continue
        features = _anchor_features(records, anchor, cutoff=cutoff, zone=zone, min_coverage=config.min_coverage, index=index, availability_mode=availability_mode)
        outcomes = _analogue_outcomes(records, anchor, cutoff=cutoff, zone=zone, config=config, index=index)
        if not outcomes:
            continue
        rows.append({"date": day, "anchor": anchor, "features": features, "outcomes": outcomes})
    return current, rows


def _robust_scales(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    keys = tuple(_FEATURE_FAMILY)
    scales: dict[str, tuple[float, float]] = {}
    for key in keys:
        if key == "site_location":
            continue
        values = [float(row["features"].get(key)) for row in rows if _finite(row["features"].get(key)) is not None]
        center = _median(values)
        spread = _quantile(values, 0.75) - _quantile(values, 0.25) if len(values) >= 2 else None
        scales[key] = (float(center or 0.0), max(float(spread or 0.0) / 1.349, 1e-6))
    return scales


_FEATURE_FAMILY = {
    "glucose_1h_mean": "glucose", "glucose_1h_slope": "glucose",
    "glucose_3h_mean": "glucose", "glucose_3h_slope": "glucose", "glucose_3h_variability": "glucose",
    "glucose_3h_low_rate": "glucose", "glucose_3h_high_rate": "glucose", "glucose_6h_mean": "glucose",
    "glucose_24h_mean": "glucose", "glucose_7d_vs_prior28_mean": "glucose",
    "delivered_bolus_3h": "insulin", "delivered_bolus_6h": "insulin", "delivered_bolus_24h": "insulin",
    "delivered_bolus_7d_daily": "insulin", "delivered_bolus_prior28_daily": "insulin",
    "meal_carbs_3h": "carbohydrate", "meal_carbs_6h": "carbohydrate", "meal_carbs_24h": "carbohydrate",
    "meal_carbs_7d_daily": "carbohydrate", "meal_carbs_prior28_daily": "carbohydrate",
    "exercise_minutes_6h": "activity", "exercise_minutes_24h": "activity",
    "exercise_minutes_7d_daily": "activity", "exercise_minutes_prior28_daily": "activity",
    "steps_6h": "activity", "steps_24h": "activity", "steps_7d_daily": "activity", "steps_prior28_daily": "activity",
    "steps_daytime_24h": "activity", "steps_overnight_24h": "activity",
    "heart_rate_6h_mean": "physiology", "heart_rate_24h_mean": "physiology",
    "heart_rate_7d_mean": "physiology", "heart_rate_prior28_mean": "physiology",
    "heart_rate_deviation_24h": "physiology", "hrv_sdnn_24h_mean": "physiology", "hrv_sdnn_deviation_24h": "physiology",
    "workout_minutes_6h": "activity", "workout_minutes_24h": "activity",
    "workout_average_hr_24h": "physiology", "workout_count_24h": "activity",
    "evening_workout": "activity",
    "sleep_hours": "sleep", "sleep_7d_mean": "sleep", "sleep_prior28_mean": "sleep",
    "sleep_variability_7d": "sleep", "sleep_shortfall_7d_hours": "sleep",
    "sleep_rem_fraction": "sleep", "sleep_deep_fraction": "sleep",
    "sleep_fragmentation_per_hour": "sleep", "sleep_stage_efficiency": "sleep",
    "mood_valence_24h": "mood", "mood_arousal_24h": "mood",
    "mood_valence_last": "mood", "mood_valence_last_age_hours": "mood",
    "mood_arousal_last": "mood", "mood_arousal_last_age_hours": "mood",
    "mood_valence_24h_std": "mood", "mood_arousal_24h_std": "mood",
    "mood_valence_7d_mean": "mood", "mood_valence_7d_std": "mood",
    "mood_arousal_7d_mean": "mood", "mood_arousal_7d_std": "mood",
    "mood_valence_7d_vs_prior28": "mood", "mood_arousal_7d_vs_prior28": "mood",
    "mood_valence_24h_mean": "mood", "mood_arousal_24h_mean": "mood",
    "body_temperature_delta": "temperature", "wrist_temperature_delta": "temperature",
    "body_temperature_7d_vs_prior28": "temperature", "wrist_temperature_7d_vs_prior28": "temperature",
    "days_since_period": "cycle", "site_age_days": "site",
    "cycle_days": "cycle", "cycle_last_interval_days": "cycle", "cycle_median_interval_days": "cycle",
    "cycle_interval_std_days": "cycle", "cycle_elapsed_vs_prior_median_days": "cycle",
    "site_previous_dwell_days": "site", "site_median_dwell_days": "site", "site_dwell_std_days": "site",
    "site_age_vs_prior_median_days": "site", "site_location": "site",
    "flow_severity_7d_mean": "cycle", "flow_severity_last_7d": "cycle", "flow_last_age_hours": "cycle",
    "cramp_severity_24h_mean": "cycle", "cramp_severity_last_7d": "cycle", "cramp_last_age_hours": "cycle",
}

_FEATURE_LABEL = {
    "glucose_1h_mean": "1-hour glucose level", "glucose_1h_slope": "1-hour glucose trend",
    "glucose_3h_mean": "3-hour glucose level", "glucose_3h_slope": "3-hour glucose trend",
    "glucose_3h_variability": "3-hour glucose variability", "glucose_3h_low_rate": "recent observed low burden",
    "glucose_3h_high_rate": "recent observed high burden", "glucose_6h_mean": "6-hour glucose level",
    "glucose_24h_mean": "prior-day glucose level", "glucose_7d_vs_prior28_mean": "recent versus prior glucose level",
    "delivered_bolus_3h": "3-hour recorded delivered bolus", "delivered_bolus_6h": "6-hour recorded delivered bolus",
    "delivered_bolus_24h": "24-hour recorded delivered bolus", "delivered_bolus_7d_daily": "recent recorded delivered bolus",
    "delivered_bolus_prior28_daily": "prior recorded delivered bolus", "meal_carbs_3h": "3-hour recorded carbohydrates",
    "meal_carbs_6h": "6-hour recorded carbohydrates", "meal_carbs_24h": "24-hour recorded carbohydrates",
    "meal_carbs_7d_daily": "recent recorded carbohydrates", "meal_carbs_prior28_daily": "prior recorded carbohydrates",
    "exercise_minutes_6h": "6-hour recorded activity", "exercise_minutes_24h": "24-hour recorded activity",
    "exercise_minutes_7d_daily": "recent recorded activity", "exercise_minutes_prior28_daily": "prior recorded activity",
    "steps_6h": "6-hour recorded steps", "steps_24h": "24-hour recorded steps",
    "steps_7d_daily": "recent recorded steps", "steps_prior28_daily": "prior recorded steps",
    "steps_daytime_24h": "recorded daytime steps", "steps_overnight_24h": "recorded overnight steps",
    "heart_rate_6h_mean": "6-hour recorded heart rate", "heart_rate_24h_mean": "prior-day recorded heart rate",
    "heart_rate_7d_mean": "recent recorded heart rate", "heart_rate_prior28_mean": "prior recorded heart rate",
    "heart_rate_deviation_24h": "24-hour heart-rate deviation from earlier personal baseline",
    "hrv_sdnn_24h_mean": "24-hour recorded HRV SDNN", "hrv_sdnn_deviation_24h": "24-hour HRV SDNN deviation from earlier personal baseline",
    "workout_minutes_6h": "6-hour recorded workout duration", "workout_minutes_24h": "24-hour recorded workout duration",
    "workout_average_hr_24h": "recorded workout average heart rate", "workout_count_24h": "recorded workout count",
    "evening_workout": "recorded evening workout",
    "sleep_hours": "latest completed sleep",
    "sleep_variability_7d": "completed sleep variability over the preceding 7 days",
    "sleep_shortfall_7d_hours": "recorded sleep shortfall over the preceding 7 days versus the earlier personal baseline",
    "sleep_rem_fraction": "latest completed sleep REM fraction", "sleep_deep_fraction": "latest completed sleep deep-sleep fraction",
    "sleep_fragmentation_per_hour": "latest completed sleep awake bouts per asleep hour",
    "sleep_stage_efficiency": "latest completed sleep efficiency from recorded stages",
    "mood_valence_24h": "24-hour recorded mood valence", "mood_arousal_24h": "24-hour recorded mood arousal",
    "mood_valence_last": "last recorded mood valence", "mood_valence_last_age_hours": "age of last recorded mood valence",
    "mood_arousal_last": "last recorded mood arousal", "mood_arousal_last_age_hours": "age of last recorded mood arousal",
    "mood_valence_24h_std": "24-hour mood-valence variability", "mood_arousal_24h_std": "24-hour mood-arousal variability",
    "mood_valence_7d_mean": "7-day recorded mood valence", "mood_valence_7d_std": "7-day mood-valence variability",
    "mood_arousal_7d_mean": "7-day recorded mood arousal", "mood_arousal_7d_std": "7-day mood-arousal variability",
    "mood_valence_7d_vs_prior28": "recent versus earlier observed mood valence",
    "mood_arousal_7d_vs_prior28": "recent versus earlier observed mood arousal",
    "sleep_7d_mean": "recent completed sleep", "sleep_prior28_mean": "prior completed sleep",
    "body_temperature_delta": "body temperature deviation", "wrist_temperature_delta": "sleeping-wrist temperature deviation",
    "body_temperature_7d_vs_prior28": "recent versus prior body temperature",
    "wrist_temperature_7d_vs_prior28": "recent versus prior sleeping-wrist temperature",
    "days_since_period": "time since recorded cycle onset", "cycle_days": "time since recorded cycle onset",
    "cycle_last_interval_days": "last completed recorded cycle interval", "cycle_median_interval_days": "median completed recorded cycle interval",
    "cycle_interval_std_days": "variability in completed recorded cycle intervals", "cycle_elapsed_vs_prior_median_days": "current cycle elapsed time versus earlier completed intervals",
    "site_age_days": "time since recorded site change", "site_previous_dwell_days": "previous completed site dwell",
    "site_median_dwell_days": "median completed site dwell", "site_dwell_std_days": "variability in completed site dwell",
    "site_age_vs_prior_median_days": "current site age versus prior completed dwell", "site_location": "recorded infusion-site location",
    "flow_severity_7d_mean": "recorded menstrual flow severity over the preceding week",
    "flow_severity_last_7d": "last recorded menstrual flow severity", "flow_last_age_hours": "age of last recorded flow day",
    "cramp_severity_24h_mean": "recorded cramp severity over the preceding day",
    "cramp_severity_last_7d": "last recorded cramp severity", "cramp_last_age_hours": "age of last recorded cramp observation",
    "therapy_profile_fingerprint": "same observed therapy profile",
}

_ABSOLUTE_CALIPER = {
    "glucose_1h_mean": 60.0, "glucose_3h_mean": 60.0, "glucose_6h_mean": 60.0, "glucose_24h_mean": 70.0,
    "glucose_1h_slope": 30.0, "glucose_3h_slope": 30.0,
    "delivered_bolus_3h": 6.0, "delivered_bolus_6h": 8.0, "delivered_bolus_24h": 20.0,
    "meal_carbs_3h": 60.0, "meal_carbs_6h": 90.0, "meal_carbs_24h": 180.0,
    "body_temperature_delta": 1.5, "wrist_temperature_delta": 1.5,
    "body_temperature_7d_vs_prior28": 1.5, "wrist_temperature_7d_vs_prior28": 1.5,
}


def _analogue_distance(current: Mapping[str, Any], historical: Mapping[str, Any], scales: Mapping[str, tuple[float, float]], threshold: float) -> tuple[float, list[str], list[str], list[str], int]:
    groups: dict[str, list[float]] = defaultdict(list)
    matched: list[str] = []; differences: list[str] = []; unavailable: list[str] = []
    # A current analogue query must have an observed 3-hour CGM context, and
    # every historical neighbor must have the same mature query window.  This
    # permits honest glucose-only analogues while refusing incomplete anchors.
    if not current.get("glucose_valid") or not historical.get("glucose_valid"):
        return float("inf"), matched, differences, unavailable, len(groups)
    current_profile = current.get("therapy_profile_fingerprint")
    historical_profile = historical.get("therapy_profile_fingerprint")
    if current_profile is not None and historical_profile is not None:
        if current_profile != historical_profile:
            return float("inf"), matched, ["therapy_profile_fingerprint"], unavailable, len(groups)
        groups["insulin"].append(0.0)
        matched.append("therapy_profile_fingerprint")
    else:
        unavailable.append("therapy_profile_fingerprint")
    for key, family in _FEATURE_FAMILY.items():
        if key == "site_location":
            left_category = current.get(key); right_category = historical.get(key)
            if isinstance(left_category, str) and isinstance(right_category, str):
                difference = left_category != right_category
                groups[family].append(1.0 if difference else 0.0)
                matched.append(key)
                if difference:
                    differences.append(key)
            else:
                unavailable.append(key)
            continue
        left = _finite(current.get(key)); right = _finite(historical.get(key))
        if left is None or right is None:
            unavailable.append(key)
            continue
        center, scale = scales.get(key, (0.0, 1.0))
        absolute_caliper = _ABSOLUTE_CALIPER.get(key)
        if absolute_caliper is not None and abs(float(left) - float(right)) > absolute_caliper:
            return float("inf"), matched, differences, unavailable, len(groups)
        z = abs((float(left) - float(right)) / max(scale, 1e-6))
        if z > 3.5:
            return float("inf"), matched, differences, unavailable, len(groups)
        groups[family].append(z)
        matched.append(key)
        if z > 1.5:
            differences.append(key)
    if len(groups) < 2:
        # Glucose-only histories are useful, but the missing-context penalty
        # and tighter absolute threshold must remain visible to callers.
        if "glucose" not in groups:
            return float("inf"), matched, differences, unavailable, len(groups)
        glucose_values = groups["glucose"]
        if len(glucose_values) < 3:
            return float("inf"), matched, differences, unavailable, len(groups)
        distance = math.sqrt(sum(value * value for value in glucose_values) / len(glucose_values))
        distance += 0.75 * (len(_FAMILY_NAMES) - len(groups)) / len(_FAMILY_NAMES)
        if distance > threshold * 0.75:
            return float("inf"), matched, differences, unavailable, len(groups)
        return distance, matched, differences, unavailable, len(groups)
    family_values = [sum(values) / len(values) for values in groups.values()]
    missing_penalty = 0.75 * (len(_FAMILY_NAMES) - len(groups)) / len(_FAMILY_NAMES)
    distance = math.sqrt(sum(value * value for value in family_values) / len(family_values)) + missing_penalty
    if distance > threshold:
        return float("inf"), matched, differences, unavailable, len(groups)
    return distance, matched, differences, unavailable, len(groups)


def _analogue_episode(records: Sequence[BriefRecord], row: Mapping[str, Any], *, cutoff: datetime) -> dict[str, Any]:
    anchor = row["anchor"]
    index = _RecordIndex.build(records)
    end = _add_elapsed(anchor, hours=24)
    points, _ = _episode_trace(records, anchor, end, cutoff, anchor=anchor, index=index)
    events = _episode_events(records, anchor, end, anchor, cutoff=cutoff)
    return {
        "id": f"analogue-{row['date'].isoformat()}", "label": f"Recorded day · {row['date'].isoformat()}",
        "group": "similar_history", "start": _iso(anchor),
        "outcomeEnds": {horizon: _iso(outcome["end"]) for horizon, outcome in row["outcomes"].items()},
        "outcomeWindows": {horizon: {
            "start": _iso(outcome.get("start", anchor)), "end": _iso(outcome["end"]),
            "coverage": outcome.get("coverage"), "observedBins": outcome.get("observedBins"),
            "expectedBins": outcome.get("expectedBins"), "runs": outcome.get("runs", {"low": [], "high": []}),
        } for horizon, outcome in row["outcomes"].items()},
        "points": points, "events": events,
    }


def _analogue_horizon_label(horizon: str, rows: Sequence[Mapping[str, Any]], zone: ZoneInfo) -> str:
    """Use shared local clock labels only when every eligible window has them."""
    if horizon == "24h":
        return "during the following 24 hours"
    bounds = []
    for row in rows:
        outcome = row["outcomes"][horizon]
        start = outcome["start"].astimezone(zone); end = outcome["end"].astimezone(zone)
        if start.utcoffset() != end.utcoffset():
            break
        bounds.append((start.strftime("%H:%M"), end.strftime("%H:%M")))
    if len(bounds) == len(rows) and len(set(bounds)) == 1 and bounds:
        return f"between {bounds[0][0]} and {bounds[0][1]} local time"
    return {
        "0_3h": "during the following 0–3 hours", "3_6h": "during the following 3–6 hours",
        "overnight": "during the later overnight window", "24h": "during the following 24 hours",
    }[horizon]


def _retrieve_analogues(
    records: Sequence[BriefRecord], *, cutoff: datetime, zone: ZoneInfo, query_anchor: datetime,
    config: BriefConfig, availability_mode: str = "strict_as_of_anchor",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    current, rows = _analogue_rows(records, cutoff=cutoff, zone=zone, query_anchor=query_anchor, config=config, availability_mode=availability_mode)
    diagnostics = {"candidateCount": len(rows), "acceptedCount": 0, "status": "insufficient"}
    if not rows:
        return None, diagnostics
    scales = _robust_scales(rows)
    accepted: list[dict[str, Any]] = []
    for row in rows:
        distance, matched, differences, unavailable, family_count = _analogue_distance(current, row["features"], scales, config.v3_analogue_distance)
        if math.isfinite(distance):
            accepted.append({**row, "distance": distance, "matched": matched, "differences": differences, "unavailable": unavailable, "familyCount": family_count})
    accepted.sort(key=lambda row: (row["distance"], row["date"]))
    deduped: list[dict[str, Any]] = []
    for row in accepted:
        candidate_24 = row["outcomes"].get("24h")
        overlaps = False
        if candidate_24 is not None:
            for prior in deduped:
                prior_24 = prior["outcomes"].get("24h")
                if prior_24 is None:
                    continue
                if candidate_24["start"].astimezone(UTC) < prior_24["end"].astimezone(UTC) and prior_24["start"].astimezone(UTC) < candidate_24["end"].astimezone(UTC):
                    overlaps = True
                    break
        if not overlaps:
            deduped.append(row)
    accepted = deduped[: max(config.v3_analogue_neighbors, config.v3_min_analogue_neighbors)]
    if len(accepted) < config.v3_min_analogue_neighbors:
        diagnostics.update({"acceptedCount": len(accepted), "status": "out_of_distribution"})
        return None, diagnostics
    weights = [1.0 / (1.0 + float(row["distance"]) ** 2) for row in accepted]
    effective = sum(weights) ** 2 / max(1e-9, sum(weight * weight for weight in weights))
    if effective < max(2.0, config.v3_min_analogue_neighbors - 0.25):
        diagnostics.update({"acceptedCount": len(accepted), "effectiveCount": effective, "status": "insufficient_effective_neighbors"})
        return None, diagnostics
    summaries: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    horizon_support: dict[str, dict[str, float | int | str]] = {}
    for horizon in _OUTCOME_ORDER:
        horizon_rows = [row for row in accepted if horizon in row["outcomes"]]
        values = [row["outcomes"][horizon]["mean"] for row in horizon_rows]
        horizon_weights = [1.0 / (1.0 + float(row["distance"]) ** 2) for row in horizon_rows]
        horizon_effective = sum(horizon_weights) ** 2 / max(1e-9, sum(weight * weight for weight in horizon_weights)) if horizon_weights else 0.0
        ready = len(values) >= config.v3_min_analogue_neighbors and horizon_effective >= max(2.0, config.v3_min_analogue_neighbors - 0.25)
        if horizon_rows:
            label = _analogue_horizon_label(horizon, horizon_rows, zone)
            for kind in ("low", "high"):
                event_dates = sorted({row["date"].isoformat() for row in horizon_rows if row["outcomes"][horizon].get("runs", {}).get(kind)})
                event_summaries.append({
                    "horizon": horizon, "horizonLabel": label, "outcomeKind": kind,
                    "eventDays": len(event_dates), "eligibleDays": len({row["date"] for row in horizon_rows}),
                    "thresholdMgDl": 70 if kind == "low" else 180,
                    "minimumDurationMinutes": 15, "supportStatus": "ready" if ready else "insufficient",
                    "eligibleDates": sorted({row["date"].isoformat() for row in horizon_rows}),
                    "eventDates": event_dates,
                })
        if not ready:
            horizon_support[horizon] = {"count": len(values), "effectiveCount": horizon_effective, "status": "insufficient"}
            continue
        horizon_support[horizon] = {"count": len(values), "effectiveCount": horizon_effective, "status": "ready"}
        summaries.append({
            "label": "Mean glucose after similar recorded situations", "horizon": horizon, "unit": "mg/dL",
            "median": float(_median(values)), "lower": float(_quantile(values, 0.25)), "upper": float(_quantile(values, 0.75)),
            "rangeKind": "observed_interquartile", "count": len(values),
        })
    if not summaries:
        diagnostics.update({"acceptedCount": len(accepted), "effectiveCount": effective, "status": "immature_outcomes", "horizonSupport": horizon_support})
        return None, diagnostics
    episodes = [_analogue_episode(records, row, cutoff=cutoff) for row in accepted]
    matched_sets = [set(row["matched"]) for row in accepted]
    matched_keys = sorted(set.intersection(*matched_sets)) if matched_sets else []
    difference_keys = sorted({key for row in accepted for key in row["differences"]})
    unavailable_keys = sorted({key for row in accepted for key in row["unavailable"]})
    matched = [_FEATURE_LABEL.get(key, key.replace("_", " ")) for key in matched_keys]
    differences = [_FEATURE_LABEL.get(key, key.replace("_", " ")) for key in difference_keys]
    unavailable = [_FEATURE_LABEL.get(key, key.replace("_", " ")) for key in unavailable_keys]
    unavailable.append("measured basal delivery")
    matched_observed_values: list[dict[str, Any]] = []
    for key in matched_keys:
        definition = NUMERIC_FEATURES.get(key)
        current_value = _finite(current.get(key))
        historical_values = [
            value for row in accepted
            if (value := _finite(row["features"].get(key))) is not None
        ]
        if definition is None or current_value is None or not historical_values:
            continue
        matched_observed_values.append({
            "feature": key, "label": _FEATURE_LABEL.get(key, definition[0]), "unit": definition[1],
            "currentValue": current_value, "historicalMedian": float(_median(historical_values)),
            "historicalLowerQuartile": float(_quantile(historical_values, 0.25)),
            "historicalUpperQuartile": float(_quantile(historical_values, 0.75)),
            "historicalCount": len(historical_values),
        })
    first = summaries[0]
    evidence = _make_evidence(
        title="Similar recorded situations", source_label="Recorded CGM and available context/event observations",
        window_label="Same local clock hour · mature historical outcomes", summary=(
            f"{len(accepted)} earlier recorded situations were comparable on {len(matched)} observed feature fields. Their subsequent {first['horizon']} mean glucose had an observed middle 50% from {first['lower']:.0f} to {first['upper']:.0f} mg/dL."
        ), limitations=[
            "These are what followed on similar recorded days, not probabilities or a forecast.",
            (
                "Similarity uses only information recorded by each historical anchor; later insulin or glucose is shown as an outcome."
                if availability_mode == "strict_as_of_anchor"
                else "Similarity is a retrospective reconstruction: it may use pre-anchor observations imported later, but never post-anchor events or outcomes to select neighbors."
            ),
            "Unrecorded activity, meals, physiology, and other context can differ. Observed ranges are not confidence intervals; a glucose-only match has no evidence about missing context streams.",
            "Insulin-context fields use recorded delivered boluses and observed profile identity; measured basal delivery was unavailable.",
        ], facts=[
            {"label": "Comparable recorded situations", "value": str(len(accepted))},
            {"label": "Effective weighted count", "value": f"{effective:.1f}"},
            {"label": "Closest standardized distance", "value": f"{accepted[0]['distance']:.2f}"},
        ], analysis_kind="current_context_analogue", evidence_level="similar_history", method_label="family-balanced robust distance with coverage/caliper abstention",
        availability_mode=availability_mode, support={"unit": "historical anchors", "count": len(accepted)},
        similarity={
            "neighborCount": len(accepted), "effectiveCount": effective,
            "matchedFeatures": matched, "matchedObservedValues": matched_observed_values,
            "differences": differences, "unavailableFeatures": unavailable,
            "glucoseOnly": len({ _FEATURE_FAMILY.get(key) for key in matched_keys }) == 1,
        },
        outcome_summary=summaries, episodes=episodes, matched_on=matched,
    )
    evidence["eventSummaries"] = event_summaries
    diagnostics.update({"acceptedCount": len(accepted), "effectiveCount": effective, "status": "ready", "matchedFeatureCount": len(matched), "unavailableFeatureCount": len(unavailable), "horizonSupport": horizon_support})
    return {"evidence": evidence, "accepted": accepted, "effectiveCount": effective}, diagnostics


def _legacy_current_items(legacy: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []; evidence: dict[str, dict[str, Any]] = {}
    for item in legacy.get("items", []) or []:
        if item.get("kind") not in {"context", "trend"}:
            continue
        evidence_id = item.get("evidenceId")
        detail = (legacy.get("evidence", {}) or {}).get(evidence_id, {})
        if evidence_id and detail:
            evidence[evidence_id] = detail
            items.append(dict(item))
    return items, evidence


def _item_from_evidence(identifier: str, kind: str, title: str, text: str, evidence_id: str, evidence_level: str, *, message_facts: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {"id": identifier, "kind": kind, "title": title, "text": text, "evidenceId": evidence_id, "evidenceLevel": evidence_level}
    if message_facts is not None:
        item["messageFacts"] = message_facts
    return item


def _fact_item(identifier: str, kind: str, title: str, claim: str, evidence_id: str, evidence_level: str, *, required: Sequence[str], qualifiers: Sequence[str] = (), **typed: Any) -> dict[str, Any]:
    facts = {"version": 1, "kind": kind, "plainClaim": claim, "requiredPhrases": list(required), "allowedQualifiers": list(qualifiers), **typed}
    return _item_from_evidence(identifier, kind, title, f"{claim} [See recorded evidence](evidence://{evidence_id}).", evidence_id, evidence_level, message_facts=facts)


def _clock_item(item: Mapping[str, Any], evidence_id: str, *, zone: ZoneInfo) -> dict[str, Any]:
    label, unit = _target_label(item["target"])
    effect = float(item["effect"])
    direction = ("more" if effect > 0 else "less") if unit.startswith("fraction") else ("higher" if effect > 0 else "lower")
    interval = _clock_label(int(item["startHour"]), int(item["width"]))
    comparison_labels: set[str] = set()
    for row in item.get("confirmation", []):
        start = row["comparisonStart"].astimezone(zone)
        end = _add_elapsed(start, hours=int(item["width"])).astimezone(zone)
        if start.utcoffset() != end.utcoffset():
            comparison_labels.clear()
            break
        comparison_labels.add(f"{start:%H:%M}–{end:%H:%M}")
    comparison = next(iter(comparison_labels)) if len(comparison_labels) == 1 else "the neighboring same-length window"
    comparison_phrase = f"between {comparison}" if len(comparison_labels) == 1 else "in the neighboring same-length window"
    if unit.startswith("fraction"):
        claim = f"On the days compared, there was {direction} observed {label.lower()} between {interval} than {comparison_phrase}."
    else:
        claim = f"On the days compared, {label.lower()} was {direction} between {interval} than {comparison_phrase}."
    return _fact_item("brief-recurring-" + item["id"], "recurring", "A repeated clock interval", claim, evidence_id, "retrospectively_replicated", required=[interval, label.lower(), direction, comparison], qualifiers=["observed", "recorded", "on the days compared"], outcomeKind=item["target"], horizon=interval, horizonLabel=interval, direction=direction, comparisonLabel=comparison)


def _event_item(item: Mapping[str, Any], evidence_id: str) -> dict[str, Any]:
    condition = " and ".join(_condition_descriptions(item, event=False))
    descriptor = dict(item.get("outcomeDescriptor") or OUTCOME_DESCRIPTORS.get(str(item["target"]), {}))
    horizon = str(descriptor.get("window", "the recorded response window"))
    target_label = str(descriptor.get("label", item["target"]))
    unit = str(descriptor.get("unit", "mg/dL"))
    direction = "higher" if item["effect"] > 0 else "lower"
    claim = f"With {condition}, {target_label.lower()} was {direction} than in otherwise similar recorded episodes."
    if descriptor.get("episodeKind") == "pre_meal":
        claim = f"Before later recorded meals, when {condition}, the retrospective {target_label.lower()} was {direction} than in otherwise similar episodes."
    return _fact_item("brief-association-" + item["id"], "association", "A repeated recorded response", claim, evidence_id, "retrospectively_replicated", required=[condition, target_label.lower(), direction, "otherwise similar recorded episodes"], qualifiers=["recorded", "retrospective", "later comparable episodes"], outcomeKind=item["target"], outcomeUnit=unit, horizon=item["target"], horizonLabel=horizon, contextLabel=condition, direction=direction, comparisonLabel="otherwise similar recorded episodes", episodeKind=descriptor.get("episodeKind"), correctionClassification=item.get("correctionClassification"))


def _daily_item(item: Mapping[str, Any], evidence_id: str) -> dict[str, Any]:
    condition = " and ".join(_condition_descriptions(item, event=False))
    descriptor = dict(item.get("outcomeDescriptor") or OUTCOME_DESCRIPTORS.get(str(item["target"]), {}))
    horizon = str(descriptor.get("window", "the following 24 hours"))
    target_label = str(descriptor.get("label", item["target"]))
    unit = str(descriptor.get("unit", "mg/dL"))
    direction = "higher" if item["effect"] > 0 else "lower"
    claim = f"When records showed {condition}, {target_label.lower()} was {direction} than on otherwise similar days."
    return _fact_item("brief-association-" + item["id"], "association", "A repeated daily context", claim, evidence_id, "retrospectively_replicated", required=[condition, target_label.lower(), direction, horizon, "otherwise similar days"], qualifiers=["recorded", "later comparable days"], outcomeKind=item["target"], outcomeUnit=unit, horizon=item["target"], horizonLabel=horizon, contextLabel=condition, direction=direction, comparisonLabel="otherwise similar days")


def _analogue_item(result: Mapping[str, Any], evidence_id: str, *, outcome_first: bool = True) -> dict[str, Any] | None:
    evidence = result.get("evidence", {})
    summaries = evidence.get("eventSummaries") or []
    chosen = next((summary for kind in ("low", "high") for horizon in _OUTCOME_ORDER
                   for summary in summaries if summary["outcomeKind"] == kind
                   and summary["horizon"] == horizon and summary["supportStatus"] == "ready"
                   and summary["eventDays"] >= 2), None)
    if chosen is None:
        if outcome_first:
            return None
        claim = "Earlier days with similar recorded context have observed glucose summaries to review."
        return _fact_item("brief-analogue", "analogue", "Similar days behind this moment", claim, evidence_id, "similar_history", required=["similar recorded context"], qualifiers=["earlier days"])
    kind = chosen["outcomeKind"]
    event_days = int(chosen["eventDays"]); eligible_days = int(chosen["eligibleDays"])
    threshold = int(chosen["thresholdMgDl"]); label = str(chosen["horizonLabel"])
    outcome_word = "Lows" if kind == "low" else "High glucose"
    context_label = "days with similar recent glucose" if evidence.get("similarity", {}).get("glucoseOnly") else "similar days"
    claim = f"{outcome_word} showed up on {event_days} of {eligible_days} {context_label} {label}."
    return _fact_item("brief-analogue", "analogue", "What followed on similar days", claim, evidence_id, "similar_history", required=[outcome_word, f"{event_days} of {eligible_days}", context_label, label], qualifiers=["CGM showed", "observed", "historical"], outcomeKind=kind, eventDays=event_days, eligibleDays=eligible_days, thresholdMgDl=threshold, minimumDurationMinutes=15, horizon=chosen["horizon"], horizonLabel=label, contextLabel=context_label)


def _current_context(records: Sequence[BriefRecord], *, anchor: datetime, zone: ZoneInfo, config: BriefConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    index = _RecordIndex.build(records)
    anchor_features = _anchor_features(records, anchor, cutoff=anchor, zone=zone, min_coverage=config.min_coverage, index=index)
    context = _daily_rule_context(
        anchor_features, anchor=anchor, zone=zone,
    )
    mean = _finite(context.get("glucose_3h_mean")); slope = _finite(context.get("glucose_3h_slope"))
    if mean is None or slope is None or not context.get("glucose_valid"):
        return None
    start = _add_elapsed(anchor, hours=-3)
    points, coverage = _episode_trace(records, start, anchor, anchor, anchor=start, index=index)
    if not points:
        return None
    direction = "rising" if slope > 2 else "falling" if slope < -2 else "fairly steady"
    extra_facts: list[dict[str, str]] = []
    phrases: list[str] = []
    sleep_records = [
        record for record in records if record.metric == "sleep_hours" and record.end is not None
        and _known_before(record, anchor) and _add_elapsed(anchor, hours=-36) <= record.end_or_start <= anchor
    ]
    if sleep_records:
        sleep_record = max(sleep_records, key=lambda record: record.end_or_start)
        local_start = sleep_record.start.astimezone(zone); local_end = sleep_record.end_or_start.astimezone(zone)
        extra_facts.append({
            "label": "Latest completed recorded sleep",
            "value": f"{sleep_record.value:.1f} hours · {local_start.isoformat(timespec='minutes')} to {local_end.isoformat(timespec='minutes')}",
        })
        phrases.append(f"{sleep_record.value:.1f} hours of completed sleep")
    for key, label in (("body_temperature_delta", "Body temperature versus recorded values in the previous two weeks"), ("wrist_temperature_delta", "Sleeping-wrist temperature versus recorded values in the previous two weeks")):
        value = _finite(context.get(key))
        if value is not None and len(extra_facts) < 3:
            extra_facts.append({"label": label, "value": f"{value:+.2f} °C"})
            phrases.append(f"{label.lower()} {value:+.2f} °C")
            break
    activity = _finite(context.get("exercise_minutes_24h")); steps = _finite(context.get("steps_24h"))
    if len(extra_facts) < 3 and (activity is not None or steps is not None):
        parts = []
        if activity is not None:
            parts.append(f"{activity:.0f} activity minutes")
        if steps is not None:
            parts.append(f"{steps:.0f} steps")
        value = " · ".join(parts)
        extra_facts.append({"label": "Recorded activity in previous 24 hours", "value": value})
        phrases.append(value)
    site_age = _finite(context.get("site_age_days"))
    if len(extra_facts) < 3 and site_age is not None:
        extra_facts.append({"label": "Time since recorded site change", "value": f"{site_age:.1f} days"})
        phrases.append(f"recorded site age {site_age:.1f} days")
    context_suffix = f" Also recorded: {'; '.join(phrases[:3])}." if phrases else ""
    evidence_id = "v3-current-context"
    evidence = _make_evidence(
        title="Recent recorded context", source_label="Recorded CGM and available context observations", window_label="Previous 3 hours and latest completed context",
        summary=f"Recorded glucose averaged {mean:.0f} mg/dL over the previous 3 hours and was {direction} across the observed samples.{context_suffix}",
        limitations=["This describes the recent recorded window; it is not a forecast or treatment recommendation.", "Missing samples remain unknown and are not interpolated as observations."],
        facts=[{"label": "Recorded 3-hour mean", "value": f"{mean:.0f} mg/dL"}, {"label": "Observed coverage", "value": f"{coverage * 100:.0f}%"}] + extra_facts,
        analysis_kind="current_recorded_context", evidence_level="observed_history", method_label="observed 3-hour CGM and available completed-context summary",
        availability_mode="strict_as_of_anchor", support={"unit": "observed CGM samples", "count": len(points)},
        episodes=[{"id": "current-context", "label": "Recent recorded window", "group": "current_context", "start": _iso(start), "outcomeEnd": _iso(anchor), "points": points, "events": []}],
    )
    claim = f"Over the previous 3 hours, recorded glucose averaged {mean:.0f} mg/dL and was {direction}."
    item = _fact_item(
        "brief-current-context", "context", "Your recent recorded context", claim,
        evidence_id, "observed_history", required=["previous 3 hours", f"{mean:.0f} mg/dL", direction],
        qualifiers=["recorded", "observed"], horizon="prior_3h", horizonLabel="previous 3 hours",
        direction=direction,
    )
    return item, evidence, context


def _finding_is_currently_relevant(finding: Mapping[str, Any], current: Mapping[str, Any], anchor: datetime, zone: ZoneInfo) -> bool:
    if finding.get("engine") == "clock":
        local = anchor.astimezone(zone)
        hour = local.hour + local.minute / 60.0
        return float(finding["startHour"]) <= hour < float(finding["startHour"] + finding["width"])
    if finding.get("engine") != "daily":
        # Meal-event findings require a current aligned meal episode; aggregate
        # 6-hour carbohydrate or bolus totals do not establish one.
        return False
    if finding.get("ruleDescriptors"):
        flags = [evaluate_rule(current, rule) for rule in finding["ruleDescriptors"]]
        return bool(flags) and all(flag is True for flag in flags)
    row = {"context": {
        "site_age_days": current.get("site_age_days"),
        "activity_7d_daily": current.get("exercise_minutes_7d_daily"),
        "body_temperature_context": current.get("body_temperature_7d_vs_prior28"),
        "wrist_temperature_context": current.get("wrist_temperature_7d_vs_prior28"),
    }}
    flags = [_event_flag(row, name, finding.get("predicateThresholds", {})) for name in finding.get("predicates", ())]
    return bool(flags) and all(flag is True for flag in flags)


def build_personal_context(
    records: Iterable[Mapping[str, Any]], *, as_of: datetime | str, timezone: str | None = None, config: BriefConfig | None = None,
) -> dict[str, Any]:
    """Build the v3 brief through the stable ``insite.brief.v1`` contract."""
    cfg = config or BriefConfig()
    raw_records = list(records)
    cutoff = _parse_datetime(as_of)
    normalized, report = normalize_brief_records(raw_records, availability_cutoff=cutoff)
    normalized = _collapse_sleep_intervals(normalized)
    zone = _safe_zone(timezone or (normalized[0].timezone if normalized else cfg.timezone or "UTC"))
    local_as_of = cutoff.astimezone(zone)
    # The current daily anchor rolls over at 06:00 local.  This is also the
    # freshness boundary: a record from an older local day cannot make a
    # stopped collection look current.
    current_day = local_as_of.date() if local_as_of.hour >= 6 else local_as_of.date() - timedelta(days=1)
    observed_days = [_local_date(record.start, zone) for record in normalized if record.metric == "glucose"]
    synthetic = any(bool(raw.get("syntheticDemo")) for raw in raw_records if isinstance(raw, Mapping))
    diagnostics: dict[str, Any] = {
        "datasetRevision": cfg.dataset_revision,
        "acceptedRecords": report.accepted, "rejectedRecords": report.rejected,
        "analysisOrigin": cutoff.date().isoformat(), "currentAnchor": current_day.isoformat(),
    }
    diagnostics.update({
        "engineRevision": ENGINE_REVISION, "availabilityMode": "as_of_recorded_availability",
        "contextAssociationAvailabilityMode": "strict_as_of_anchor",
        "analogueAvailabilityMode": str(getattr(cfg, "v3_availability_mode", "strict_as_of_anchor")),
        "acceptedRecords": report.accepted, "rejectedRecords": report.rejected,
        "v3MinimumUnitRule": f"at least {cfg.v3_min_units} complete units per discovery/confirmation arm",
        "featureFamilies": list(_FAMILY_NAMES),
    })
    if not normalized or not any(record.metric == "glucose" for record in normalized):
        return {
            "schema": BRIEF_SCHEMA, "generatedAt": _iso(cutoff), "asOf": _iso(cutoff),
            "timezone": zone.key, "synthetic": synthetic, "title": "Your day, in context",
            "intro": "There is not enough timestamped glucose history for a daily brief yet.",
            "status": "insufficient", "items": [], "exploreItems": [], "evidence": {},
            "evidenceFlags": {"heuristic": True, "method": ENGINE_REVISION},
            "analysisDiagnostics": diagnostics, "datasetRevision": cfg.dataset_revision,
            "engineRevision": ENGINE_REVISION,
        }
    if observed_days and max(observed_days) < current_day:
        return {
            "schema": BRIEF_SCHEMA, "generatedAt": _iso(cutoff), "asOf": _iso(cutoff),
            "timezone": zone.key, "synthetic": synthetic, "title": "Your day, in context",
            "intro": "Waiting for recent glucose data before making today's brief.",
            "status": "stale", "items": [], "exploreItems": [], "evidence": {},
            "evidenceFlags": {"heuristic": True, "method": ENGINE_REVISION},
            "analysisDiagnostics": diagnostics, "datasetRevision": cfg.dataset_revision,
            "engineRevision": ENGINE_REVISION,
        }
    items: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    clock_candidates, clock_context, clock_diagnostics = _prepare_clock_family(normalized, cutoff=cutoff, zone=zone, current_day=current_day, config=cfg)
    event_candidates, event_context, event_diagnostics = _prepare_event_family(normalized, cutoff=cutoff, zone=zone, config=cfg)
    daily_candidates, daily_context, daily_diagnostics = _prepare_daily_family(normalized, cutoff=cutoff, zone=zone, config=cfg)
    frozen_family = _bounded_global_family(
        clock_candidates=clock_candidates, event_candidates=event_candidates,
        daily_candidates=daily_candidates, maximum=cfg.v3_max_candidates,
    )
    for selected in frozen_family:
        selected_context = {
            "clock": clock_diagnostics,
            "event": event_context.get("search", {}),
            "daily": daily_context.get("search", {}),
        }.get(str(selected.get("engine")), {})
        per_target = selected_context.get("targetDiagnostics", {}) if isinstance(selected_context, Mapping) else {}
        target_diag = per_target.get(str(selected.get("target"))) if isinstance(per_target, Mapping) else None
        if isinstance(target_diag, dict):
            target_diag["selectedForConfirmation"] = int(target_diag.get("selectedForConfirmation", 0)) + 1
    clock_findings, event_findings, daily_findings = _confirm_global_family(
        frozen_family, event_context=event_context, clock_context=clock_context, daily_context=daily_context,
        cutoff=cutoff, zone=zone, config=cfg,
    ) if frozen_family else ([], [], [])
    event_supported_count = len(event_findings); daily_supported_count = len(daily_findings)
    event_findings = _dedupe_association_findings(event_findings)
    daily_findings = _dedupe_association_findings(daily_findings)
    global_family_size = len(frozen_family)
    clock_diagnostics.update({"tested": sum(item["engine"] == "clock" for item in frozen_family), "selected": len(clock_findings), "globalFamilySize": global_family_size})
    event_diagnostics.update({"tested": sum(item["engine"] == "event" for item in frozen_family), "supported": event_supported_count, "releasedAfterDeduplication": len(event_findings), "globalFamilySize": global_family_size})
    daily_diagnostics.update({"tested": sum(item["engine"] == "daily" for item in frozen_family), "supported": daily_supported_count, "releasedAfterDeduplication": len(daily_findings), "globalFamilySize": global_family_size})
    query_anchor = cutoff
    availability_mode = str(getattr(cfg, "v3_availability_mode", "strict_as_of_anchor"))
    analogue, analogue_diagnostics = _retrieve_analogues(normalized, cutoff=cutoff, zone=zone, query_anchor=query_anchor, config=cfg, availability_mode=availability_mode)
    diagnostics["clockWindows"] = clock_diagnostics; diagnostics["eventResponses"] = event_diagnostics
    diagnostics["dailyContextResponses"] = daily_diagnostics; diagnostics["analogueRetrieval"] = analogue_diagnostics
    diagnostics["globalConfirmationFamily"] = {
        "size": global_family_size, "candidateIds": [item["id"] for item in frozen_family],
        "discoverySearchCount": clock_diagnostics.get("discoverySearchCount", 0) + event_diagnostics.get("discoverySearchCount", 0) + daily_diagnostics.get("discoverySearchCount", 0),
        "adjustment": "Bonferroni-adjusted block percentile intervals", "alpha": cfg.alpha,
    }
    diagnostics["featureRegistryV3"] = {
        family: [key for key, registered_family in _FEATURE_FAMILY.items() if registered_family == family]
        for family in _FAMILY_NAMES
    }
    diagnostics["featureRegistryV3"]["insulin"].append("therapy_profile_fingerprint")
    patterns: list[dict[str, Any]] = []
    pattern_by_finding_id: dict[str, dict[str, Any]] = {}
    for finding in clock_findings:
        evidence_id = "v3-clock-" + finding["id"]
        evidence[evidence_id] = _clock_evidence(finding, config=cfg)
        pattern_by_finding_id[finding["id"]] = _clock_item(finding, evidence_id, zone=zone)
    for finding in event_findings:
        evidence_id = "v3-event-" + finding["id"]
        evidence[evidence_id] = _event_evidence(finding, target=finding["target"], config=cfg)
        pattern_by_finding_id[finding["id"]] = _event_item(finding, evidence_id)
    for finding in daily_findings:
        evidence_id = "v3-daily-" + finding["id"]
        evidence[evidence_id] = _daily_evidence(finding, config=cfg)
        pattern_by_finding_id[finding["id"]] = _daily_item(finding, evidence_id)
    current = _current_context(normalized, anchor=cutoff, zone=zone, config=cfg)
    generated: list[dict[str, Any]] = []
    context_item: dict[str, Any] | None = None
    if current is not None:
        current_item, current_evidence, current_features = current
        evidence[current_item["evidenceId"]] = current_evidence
        context_item = current_item
        relevant_daily_ids = {
            str(finding["id"])
            for finding in daily_findings
            if _finding_is_currently_relevant(finding, current_features, cutoff, zone)
        }
        diagnostics["todayContextRuleMatching"] = {
            "status": "evaluated", "confirmedDailyRulesChecked": len(daily_findings),
            "matchedConfirmedDailyRules": len(relevant_daily_ids),
            "featureBuilder": "shared_historical_current_daily_rule_context",
            "searchScope": "apply_frozen_confirmed_daily_rules_to_current_as_of_features",
        }
    else:
        current_features = {}
        relevant_daily_ids = set()
        diagnostics["todayContextRuleMatching"] = {
            "status": "current_context_unavailable", "confirmedDailyRulesChecked": 0,
            "matchedConfirmedDailyRules": 0,
            "featureBuilder": "shared_historical_current_daily_rule_context",
            "searchScope": "apply_frozen_confirmed_daily_rules_to_current_as_of_features",
        }
    supported_findings = [item for item in frozen_family if item["id"] in pattern_by_finding_id]
    current_relevant_ids = {
        str(finding["id"])
        for finding in supported_findings
        if _finding_is_currently_relevant(finding, current_features, cutoff, zone)
    }
    supported_findings.sort(
        key=lambda finding: (
            str(finding["id"]) not in current_relevant_ids,
            [item["id"] for item in frozen_family].index(finding["id"]),
        )
    )
    patterns = [pattern_by_finding_id[item["id"]] for item in supported_findings]
    analogue_item: dict[str, Any] | None = None
    analogue_explore: dict[str, Any] | None = None
    if analogue is not None:
        evidence_id = "v3-analogue"
        evidence[evidence_id] = analogue["evidence"]
        analogue_item = _analogue_item(analogue, evidence_id)
        if analogue_item is None:
            analogue_explore = _analogue_item(analogue, evidence_id, outcome_first=False)
    if analogue_item is not None:
        generated.append(analogue_item)
    generated.extend(patterns)
    # Current context and an uneventful mean/range summary remain reviewable
    # without occupying one of the useful main claims.
    combined = generated + ([analogue_explore] if analogue_explore else []) + ([context_item] if context_item else []) + items
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in combined:
        if item.get("id") in seen_ids:
            continue
        seen_ids.add(str(item.get("id"))); deduped.append(item)
    main: list[dict[str, Any]] = []
    main_kinds: set[str] = set()
    for item in generated:
        kind = str(item["kind"])
        if kind not in main_kinds and len(main) < 3:
            main.append(item); main_kinds.add(kind)
    main_ids = {item["id"] for item in main}
    explore = [item for item in deduped if item["id"] not in main_ids]
    valid_evidence_ids = {item.get("evidenceId") for item in deduped}
    evidence = {key: value for key, value in evidence.items() if key in valid_evidence_ids}
    result = {
        "schema": BRIEF_SCHEMA, "generatedAt": _iso(cutoff), "asOf": _iso(cutoff),
        "timezone": zone.key, "synthetic": synthetic, "title": "Your day, in context",
        "datasetRevision": cfg.dataset_revision,
    }
    result.update({
        "engineRevision": ENGINE_REVISION, "items": main, "exploreItems": explore, "evidence": evidence,
        "analysisDiagnostics": diagnostics,
        "evidenceFlags": {"heuristic": True, "method": ENGINE_REVISION, "retrospective": True, "causal": False},
    })
    if main:
        result["status"] = "ready"
        result["intro"] = "Observed outcomes and repeated patterns from your recorded history are linked below."
    elif result.get("status") not in {"stale", "insufficient"}:
        result["status"] = "quiet"
        result["intro"] = "No supported recurring or comparable-history finding is available from the recorded windows yet."
    relevant_item_ids = {
        str(pattern_by_finding_id[finding_id]["id"])
        for finding_id in current_relevant_ids if finding_id in pattern_by_finding_id
    }
    return present_brief(result, current_relevant_ids=relevant_item_ids)


__all__ = ["ENGINE_REVISION", "build_personal_context"]
