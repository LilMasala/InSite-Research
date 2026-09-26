"""Evidence-grounded daily briefs from timestamped personal observations.

This module is deliberately independent from Firebase, HealthKit, Firestore,
and language-model providers.  ``build_daily_brief`` accepts JSON-like
observations and returns the owner-readable ``insite.brief.v1`` document.  It
does not make treatment recommendations, forecast an unseen future, or fill a
missing stream with a negative value.

The implementation is intentionally conservative.  Context is computed only
from records known by the analysis cutoff, historical outcomes are computed
only after their complete horizon is available, and every displayed sentence
has an evidence sheet in the returned document.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import random
import statistics
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

BRIEF_SCHEMA = "insite.brief.v1"
ENGINE_REVISION = "daily-brief-v2-multiscale"
DATASET_REVISION = "synthetic-daily-brief-v2"
UTC = timezone.utc


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _iso(value: datetime | date) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _safe_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _local_date(value: datetime, zone: ZoneInfo) -> date:
    return value.astimezone(zone).date()


def _local_datetime(day: date, clock: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, clock, tzinfo=zone).astimezone(UTC)


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _median(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if _finite(value) is not None]
    return statistics.median(clean) if clean else None


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if _finite(value) is not None]
    return sum(clean) / len(clean) if clean else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _week(day: date) -> str:
    return (day - timedelta(days=day.weekday())).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert nested dataclass-adjacent values without allowing NaN."""
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@dataclass(frozen=True)
class BriefConfig:
    """Versioned analysis choices.

    ``min_pairs`` and ``min_weeks`` are deliberately explicit so small pilot
    accounts return ``insufficient`` rather than silently relaxing support.
    Tests may use a smaller configuration; the default is the production
    contract from the implementation plan.
    """

    timezone: str | None = None
    min_coverage: float = 0.80
    min_pairs: int = 12
    min_weeks: int = 6
    block_bootstrap_draws: int = 2000
    alpha: float = 0.05
    current_days: int = 7
    prior_days: int = 28
    long_prior_days: int = 56
    context_days: int = 56
    purge_days: int = 64
    max_evidence_episodes: int = 12
    subgroup_max_depth: int = 2
    subgroup_beam_width: int = 20
    max_locked_candidates: int = 10
    seed: int = 2718
    dataset_revision: str = DATASET_REVISION
    # The v3 personal-context engine is the default application path.  The
    # legacy implementation remains available for reproducibility and for
    # comparing old synthetic fixtures; it is never selected implicitly by a
    # worker or report entrypoint.
    engine_mode: str = "personal-context-v3"
    v3_min_units: int = 6
    v3_block_days: int = 2
    v3_max_candidates: int = 6
    # Native numeric subgroup search for event-aligned and daily responses.
    # Search boundaries and selectors are fitted on discovery data only.
    v3_search_depth: int = 4
    v3_search_bins: int = 6
    v3_search_beam_width: int = 256
    v3_search_shortlist: int = 128
    v3_window_widths: tuple[int, ...] = (2, 3, 4)
    v3_analogue_neighbors: int = 8
    v3_min_analogue_neighbors: int = 3
    v3_analogue_distance: float = 3.0
    v3_ridge_alpha: float = 4.0
    # Strict chronology is the default.  Retrospective reconstruction is an
    # explicit analysis mode for backfilled records and is labelled in the
    # resulting evidence; it is never silently inferred from availability.
    v3_availability_mode: str = "strict_as_of_anchor"
    # Percentile block intervals use a frozen confirmation family.  Keep at
    # least 20 draws in each Bonferroni tail at the default family bound.
    v3_bootstrap_draws: int = 10000

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_coverage": self.min_coverage,
            "min_pairs": self.min_pairs,
            "min_weeks": self.min_weeks,
            "block_bootstrap_draws": self.block_bootstrap_draws,
            "alpha": self.alpha,
            "current_days": self.current_days,
            "prior_days": self.prior_days,
            "long_prior_days": self.long_prior_days,
            "context_days": self.context_days,
            "purge_days": self.purge_days,
            "max_evidence_episodes": self.max_evidence_episodes,
            "subgroup_max_depth": self.subgroup_max_depth,
            "subgroup_beam_width": self.subgroup_beam_width,
            "max_locked_candidates": self.max_locked_candidates,
            "dataset_revision": self.dataset_revision,
            "engine_mode": self.engine_mode,
            "v3_min_units": self.v3_min_units,
            "v3_block_days": self.v3_block_days,
            "v3_max_candidates": self.v3_max_candidates,
            "v3_search_depth": self.v3_search_depth,
            "v3_search_bins": self.v3_search_bins,
            "v3_search_beam_width": self.v3_search_beam_width,
            "v3_search_shortlist": self.v3_search_shortlist,
            "v3_window_widths": list(self.v3_window_widths),
            "v3_analogue_neighbors": self.v3_analogue_neighbors,
            "v3_min_analogue_neighbors": self.v3_min_analogue_neighbors,
            "v3_analogue_distance": self.v3_analogue_distance,
            "v3_ridge_alpha": self.v3_ridge_alpha,
            "v3_availability_mode": self.v3_availability_mode,
            "v3_bootstrap_draws": self.v3_bootstrap_draws,
        }


@dataclass(frozen=True)
class BriefRecord:
    id: str
    metric: str
    value: float
    unit: str
    start: datetime
    end: datetime | None
    timezone: str
    source: str
    kind: str
    available_at: datetime | None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def known_at(self) -> datetime:
        # An interval's end is the minimum honest time at which its value can
        # be used.  available_at may be later, but never makes it earlier.
        return max(self.end or self.start, self.available_at or self.end or self.start)

    @property
    def local_start(self) -> datetime:
        return self.start.astimezone(_safe_zone(self.timezone))

    @property
    def end_or_start(self) -> datetime:
        """The completed endpoint used by interval-aware context logic."""
        return self.end or self.start


@dataclass(frozen=True)
class NormalizationReport:
    accepted: int
    rejected: int
    duplicate_ids: int
    wrong_units: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _RecordIndex:
    records: tuple[BriefRecord, ...]
    starts: tuple[datetime, ...]

    @classmethod
    def build(cls, records: Iterable[BriefRecord]) -> "_RecordIndex":
        ordered = tuple(sorted(records, key=lambda record: (record.start, record.id)))
        return cls(ordered, tuple(record.start for record in ordered))

    def between(self, start: datetime, end: datetime) -> tuple[BriefRecord, ...]:
        left = bisect_left(self.starts, start)
        right = bisect_left(self.starts, end)
        return self.records[left:right]


def _attr(record: BriefRecord, *names: str) -> Any:
    for name in names:
        if name in record.attributes:
            return record.attributes[name]
    return None


def _canonical_metric(metric: str) -> str:
    value = metric.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "blood_glucose": "glucose",
        "cgm": "glucose",
        "carbs": "meal_carbs",
        "carb_grams": "meal_carbs",
        "carbohydrate": "meal_carbs",
        "meal": "meal_carbs",
        "bolus": "delivered_bolus",
        "insulin_delivered": "delivered_bolus",
        "insulin_delivery": "delivered_bolus",
        "delivered_insulin": "delivered_bolus",
        "sleep": "sleep_hours",
        "exercise": "exercise_minutes",
        "wrist_temperature": "sleeping_wrist_temperature",
        "sleeping_wrist_temp": "sleeping_wrist_temperature",
        "temperature_wrist": "sleeping_wrist_temperature",
        "temperature_body": "body_temperature",
        "cycle_start": "period_onset",
        "menstrual_onset": "period_onset",
        "period_start": "period_onset",
        "exercise_session_minutes": "exercise_minutes",
    }
    return aliases.get(value, value)


_UNITS: dict[str, set[str]] = {
    "glucose": {"mg/dl", "mgdl", "milligrams_per_deciliter"},
    "meal_carbs": {"g", "gram", "grams"},
    "fasting_interval": {"h", "hr", "hour", "hours"},
    "delivered_bolus": {"u", "unit", "units", "iu"},
    "sleep_hours": {"h", "hr", "hour", "hours"},
    "sleep_asleep_interval": {"h", "hr", "hour", "hours"},
    "sleep_in_bed_interval": {"h", "hr", "hour", "hours"},
    "sleep_awake_interval": {"h", "hr", "hour", "hours"},
    "sleep_core_interval": {"h", "hr", "hour", "hours"},
    "sleep_deep_interval": {"h", "hr", "hour", "hours"},
    "sleep_rem_interval": {"h", "hr", "hour", "hours"},
    "sleep_unspecified_interval": {"h", "hr", "hour", "hours"},
    "hrv_sdnn": {"ms", "millisecond", "milliseconds"},
    "workout_minutes": {"min", "minute", "minutes"},
    "workout_average_hr": {"bpm", "beats_per_minute"},
    "workout_max_hr": {"bpm", "beats_per_minute"},
    "workout_energy_kcal": {"kcal", "kilocalorie", "kilocalories"},
    "mood_valence": {"score", "unitless"},
    "mood_arousal": {"score", "unitless"},
    "exercise_minutes": {"min", "minute", "minutes"},
    "steps": {"count", "counts", "step", "steps"},
    "heart_rate": {"bpm", "beats_per_minute"},
    "resting_hr": {"bpm", "beats_per_minute"},
    "body_temperature": {"c", "°c", "celsius", "degc"},
    "sleeping_wrist_temperature": {"c", "°c", "celsius", "degc"},
    "period_onset": {"event", "bool", "day", "days"},
    "exercise_coverage": {"fraction", "bool", "day", "days"},
}


def normalize_brief_records(
    records: Iterable[Mapping[str, Any]],
    *,
    availability_cutoff: datetime | str | None = None,
) -> tuple[list[BriefRecord], NormalizationReport]:
    """Parse timestamped records and reject malformed/incorrect units.

    Unknown auxiliary metrics are retained only when they have a finite value;
    they cannot become a context predicate without an explicit registry entry.
    Missing availability never moves a record earlier than its own end.
    """
    cutoff = _parse_datetime(availability_cutoff) if availability_cutoff is not None else None
    latest: dict[str, BriefRecord] = {}
    rejected = wrong_units = duplicates = 0
    reasons: list[str] = []
    for raw in records:
        try:
            required = ("id", "metric", "value", "unit", "start", "timezone")
            missing = [name for name in required if name not in raw]
            if missing:
                raise ValueError("missing:" + ",".join(missing))
            metric = _canonical_metric(str(raw["metric"]))
            value = _finite(raw["value"])
            if value is None:
                raise ValueError("nonfinite_or_boolean_value")
            unit = str(raw["unit"]).strip().lower()
            allowed = _UNITS.get(metric)
            if allowed is not None and unit not in allowed:
                wrong_units += 1
                raise ValueError(f"wrong_unit:{metric}")
            start = _parse_datetime(raw["start"])
            end = _parse_datetime(raw["end"]) if raw.get("end") else None
            if end is not None and end < start:
                raise ValueError("end_before_start")
            available = _parse_datetime(raw["available_at"]) if raw.get("available_at") else None
            attributes: dict[str, Any] = {}
            if isinstance(raw.get("metadata"), Mapping):
                attributes.update(raw["metadata"])
            for key in (
                "event_id", "eventId", "meal_id", "mealId", "session_id", "sessionId",
                "intensity", "vigorous", "met", "explicitCycleStart", "explicit_cycle_start",
                "coverage", "session_count", "sessionCount", "observed", "datasetId", "dataset_id",
                "source_provenance", "confirmed", "profile", "syntheticDemo", "demoLabel", "revision",
                "location", "correction_classification", "correction_only", "correction_component_units",
                "fasting_provenance", "fasting_status", "sampleCount", "observedFlowValues",
                "crampObservations", "severity", "source_provenance_present",
            ):
                if key in raw and key not in attributes:
                    attributes[key] = raw[key]
            item = BriefRecord(
                id=str(raw["id"]), metric=metric, value=value, unit=unit,
                start=start, end=end, timezone=str(raw["timezone"]),
                source=str(raw.get("source", "unknown")), kind=str(raw.get("kind", "measured")),
                available_at=available, attributes=attributes,
            )
            if cutoff is not None and item.known_at > cutoff:
                continue
            previous = latest.get(item.id)
            revision = int(raw.get("revision", 0) or 0)
            if previous is None:
                latest[item.id] = item
                continue
            previous_revision = int(previous.attributes.get("revision", 0) or 0)
            if revision > previous_revision:
                latest[item.id] = item
            elif revision == previous_revision:
                duplicates += 1
                if previous.value != item.value or previous.start != item.start:
                    rejected += 1
                    # Never place owner/event identifiers in diagnostics;
                    # the report is persisted and may be shown to clients.
                    reasons.append("conflicting_revision")
        except (TypeError, ValueError, OverflowError) as exc:
            rejected += 1
            reason = str(exc).split(":", 1)[0].strip() or "invalid_record"
            reasons.append(reason[:80])
    values = sorted(latest.values(), key=lambda item: (item.start, item.metric, item.id))
    return values, NormalizationReport(len(values), rejected, duplicates, wrong_units, tuple(reasons[:20]))


def _available(records: Iterable[BriefRecord], cutoff: datetime) -> list[BriefRecord]:
    return [record for record in records if record.known_at <= cutoff]


def _metric(records: Iterable[BriefRecord], names: set[str]) -> list[BriefRecord]:
    return [record for record in records if record.metric in names]


def _series(
    records: Iterable[BriefRecord],
    start: datetime,
    end: datetime,
    *,
    cutoff: datetime,
    min_coverage: float,
    glucose_only: bool = True,
    index: _RecordIndex | None = None,
) -> dict[str, Any]:
    """Return a five-minute median series with explicit gaps preserved."""
    source = index.between(start, end) if index is not None else records
    candidates = [
        record for record in source
        if record.start >= start and record.start < end and record.known_at <= cutoff
        and (not glucose_only or record.metric == "glucose")
    ]
    bins: dict[int, list[float]] = defaultdict(list)
    for record in candidates:
        value = _finite(record.value)
        if value is None:
            continue
        index = int((record.start - start).total_seconds() // 300)
        bins[index].append(value)
    expected = max(1, int(round((end - start).total_seconds() / 300)))
    values: list[dict[str, float]] = []
    for index in sorted(bins):
        value = _median(bins[index])
        if value is not None:
            values.append({"minute": float(index * 5), "glucose": float(value)})
    coverage = len(values) / expected
    return {
        "start": start, "end": end, "expectedBins": expected,
        "observedBins": len(values), "coverage": coverage,
        "valid": coverage >= min_coverage, "points": values,
    }


def _outcome(series: Mapping[str, Any], start: datetime, end: datetime) -> dict[str, Any] | None:
    if not series.get("valid"):
        return None
    points = list(series.get("points", []))
    glucose = [float(point["glucose"]) for point in points]
    if not glucose:
        return None
    glucose_mean = sum(glucose) / len(glucose)
    return {
        "points": points,
        "coverage": float(series["coverage"]),
        "observedBins": int(series["observedBins"]),
        "expectedBins": int(series["expectedBins"]),
        "mean": glucose_mean,
        # Population variance over observed five-minute medians only. Missing
        # bins are absent from the calculation, never imputed.
        "variance": float(sum((value - glucose_mean) ** 2 for value in glucose) / len(glucose)),
        "lowMinutes": float(sum(value < 70 for value in glucose) * 5),
        "veryLowMinutes": float(sum(value < 54 for value in glucose) * 5),
        "highMinutes": float(sum(value > 180 for value in glucose) * 5),
        "tir": float(sum(70 <= value <= 180 for value in glucose) / len(glucose)),
        "start": start, "end": end,
    }


def _day_bounds(day: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    start = _local_datetime(day, time(0, 0), zone)
    end = _local_datetime(day + timedelta(days=1), time(0, 0), zone)
    return start, end


def _night_bounds(day: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    return _local_datetime(day, time(0, 0), zone), _local_datetime(day, time(6, 0), zone)


def _anchor_cutoff(day: date, zone: ZoneInfo) -> datetime:
    return _local_datetime(day, time(6, 0), zone)


def _record_date(record: BriefRecord, zone: ZoneInfo) -> date:
    return record.start.astimezone(zone).date()


def _context_observations(records: list[BriefRecord], anchor: datetime) -> list[BriefRecord]:
    return [record for record in records if record.known_at <= anchor and record.end_or_start <= anchor]


def _latest_sleep(records: list[BriefRecord], anchor: datetime, zone: ZoneInfo) -> BriefRecord | None:
    candidates = [
        record for record in records
        if record.metric == "sleep_hours" and record.known_at <= anchor
        and record.end_or_start <= anchor and record.end_or_start >= anchor - timedelta(hours=24)
    ]
    return max(candidates, key=lambda record: record.end_or_start, default=None)


def _temperature_delta(
    records: list[BriefRecord], *, metric: str, anchor: datetime, zone: ZoneInfo, days: int = 14
) -> tuple[float | None, float | None, int]:
    current_day = _local_date(anchor, zone)
    prior_start = current_day - timedelta(days=days)
    current_values = [
        record.value for record in records
        if record.metric == metric and record.known_at <= anchor
        and _record_date(record, zone) == current_day and record.start < anchor
    ]
    baseline_by_day: dict[date, list[float]] = defaultdict(list)
    for record in records:
        day = _record_date(record, zone)
        if record.metric != metric or record.known_at > anchor or not (prior_start <= day < current_day):
            continue
        value = _finite(record.value)
        if value is not None:
            baseline_by_day[day].append(value)
    baseline_days = [_median(values) for values in baseline_by_day.values()]
    baseline_days = [value for value in baseline_days if value is not None]
    baseline = _median(baseline_days) if len(baseline_days) >= 7 else None
    current = _median(current_values)
    return (current - baseline if current is not None and baseline is not None else None, baseline, len(baseline_days))


def _period_onset_days(records: list[BriefRecord], zone: ZoneInfo, cutoff: datetime) -> list[date]:
    onsets: set[date] = set()
    for record in records:
        if record.metric != "period_onset" or record.known_at > cutoff:
            continue
        explicit = _attr(record, "explicitCycleStart", "explicit_cycle_start", "confirmed")
        if explicit is False:
            continue
        onsets.add(_record_date(record, zone))
    return sorted(onsets)


def _days_since_period(day: date, onsets: Sequence[date]) -> int | None:
    prior = [onset for onset in onsets if onset <= day]
    return (day - prior[-1]).days if prior else None


def _is_vigorous(record: BriefRecord) -> bool:
    intensity = str(_attr(record, "intensity") or "").strip().lower()
    if intensity in {"vigorous", "high", "hard", "very_high"}:
        return True
    flag = _attr(record, "vigorous")
    if isinstance(flag, bool):
        return flag
    met = _finite(_attr(record, "met"))
    return met is not None and met >= 6.0


def _exercise_day_summary(records: list[BriefRecord], day: date, zone: ZoneInfo) -> dict[str, Any]:
    sessions: list[BriefRecord] = []
    coverage_record = False
    minutes = 0.0
    for record in records:
        if _record_date(record, zone) != day:
            continue
        if record.metric == "exercise_minutes":
            sessions.append(record)
            minutes += max(0.0, record.value)
        elif record.metric == "exercise_coverage":
            value = _finite(record.value)
            coverage_record = value is not None and value > 0
    # A daily summary is one observed session unless an explicit count exists.
    session_count_value = None
    if sessions:
        explicit_counts = [_finite(_attr(record, "session_count", "sessionCount")) for record in sessions]
        explicit_counts = [value for value in explicit_counts if value is not None]
        session_count_value = int(round(sum(explicit_counts))) if explicit_counts else len(sessions)
    return {
        "valid": bool(sessions or coverage_record),
        "minutes": minutes if sessions else 0.0,
        "sessions": session_count_value if session_count_value is not None else 0,
        "vigorous": any(_is_vigorous(record) for record in sessions),
        "coverage": bool(sessions or coverage_record),
    }


def _sleep_day_summary(records: list[BriefRecord], day: date, zone: ZoneInfo, cutoff: datetime) -> float | None:
    # Sleep assigned to the waking/local end day.  Only completed intervals are
    # allowed; a point-only sleep record remains unavailable for this engine.
    candidates: list[BriefRecord] = []
    for record in records:
        if record.metric != "sleep_hours" or record.known_at > cutoff or record.end is None:
            continue
        if _record_date(BriefRecord(record.id, record.metric, record.value, record.unit, record.end, None, record.timezone, record.source, record.kind, None, {}), zone) == day:
            candidates.append(record)
        elif record.end.astimezone(zone).date() == day:
            candidates.append(record)
    return max((record.value for record in candidates), default=None)


def _temp_day_summary(records: list[BriefRecord], day: date, zone: ZoneInfo, metric: str, cutoff: datetime) -> float | None:
    values = [record.value for record in records if record.metric == metric and record.known_at <= cutoff and _record_date(record, zone) == day]
    return _median(values)


def _build_daily_summaries(records: list[BriefRecord], zone: ZoneInfo, cutoff: datetime) -> dict[date, dict[str, Any]]:
    """Index low-cardinality context streams once per local day."""
    summaries: dict[date, dict[str, Any]] = defaultdict(lambda: {
        "exercise": [], "sleep": [], "wrist": [], "body": [], "bolus": [], "glucosePrior3h": [],
    })
    for record in records:
        if record.known_at > cutoff:
            continue
        if record.metric == "exercise_minutes":
            summaries[_record_date(record, zone)]["exercise"].append(record)
        elif record.metric == "exercise_coverage":
            summaries[_record_date(record, zone)]["exercise"].append(record)
        elif record.metric == "sleep_hours" and record.end is not None:
            summaries[record.end.astimezone(zone).date()]["sleep"].append(record)
        elif record.metric == "sleeping_wrist_temperature":
            summaries[_record_date(record, zone)]["wrist"].append(record)
        elif record.metric == "body_temperature":
            summaries[_record_date(record, zone)]["body"].append(record)
        elif record.metric == "delivered_bolus" and _is_actual_delivery(record):
            summaries[_record_date(record, zone)]["bolus"].append(record)
        elif record.metric == "glucose":
            local = record.start.astimezone(zone)
            local_anchor = datetime.combine(local.date(), time(6, 0), tzinfo=zone)
            if time(3, 0) <= local.time() < time(6, 0) and record.start < local_anchor:
                summaries[local.date()]["glucosePrior3h"].append(record)
    indexed: dict[date, dict[str, Any]] = {}
    for day, values in summaries.items():
        exercise = values["exercise"]
        sessions = [record for record in exercise if record.metric == "exercise_minutes"]
        explicit_counts = [_finite(_attr(record, "session_count", "sessionCount")) for record in sessions]
        explicit_counts = [value for value in explicit_counts if value is not None]
        boluses = values["bolus"]
        glucose = sorted(values["glucosePrior3h"], key=lambda record: record.start)
        glucose_values = [float(record.value) for record in glucose if _finite(record.value) is not None]
        glucose_valid = len(glucose_values) >= 29  # >=80% of the 36 five-minute bins
        glucose_mean = _mean(glucose_values) if glucose_valid else None
        first_half = _mean(glucose_values[: len(glucose_values) // 2]) if glucose_valid else None
        last_half = _mean(glucose_values[len(glucose_values) // 2 :]) if glucose_valid else None
        indexed[day] = {
            "exerciseValid": bool(exercise), "exerciseMinutes": sum(max(0.0, record.value) for record in sessions) if sessions else 0.0,
            "exerciseSessions": sum(explicit_counts) if explicit_counts else len(sessions),
            "vigorous": any(_is_vigorous(record) for record in sessions),
            "sleepHours": max((record.value for record in values["sleep"]), default=None),
            "wristTemperature": _median(record.value for record in values["wrist"]),
            "bodyTemperature": _median(record.value for record in values["body"]),
            "sleepValid": bool(values["sleep"]), "wristValid": bool(values["wrist"]), "bodyValid": bool(values["body"]),
            # This is recorded delivered bolus only.  It is deliberately not
            # an estimate of total insulin, basal delivery, or insulin-on-board.
            "deliveredBolus": sum(max(0.0, record.value) for record in boluses) if boluses else None,
            "bolusValid": bool(boluses), "bolusEvents": len(boluses),
            # These are as-of 03:00–06:00 local glucose summaries.  Missing
            # bins leave the complete feature unknown rather than zero.
            "glucosePrior3hMean": glucose_mean,
            "glucosePrior3hTrend": (last_half - first_half) if first_half is not None and last_half is not None else None,
            "glucosePrior3hLowMinutes": float(sum(value < 70 for value in glucose_values) * 5) if glucose_valid else None,
            "glucoseValid": glucose_valid, "glucoseObservedBins": len(glucose_values),
        }
    return indexed


def _context_windows(
    records: list[BriefRecord], anchor_day: date, zone: ZoneInfo, cutoff: datetime, config: BriefConfig,
    daily_summaries: Mapping[date, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return context-window evidence and predicate values for one anchor."""
    def summary(start: date, end: date) -> dict[str, Any]:
        days = _date_range(start, end)
        if daily_summaries is not None:
            exercise = [
                {
                    "valid": bool(daily_summaries.get(day, {}).get("exerciseValid", False)),
                    "minutes": daily_summaries.get(day, {}).get("exerciseMinutes", 0.0),
                    "sessions": daily_summaries.get(day, {}).get("exerciseSessions", 0.0),
                }
                for day in days
            ]
        else:
            exercise = [_exercise_day_summary(records, day, zone) for day in days]
        exercise_valid = [item for item in exercise if item["valid"]]
        sleep = [
            daily_summaries.get(day, {}).get("sleepHours") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [_sleep_day_summary(records, day, zone, cutoff) for day in days]
        sleep_valid = [float(value) for value in sleep if value is not None]
        wrist = [
            daily_summaries.get(day, {}).get("wristTemperature") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [_temp_day_summary(records, day, zone, "sleeping_wrist_temperature", cutoff) for day in days]
        wrist_valid = [float(value) for value in wrist if value is not None]
        body = [
            daily_summaries.get(day, {}).get("bodyTemperature") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [_temp_day_summary(records, day, zone, "body_temperature", cutoff) for day in days]
        body_valid = [float(value) for value in body if value is not None]
        delivered_bolus = [
            daily_summaries.get(day, {}).get("deliveredBolus") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [None for _ in days]
        bolus_valid = [float(value) for value in delivered_bolus if value is not None]
        glucose_mean = [
            daily_summaries.get(day, {}).get("glucosePrior3hMean") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [None for _ in days]
        glucose_trend = [
            daily_summaries.get(day, {}).get("glucosePrior3hTrend") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [None for _ in days]
        glucose_low = [
            daily_summaries.get(day, {}).get("glucosePrior3hLowMinutes") if day in daily_summaries else None
            for day in days
        ] if daily_summaries is not None else [None for _ in days]
        glucose_mean_valid = [float(value) for value in glucose_mean if value is not None]
        glucose_trend_valid = [float(value) for value in glucose_trend if value is not None]
        glucose_low_valid = [float(value) for value in glucose_low if value is not None]
        expected = len(days)
        return {
            "start": start, "end": end, "expected": expected,
            "exerciseValid": len(exercise_valid),
            "exerciseMinutes": sum(item["minutes"] for item in exercise_valid) / max(1, len(exercise_valid)),
            "exerciseSessions": sum(item["sessions"] for item in exercise_valid) / max(1, len(exercise_valid)),
            "exerciseWeeks": sum(item["sessions"] for item in exercise_valid) / max(1, len(exercise_valid)) * 7,
            "sleepValid": len(sleep_valid), "sleepHours": _median(sleep_valid),
            "wristValid": len(wrist_valid), "wristTemperature": _median(wrist_valid),
            "bodyValid": len(body_valid), "bodyTemperature": _median(body_valid),
            "bolusValid": len(bolus_valid), "deliveredBolus": _mean(bolus_valid),
            "glucoseValid": len(glucose_mean_valid), "glucosePrior3hMean": _mean(glucose_mean_valid),
            "glucosePrior3hTrend": _mean(glucose_trend_valid), "glucosePrior3hLowMinutes": _mean(glucose_low_valid),
        }

    recent_start = anchor_day - timedelta(days=config.current_days)
    recent_end = anchor_day - timedelta(days=1)
    prior_start = recent_start - timedelta(days=config.prior_days)
    prior_end = recent_start - timedelta(days=1)
    long_recent_start = anchor_day - timedelta(days=config.prior_days)
    long_recent_end = anchor_day - timedelta(days=config.current_days + 1)
    long_prior_start = long_recent_start - timedelta(days=config.long_prior_days)
    long_prior_end = long_recent_start - timedelta(days=1)
    recent = summary(recent_start, recent_end)
    prior = summary(prior_start, prior_end)
    long_recent = summary(long_recent_start, long_recent_end)
    long_prior = summary(long_prior_start, long_prior_end)

    windows: list[dict[str, Any]] = []
    def add_window(label: str, metric: str, unit: str, left: dict[str, Any], right: dict[str, Any], left_key: str, right_key: str, required_left: int, required_right: int) -> tuple[float | None, float | None]:
        left_value = left.get(left_key)
        right_value = right.get(right_key)
        # Explicit valid counts avoid interpreting a missing stream as zero.
        valid_map = {
            "exerciseMinutes": "exerciseValid", "exerciseSessions": "exerciseValid", "exerciseWeeks": "exerciseValid",
            "sleepHours": "sleepValid", "wristTemperature": "wristValid", "bodyTemperature": "bodyValid",
            "deliveredBolus": "bolusValid", "glucosePrior3hMean": "glucoseValid",
            "glucosePrior3hTrend": "glucoseValid", "glucosePrior3hLowMinutes": "glucoseValid",
        }
        left_valid = int(left.get(valid_map[left_key], 0))
        right_valid = int(right.get(valid_map[right_key], 0))
        if left_value is None or right_value is None or left_valid < required_left or right_valid < required_right:
            return None, None
        windows.append({
            "label": label, "start": left["start"], "end": left["end"], "validDays": left_valid,
            "expectedDays": left["expected"], "metric": metric, "unit": unit,
            "value": float(left_value), "baselineValue": float(right_value),
            "baselineStart": right["start"], "baselineEnd": right["end"],
            "baselineValidDays": right_valid, "baselineExpectedDays": right["expected"],
        })
        return float(left_value), float(right_value)

    exercise7, exercise28 = add_window("Recent 7 days vs preceding 28", "exercise_frequency", "sessions/week", recent, prior, "exerciseWeeks", "exerciseWeeks", 5, 20)
    minutes7, minutes28 = add_window("Recent 7 days vs preceding 28", "exercise_minutes", "minutes/day", recent, prior, "exerciseMinutes", "exerciseMinutes", 5, 20)
    wrist7, wrist28 = add_window("Recent 7 days vs preceding 28", "sleeping_wrist_temperature", "°C", recent, prior, "wristTemperature", "wristTemperature", 5, 20)
    sleep7, sleep28 = add_window("Recent 7 days vs preceding 28", "sleep_duration", "hours/night", recent, prior, "sleepHours", "sleepHours", 5, 20)
    exercise28long, exercise56 = add_window("Recent 28 days vs preceding 56", "exercise_frequency", "sessions/week", long_recent, long_prior, "exerciseWeeks", "exerciseWeeks", 20, 40)
    body7, body28 = add_window("Recent 7 days vs preceding 28", "body_temperature", "°C", recent, prior, "bodyTemperature", "bodyTemperature", 5, 20)
    bolus7, bolus28 = add_window("Recent 7 days vs preceding 28", "recorded_delivered_bolus", "U/day", recent, prior, "deliveredBolus", "deliveredBolus", 5, 20)
    glucose_mean7, glucose_mean28 = add_window("Recent 7 days vs preceding 28", "glucose_prior3h_mean", "mg/dL", recent, prior, "glucosePrior3hMean", "glucosePrior3hMean", 5, 20)
    glucose_trend7, glucose_trend28 = add_window("Recent 7 days vs preceding 28", "glucose_prior3h_trend", "mg/dL", recent, prior, "glucosePrior3hTrend", "glucosePrior3hTrend", 5, 20)
    glucose_low7, glucose_low28 = add_window("Recent 7 days vs preceding 28", "glucose_prior3h_low_minutes", "minutes", recent, prior, "glucosePrior3hLowMinutes", "glucosePrior3hLowMinutes", 5, 20)
    predicates: dict[str, bool | None] = {
        "exercise_frequency_down_7v28": exercise7 is not None and exercise28 is not None and exercise7 <= exercise28 - 2.0,
        "exercise_minutes_down_7v28": minutes7 is not None and minutes28 is not None and minutes7 <= minutes28 - 30.0,
        "wrist_temperature_lower_7v28": wrist7 is not None and wrist28 is not None and wrist7 <= wrist28 - 0.2,
        "sleep_duration_down_7v28": sleep7 is not None and sleep28 is not None and sleep7 <= sleep28 - 0.75,
        "exercise_frequency_down_28v56": exercise28long is not None and exercise56 is not None and exercise28long <= exercise56 - 2.0,
        "delivered_bolus_down_7v28": bolus7 is not None and bolus28 is not None and bolus28 > 0 and bolus7 <= bolus28 * 0.80,
        "glucose_prior3h_mean_up_7v28": glucose_mean7 is not None and glucose_mean28 is not None and glucose_mean7 >= glucose_mean28 + 15.0,
        "glucose_prior3h_trend_up_7v28": glucose_trend7 is not None and glucose_trend28 is not None and glucose_trend7 >= glucose_trend28 + 10.0,
        "glucose_prior3h_low_minutes_up_7v28": glucose_low7 is not None and glucose_low28 is not None and glucose_low7 >= glucose_low28 + 10.0,
    }
    # Mark a comparator unknown when the required coverage existed in neither arm.
    if exercise7 is None or exercise28 is None:
        predicates["exercise_frequency_down_7v28"] = None
    if minutes7 is None or minutes28 is None:
        predicates["exercise_minutes_down_7v28"] = None
    if wrist7 is None or wrist28 is None:
        predicates["wrist_temperature_lower_7v28"] = None
    if sleep7 is None or sleep28 is None:
        predicates["sleep_duration_down_7v28"] = None
    if exercise28long is None or exercise56 is None:
        predicates["exercise_frequency_down_28v56"] = None
    if bolus7 is None or bolus28 is None:
        predicates["delivered_bolus_down_7v28"] = None
    if glucose_mean7 is None or glucose_mean28 is None:
        predicates["glucose_prior3h_mean_up_7v28"] = None
    if glucose_trend7 is None or glucose_trend28 is None:
        predicates["glucose_prior3h_trend_up_7v28"] = None
    if glucose_low7 is None or glucose_low28 is None:
        predicates["glucose_prior3h_low_minutes_up_7v28"] = None
    return {"windows": windows, "recent": recent, "prior": prior, "longRecent": long_recent, "longPrior": long_prior, "bodyRecent": body7, "bodyPrior": body28}, predicates


def _current_sleep_and_context(records: list[BriefRecord], anchor: datetime, zone: ZoneInfo, cutoff: datetime) -> dict[str, Any]:
    sleep = _latest_sleep(records, anchor, zone)
    wrist_delta, wrist_baseline, wrist_days = _temperature_delta(records, metric="sleeping_wrist_temperature", anchor=anchor, zone=zone)
    body_delta, body_baseline, body_days = _temperature_delta(records, metric="body_temperature", anchor=anchor, zone=zone)
    evening_start = _local_datetime(_local_date(anchor, zone) - timedelta(days=1), time(16, 0), zone)
    evening_end = _local_datetime(_local_date(anchor, zone), time(0, 0), zone)
    exercise = [
        record for record in records if record.metric == "exercise_minutes" and record.known_at <= cutoff
        and evening_start <= record.start <= evening_end and _is_vigorous(record)
    ]
    return {
        "sleepHours": float(sleep.value) if sleep else None,
        "sleepEnd": sleep.end_or_start if sleep else None,
        "wristTemperatureDelta": wrist_delta, "wristTemperatureBaseline": wrist_baseline, "wristValidDays": wrist_days,
        "bodyTemperatureDelta": body_delta, "bodyTemperatureBaseline": body_baseline, "bodyValidDays": body_days,
        "vigorousEvening": bool(exercise),
        "vigorousEveningCount": len(exercise),
    }


def _is_actual_delivery(record: BriefRecord) -> bool:
    """Accept actual delivered events without treating commands as delivery."""
    label = _attr(record, "delivery", "delivery_type", "deliveryType")
    if label is None:
        # The canonical delivered_bolus metric already means a delivered event.
        return True
    return str(label).strip().lower() not in {"commanded", "scheduled", "requested", "planned"}


def _event_id(record: BriefRecord) -> str | None:
    value = _attr(record, "event_id", "eventId", "meal_id", "mealId")
    return str(value) if value not in (None, "") else None


def _local_minutes(record: BriefRecord, zone: ZoneInfo) -> int:
    local = record.start.astimezone(zone)
    return local.hour * 60 + local.minute


def _start_glucose(records: list[BriefRecord], anchor: datetime, cutoff: datetime, index: _RecordIndex | None = None) -> float | None:
    source = index.between(anchor - timedelta(minutes=15), anchor) if index is not None else records
    values = [
        record.value for record in source
        if record.metric == "glucose" and record.known_at <= cutoff
        and anchor - timedelta(minutes=15) <= record.start < anchor
    ]
    return _median(values) if len(values) >= 2 else None


def _sum_events(
    records: list[BriefRecord], *, metrics: set[str], start: datetime, end: datetime, cutoff: datetime, index: _RecordIndex | None = None
) -> float | None:
    source = index.between(start, end) if index is not None else records
    values = [record.value for record in source if record.metric in metrics and start <= record.start < end and record.known_at <= cutoff]
    return sum(values) if values else None


def _meal_episodes(records: list[BriefRecord], zone: ZoneInfo, cutoff: datetime, config: BriefConfig, index: _RecordIndex | None = None) -> list[dict[str, Any]]:
    meals = [
        record for record in records
        if record.metric == "meal_carbs" and record.value > 0 and record.known_at <= cutoff
        and 5 * 60 <= _local_minutes(record, zone) < 11 * 60
    ]
    boluses = [record for record in records if record.metric == "delivered_bolus" and record.value > 0 and record.known_at <= cutoff]
    episodes: list[dict[str, Any]] = []
    for meal in sorted(meals, key=lambda record: record.start):
        event_id = _event_id(meal)
        candidates = [
            record for record in boluses
            if abs((record.start - meal.start).total_seconds()) <= 15 * 60
            and (event_id is None or _event_id(record) in {None, event_id})
        ]
        if not candidates:
            continue
        bolus = min(candidates, key=lambda record: abs((record.start - meal.start).total_seconds()))
        # A second nearby meal is an overlapping event.  Keep its raw episode
        # available to a detail sheet, but exclude it from isolated matching.
        overlaps = [
            other for other in meals if other.id != meal.id and meal.start < other.start < meal.start + timedelta(hours=3)
        ]
        anchor = meal.start
        start = anchor - timedelta(minutes=15)
        response_end = anchor + timedelta(hours=3)
        baseline = _start_glucose(records, anchor, cutoff, index)
        response = _series(records, anchor, response_end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        outcome = _outcome(response, anchor, response_end)
        if baseline is None or outcome is None:
            continue
        outcome["meanRise"] = float(outcome["mean"] - baseline)
        local_day = _record_date(meal, zone)
        episodes.append({
            "id": f"meal-{meal.id}", "date": local_day, "anchor": anchor,
            "meal": meal, "bolus": bolus, "carbs": float(meal.value), "bolusUnits": float(bolus.value),
            "mealMinute": _local_minutes(meal, zone), "startingGlucose": float(baseline),
            "isolated": not overlaps, "overlapIds": [item.id for item in overlaps],
            "outcome": outcome, "trace": response["points"],
            "sourceLabel": "inferred meal entry + recorded delivered bolus",
        })
    return episodes


def _daily_outcomes(
    records: list[BriefRecord], day: date, zone: ZoneInfo, cutoff: datetime, config: BriefConfig, index: _RecordIndex | None = None
) -> dict[str, Any]:
    night_start, night_end = _night_bounds(day, zone)
    night_series = _series(records, night_start, night_end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
    night = _outcome(night_series, night_start, night_end)
    day_start, day_end = _day_bounds(day, zone)
    next24_series = _series(records, day_start, day_end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
    next24 = _outcome(next24_series, day_start, day_end)

    day_outcomes: list[dict[str, Any]] = []
    for offset in range(1, 8):
        target_day = day + timedelta(days=offset)
        start, end = _day_bounds(target_day, zone)
        series = _series(records, start, end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        outcome = _outcome(series, start, end)
        if outcome is not None:
            outcome = dict(outcome)
            outcome["date"] = target_day
            day_outcomes.append(outcome)
    next7 = None
    if len(day_outcomes) >= 5:
        next7 = {
            "validDays": len(day_outcomes), "expectedDays": 7,
            "lowMinutes": _mean(item["lowMinutes"] for item in day_outcomes),
            "veryLowMinutes": _mean(item["veryLowMinutes"] for item in day_outcomes),
            "highMinutes": _mean(item["highMinutes"] for item in day_outcomes),
            "tir": _mean(item["tir"] for item in day_outcomes),
            "points": [
                {"minute": float(i * 1440 + point["minute"]), "glucose": point["glucose"]}
                for i, item in enumerate(day_outcomes) for point in item["points"]
            ][: 7 * 288],
            "start": day_outcomes[0]["start"], "end": day_outcomes[-1]["end"],
            "dates": [item["date"] for item in day_outcomes],
        }
    return {"night": night, "next24h": next24, "next7d": next7}


def _vigorous_evening(records: list[BriefRecord], day: date, zone: ZoneInfo, cutoff: datetime) -> bool | None:
    start = _local_datetime(day - timedelta(days=1), time(16, 0), zone)
    end = _local_datetime(day, time(0, 0), zone)
    values = [
        record for record in records
        if record.metric == "exercise_minutes" and start <= record.start < end and record.known_at <= cutoff
    ]
    if not values:
        return None
    return any(_is_vigorous(record) for record in values)


def _context_for_day(
    records: list[BriefRecord], day: date, zone: ZoneInfo, cutoff: datetime, onsets: Sequence[date], config: BriefConfig,
    daily_summaries: Mapping[date, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = _anchor_cutoff(day, zone)
    sleep = _latest_sleep(records, anchor, zone)
    sleep_hours = float(sleep.value) if sleep else None
    cycle_days = _days_since_period(day, onsets)
    windows, predicates = _context_windows(records, day, zone, cutoff, config, daily_summaries)
    prior_vigorous = _vigorous_evening(records, day, zone, cutoff)
    values: dict[str, Any] = {
        "sleepHours": sleep_hours, "shortSleep": sleep_hours < 6.0 if sleep_hours is not None else None,
        "recordedCycleEarly": 0 <= cycle_days <= 7 if cycle_days is not None else None,
        "cycleDays": cycle_days, "priorVigorousEvening": prior_vigorous,
        "contextWindows": windows["windows"], "summary": windows,
        "predicates": dict(predicates),
    }
    recent_summary = windows["recent"]
    values.update({
        # Explicit as-of features for discovery.  They are summaries of
        # observed records before the 06:00 local anchor, never target values.
        "recentGlucosePrior3hMean": recent_summary.get("glucosePrior3hMean"),
        "recentGlucosePrior3hTrend": recent_summary.get("glucosePrior3hTrend"),
        "recentGlucosePrior3hLowMinutes": recent_summary.get("glucosePrior3hLowMinutes"),
        "recentDeliveredBolus": recent_summary.get("deliveredBolus"),
    })
    values["shortSleepAndExerciseDown"] = (
        values["shortSleep"] is True and predicates.get("exercise_frequency_down_7v28") is True
        if values["shortSleep"] is not None and predicates.get("exercise_frequency_down_7v28") is not None else None
    )
    return values, predicates


def _anchor_fields(
    records: list[BriefRecord], day: date, zone: ZoneInfo, cutoff: datetime, meal: dict[str, Any] | None = None, index: _RecordIndex | None = None
) -> dict[str, float | None]:
    day_start, _ = _day_bounds(day, zone)
    anchor_glucose = _start_glucose(records, day_start + timedelta(hours=6), cutoff, index)
    # Daily rows describe the upcoming night and next-day horizons, so their
    # matching window is the same local day's recorded 16:00–24:00 period.
    # Morning meal rows use only the meal fields above; retaining this field
    # also allows a future target family to match the preceding evening
    # explicitly without treating missing insulin as zero.
    evening_start = _local_datetime(day, time(16, 0), zone)
    evening_end = _local_datetime(day + timedelta(days=1), time(0, 0), zone)
    meal_carbs = meal["carbs"] if meal else None
    meal_bolus = meal["bolusUnits"] if meal else None
    meal_minute = float(meal["mealMinute"]) if meal else None
    return {
        "startingGlucose": float(meal["startingGlucose"]) if meal else anchor_glucose,
        "anchorGlucose": anchor_glucose,
        "mealCarbs": meal_carbs,
        "mealBolus": meal_bolus,
        "mealMinute": meal_minute,
        "eveningCarbs": _sum_events(records, metrics={"meal_carbs"}, start=evening_start, end=evening_end, cutoff=cutoff, index=index),
        "eveningBolus": _sum_events(records, metrics={"delivered_bolus"}, start=evening_start, end=evening_end, cutoff=cutoff, index=index),
    }


_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"id": "short_sleep", "predicates": ("shortSleep",), "label": "shorter completed sleep"},
    {"id": "recorded_cycle_early", "predicates": ("recordedCycleEarly",), "label": "an explicitly recorded early-cycle day"},
    {"id": "prior_vigorous_evening", "predicates": ("priorVigorousEvening",), "label": "recorded vigorous activity the preceding evening"},
    {"id": "exercise_frequency_down_7v28", "predicates": ("exercise_frequency_down_7v28",), "label": "fewer recorded exercise sessions in the recent 7 days"},
    {"id": "exercise_minutes_down_7v28", "predicates": ("exercise_minutes_down_7v28",), "label": "fewer recorded exercise minutes in the recent 7 days"},
    {"id": "wrist_temperature_lower_7v28", "predicates": ("wrist_temperature_lower_7v28",), "label": "a lower recent sleeping-wrist temperature level"},
    {"id": "sleep_duration_down_7v28", "predicates": ("sleep_duration_down_7v28",), "label": "a lower recent sleep-duration average"},
    {"id": "exercise_frequency_down_28v56", "predicates": ("exercise_frequency_down_28v56",), "label": "fewer recorded exercise sessions across recent weeks"},
    {"id": "short_sleep_and_exercise_down", "predicates": ("shortSleepAndExerciseDown",), "label": "shorter sleep alongside fewer recorded exercise sessions"},
    {"id": "delivered_bolus_down_7v28", "predicates": ("delivered_bolus_down_7v28",), "label": "lower recorded delivered bolus across recent weeks"},
    {"id": "glucose_prior3h_mean_up_7v28", "predicates": ("glucose_prior3h_mean_up_7v28",), "label": "higher as-of glucose in the preceding 3 hours"},
    {"id": "glucose_prior3h_trend_up_7v28", "predicates": ("glucose_prior3h_trend_up_7v28",), "label": "a higher as-of glucose trend in the preceding 3 hours"},
    {"id": "glucose_prior3h_low_minutes_up_7v28", "predicates": ("glucose_prior3h_low_minutes_up_7v28",), "label": "more recorded low minutes in the preceding 3 hours"},
)


_OUTCOMES: tuple[dict[str, Any], ...] = (
    {"id": "meal_mean_rise", "horizon": "meal_0_3h", "label": "the observed 3-hour post-meal glucose rise", "unit": "mg/dL"},
    {"id": "meal_high_minutes", "horizon": "meal_0_3h", "label": "minutes above 180 mg/dL in the 3-hour meal window", "unit": "minutes"},
    {"id": "night_low_minutes", "horizon": "upcoming_night_00_06", "label": "minutes below 70 mg/dL in the upcoming night", "unit": "minutes"},
    {"id": "next24_low_minutes", "horizon": "next_24h", "label": "minutes below 70 mg/dL in the following day", "unit": "minutes"},
    {"id": "next24_tir", "horizon": "next_24h", "label": "time in range in the following day", "unit": "fraction"},
    {"id": "next7_low_minutes", "horizon": "next_7d", "label": "average daily minutes below 70 mg/dL over the following week", "unit": "minutes/day"},
    {"id": "next7_tir", "horizon": "next_7d", "label": "average daily time in range over the following week", "unit": "fraction"},
)


def _target_value(row: Mapping[str, Any], target: str) -> float | None:
    value = row.get("outcomes", {}).get(target)
    return _finite(value)


def _threshold(target: str) -> float:
    if target == "meal_mean_rise": return 8.0
    if target == "meal_high_minutes": return 15.0
    if target in {"night_low_minutes", "next24_low_minutes", "next7_low_minutes"}: return 10.0
    return 0.05


def _target_fields(target: str) -> tuple[str, ...]:
    if target.startswith("meal_"):
        return ("startingGlucose", "mealCarbs", "mealBolus", "mealMinute")
    return ("anchorGlucose", "eveningCarbs", "eveningBolus")


def _match_pairs(rows: list[dict[str, Any]], candidate: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    predicates = tuple(candidate["predicates"])
    eligible = []
    fields = _target_fields(target)
    for row in rows:
        flags = [row["context"].get(name) for name in predicates]
        if any(flag is None for flag in flags):
            continue
        if _target_value(row, target) is None:
            continue
        match_fields = row.get("match", {})
        if any(_finite(match_fields.get(field)) is None for field in fields):
            continue
        eligible.append((row, bool(all(flags))))
    positives = sorted([row for row, match in eligible if match], key=lambda row: (row["date"], row["id"]))
    negatives = [row for row, match in eligible if not match]
    available = {row["id"]: row for row in negatives}
    pairs: list[dict[str, Any]] = []
    for positive in positives:
        options: list[tuple[float, dict[str, Any]]] = []
        for negative in available.values():
            distance = 0.0
            valid = True
            for field_name in fields:
                left = _finite(positive["match"].get(field_name)); right = _finite(negative["match"].get(field_name))
                if left is None or right is None:
                    valid = False; break
                caliper = {"startingGlucose": 20.0, "anchorGlucose": 20.0, "mealCarbs": 10.0, "mealBolus": 1.0, "mealMinute": 90.0, "eveningCarbs": 20.0, "eveningBolus": 1.0}.get(field_name, 20.0)
                difference = abs(left - right)
                if difference > caliper:
                    valid = False; break
                distance += difference / max(caliper, 1e-9)
            if valid:
                options.append((distance, negative))
        if not options:
            continue
        _, negative = min(options, key=lambda item: (item[0], item[1]["date"], item[1]["id"]))
        available.pop(negative["id"], None)
        left = _target_value(positive, target); right = _target_value(negative, target)
        if left is None or right is None:
            continue
        pairs.append({
            "context": positive, "comparison": negative,
            "difference": float(left - right), "week": _week(positive["date"]),
        })
    return pairs


def _block_statistics(pairs: list[dict[str, Any]], *, draws: int, seed: int) -> tuple[float, float | None, float | None]:
    if not pairs:
        return 1.0, None, None
    observed = _mean(pair["difference"] for pair in pairs) or 0.0
    blocks: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        blocks[pair["week"]].append(float(pair["difference"]))
    block_values = list(blocks.values())
    rng = random.Random(seed)
    boot: list[float] = []
    sign_values: list[float] = []
    for _ in range(max(1, draws)):
        sampled = [block_values[rng.randrange(len(block_values))] for _ in block_values]
        boot.append(float(_mean(value for block in sampled for value in block) or 0.0))
        signs = [(-1.0 if rng.random() < 0.5 else 1.0) for _ in pairs]
        sign_values.append(float(_mean(sign * pair["difference"] for sign, pair in zip(signs, pairs)) or 0.0))
    extreme = sum(abs(value) >= abs(observed) for value in sign_values)
    p_value = (1.0 + extreme) / (len(sign_values) + 1.0)
    return p_value, _percentile(boot, 0.025), _percentile(boot, 0.975)


def _holm(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (total - rank)))
        adjusted[index] = running
    return adjusted


def _trace_episode(row: Mapping[str, Any], target: str, *, group: str, cap: int) -> dict[str, Any] | None:
    trace = row.get("traces", {}).get(target)
    if not trace:
        return None
    points = [
        {"minute": float(point["minute"]), "glucose": float(point["glucose"])}
        for point in trace[:288]
        if _finite(point.get("minute")) is not None and _finite(point.get("glucose")) is not None
    ]
    if not points:
        return None
    return {
        "id": str(row["id"]), "label": f"{group.title()} · {row['date'].isoformat()}",
        "group": group, "start": row["date"].isoformat(), "points": points,
    }


def _window_serialized(window: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": window["label"], "start": _iso(window["start"]), "end": _iso(window["end"]),
        "validDays": int(window["validDays"]), "expectedDays": int(window["expectedDays"]),
        "metric": window["metric"], "unit": window["unit"], "value": float(window["value"]),
        "baselineValue": float(window["baselineValue"]), "baselineStart": _iso(window["baselineStart"]),
        "baselineEnd": _iso(window["baselineEnd"]), "baselineValidDays": int(window["baselineValidDays"]),
        "baselineExpectedDays": int(window["baselineExpectedDays"]),
    }


def _base_evidence(
    *, title: str, source_label: str, window_label: str, summary: str, limitations: list[str], facts: list[dict[str, Any]],
    context_windows: list[dict[str, Any]] | None = None, outcome_horizon: str | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "title": title, "sourceLabel": source_label, "windowLabel": window_label,
        "summary": summary, "limitations": limitations, "facts": facts,
        "episodes": [], "contextPredicates": [], "matchedOn": [],
    }
    if context_windows:
        evidence["contextWindows"] = [_window_serialized(window) for window in context_windows]
    if outcome_horizon:
        evidence["outcomeHorizon"] = outcome_horizon
    return evidence


def _display_horizon(value: str) -> str:
    labels = {
        "meal_0_3h": "meal 0–3h",
        "upcoming_night_00_06": "upcoming night 00:00–06:00",
        "night_00_06": "night 00:00–06:00",
        "next_24h": "next 24 hours",
        "next_7d": "next 7 days",
    }
    return labels.get(value, value.replace("_", " "))


def _recent_night_trend(
    records: list[BriefRecord], current_night: date, zone: ZoneInfo, cutoff: datetime, config: BriefConfig, index: _RecordIndex | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    all_nights: dict[date, dict[str, Any]] = {}
    for day in _date_range(current_night - timedelta(days=34), current_night):
        start, end = _night_bounds(day, zone)
        series = _series(records, start, end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
        outcome = _outcome(series, start, end)
        if outcome is not None:
            all_nights[day] = {"date": day, "outcome": outcome, "trace": series["points"]}
    recent_dates = [day for day in _date_range(current_night - timedelta(days=6), current_night) if day in all_nights]
    baseline_dates = [day for day in _date_range(current_night - timedelta(days=34), current_night - timedelta(days=7)) if day in all_nights]
    recent = [all_nights[day] for day in recent_dates]
    baseline = [all_nights[day] for day in baseline_dates]
    diagnostic = {
        "recentValidNights": len(recent), "recentExpectedNights": 7,
        "baselineValidNights": len(baseline), "baselineExpectedNights": 28,
        "recentDates": [day.isoformat() for day in recent_dates],
        "baselineDates": [day.isoformat() for day in baseline_dates],
    }
    if len(recent) < 5 or len(baseline) < 14:
        return None, diagnostic
    recent_low = _mean(item["outcome"]["lowMinutes"] for item in recent) or 0.0
    baseline_low = _mean(item["outcome"]["lowMinutes"] for item in baseline) or 0.0
    qualifying = [item for item in recent if item["outcome"]["lowMinutes"] >= 15.0]
    diagnostic.update({"recentLowMinutes": recent_low, "baselineLowMinutes": baseline_low, "qualifyingRecentNights": len(qualifying)})
    if len(qualifying) < 3 or recent_low - baseline_low < 10.0:
        return None, diagnostic
    evidence = _base_evidence(
        title="Recent overnight glucose observations",
        source_label="Recorded CGM observations",
        window_label=f"Last 7 complete nights vs preceding 28 · {len(recent)} / {len(baseline)} valid nights",
        summary=f"Glucose was below 70 mg/dL on {len(qualifying)} of the last {len(recent)} valid nights; the recorded average was {recent_low:.0f} minutes vs {baseline_low:.0f} minutes in the preceding window.",
        limitations=["This is a recent descriptive change, not a tested explanation or predictive alert.", "Minutes use only observed five-minute bins; gaps are excluded."],
        facts=[
            {"label": "Recent nights below 70", "value": f"{len(qualifying)} of {len(recent)}"},
            {"label": "Recent average low minutes", "value": f"{recent_low:.0f} minutes/night"},
            {"label": "Preceding average low minutes", "value": f"{baseline_low:.0f} minutes/night"},
            {"label": "Coverage", "value": f"{len(recent)} / 7 recent · {len(baseline)} / 28 baseline"},
        ], outcome_horizon="upcoming_night_00_06",
    )
    evidence["episodes"] = [
        {"id": f"night-{item['date'].isoformat()}", "label": f"Night · {item['date'].isoformat()}", "group": "recent", "start": item["date"].isoformat(), "points": item["trace"][:288]}
        for item in recent[: config.max_evidence_episodes]
    ]
    return {"evidence": evidence, "recentLow": recent_low, "baselineLow": baseline_low, "recent": recent, "baseline": baseline}, diagnostic


def _build_rows(
    records: list[BriefRecord], zone: ZoneInfo, cutoff: datetime, config: BriefConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[date, dict[str, Any]], dict[date, dict[str, Any]]]:
    dates = sorted({_record_date(record, zone) for record in records if record.metric == "glucose" and record.known_at <= cutoff})
    if not dates:
        return [], [], {}, {}
    onsets = _period_onset_days(records, zone, cutoff)
    daily_summaries = _build_daily_summaries(records, zone, cutoff)
    index = _RecordIndex.build(records)
    context_cache: dict[date, dict[str, Any]] = {}
    outcomes_cache: dict[date, dict[str, Any]] = {}
    for day in dates:
        context, _ = _context_for_day(records, day, zone, cutoff, onsets, config, daily_summaries)
        context_cache[day] = context
        outcomes_cache[day] = _daily_outcomes(records, day, zone, cutoff, config, index)
    meals = _meal_episodes(records, zone, cutoff, config, index)
    meal_rows: list[dict[str, Any]] = []
    for episode in meals:
        if not episode["isolated"]:
            continue
        day = episode["date"]
        context = context_cache.get(day)
        if context is None:
            continue
        meal_outcome = episode["outcome"]
        row = {
            "id": episode["id"], "date": day, "context": context,
            "match": _anchor_fields(records, day, zone, cutoff, meal=episode, index=index),
            "outcomes": {
                "meal_mean_rise": meal_outcome["meanRise"],
                "meal_high_minutes": meal_outcome["highMinutes"],
            },
            "traces": {"meal_mean_rise": episode["trace"], "meal_high_minutes": episode["trace"]},
            "sourceLabel": episode["sourceLabel"], "episode": episode,
        }
        meal_rows.append(row)

    daily_rows: list[dict[str, Any]] = []
    for day in dates:
        context = context_cache[day]
        outcomes = outcomes_cache[day]
        night = outcomes.get("night"); next24 = outcomes.get("next24h"); next7 = outcomes.get("next7d")
        daily_rows.append({
            "id": f"day-{day.isoformat()}", "date": day, "context": context,
            "match": _anchor_fields(records, day, zone, cutoff, index=index),
            "outcomes": {
                "night_low_minutes": night["lowMinutes"] if night else None,
                "next24_low_minutes": next24["lowMinutes"] if next24 else None,
                "next24_tir": next24["tir"] if next24 else None,
                "next7_low_minutes": next7["lowMinutes"] if next7 else None,
                "next7_tir": next7["tir"] if next7 else None,
            },
            "traces": {
                "night_low_minutes": night["points"] if night else [],
                "next24_low_minutes": next24["points"] if next24 else [],
                "next24_tir": next24["points"] if next24 else [],
                "next7_low_minutes": next7["points"] if next7 else [],
                "next7_tir": next7["points"] if next7 else [],
            },
            "sourceLabel": "recorded glucose and context observations",
        })
    return daily_rows, meal_rows, context_cache, daily_summaries


def _partition_rows(
    rows: list[dict[str, Any]], *, current_day: date, horizon_days: int, config: BriefConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible = sorted({row["date"] for row in rows if row["date"] < current_day - timedelta(days=7)})
    if len(eligible) < max(config.min_pairs * 2, 24):
        return [], [], {"status": "insufficient_dates", "eligibleDates": len(eligible)}
    split_index = max(1, int(len(eligible) * 0.60))
    discovery_dates = eligible[:split_index]
    discovery_end = discovery_dates[-1]
    # Exposure windows are up to 56 days and the longest outcome is a week.
    # The first confirmation exposure must start after the latest discovery
    # outcome ends; otherwise a context history can straddle the split.
    earliest_confirmation = discovery_end + timedelta(days=max(config.purge_days, horizon_days + config.context_days + 1))
    confirmation_dates = [day for day in eligible if day >= earliest_confirmation]
    if not confirmation_dates:
        return [], [], {"status": "insufficient_purge", "eligibleDates": len(eligible), "discoveryEnd": discovery_end.isoformat()}
    discovery = [row for row in rows if row["date"] in set(discovery_dates)]
    confirmation = [row for row in rows if row["date"] in set(confirmation_dates)]
    return discovery, confirmation, {
        "status": "ready", "discoveryStart": discovery_dates[0].isoformat(), "discoveryEnd": discovery_end.isoformat(),
        "confirmationStart": confirmation_dates[0].isoformat(), "confirmationEnd": confirmation_dates[-1].isoformat(),
        "purgeDays": max(config.purge_days, horizon_days + config.context_days + 1),
    }


def _disjoint_horizon_rows(rows: Sequence[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    """Use one anchor per local week for the seven-day outcome family."""
    if not target.startswith("next7"):
        return list(rows)
    by_week: dict[str, dict[str, Any]] = {}
    for row in rows:
        week = _week(row["date"])
        prior = by_week.get(week)
        if prior is None or (row["date"], row["id"]) < (prior["date"], prior["id"]):
            by_week[week] = row
    return sorted(by_week.values(), key=lambda row: (row["date"], row["id"]))


def _row_predicate(row: Mapping[str, Any], predicate: str) -> bool | None:
    context = row.get("context", {})
    direct = context.get(predicate)
    if isinstance(direct, bool):
        return direct
    value = context.get("predicates", {}).get(predicate)
    return value if isinstance(value, bool) else None


def _candidate_library() -> tuple[dict[str, Any], ...]:
    """Return single predicates plus the explicitly allowed depth-two seed."""
    singles: dict[str, dict[str, Any]] = {}
    for candidate in _CANDIDATES:
        predicates = tuple(candidate["predicates"])
        if len(predicates) == 1:
            singles[predicates[0]] = dict(candidate)
    output = list(singles.values())
    # The short-sleep/activity interaction is a registered feature family, not
    # a truth-label shortcut.  Other pairs may be discovered by the bounded
    # subgroup search below.
    output.extend(
        dict(candidate) for candidate in _CANDIDATES
        if len(tuple(candidate["predicates"])) == 2
    )
    return tuple(output)


def _candidate_from_predicates(predicates: Sequence[str], labels: Mapping[str, str]) -> dict[str, Any]:
    ordered = tuple(predicates)
    candidate_id = "__".join(ordered)
    return {
        "id": candidate_id,
        "predicates": ordered,
        "label": " alongside ".join(labels.get(predicate, predicate) for predicate in ordered),
    }


def _discovery_score(rows: Sequence[dict[str, Any]], candidate: Mapping[str, Any], target: str) -> tuple[float, int, int, int]:
    positive: list[float] = []
    negative: list[float] = []
    positive_weeks: set[str] = set()
    negative_weeks: set[str] = set()
    for row in rows:
        value = _target_value(row, target)
        if value is None:
            continue
        flags = [_row_predicate(row, name) for name in candidate["predicates"]]
        if any(flag is None for flag in flags):
            continue
        if all(flags):
            positive.append(value); positive_weeks.add(_week(row["date"]))
        else:
            negative.append(value); negative_weeks.add(_week(row["date"]))
    if not positive or not negative:
        return 0.0, len(positive), len(negative), len(positive_weeks)
    return abs(float(_mean(positive) - _mean(negative))), len(positive), len(negative), len(positive_weeks)


def _bounded_subgroup_candidates(
    rows: Sequence[dict[str, Any]], target: str, config: BriefConfig,
) -> tuple[list[dict[str, Any]], str]:
    """Discover a finite positive registry using pysubgroup when available.

    The data frame passed to the optional library contains only rows where all
    registry predicates are observed.  Thus an unavailable stream never turns
    into a False selector.  The deterministic beam is the explicitly tested
    fallback for the minimal worker image; it searches the same finite
    registry and records that configuration in diagnostics.
    """
    library = _candidate_library()
    labels: dict[str, str] = {}
    # Prefer the readable singleton label when a predicate also appears in
    # the registered depth-two interaction.
    for candidate in library:
        if len(tuple(candidate["predicates"])) == 1:
            labels.setdefault(candidate["predicates"][0], candidate["label"])
    for candidate in library:
        for predicate in candidate["predicates"]:
            labels.setdefault(predicate, candidate["label"])
    names = tuple(labels)
    candidates: list[dict[str, Any]] = []
    for name in names:
        candidates.append(_candidate_from_predicates((name,), labels))
    if config.subgroup_max_depth >= 2:
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                candidates.append(_candidate_from_predicates((first, second), labels))
    scores = [(candidate, *_discovery_score(rows, candidate, target)) for candidate in candidates]
    scores = [item for item in scores if item[1] > 0 and item[2] >= 2 and item[3] >= 2]
    scores.sort(key=lambda item: (-item[1], -item[2], len(item[0]["predicates"]), item[0]["id"]))
    fallback = [item[0] for item in scores[: max(1, config.subgroup_beam_width)]]

    # The project pins pysubgroup for verification and the normal worker
    # image.  If an offline/minimal service image cannot import it, the
    # deterministic registry below remains an explicit degraded path;
    # confirmation still uses the product's own matching, purge, uncertainty
    # and multiplicity gates.  No textual negation is accepted.
    try:
        import pandas as pd  # type: ignore
        import pysubgroup as ps  # type: ignore
        searchable = [
            row for row in rows
            if _target_value(row, target) is not None
            and all(_row_predicate(row, name) is not None for name in names)
        ]
        if searchable:
            columns = {
                f"p{index}": [bool(_row_predicate(row, name)) for row in searchable]
                for index, name in enumerate(names)
            }
            columns[target] = [float(_target_value(row, target)) for row in searchable]
            frame = pd.DataFrame(columns)
            # The registry is intentionally one-sided: every feature is a
            # positive, observed predicate.  ``create_selectors`` also emits
            # the complementary ``pN==False`` selectors for boolean columns;
            # remove those before the beam so the library cannot turn an
            # unavailable or absent context into a discovered negative story.
            selectors = [
                selector for selector in ps.create_selectors(frame, ignore=[target])
                if bool(getattr(selector, "attribute_value", False))
            ]
            task = ps.SubgroupDiscoveryTask(
                frame, ps.NumericTarget(target), selectors, ps.StandardQFNumeric(a=1.0),
                result_set_size=config.subgroup_beam_width, depth=config.subgroup_max_depth,
                constraints=[ps.MinSupportConstraint(2)],
            )
            table = ps.BeamSearch(beam_width=config.subgroup_beam_width).execute(task).to_dataframe()
            ranked: list[dict[str, Any]] = []
            for _, result in table.iterrows():
                # pysubgroup 0.9 calls this column ``subgroup``; older
                # releases used ``description``.  Keep both spellings so a
                # pinned worker and a developer environment rank identically.
                description = str(result.get("subgroup", result.get("description", "")))
                lowered = description.lower()
                if "false" in lowered or "not(" in lowered or "~" in lowered:
                    continue
                # Use token boundaries: ``p1`` must not accidentally match
                # the selector token ``p10`` when the registry grows.
                selected = tuple(
                    names[index] for index in range(len(names))
                    if re.search(rf"\bp{index}\b", description)
                )
                if not selected or len(selected) > config.subgroup_max_depth:
                    continue
                candidate = _candidate_from_predicates(selected, labels)
                score = _discovery_score(rows, candidate, target)
                if score[1] >= 2 and score[2] >= 2:
                    ranked.append(candidate)
            if ranked:
                # Retain the explicit registered interaction even if the
                # library's quality score ranks it just outside the beam.
                by_id = {candidate["id"]: candidate for candidate in ranked + fallback}
                return list(by_id.values())[: max(1, config.subgroup_beam_width)], "pysubgroup"
    except (ImportError, AttributeError, TypeError, ValueError, KeyError):
        pass
    return fallback, "deterministic_finite_beam_pysubgroup_unavailable"


def _find_associations(
    rows_by_family: Sequence[tuple[list[dict[str, Any]], str]],
    *, current_day: date, config: BriefConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovery_candidates: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "tested": 0, "insufficient": 0, "discoveryCandidates": 0, "lockedCandidates": 0,
        "confirmed": 0, "byHorizon": {}, "searchedRegistry": [],
        "searchEngine": "unavailable",
        "subgroupDepth": config.subgroup_max_depth, "subgroupBeamWidth": config.subgroup_beam_width,
        "multiplicityFamilySize": 0,
    }
    searched_registry: set[str] = set()
    engines: set[str] = set()
    for rows, family in rows_by_family:
        for outcome_spec in _OUTCOMES:
            target = outcome_spec["id"]
            if family == "meal" and not target.startswith("meal_"):
                continue
            if family == "daily" and target.startswith("meal_"):
                continue
            horizon_days = 7 if target.startswith("next7") else 1
            target_rows = _disjoint_horizon_rows(rows, target)
            discovery, confirmation, partition = _partition_rows(target_rows, current_day=current_day, horizon_days=horizon_days, config=config)
            if target.startswith("next7"):
                diagnostics["next7DisjointAnchorCount"] = len(target_rows)
            if not discovery or not confirmation:
                diagnostics["insufficient"] += 1
                continue
            proposed, engine = _bounded_subgroup_candidates(discovery, target, config)
            engines.add(engine)
            for candidate in proposed:
                searched_registry.add(str(candidate["id"]))
                diagnostics["tested"] += 1
                pairs_d = _match_pairs(discovery, candidate, target)
                d_effect = _mean(pair["difference"] for pair in pairs_d)
                if d_effect is None or len(pairs_d) < config.min_pairs or len({_week(pair["context"]["date"]) for pair in pairs_d}) < config.min_weeks:
                    diagnostics["insufficient"] += 1
                    continue
                if abs(d_effect) < _threshold(target):
                    continue
                diagnostics["discoveryCandidates"] += 1
                discovery_candidates.append({
                    "candidate": candidate, "target": target, "horizon": outcome_spec["horizon"], "unit": outcome_spec["unit"],
                    "label": outcome_spec["label"], "family": family, "discovery": pairs_d,
                    "discoveryEffect": float(d_effect), "partition": partition, "confirmationRows": confirmation,
                })
    # Freeze the complete searched family before observing confirmation
    # outcomes.  Identical candidate/target/family identities are retained
    # once; Holm correction below covers every locked hypothesis.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in discovery_candidates:
        identity = (item["family"], item["target"], item["candidate"]["id"])
        prior = unique.get(identity)
        if prior is None or abs(item["discoveryEffect"]) > abs(prior["discoveryEffect"]):
            unique[identity] = item
    locked = sorted(
        unique.values(),
        key=lambda item: (-abs(item["discoveryEffect"]), item["family"], item["target"], item["candidate"]["id"]),
    )[: max(1, config.max_locked_candidates)]
    diagnostics["lockedCandidates"] = len(locked)
    diagnostics["multiplicityFamilySize"] = len(locked)
    diagnostics["searchedRegistry"] = sorted(searched_registry)
    if engines == {"pysubgroup"}:
        diagnostics["searchEngine"] = "pysubgroup"
    elif len(engines) == 1:
        diagnostics["searchEngine"] = next(iter(engines))
    elif engines:
        diagnostics["searchEngine"] = "mixed:" + ",".join(sorted(engines))
    else:
        diagnostics["searchEngine"] = "unavailable"
    candidates: list[dict[str, Any]] = []
    for item in locked:
        pairs_c = _match_pairs(item["confirmationRows"], item["candidate"], item["target"])
        c_effect = _mean(pair["difference"] for pair in pairs_c)
        if c_effect is None:
            diagnostics["insufficient"] += 1
            continue
        p_value, low, high = _block_statistics(
            pairs_c, draws=config.block_bootstrap_draws, seed=config.seed + len(candidates)
        )
        candidates.append({
            **item, "confirmation": pairs_c, "effect": float(c_effect),
            "low": low, "high": high, "p": p_value,
        })
    adjusted = _holm([float(item["p"]) for item in candidates])
    findings: list[dict[str, Any]] = []
    for item, q_value in zip(candidates, adjusted):
        same_direction = item["discoveryEffect"] * item["effect"] > 0
        excludes_zero = item["low"] is not None and item["high"] is not None and (item["low"] > 0 or item["high"] < 0)
        supported = q_value <= config.alpha and excludes_zero and same_direction and abs(item["effect"]) >= _threshold(item["target"])
        item["adjustedP"] = q_value
        item["supported"] = supported
        if supported:
            diagnostics["confirmed"] += 1
            horizon = item["horizon"]
            diagnostics["byHorizon"][horizon] = diagnostics["byHorizon"].get(horizon, 0) + 1
            diagnostics.setdefault("confirmedCandidates", []).append({
                "family": item["family"],
                "candidateId": item["candidate"]["id"],
                "target": item["target"],
                "horizon": horizon,
                "direction": "higher" if item["effect"] > 0 else "lower",
            })
            findings.append(item)
    return findings, diagnostics


def _current_predicate_value(context: Mapping[str, Any], predicate: str) -> bool | None:
    if predicate in context:
        value = context.get(predicate)
        return value if isinstance(value, bool) else None
    return context.get("predicates", {}).get(predicate)


def _association_evidence(
    item: Mapping[str, Any], *, current_context: Mapping[str, Any], config: BriefConfig
) -> dict[str, Any]:
    candidate = item["candidate"]
    target = item["target"]
    scale = 100.0 if item["unit"] == "fraction" else 1.0
    unit = "percentage points" if scale == 100.0 else item["unit"]
    effect = float(item["effect"]) * scale
    low = item.get("low") * scale if item.get("low") is not None else None
    high = item.get("high") * scale if item.get("high") is not None else None
    direction = "higher" if effect > 0 else "lower"
    context_label = candidate["label"]
    limitations = [
        "This is an association between recorded groups, not a treatment effect or recommendation.",
        "Matching used predeclared calipers; it does not establish complete confound removal.",
        "The confirmation interval uses a seven-day block bootstrap and sign-flip test with Holm correction across the searched family.",
        "Repeated future analysis snapshots still require separate error-control validation.",
    ]
    display_horizon = _display_horizon(item["horizon"])
    summary = f"After {context_label}, {item['label']} was {abs(effect):.1f} {unit} {direction} in the historical comparison."
    facts = [
        {"label": "Recorded difference", "value": f"{effect:+.1f} {unit}"},
        {"label": "Context support", "value": f"{len(item['confirmation'])} matched pairs"},
        {"label": "Comparison", "value": f"{item['partition'].get('confirmationStart', 'recorded')} to {item['partition'].get('confirmationEnd', 'recorded')}"},
        {"label": "Uncertainty", "value": f"[{low:.1f}, {high:.1f}] {unit}" if low is not None and high is not None else "Unavailable"},
    ]
    evidence = _base_evidence(
        title=f"Earlier {display_horizon} observations",
        source_label="Recorded CGM and context observations",
        window_label=f"Discovery {item['partition'].get('discoveryStart', '—')}–{item['partition'].get('discoveryEnd', '—')}; confirmation {item['partition'].get('confirmationStart', '—')}–{item['partition'].get('confirmationEnd', '—')}",
        summary=summary, limitations=limitations, facts=facts,
        context_windows=[], outcome_horizon=item["horizon"],
    )
    evidence["comparison"] = {
        "outcome": target, "unit": unit, "effect": effect, "low": low, "high": high,
        "contextN": len(item["confirmation"]), "comparisonN": len(item["confirmation"]),
    }
    evidence["comparisons"] = [evidence["comparison"]]
    evidence["contextPredicates"] = list(candidate["predicates"])
    fields = list(_target_fields(target))
    evidence["matchedOn"] = fields
    episodes: list[dict[str, Any]] = []
    # The UI can show actual representative traces without exposing source
    # document IDs.  Dates are the useful owner-facing identity of a day.
    for pair in item["confirmation"][: max(1, config.max_evidence_episodes // 2)]:
        for group, row in (("context", pair["context"]), ("comparison", pair["comparison"])):
            trace = row.get("traces", {}).get(target, [])
            if trace:
                episodes.append({
                    "id": f"{target}-{group}-{row['date'].isoformat()}",
                    "label": f"{group.title()} · {row['date'].isoformat()}", "group": group,
                    "start": row["date"].isoformat(), "points": trace[:288],
                })
    evidence["episodes"] = episodes[: config.max_evidence_episodes]
    return evidence


def _last_night_evidence(
    records: list[BriefRecord], *, night_day: date, zone: ZoneInfo, cutoff: datetime, config: BriefConfig, current_context: Mapping[str, Any], index: _RecordIndex | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    start, end = _night_bounds(night_day, zone)
    series = _series(records, start, end, cutoff=cutoff, min_coverage=config.min_coverage, index=index)
    outcome = _outcome(series, start, end)
    if outcome is None:
        return None, None
    low = outcome["lowMinutes"]
    facts: list[dict[str, Any]] = [
        {"label": "Observed below 70 mg/dL", "value": f"{low:.0f} minutes"},
        {"label": "CGM coverage", "value": f"{outcome['observedBins']} of {outcome['expectedBins']} five-minute bins ({outcome['coverage']:.0%})"},
    ]
    sleep_hours = current_context.get("sleepHours")
    if sleep_hours is not None:
        facts.append({"label": "Completed sleep", "value": f"{sleep_hours:.1f} hours"})
    wrist_delta = current_context.get("wristTemperatureDelta")
    if wrist_delta is not None:
        facts.append({"label": "Sleeping-wrist temperature", "value": f"{wrist_delta:+.1f} °C vs preceding baseline"})
    body_delta = current_context.get("bodyTemperatureDelta")
    if body_delta is not None:
        facts.append({"label": "Body temperature", "value": f"{body_delta:+.1f} °C vs preceding baseline"})
    if current_context.get("vigorousEvening"):
        facts.append({"label": "Preceding evening", "value": "Vigorous activity was recorded"})
    evidence = _base_evidence(
        title=f"Last night · {night_day.isoformat()}", source_label="Recorded CGM and available context observations",
        window_label=f"00:00–06:00 local · {night_day.isoformat()}",
        summary=f"The recorded overnight window had {low:.0f} minutes below 70 mg/dL using observed five-minute bins.",
        limitations=["This sheet describes the completed window; it is not a predictive alert.", "Missing CGM bins remain gaps and are not connected."],
        facts=facts, context_windows=[], outcome_horizon="night_00_06",
    )
    evidence["episodes"] = [{
        "id": f"night-recent-{night_day.isoformat()}", "label": f"Recorded night · {night_day.isoformat()}", "group": "recent", "start": night_day.isoformat(), "points": series["points"][:288]
    }]
    return evidence, f"Your recorded glucose had {low:.0f} minutes below 70 mg/dL last night."


def _last_night_item_text(evidence: Mapping[str, Any], night_day: date) -> str:
    values = {fact.get("label"): fact.get("value") for fact in evidence.get("facts", [])}
    low = values.get("Observed below 70 mg/dL")
    sleep = values.get("Completed sleep")
    if low and sleep:
        return f"You slept {sleep} and recorded {low} below 70 mg/dL overnight."
    if low:
        return f"You recorded {low} below 70 mg/dL overnight."
    return f"Last night’s recorded window is available for review."


def _brief_synthetic(records: list[BriefRecord]) -> bool:
    if not records:
        return False
    return all(
        bool(record.attributes.get("syntheticDemo") or record.source in {"synthetic-demo", "synthetic"})
        for record in records
    )


def build_daily_brief(
    records: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime | str,
    timezone: str | None = None,
    config: BriefConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic owner-readable daily brief.

    The current anchor's future outcomes are intentionally absent.  A current
    brief can link to what followed *analogous earlier contexts*, but it never
    uses today's future or synthetic future records to create a forecast.
    """
    cfg = config or BriefConfig()
    if cfg.engine_mode != "legacy":
        # Keep the public entrypoint stable while making the v3 engine the
        # default path used by the worker and local reports.  The import is
        # intentionally local: the legacy implementation below remains a
        # dependency-free fallback and the new module reuses its contracts.
        from .personal_context_engine import build_personal_context
        return build_personal_context(records, as_of=as_of, timezone=timezone, config=cfg)
    cutoff = _parse_datetime(as_of)
    parsed, report = normalize_brief_records(records, availability_cutoff=cutoff)
    zone_name = timezone or (parsed[0].timezone if parsed else cfg.timezone or "UTC")
    zone = _safe_zone(zone_name)
    local_as_of = cutoff.astimezone(zone)
    current_day = local_as_of.date() if local_as_of.hour >= 6 else local_as_of.date() - timedelta(days=1)
    # Never relabel an old night as "last night" when collection has stopped.
    observed_days = [_record_date(record, zone) for record in parsed if record.metric == "glucose"]
    stale = bool(observed_days and max(observed_days) < current_day)
    synthetic = _brief_synthetic(parsed)
    diagnostics: dict[str, Any] = {
        "engineRevision": ENGINE_REVISION, "datasetRevision": cfg.dataset_revision,
        "acceptedRecords": report.accepted, "rejectedRecords": report.rejected,
        "wrongUnitRecords": report.wrong_units, "duplicateRecords": report.duplicate_ids,
        "analysisOrigin": cutoff.date().isoformat(), "currentAnchor": current_day.isoformat(),
    }
    empty_base = {
        "schema": BRIEF_SCHEMA, "generatedAt": _iso(cutoff), "asOf": _iso(cutoff), "timezone": zone.key,
        "synthetic": synthetic, "title": "Your day, in context", "intro": "A brief from recorded observations.",
        "items": [], "evidence": {}, "evidenceFlags": {"heuristic": True, "method": ENGINE_REVISION},
        "analysisDiagnostics": diagnostics, "datasetRevision": cfg.dataset_revision,
    }
    if not parsed or not any(record.metric == "glucose" for record in parsed):
        empty_base["status"] = "insufficient"
        empty_base["intro"] = "There is not enough timestamped glucose history for a daily brief yet."
        return empty_base

    if stale:
        empty_base["status"] = "stale"
        empty_base["intro"] = "Waiting for recent glucose data before making today's brief."
        return empty_base

    daily_rows, meal_rows, context_cache, daily_summaries = _build_rows(parsed, zone, cutoff, cfg)
    record_index = _RecordIndex.build(parsed)
    current_context, _ = _context_for_day(parsed, current_day, zone, cutoff, _period_onset_days(parsed, zone, cutoff), cfg, daily_summaries)
    current_context.update(_current_sleep_and_context(parsed, cutoff, zone, cutoff))
    diagnostics["dailyRows"] = len(daily_rows); diagnostics["mealEpisodes"] = len(meal_rows)
    trend, trend_diagnostics = _recent_night_trend(parsed, current_day, zone, cutoff, cfg, record_index)
    diagnostics["recentNightTrend"] = trend_diagnostics
    rows_by_family = [(meal_rows, "meal"), (daily_rows, "daily")]
    findings, association_diagnostics = _find_associations(rows_by_family, current_day=current_day, config=cfg)
    diagnostics["associations"] = association_diagnostics
    diagnostics["featureRegistry"] = [
        candidate["id"] for candidate in _CANDIDATES
    ] + [
        "glucose_prior3h_mean_as_of", "glucose_prior3h_trend_as_of",
        "glucose_prior3h_low_minutes_as_of", "recorded_delivered_bolus_7v28",
    ]
    diagnostics["outcomeHorizons"] = [spec["horizon"] for spec in _OUTCOMES]

    items: list[dict[str, Any]] = []
    evidence_map: dict[str, dict[str, Any]] = {}
    # Recent nocturnal lows are selected first, but the trend does not claim a
    # reason.  It links to the recorded last-seven/baseline evidence sheet.
    if trend is not None:
        evidence_id = "recent-night-trend"
        evidence_map[evidence_id] = trend["evidence"]
        items.append({
            "id": "brief-recent-nights", "kind": "trend", "title": "Recent nights",
            "text": f"Your glucose was below 70 mg/dL on {trend['evidence']['facts'][0]['value'].split(' of ')[0]} of the last 7 valid nights.", "evidenceId": evidence_id,
        })

    # Select one relevant supported association.  Exact predicate matches are
    # required, and support is retained for the detail sheet rather than
    # turning a historical association into a current prediction.
    relevant: list[dict[str, Any]] = []
    for finding in findings:
        candidate = finding["candidate"]
        flags = [_current_predicate_value(current_context, name) for name in candidate["predicates"]]
        if flags and all(flag is True for flag in flags):
            relevant.append(finding)
    relevant.sort(key=lambda item: (0 if item["horizon"] == "meal_0_3h" else 1, abs(item["effect"])))
    if relevant and len(items) < 2:
        finding = relevant[0]
        evidence_id = "history-" + hashlib.sha256((finding["candidate"]["id"] + finding["target"]).encode()).hexdigest()[:12]
        evidence_map[evidence_id] = _association_evidence(finding, current_context=current_context, config=cfg)
        evidence = evidence_map[evidence_id]
        item_title = {"meal_0_3h": "Those mornings", "next_24h": "Similar days", "next_7d": "Similar weeks"}.get(finding["horizon"], "Similar nights")
        items.append({
            "id": "brief-history-" + finding["candidate"]["id"], "kind": "history", "title": item_title,
            "text": evidence["summary"], "evidenceId": evidence_id,
        })

    last_evidence, _ = _last_night_evidence(parsed, night_day=current_day, zone=zone, cutoff=cutoff, config=cfg, current_context=current_context, index=record_index)
    if last_evidence is not None and len(items) < 3:
        evidence_id = "last-night"
        evidence_map[evidence_id] = last_evidence
        items.append({
            "id": "brief-last-night", "kind": "context", "title": "Last night",
            "text": _last_night_item_text(last_evidence, current_day), "evidenceId": evidence_id,
        })

    if not items:
        # A quiet result is a valid outcome.  It does not invent a pattern or
        # say that an absent stream was normal.
        latest = max((record.known_at for record in parsed), default=cutoff)
        age_hours = max(0.0, (cutoff - latest).total_seconds() / 3600.0)
        empty_base["status"] = "stale" if age_hours > 36 else ("insufficient" if len(daily_rows) < 7 else "quiet")
        empty_base["intro"] = "No supported daily finding is available from the recorded windows yet."
    else:
        latest = max((record.known_at for record in parsed), default=cutoff)
        age_hours = max(0.0, (cutoff - latest).total_seconds() / 3600.0)
        empty_base["status"] = "stale" if age_hours > 36 else "ready"
        empty_base["intro"] = "Recorded context and earlier comparable windows are linked below."
    empty_base["items"] = items[:3]
    empty_base["evidence"] = evidence_map
    items.sort(key=lambda item: {"context": 0, "trend": 1, "history": 2}.get(item["kind"], 3))
    empty_base["items"] = items[:3]
    return _jsonable(empty_base)


__all__ = [
    "BRIEF_SCHEMA", "ENGINE_REVISION", "DATASET_REVISION", "BriefConfig", "BriefRecord",
    "NormalizationReport", "normalize_brief_records", "build_daily_brief",
]
