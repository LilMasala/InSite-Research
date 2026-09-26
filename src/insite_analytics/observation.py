"""Timestamp-preserving observation and analysis-unit contracts.

The iOS v1 dense day format can contain replicated hourly means.  This module
does not guess which bins are real.  It accepts the v2 timestamped sidecar (or
explicitly marked legacy records) and reports quality rather than fabricating
coverage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any, Iterable
from zoneinfo import ZoneInfo

UTC = timezone.utc
OBSERVATION_SCHEMA = "insite.observations.v2"


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


@dataclass(frozen=True)
class Observation:
    id: str
    metric: str
    value: float | str
    unit: str
    start: datetime
    end: datetime | None
    timezone: str
    source: str
    kind: str
    available_at: datetime | None = None
    quality: str = "observed"
    revision: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Observation":
        required = ("id", "metric", "value", "unit", "start", "timezone", "source", "kind")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"observation missing fields: {','.join(missing)}")
        start = _utc(raw["start"])
        end = _utc(raw["end"]) if raw.get("end") else None
        if end is not None and end < start:
            raise ValueError("observation end precedes start")
        if raw.get("kind") == "measured" and raw.get("quality", "observed") == "legacy_dense":
            raise ValueError("legacy dense values cannot be used as measured observations")
        value = raw["value"]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("observation value must be scalar")
        if isinstance(value, (int, float)) and _finite(value) is None:
            raise ValueError("numeric observation value must be finite")
        available = _utc(raw["available_at"]) if raw.get("available_at") else None
        return cls(
            id=str(raw["id"]), metric=str(raw["metric"]), value=value, unit=str(raw["unit"]),
            start=start, end=end, timezone=str(raw["timezone"]), source=str(raw["source"]),
            kind=str(raw["kind"]), available_at=available, quality=str(raw.get("quality", "observed")),
            revision=int(raw.get("revision", 0)),
        )


@dataclass(frozen=True)
class QualityReport:
    schema: str
    accepted: int
    rejected: int
    duplicate_ids: int
    legacy_records: int
    conflicts: int
    reasons: tuple[str, ...] = ()

    @property
    def eligible_for_raw_coverage(self) -> bool:
        return self.schema == OBSERVATION_SCHEMA and self.accepted > 0 and self.legacy_records == 0


@dataclass
class NormalizedTimeline:
    observations: list[Observation]
    quality: QualityReport

    def metric(self, name: str, *, kind: str | None = "measured") -> list[Observation]:
        return [o for o in self.observations if o.metric == name and (kind is None or o.kind == kind)]

    def available_as_of(self, cutoff: datetime | str | None) -> "NormalizedTimeline":
        if cutoff is None:
            return self
        instant = _utc(cutoff)
        values = [o for o in self.observations if o.available_at is None or o.available_at <= instant]
        return NormalizedTimeline(values, self.quality)


def normalize_observations(
    records: Iterable[dict[str, Any]],
    *,
    availability_cutoff: datetime | str | None = None,
) -> NormalizedTimeline:
    """Normalize, deduplicate, and quality-check v2 observations.

    Legacy dense records are counted and rejected for analysis.  Duplicate IDs
    keep the newest revision; conflicting same-revision values are rejected.
    """
    source = list(records)
    schema = OBSERVATION_SCHEMA if all(r.get("schema") == OBSERVATION_SCHEMA for r in source) else "legacy"
    by_id: dict[str, Observation] = {}
    rejected = 0
    legacy = sum(1 for r in source if r.get("quality") == "legacy_dense" or r.get("schema") != OBSERVATION_SCHEMA)
    conflicts = 0
    reasons: list[str] = []
    for raw in source:
        try:
            obs = Observation.from_dict(raw)
        except ValueError as exc:
            rejected += 1
            reasons.append(str(exc))
            continue
        previous = by_id.get(obs.id)
        if previous is None or obs.revision > previous.revision:
            by_id[obs.id] = obs
        elif obs.revision == previous.revision and (obs.value != previous.value or obs.start != previous.start):
            conflicts += 1
            rejected += 1
            reasons.append(f"conflicting revision for {obs.id}")
    values = list(by_id.values())
    if availability_cutoff is not None:
        cutoff = _utc(availability_cutoff)
        values = [o for o in values if o.available_at is None or o.available_at <= cutoff]
    values.sort(key=lambda o: (o.start, o.metric, o.id))
    return NormalizedTimeline(
        values,
        QualityReport(schema, len(values), rejected, len(source) - len(by_id), legacy, conflicts, tuple(reasons[:10])),
    )


def _local_window(start: datetime, tz: str, window: str) -> tuple[date, datetime, datetime]:
    zone = ZoneInfo(tz)
    local = start.astimezone(zone)
    if window == "day":
        anchor = local.replace(hour=6, minute=0, second=0, microsecond=0)
        if local < anchor:
            anchor -= timedelta(days=1)
        end = anchor + timedelta(hours=12)
    elif window == "night":
        anchor = local.replace(hour=18, minute=0, second=0, microsecond=0)
        if local < anchor:
            anchor -= timedelta(days=1)
        end = anchor + timedelta(hours=12)
    elif window == "full_day":
        anchor = local.replace(hour=0, minute=0, second=0, microsecond=0)
        end = anchor + timedelta(days=1)
    else:
        raise ValueError(f"unsupported analysis window: {window}")
    return anchor.date(), anchor.astimezone(UTC), end.astimezone(UTC)


def build_analysis_units(
    timeline: NormalizedTimeline,
    *,
    windows: tuple[str, ...] = ("day", "night", "full_day"),
    glucose_metric: str = "glucose",
    bin_minutes: int = 5,
) -> list[dict[str, Any]]:
    """Build one row per local day/window with honest occupied-minute coverage."""
    glucose = timeline.metric(glucose_metric, kind="measured")
    if not glucose:
        return []
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for obs in glucose:
        local_date_by_window = []
        for window in windows:
            key, start, end = _local_window(obs.start, obs.timezone, window)
            local_date_by_window.append((window, key, start, end))
        for window, key, start, end in local_date_by_window:
            row_key = (window, key.isoformat(), obs.timezone)
            # Build lazily in a dictionary to avoid assuming one global timezone.
            found = rows_by_key.get(row_key)
            if found is None:
                found = {"_key": row_key, "window": window, "local_date": key.isoformat(), "timezone": obs.timezone, "start_utc": start, "end_utc": end, "glucose": []}
                rows_by_key[row_key] = found
            if start <= obs.start < end:
                found["glucose"].append(obs)
    clean: list[dict[str, Any]] = []
    for row in rows_by_key.values():
        samples = sorted(row.pop("glucose"), key=lambda o: o.start)
        values_by_bin: dict[int, list[float]] = defaultdict(list)
        for sample in samples:
            value = _finite(sample.value)
            if value is None:
                continue
            bin_number = int((sample.start - row["start_utc"]).total_seconds() // (bin_minutes * 60))
            values_by_bin[bin_number].append(value)
        unique_bins = set(values_by_bin)
        # Each occupied interval contributes one robust value.  A duplicated
        # sample (or a burst of readings in one bin) must not change the day's
        # mean/TIR merely by increasing its row count.
        values = [median(values_by_bin[bin_number]) for bin_number in sorted(values_by_bin)]
        expected = max(1, int((row["end_utc"] - row["start_utc"]).total_seconds() // (bin_minutes * 60)))
        row.update({
            "sample_count": len(samples),
            "occupied_bins": len(unique_bins),
            "expected_bins": expected,
            "coverage": len(unique_bins) / expected,
            "mean_glucose": sum(values) / len(values) if values else None,
            "tir": sum(70 <= v <= 180 for v in values) / len(values) if values else None,
            "tbr70": sum(v < 70 for v in values) / len(values) if values else None,
            "tbr54": sum(v < 54 for v in values) / len(values) if values else None,
            "tar180": sum(v > 180 for v in values) / len(values) if values else None,
            "values": values,
        })
        row.pop("_key", None)
        clean.append(row)
    return sorted(clean, key=lambda r: (r["local_date"], r["window"]))


def attach_context(
    rows: list[dict[str, Any]],
    timeline: NormalizedTimeline,
    *,
    baseline_days: int = 14,
) -> list[dict[str, Any]]:
    """Attach only source-observed context to analysis rows.

    Context is summarized from observations ending before the row's outcome
    window.  It intentionally does not infer cycle phase or site changes.
    """
    by_metric: dict[str, list[Observation]] = defaultdict(list)
    for obs in timeline.observations:
        by_metric[obs.metric].append(obs)
    for values in by_metric.values():
        values.sort(key=lambda o: o.start)

    def prior_values(metric: str, row: dict[str, Any], *, days: int = baseline_days) -> list[float]:
        candidates = by_metric.get(metric, [])
        end = row["start_utc"]
        start = end - timedelta(days=days)
        values: list[float] = []
        for observation in candidates:
            # Interval-valued context (especially sleep) is usable only after
            # the interval has completed. A point observation retains the
            # ordinary start-time semantics.
            completed_at = observation.end or observation.start
            if completed_at > end or completed_at < start:
                continue
            numeric = _finite(observation.value)
            if numeric is not None:
                values.append(numeric)
        return values

    result = []
    for source in rows:
        row = dict(source)
        # Sleep is intentionally limited to the immediately preceding,
        # completed 24-hour period. This prevents an old sleep sample from
        # being carried into a later outcome window.
        sleep = prior_values("sleep_hours", row, days=1)
        active = prior_values("exercise_minutes", row)
        hr = prior_values("resting_hr", row)
        temp = prior_values("body_temperature", row)
        row["prior_sleep_hours"] = sleep[-1] if sleep else None
        row["morning_active_minutes"] = sum(active[-12:]) if active else None
        row["resting_hr_delta"] = (hr[-1] - median(hr)) if hr else None
        row["body_temperature_delta"] = (temp[-1] - median(temp)) if temp else None
        # Explicit cycle observations only; no 28-day phase synthesis.
        cycle = prior_values("days_since_period", row)
        row["days_since_period"] = cycle[-1] if cycle else None
        site = prior_values("site_age_days", row)
        row["site_age_days"] = site[-1] if site else None
        result.append(row)
    return result
