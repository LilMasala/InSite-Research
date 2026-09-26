"""Adapter and readiness gate for the existing physiological twin."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from .observation import NormalizedTimeline


@dataclass(frozen=True)
class TwinEligibility:
    status: str
    reason_codes: tuple[str, ...]
    training_days: int
    evaluation_days: int
    cgm_coverage: float
    basal_coverage: float
    actual_delivery: bool
    used_inputs: tuple[str, ...]
    unavailable_inputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwinInputBundle:
    raw_day_records: tuple[dict[str, Any], ...]
    therapy_settings: tuple[dict[str, Any], ...]
    sleep_daily: tuple[dict[str, Any], ...]
    eligibility: TwinEligibility
    app_profile: str = "app_v1"


def overlay_timestamped_glucose(records: list[dict[str, Any]], observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Place v2 measured glucose on a legacy-compatible grid without filling gaps."""
    out = deepcopy(records)

    def parse(value: str) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)

    samples = [o for o in observations if o.get("metric") in {"glucose", "blood_glucose"} and o.get("kind", "measured") == "measured" and o.get("quality", "observed") != "legacy_dense"]
    for record in out:
        bins = int(record["bin_count"])
        block = record.setdefault("dense", {}).setdefault("cgm_mgdl", {})
        block["values"] = [None] * bins
        block["observed_flag"] = [0] * bins
        anchor = parse(record["utc_anchor"])
        for sample in samples:
            try:
                when = parse(sample["start"])
                value = float(sample["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if when < anchor or when >= anchor + timedelta(minutes=5 * bins):
                continue
            index = int((when - anchor).total_seconds() // 300)
            if 0 <= index < bins and block["observed_flag"][index] == 0:
                block["values"][index] = value
                block["observed_flag"][index] = 1
    return out


def assess_twin_eligibility(
    timeline: NormalizedTimeline,
    *,
    complete_days: int,
    training_days: int | None = None,
    evaluation_days: int | None = None,
    cgm_coverage: float = 0.0,
    basal_coverage: float = 0.0,
    actual_delivery: bool = False,
) -> TwinEligibility:
    """Return a conservative readiness result without modifying source data."""
    train = training_days if training_days is not None else max(0, complete_days - 7)
    evaluation = evaluation_days if evaluation_days is not None else min(7, complete_days)
    reasons: list[str] = []
    if timeline.quality.legacy_records:
        reasons.append("legacy_dense_observations")
    if complete_days < 21:
        reasons.append("need_21_complete_days")
    if train < 14:
        reasons.append("need_14_training_days")
    if evaluation < 5:
        reasons.append("need_5_evaluation_days")
    if cgm_coverage < 0.70:
        reasons.append("cgm_coverage_below_70_percent")
    if basal_coverage < 0.90:
        reasons.append("basal_delivery_coverage_below_90_percent")
    if not actual_delivery:
        reasons.append("actual_delivery_unavailable")
    return TwinEligibility(
        status="ready" if not reasons else "waiting_for_delivery_data" if "actual_delivery_unavailable" in reasons else "waiting_for_data",
        reason_codes=tuple(reasons), training_days=train, evaluation_days=evaluation,
        cgm_coverage=cgm_coverage, basal_coverage=basal_coverage, actual_delivery=actual_delivery,
        used_inputs=("timestamped_cgm", "delivered_bolus", "delivered_basal", "recorded_meals", "sleep_daily"),
        unavailable_inputs=("prospective_forecast", "settings_counterfactuals"),
    )


def build_twin_inputs(
    timeline: NormalizedTimeline,
    *,
    raw_day_records: list[dict[str, Any]],
    therapy_settings: list[dict[str, Any]] | None = None,
    sleep_daily: list[dict[str, Any]] | None = None,
    complete_days: int,
    cgm_coverage: float,
    basal_coverage: float,
    actual_delivery: bool,
) -> TwinInputBundle:
    eligibility = assess_twin_eligibility(
        timeline, complete_days=complete_days, cgm_coverage=cgm_coverage,
        basal_coverage=basal_coverage, actual_delivery=actual_delivery,
    )
    return TwinInputBundle(
        tuple(raw_day_records), tuple(therapy_settings or []), tuple(sleep_daily or []), eligibility,
    )
