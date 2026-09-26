"""Discovery-only numeric subgroup proposal using pysubgroup.

This adapter keeps the mining implementation in pysubgroup. It supplies
discovery-fitted interval selectors and a candidate-aware numeric quality
function so rows missing one of a rule's features are unknown for that rule,
not comparator negatives.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pysubgroup as ps


FEATURES: dict[str, tuple[str, str]] = {
    "glucose_1h_mean": ("preceding 1-hour glucose", "mg/dL"),
    "glucose_1h_slope": ("preceding 1-hour glucose trend", "mg/dL per hour"),
    "glucose_3h_mean": ("preceding 3-hour glucose", "mg/dL"),
    "glucose_3h_slope": ("preceding 3-hour glucose trend", "mg/dL per hour"),
    "glucose_3h_variability": ("preceding 3-hour glucose variability", "mg/dL"),
    "glucose_3h_low_rate": ("preceding 3-hour observed low fraction", "fraction"),
    "glucose_3h_high_rate": ("preceding 3-hour observed high fraction", "fraction"),
    "glucose_6h_mean": ("preceding 6-hour glucose", "mg/dL"),
    "glucose_24h_mean": ("preceding 24-hour glucose", "mg/dL"),
    "glucose_7d_vs_prior28_mean": ("recent versus earlier glucose", "mg/dL"),
    "delivered_bolus_3h": ("recorded delivered bolus in 3 hours", "units"),
    "delivered_bolus_6h": ("recorded delivered bolus in 6 hours", "units"),
    "delivered_bolus_24h": ("recorded delivered bolus in 24 hours", "units"),
    "delivered_bolus_7d_daily": ("recent average daily delivered bolus", "units per day"),
    "delivered_bolus_prior28_daily": ("earlier average daily delivered bolus", "units per day"),
    "meal_carbs_3h": ("recorded carbohydrates in 3 hours", "g"),
    "meal_carbs_6h": ("recorded carbohydrates in 6 hours", "g"),
    "meal_carbs_24h": ("recorded carbohydrates in 24 hours", "g"),
    "meal_carbs_7d_daily": ("recent average daily carbohydrates", "g per day"),
    "meal_carbs_prior28_daily": ("earlier average daily carbohydrates", "g per day"),
    "exercise_minutes_6h": ("recorded activity in 6 hours", "minutes"),
    "exercise_minutes_24h": ("recorded activity in 24 hours", "minutes"),
    "exercise_minutes_7d_daily": ("recent average daily activity", "minutes per day"),
    "exercise_minutes_prior28_daily": ("earlier average daily activity", "minutes per day"),
    "steps_6h": ("recorded steps in 6 hours", "steps"),
    "steps_24h": ("recorded steps in 24 hours", "steps"),
    "steps_7d_daily": ("recent average daily steps", "steps per day"),
    "steps_prior28_daily": ("earlier average daily steps", "steps per day"),
    "heart_rate_6h_mean": ("preceding 6-hour heart rate", "bpm"),
    "heart_rate_24h_mean": ("preceding 24-hour heart rate", "bpm"),
    "heart_rate_7d_mean": ("recent heart rate", "bpm"),
    "heart_rate_prior28_mean": ("earlier heart rate", "bpm"),
    "heart_rate_deviation_24h": ("heart-rate deviation", "bpm"),
    "hrv_sdnn_24h_mean": ("preceding 24-hour HRV SDNN", "ms"),
    "hrv_sdnn_deviation_24h": ("HRV SDNN deviation", "ms"),
    "steps_daytime_24h": ("recorded daytime steps", "steps"),
    "steps_overnight_24h": ("recorded overnight steps", "steps"),
    "body_temperature_7d_vs_prior28": ("recent body temperature versus earlier baseline", "°C"),
    "wrist_temperature_7d_vs_prior28": ("recent sleeping-wrist temperature versus earlier baseline", "°C"),
    "sleep_7d_mean": ("recent average completed sleep", "hours"),
    "sleep_prior28_mean": ("earlier average completed sleep", "hours"),
    "days_since_period": ("time since recorded cycle onset", "days"),
    "workout_minutes_6h": ("recorded workout duration in 6 hours", "minutes"),
    "workout_average_hr_24h": ("recorded workout average heart rate", "bpm"),
    "workout_count_24h": ("recorded workout count", "workouts"),
    "sleep_variability_7d": ("completed sleep variability over 7 days", "hours"),
    "sleep_shortfall_7d_hours": ("recorded sleep shortfall over 7 days versus earlier personal baseline", "hours"),
    "sleep_hours": ("completed sleep duration", "hours"),
    "sleep_7d_mean": ("recent average completed sleep", "hours"),
    "sleep_prior28_mean": ("earlier average completed sleep", "hours"),
    "sleep_rem_fraction": ("completed sleep REM fraction", "fraction"),
    "sleep_deep_fraction": ("completed sleep deep fraction", "fraction"),
    "sleep_fragmentation_per_hour": ("completed sleep awake bouts", "bouts per hour"),
    "sleep_stage_efficiency": ("completed sleep efficiency", "fraction"),
    "heart_rate_deviation_24h": ("heart-rate deviation", "bpm"),
    "hrv_sdnn_deviation_24h": ("HRV SDNN deviation", "ms"),
    "mood_valence_24h": ("recorded mood valence", "scale points"),
    "mood_arousal_24h": ("recorded mood arousal", "scale points"),
    "mood_valence_24h_mean": ("recorded mood valence over 24 hours", "scale points"),
    "mood_arousal_24h_mean": ("recorded mood arousal over 24 hours", "scale points"),
    "mood_valence_last": ("last recorded mood valence", "scale points"),
    "mood_valence_last_age_hours": ("age of last recorded mood valence", "hours"),
    "mood_arousal_last": ("last recorded mood arousal", "scale points"),
    "mood_arousal_last_age_hours": ("age of last recorded mood arousal", "hours"),
    "mood_valence_24h_std": ("24-hour recorded mood valence variability", "scale points"),
    "mood_arousal_24h_std": ("24-hour recorded mood arousal variability", "scale points"),
    "mood_valence_7d_mean": ("7-day recorded mood valence", "scale points"),
    "mood_valence_7d_std": ("7-day recorded mood valence variability", "scale points"),
    "mood_arousal_7d_mean": ("7-day recorded mood arousal", "scale points"),
    "mood_arousal_7d_std": ("7-day recorded mood arousal variability", "scale points"),
    "mood_valence_7d_vs_prior28": ("recent mood valence versus earlier observed days", "scale points"),
    "mood_arousal_7d_vs_prior28": ("recent mood arousal versus earlier observed days", "scale points"),
    "site_age_days": ("time since recorded site change", "days"),
    "site_previous_dwell_days": ("previous completed site dwell", "days"),
    "site_median_dwell_days": ("median completed site dwell", "days"),
    "site_dwell_std_days": ("variability in completed site dwell", "days"),
    "site_age_vs_prior_median_days": ("current site age versus earlier median dwell", "days"),
    "site_location": ("recorded infusion-site location", "category"),
    "cycle_days": ("time since recorded cycle onset", "days"),
    "cycle_last_interval_days": ("last completed recorded cycle interval", "days"),
    "cycle_median_interval_days": ("median completed recorded cycle interval", "days"),
    "cycle_interval_std_days": ("variability in completed recorded cycle intervals", "days"),
    "cycle_elapsed_vs_prior_median_days": ("elapsed time since onset versus earlier completed interval median", "days"),
    "flow_severity_7d_mean": ("recorded menstrual flow severity over 7 days", "ordered category score"),
    "flow_severity_last_7d": ("last recorded menstrual flow severity in 7 days", "ordered category score"),
    "flow_last_age_hours": ("age of last recorded menstrual flow day", "hours"),
    "cramp_severity_24h_mean": ("recorded cramp severity over 24 hours", "ordered category score"),
    "cramp_severity_last_7d": ("last recorded cramp severity in 7 days", "ordered category score"),
    "cramp_last_age_hours": ("age of last recorded cramp observation", "hours"),
    "days_since_period": ("time since recorded cycle onset", "days"),
    "start_glucose": ("starting glucose", "mg/dL"),
    "pre_glucose_slope": ("pre-anchor glucose trend", "mg/dL per hour"),
    "carbs": ("recorded carbohydrates", "g"),
    "bolus": ("recorded delivered bolus", "units"),
    "hour": ("local anchor time", "hour of day"),
    "active_6h": ("recorded activity before the anchor", "minutes"),
    "activity_7d_daily": ("recent average daily activity", "minutes per day"),
    "body_temperature_delta": ("body-temperature deviation", "°C"),
    "wrist_temperature_delta": ("sleeping-wrist temperature deviation", "°C"),
    "body_temperature_context": ("recent body temperature versus earlier baseline", "°C"),
    "wrist_temperature_context": ("recent sleeping-wrist temperature versus earlier baseline", "°C"),
    "steps_daytime_24h": ("recorded daytime steps", "steps"),
    "steps_overnight_24h": ("recorded overnight steps", "steps"),
    "sleep_rem_fraction": ("completed sleep REM fraction", "fraction"),
    "workout_minutes_24h": ("recorded workout duration", "minutes"),
}

SUPPLEMENTAL_FEATURES = {
    "recorded_iob_24h_mean": ("recorded_iob", "mean", "recorded pump insulin on board", "U"),
    "body_mass_24h_mean": ("body_mass", "mean", "recorded body mass", "kg"),
    "resting_heart_rate_24h_mean": ("resting_heart_rate", "mean", "recorded resting heart rate", "bpm"),
    "active_energy_24h": ("active_energy", "sum", "recorded active energy", "kcal"),
    "basal_energy_24h": ("basal_energy", "sum", "recorded resting energy", "kcal"),
    "move_minutes_24h": ("move_minutes", "sum", "recorded move time", "minutes"),
    "workout_distance_24h": ("workout_distance", "sum", "recorded workout distance", "m"),
    "workout_min_hr_24h": ("workout_min_hr", "mean", "recorded workout minimum heart rate", "bpm"),
    "workout_max_hr_24h": ("workout_max_hr", "mean", "recorded workout maximum heart rate", "bpm"),
    "workout_energy_24h": ("workout_energy_kcal", "sum", "recorded workout energy", "kcal"),
}
FEATURES.update({key: (spec[2], spec[3]) for key, spec in SUPPLEMENTAL_FEATURES.items()})
FEATURES.update({"therapy_carb_ratio": ("observed carbohydrate ratio at this time", "g/U"),
                 "therapy_isf": ("observed correction factor at this time", "mg/dL/U"),
                 "therapy_scheduled_basal": ("scheduled basal at this time", "U/hour")})

DAILY_FEATURES = tuple(dict.fromkeys((*FEATURES, *(
    "sleep_variability_7d", "sleep_shortfall_7d_hours",
    "sleep_hours", "sleep_rem_fraction", "sleep_deep_fraction",
    "sleep_fragmentation_per_hour", "sleep_stage_efficiency",
    "heart_rate_deviation_24h", "hrv_sdnn_deviation_24h",
    "mood_valence_24h", "mood_arousal_24h", "site_age_days", "cycle_days",
    "start_glucose", "pre_glucose_slope", "carbs", "bolus", "hour",
    "active_6h", "activity_7d_daily", "body_temperature_context",
    "wrist_temperature_context", "workout_minutes_24h",
))))
EVENT_FEATURES = tuple(dict.fromkeys((*DAILY_FEATURES, "body_temperature_delta", "wrist_temperature_delta")))
CATEGORICAL_FEATURES = {"site_location"}

# These pairs are explicit aliases created by the shared context builder.  A
# pair is collapsed only when the discovery rows have exactly equal values
# and missingness.  Other same-looking measurements (for example
# ``start_glucose`` and ``glucose_1h_mean``) stay separate because their
# episode/baseline semantics can differ.
_SEARCH_FEATURE_ALIASES = {
    "activity_7d_daily": "exercise_minutes_7d_daily",
    "active_6h": "exercise_minutes_6h",
    "cycle_days": "days_since_period",
    "body_temperature_context": "body_temperature_7d_vs_prior28",
    "wrist_temperature_context": "wrist_temperature_7d_vs_prior28",
    "mood_valence_24h": "mood_valence_24h_mean",
    "mood_arousal_24h": "mood_arousal_24h_mean",
}

_NumericStats = namedtuple(
    "EligibleNumericStats", ("size_sg", "mean", "eligible_size", "eligible_mean", "estimate")
)


class _BoundedMeanCache:
    """Exact row-mask means with a fixed memory ceiling."""

    def __init__(self, capacity: int = 8192):
        self.capacity = max(1, int(capacity))
        self.values: OrderedDict[int, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, mask_bits: int, values: np.ndarray, row_count: int,
            packed_byte_count: int) -> float:
        cached = self.values.get(mask_bits)
        if cached is not None:
            self.hits += 1
            self.values.move_to_end(mask_bits)
            return cached
        self.misses += 1
        if mask_bits:
            mask = np.unpackbits(
                np.frombuffer(mask_bits.to_bytes(packed_byte_count, "little"), dtype=np.uint8),
                bitorder="little", count=row_count,
            ).astype(bool, copy=False)
            mean = float(np.mean(values[mask]))
        else:
            mean = 0.0
        self.values[mask_bits] = mean
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return mean


@dataclass(frozen=True)
class PreparedNumericSearchSpace:
    """Feature-derived parts shared across targets/directions on same rows."""

    row_ids: tuple[str, ...]
    features: tuple[str, ...]
    bins: int
    rows: tuple[Mapping[str, Any], ...]
    data: pd.DataFrame
    usable_features: tuple[str, ...]
    selectors: tuple[Any, ...]
    cutpoints: Mapping[str, int]
    coverage: Mapping[str, Mapping[str, int]]
    finite_by_feature: Mapping[str, np.ndarray]
    selector_cover: Mapping[int, np.ndarray]
    canonicalized_aliases: Mapping[str, str]


class _EligibleStandardQFNumeric(ps.StandardQFNumeric):
    """Native StandardQFNumeric score with each rule's measured-row baseline."""

    def __init__(
        self, data: pd.DataFrame, target: ps.NumericTarget, *, min_support: int,
        finite_by_feature: Mapping[str, np.ndarray] | None = None,
        selector_cover: Mapping[int, np.ndarray] | None = None,
        use_bitset_masks: bool = True,
        mean_cache_capacity: int = 8192,
    ):
        super().__init__(a=0.5, centroid="mean")
        self.tpl = _NumericStats
        self.required_stat_attrs = ("size_sg", "mean")
        self._minimum_support = int(min_support)
        self._evaluations = 0
        self._max_depth = 0
        self._finite_by_feature = dict(finite_by_feature) if finite_by_feature is not None else {
            key: (data[key].notna().to_numpy(dtype=bool) if key in CATEGORICAL_FEATURES
                  else np.isfinite(data[key].to_numpy(dtype=float)))
            for key in data.columns if key != target.target_variable
        }
        self._values = data[target.target_variable].to_numpy(dtype=float)
        self._selector_cover = dict(selector_cover or {})
        self._use_bitset_masks = bool(use_bitset_masks)
        self._row_count = len(self._values)
        self._packed_byte_count = (self._row_count + 7) // 8
        self._full_mask_bits = (1 << self._row_count) - 1
        self._finite_mask_bits = {
            feature: int.from_bytes(np.packbits(mask, bitorder="little").tobytes(), "little")
            for feature, mask in self._finite_by_feature.items()
        } if self._use_bitset_masks else {}
        self._selector_mask_bits = {
            selector_id: int.from_bytes(np.packbits(mask, bitorder="little").tobytes(), "little")
            for selector_id, mask in self._selector_cover.items()
        } if self._use_bitset_masks else {}
        self._eligible_mean_cache = _BoundedMeanCache(mean_cache_capacity)
        self._positive_mean_cache = _BoundedMeanCache(mean_cache_capacity)

    def calculate_constant_statistics(self, data: pd.DataFrame, target: ps.NumericTarget) -> None:
        del target
        self.all_target_values = self._values
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup: Any, target: ps.NumericTarget, data: pd.DataFrame, statistics: Any = None) -> _NumericStats:
        del target, statistics
        selectors = tuple(getattr(subgroup, "selectors", ()))
        self._evaluations += 1
        self._max_depth = max(self._max_depth, len(selectors))
        if self._use_bitset_masks:
            eligible_bits = self._full_mask_bits
            positive_bits = self._full_mask_bits
            for selector in selectors:
                name = str(selector.attribute_name)
                eligible_bits &= self._finite_mask_bits[name]
                selector_id = id(selector)
                cover_bits = self._selector_mask_bits.get(selector_id)
                if cover_bits is None:
                    cover = self._selector_cover.get(selector_id)
                    if cover is None:
                        cover = np.asarray(selector.covers(data), dtype=bool)
                        self._selector_cover[selector_id] = cover
                    cover_bits = int.from_bytes(np.packbits(cover, bitorder="little").tobytes(), "little")
                    self._selector_mask_bits[selector_id] = cover_bits
                positive_bits &= cover_bits
            positive_bits &= eligible_bits
            n_positive = positive_bits.bit_count()
            n_eligible = eligible_bits.bit_count()
            # These are separate groups: an empty rule-positive set still
            # has a real candidate-specific eligible baseline.
            mean_positive = self._positive_mean_cache.get(
                positive_bits, self._values, self._row_count, self._packed_byte_count,
            )
            mean_eligible = self._eligible_mean_cache.get(
                eligible_bits, self._values, self._row_count, self._packed_byte_count,
            )
        else:
            eligible = np.ones(len(data), dtype=bool)
            positive = np.ones(len(data), dtype=bool)
            for selector in selectors:
                name = str(selector.attribute_name)
                eligible &= self._finite_by_feature[name]
                cover = self._selector_cover.get(id(selector))
                if cover is None:
                    cover = np.asarray(selector.covers(data), dtype=bool)
                    self._selector_cover[id(selector)] = cover
                positive &= cover
            positive &= eligible
            n_positive = int(positive.sum())
            n_eligible = int(eligible.sum())
            mean_positive = float(np.mean(self._values[positive])) if n_positive else 0.0
            mean_eligible = float(np.mean(self._values[eligible])) if n_eligible else 0.0
        return _NumericStats(n_positive, mean_positive, n_eligible, mean_eligible, 0.0)

    def evaluate(self, subgroup: Any, target: ps.NumericTarget, data: pd.DataFrame, statistics: Any = None) -> float:
        stats = statistics or self.calculate_statistics(subgroup, target, data)
        if stats.size_sg < self._minimum_support or stats.eligible_size <= stats.size_sg:
            return float("-inf")
        return float(ps.StandardQFNumeric.standard_qf_numeric(
            self.a, stats.eligible_size, stats.eligible_mean, stats.size_sg, stats.mean,
        ))


class _UniqueFeatureConstraint:
    """Do not combine multiple intervals for one feature; all their ranges
    already exist as single native IntervalSelectors in the search space."""

    is_monotone = False

    def is_satisfied(self, subgroup: Any, statistics: Any = None, data: Any = None) -> bool:
        del statistics, data
        names = [selector.attribute_name for selector in getattr(subgroup, "selectors", ())]
        return len(names) == len(set(names))


def _finite_values(rows: Sequence[Mapping[str, Any]], feature: str) -> np.ndarray:
    values = np.asarray([
        row.get("context", row).get(feature) if isinstance(row.get("context", row), Mapping) else None
        for row in rows
    ], dtype=object)
    numeric = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            numeric[index] = parsed
    return numeric


def _feature_values(rows: Sequence[Mapping[str, Any]], feature: str) -> np.ndarray:
    if feature not in CATEGORICAL_FEATURES:
        return _finite_values(rows, feature)
    values = []
    for row in rows:
        context = row.get("context", row)
        value = context.get(feature) if isinstance(context, Mapping) else None
        text = str(value).strip() if value is not None else ""
        values.append(text if text else None)
    return np.asarray(values, dtype=object)


def _make_selectors(data: pd.DataFrame, features: Sequence[str], bins: int) -> tuple[list[Any], dict[str, int]]:
    selectors: list[Any] = []
    cutpoint_counts: dict[str, int] = {}
    for feature in features:
        if feature in CATEGORICAL_FEATURES:
            values = [str(value) for value in data[feature].dropna().unique()]
            for value in sorted(values):
                selectors.append(ps.EqualitySelector(feature, value))
            cutpoint_counts[feature] = 0
            continue
        values = data[feature].to_numpy(dtype=float)
        observed = values[np.isfinite(values)]
        if not len(observed) or len(np.unique(observed)) < 2:
            continue
        unique = np.unique(observed)
        if len(unique) <= bins:
            for value in unique:
                selectors.append(ps.EqualitySelector(feature, float(value)))
            cutpoints = (unique[:-1] + unique[1:]) / 2.0
        else:
            quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)[1:-1]
            cutpoints = np.unique(np.quantile(observed, quantiles))
            cutpoints = cutpoints[(cutpoints > np.min(observed)) & (cutpoints < np.max(observed))]
        cutpoints = np.asarray(cutpoints, dtype=float)
        cutpoints = np.unique(cutpoints[np.isfinite(cutpoints)])
        cutpoint_counts[feature] = int(len(cutpoints))
        edges = [float("-inf"), *[float(value) for value in cutpoints], float("inf")]
        # Include both tails and every bounded range defined by the
        # discovery-only quantile/equality cutpoints.
        for left in range(len(edges) - 1):
            for right in range(left + 1, len(edges)):
                if left == 0 and right == len(edges) - 1:
                    continue  # the selector covering the whole finite domain
                selectors.append(ps.IntervalSelector(feature, edges[left], edges[right]))
    # Quantile ties and low-cardinality values can make several different
    # descriptors cover the same discovery rows. Keep one deterministic
    # representative per feature-specific eligible/positive mask before they
    # consume native beam capacity.
    unique_selectors: list[Any] = []
    by_feature: dict[str, dict[bytes, Any]] = {}
    for selector in selectors:
        feature = str(selector.attribute_name)
        mask = np.asarray(selector.covers(data), dtype=bool).tobytes()
        prior = by_feature.setdefault(feature, {}).get(mask)
        if prior is None:
            by_feature[feature][mask] = selector
            continue
        def rank(candidate: Any) -> tuple[Any, ...]:
            if isinstance(candidate, ps.IntervalSelector):
                low, high = float(candidate.lower_bound), float(candidate.upper_bound)
                width = high - low if math.isfinite(low) and math.isfinite(high) else float("inf")
                return (0, width, repr(candidate))
            return (1, 0.0, repr(candidate))
        if rank(selector) < rank(prior):
            by_feature[feature][mask] = selector
    for feature in features:
        unique_selectors.extend(by_feature.get(feature, {}).values())
    return unique_selectors, cutpoint_counts


def _same_feature_values(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact numeric equality including identical unknown-value positions."""
    try:
        left_numeric = np.asarray(left, dtype=float)
        right_numeric = np.asarray(right, dtype=float)
    except (TypeError, ValueError):
        return False
    return left_numeric.shape == right_numeric.shape and bool(
        np.array_equal(left_numeric, right_numeric, equal_nan=True)
    )


def prepare_numeric_search_space(
    rows: Sequence[Mapping[str, Any]], residuals: Mapping[str, float], *,
    features: Sequence[str], bins: int = 6,
) -> PreparedNumericSearchSpace:
    """Prepare the discovery-row search space once for reusable target runs.

    The selector objects, their coverage masks, and feature-missingness masks
    are deterministic functions of these rows and discovery-only boundaries.
    Reusing them between residual targets and opposite score directions is
    result-preserving.
    """
    aligned = tuple(row for row in rows if str(row.get("id")) in residuals)
    row_ids = tuple(str(row.get("id")) for row in aligned)
    feature_values = {
        feature: _feature_values(aligned, feature)
        for feature in features if feature in FEATURES
    }
    canonicalized: dict[str, str] = {}
    for alias, canonical in _SEARCH_FEATURE_ALIASES.items():
        if alias in feature_values and canonical in feature_values and _same_feature_values(
            feature_values[alias], feature_values[canonical]
        ):
            feature_values.pop(alias)
            canonicalized[alias] = canonical
    usable_features = tuple(
        feature for feature, values in feature_values.items()
        if (pd.notna(values).any() if feature in CATEGORICAL_FEATURES else np.isfinite(values).any())
    )
    data = pd.DataFrame({feature: feature_values[feature] for feature in usable_features})
    coverage = {
        feature: {
            "observed": int(pd.notna(feature_values[feature]).sum()
                            if feature in CATEGORICAL_FEATURES
                            else np.isfinite(feature_values[feature]).sum()),
            "rows": len(aligned),
        }
        for feature in usable_features
    }
    selectors, cutpoints = _make_selectors(data, usable_features, bins) if usable_features else ([], {})
    finite_by_feature = {
        key: (data[key].notna().to_numpy(dtype=bool) if key in CATEGORICAL_FEATURES
              else np.isfinite(data[key].to_numpy(dtype=float)))
        for key in usable_features
    }
    selector_cover = {
        id(selector): np.asarray(selector.covers(data), dtype=bool)
        for selector in selectors
    }
    return PreparedNumericSearchSpace(
        row_ids=row_ids, features=tuple(features), bins=max(1, int(bins)), rows=aligned,
        data=data, usable_features=usable_features, selectors=tuple(selectors),
        cutpoints=cutpoints, coverage=coverage, finite_by_feature=finite_by_feature,
        selector_cover=selector_cover, canonicalized_aliases=canonicalized,
    )


def _rule_from_selectors(selectors: Sequence[Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for selector in selectors:
        feature = str(selector.attribute_name)
        if feature in CATEGORICAL_FEATURES and isinstance(selector, ps.EqualitySelector):
            value = str(selector.attribute_value)
            rules.append({"feature": feature, "kind": "equal", "value": value,
                          "unit": FEATURES[feature][1], "featureLabel": FEATURES[feature][0]})
        elif isinstance(selector, ps.IntervalSelector):
            low = float(selector.lower_bound)
            high = float(selector.upper_bound)
            rules.append({
                "feature": feature, "kind": "interval",
                "lower": low if math.isfinite(low) else None,
                "upper": high if math.isfinite(high) else None,
                "lowerInclusive": True, "upperInclusive": False,
                "unit": FEATURES[feature][1], "featureLabel": FEATURES[feature][0],
            })
        elif isinstance(selector, ps.EqualitySelector):
            raw_value = selector.attribute_value
            value = str(raw_value) if feature in CATEGORICAL_FEATURES else float(raw_value)
            rules.append({
                "feature": feature, "kind": "equal", "value": value,
                "unit": FEATURES[feature][1], "featureLabel": FEATURES[feature][0],
            })
        else:
            continue
    return sorted(rules, key=lambda rule: (rule["feature"], rule["kind"], str(rule)))


def rule_id(rules: Sequence[Mapping[str, Any]], *, engine: str, target: str) -> str:
    canonical = json.dumps(list(rules), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{engine}-numeric-{digest}-{target}"


def _display_number(value: float) -> str:
    # Shortest general representation is readable for common boundaries
    # while retaining enough precision to expose their actual values.
    return format(float(value), ".6g")


def describe_rule(rule: Mapping[str, Any]) -> str:
    label = str(rule.get("featureLabel") or rule["feature"].replace("_", " "))
    unit = str(rule.get("unit") or "")
    suffix = f" {unit}" if unit and unit != "category" else ""
    if rule.get("kind") == "equal":
        value = str(rule["value"]) if rule.get("unit") == "category" else _display_number(float(rule["value"]))
        return f"{label} was {value}{suffix}"
    lower = rule.get("lower")
    upper = rule.get("upper")
    if lower is None:
        return f"{label} was below {_display_number(float(upper))}{suffix}"
    if upper is None:
        return f"{label} was at least {_display_number(float(lower))}{suffix}"
    return (
        f"{label} was from {_display_number(float(lower))} up to, but not including, "
        f"{_display_number(float(upper))}{suffix}"
    )


def evaluate_rule(row: Mapping[str, Any], rule: Mapping[str, Any]) -> bool | None:
    value = row.get(str(rule["feature"]))
    if rule.get("unit") == "category":
        if value is None:
            return None
        return str(value) == str(rule.get("value"))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if rule.get("kind") == "equal":
        return numeric == float(rule["value"])
    lower = rule.get("lower")
    upper = rule.get("upper")
    return (lower is None or numeric >= float(lower)) and (upper is None or numeric < float(upper))


def discover_numeric_rules(
    rows: Sequence[Mapping[str, Any]], residuals: Mapping[str, float], *,
    features: Sequence[str], engine: str, target: str, direction: int,
    depth: int = 4, bins: int = 6, beam_width: int = 256, result_size: int = 128,
    min_support: int = 3, prepared_space: PreparedNumericSearchSpace | None = None,
    _use_bitset_masks: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one native pysubgroup numeric search on discovery residuals only."""
    expected_row_ids = tuple(str(row.get("id")) for row in rows if str(row.get("id")) in residuals)
    prepared = prepared_space or prepare_numeric_search_space(rows, residuals, features=features, bins=bins)
    if (prepared.row_ids != expected_row_ids or prepared.features != tuple(features)
            or prepared.bins != max(1, int(bins))):
        raise ValueError("prepared_numeric_search_space_does_not_match_discovery_rows")
    aligned = prepared.rows
    usable_features = prepared.usable_features
    aliases = dict(prepared.canonicalized_aliases)
    base_diagnostics = {
        "rows": len(aligned), "features": len(usable_features),
        "selectors": len(prepared.selectors), "cutpoints": dict(prepared.cutpoints),
        "featureCoverage": dict(prepared.coverage),
        "canonicalizedFeatureAliases": aliases,
        "searchVocabularyCanonicalization": "explicit semantic aliases removed only when discovery values and missingness match exactly; this may change heuristic beam traversal",
    }
    if not aligned or not usable_features or not prepared.selectors:
        return [], {**base_diagnostics, "evaluated": 0, "maxDepthReached": 0, "proposals": 0}
    data = prepared.data.copy(deep=False)
    data["__target__"] = np.asarray([float(residuals[row_id]) * int(direction) for row_id in prepared.row_ids], dtype=float)
    target_definition = ps.NumericTarget("__target__")
    qf = _EligibleStandardQFNumeric(
        data, target_definition, min_support=max(1, int(min_support)),
        finite_by_feature=prepared.finite_by_feature,
        selector_cover=prepared.selector_cover,
        use_bitset_masks=_use_bitset_masks,
    )
    task = ps.SubgroupDiscoveryTask(
        data=data, target=target_definition, search_space=prepared.selectors, qf=qf,
        result_set_size=max(1, int(result_size)), depth=max(1, int(depth)),
        constraints=[ps.MinSupportConstraint(max(1, int(min_support))), _UniqueFeatureConstraint()],
    )
    result = ps.BeamSearch(beam_width=max(int(beam_width), int(result_size))).execute(task)
    proposals: dict[str, dict[str, Any]] = {}
    for quality, subgroup, _statistics in result.results:
        rules = _rule_from_selectors(getattr(subgroup, "selectors", ()))
        if not rules or float(quality) <= 0.0:
            continue
        identifier = rule_id(rules, engine=engine, target=target)
        proposal = {
            "id": identifier, "engine": engine, "target": target,
            "predicates": tuple(rule["feature"] for rule in rules),
            "ruleDescriptors": rules, "subgroupScore": float(quality),
            "searchDirection": int(direction),
        }
        prior = proposals.get(identifier)
        if prior is None or proposal["subgroupScore"] > prior["subgroupScore"]:
            proposals[identifier] = proposal
    return list(proposals.values()), {
        **base_diagnostics,
        "evaluated": qf._evaluations, "maxDepthReached": qf._max_depth,
        "meanStatisticsCache": {
            "capacityPerGroup": qf._eligible_mean_cache.capacity,
            "eligibleHits": qf._eligible_mean_cache.hits,
            "positiveHits": qf._positive_mean_cache.hits,
            "entriesPerGroupMaximum": qf._eligible_mean_cache.capacity,
        },
        "proposals": len(proposals),
        "algorithm": "pysubgroup.BeamSearch", "heuristic": True,
        "configuredDepth": max(1, int(depth)), "configuredBeamWidth": max(int(beam_width), int(result_size)),
    }
