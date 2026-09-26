from copy import deepcopy
from datetime import UTC, date, datetime, timedelta

from insite_analytics.daily_brief import BriefConfig
import insite_analytics.personal_context_engine as pce


def _trace(values=None, *, omit=()):
    values = values or {}
    omitted = set(omit)
    return [
        {"minute": float(index * 5), "glucose": float(values.get(index, 100.0))}
        for index in range(36)
        if index not in omitted
    ]


def _row(index, target="firstResponse", *, points=None, hour=9.0, profile="profile-a", outcome=-5.0):
    anchor = datetime(2025, 1, 1, 9, tzinfo=UTC) + timedelta(days=index)
    return {
        "id": f"synthetic-{index}",
        "date": date(2025, 1, 1) + timedelta(days=index),
        "anchor": anchor,
        "outcomeEnds": {target: anchor + timedelta(minutes=180)},
        target: outcome,
        "trace": _trace() if points is None else points,
        "events": [],
        "context": {"hour": hour, "therapy_profile_fingerprint": profile},
    }


def _item(pairs, target="firstResponse", episode_kind="meal"):
    return {
        "id": "synthetic-evidence",
        "effect": -8.0,
        "low": -12.0,
        "high": -4.0,
        "predicates": (),
        "pairs": pairs,
        "outcomeDescriptor": {**pce.OUTCOME_DESCRIPTORS[target], "episodeKind": episode_kind},
    }


def _pairs(context_rows, comparison_rows, target="firstResponse"):
    assert len(context_rows) == len(comparison_rows)
    return [
        {"left": left, "right": right, "difference": 0.0}
        for left, right in zip(context_rows, comparison_rows)
    ]


def test_event_evidence_aggregates_every_pair_beyond_episode_display_cap():
    left = [_row(index, profile=f"left-{index}") for index in range(15)]
    right = [_row(index + 20, profile=f"right-{index}") for index in range(15)]
    item = _item(_pairs(left, right))
    before = deepcopy(item["pairs"])

    evidence = pce._event_evidence(item, target="firstResponse", config=BriefConfig(max_evidence_episodes=12))

    summary = evidence["responseOutcomeContext"]
    assert summary == {
        "version": 1,
        "target": "firstResponse",
        "windowMinutes": 180,
        "lowThresholdMgDl": 70,
        "highThresholdMgDl": 180,
        "minimumRunMinutes": 15,
        "groups": {
            group: {
                "totalEpisodes": 15,
                "eligibleEpisodes": 15,
                "complete": True,
                "lowEpisodes": 0,
                "highEpisodes": 0,
                "fullyInRangeEpisodes": 15,
                "lowMinutesMedian": 0.0,
                "highMinutesMedian": 0.0,
                "localAnchorHourMin": 9.0,
                "localAnchorHourMax": 9.0,
                "profileKnownEpisodes": 15,
                "singleObservedProfile": False,
            }
            for group in ("context", "comparison")
        },
    }
    assert len(evidence["episodes"]) == 12
    assert item["pairs"] == before


def test_trace_bins_are_finite_deduplicated_and_gaps_are_not_filled():
    points = _trace({0: 65, 1: 65, 3: 65}, omit=(2,))
    points.extend([
        {"minute": 0.0, "glucose": 65.0},  # duplicate observed bin
        {"minute": 10.0, "glucose": float("nan")},
        {"minute": 12.0, "glucose": 60.0},  # off-grid
        {"minute": 180.0, "glucose": 60.0},
        {"minute": -5.0, "glucose": 60.0},
    ])
    before = deepcopy(points)
    normalized = pce._response_trace_points({"trace": points})
    assert len(normalized) == 35
    assert [point["minute"] for point in normalized].count(0.0) == 1
    assert 10.0 not in [point["minute"] for point in normalized]
    assert 12.0 not in [point["minute"] for point in normalized]
    assert all(0.0 <= point["minute"] < 180.0 for point in normalized)
    assert points == before

    result = pce._response_outcome_context(
        _item(_pairs([_row(0, points=points)], [_row(1)])),
        target="firstResponse", config=BriefConfig(min_coverage=0.8), episode_kind="meal",
    )
    group = result["groups"]["context"]
    assert group["eligibleEpisodes"] == 1
    assert group["lowEpisodes"] == 0  # the missing 10-minute bin breaks the run
    assert group["lowMinutesMedian"] == 15.0  # three actually observed low bins


def test_sustained_threshold_events_use_three_contiguous_bins_and_mean_shift_is_not_a_low():
    points = _trace({0: 69.9, 1: 69.9, 2: 69.9, 6: 180.1, 7: 180.1, 8: 180.1})
    result = pce._response_outcome_context(
        _item(_pairs([_row(0, points=points)], [_row(1)])),
        target="firstResponse", config=BriefConfig(min_coverage=0.8), episode_kind="meal",
    )
    group = result["groups"]["context"]
    assert group["lowEpisodes"] == 1
    assert group["highEpisodes"] == 1
    assert group["lowMinutesMedian"] == 15.0
    assert group["highMinutesMedian"] == 15.0

    shifted_without_threshold_crossing = _row(2, points=_trace({index: 75 for index in range(36)}), outcome=-35.0)
    no_low = pce._response_outcome_context(
        _item(_pairs([shifted_without_threshold_crossing], [_row(3)])),
        target="firstResponse", config=BriefConfig(min_coverage=0.8), episode_kind="meal",
    )
    assert no_low["groups"]["context"]["lowEpisodes"] == 0


def test_eligibility_in_range_completeness_hours_and_profile_flags():
    partial_trace = _trace(omit=(35,))
    context = [
        _row(0, points=_trace({index: 70 for index in range(36)}), hour=8, profile="same-profile"),
        _row(1, points=_trace({index: 180 for index in range(36)}), hour=12, profile="same-profile"),
        _row(2, points=_trace(omit=range(8)), hour=15, profile="other-profile"),
    ]
    comparison = [
        _row(3, points=_trace(), hour=7, profile="profile-x"),
        _row(4, points=_trace(), hour=None, profile=None),
        _row(5, points=_trace(), hour=11, profile="profile-y"),
    ]
    result = pce._response_outcome_context(
        _item(_pairs(context, comparison)),
        target="firstResponse", config=BriefConfig(min_coverage=0.8), episode_kind="meal",
    )

    context_group = result["groups"]["context"]
    assert context_group["totalEpisodes"] == 3
    assert context_group["eligibleEpisodes"] == 2
    assert context_group["complete"] is False
    assert context_group["fullyInRangeEpisodes"] == 2
    assert context_group["localAnchorHourMin"] == 8.0
    assert context_group["localAnchorHourMax"] == 15.0
    assert context_group["profileKnownEpisodes"] == 3
    assert context_group["singleObservedProfile"] is True

    comparison_group = result["groups"]["comparison"]
    assert comparison_group["complete"] is True
    assert comparison_group["fullyInRangeEpisodes"] == 3
    assert comparison_group["localAnchorHourMin"] is None
    assert comparison_group["localAnchorHourMax"] is None
    assert comparison_group["profileKnownEpisodes"] == 2
    assert comparison_group["singleObservedProfile"] is False
    assert "profile-x" not in str(result) and "profile-y" not in str(result)

    one_missing_bin = pce._response_outcome_context(
        _item(_pairs([_row(6, points=partial_trace)], [_row(7)])),
        target="firstResponse", config=BriefConfig(min_coverage=0.8), episode_kind="meal",
    )
    assert one_missing_bin["groups"]["context"]["eligibleEpisodes"] == 1
    assert one_missing_bin["groups"]["context"]["complete"] is True
    assert one_missing_bin["groups"]["context"]["fullyInRangeEpisodes"] == 0


def test_metadata_is_limited_to_the_two_supported_target_episode_pairs():
    item = _item([], target="firstResponse", episode_kind="meal")
    config = BriefConfig()
    assert pce._response_outcome_context(item, target="lateResponse", config=config, episode_kind="meal") is None
    assert pce._response_outcome_context(item, target="firstResponse", config=config, episode_kind="delivered_bolus") is None
    bolus_target = "bolusResponse0_3h"
    bolus_item = _item(
        _pairs([_row(0, target=bolus_target)], [_row(1, target=bolus_target)]),
        target=bolus_target, episode_kind="delivered_bolus",
    )
    assert pce._response_outcome_context(
        bolus_item, target=bolus_target, config=config, episode_kind="delivered_bolus",
    ) is not None
    evidence = pce._event_evidence(bolus_item, target=bolus_target, config=config)
    assert evidence["responseOutcomeContext"]["target"] == bolus_target
