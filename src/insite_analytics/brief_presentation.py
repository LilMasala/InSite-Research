"""Deterministic, evidence-preserving assembly of the participant brief.

This module changes only what is shown first and how related findings are
grouped. It never changes discovery, matching, confirmation, or evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence


_CLOCK = "clock_window"
_ASSOCIATION = "numeric_association"
_MIN_GROUP_JACCARD = 0.75


def _evidence_for(item: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    value = evidence.get(item.get("evidenceId"), {})
    return value if isinstance(value, Mapping) else {}


def _coverage(item: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    detail = _evidence_for(item, evidence).get("presentationCoverage")
    if not isinstance(detail, Mapping) or detail.get("complete") is not True:
        return None
    return detail


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _window_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        left_start = float(left["startHour"])
        left_width = float(left["widthHours"])
        right_start = float(right["startHour"])
        right_width = float(right["widthHours"])
    except (KeyError, TypeError, ValueError):
        return False
    if left_width <= 0 or right_width <= 0:
        return False
    # Clock windows are local-day bins and are never allowed to cross midnight.
    return max(left_start, right_start) < min(left_start + left_width, right_start + right_width)


def _compatible(left: Mapping[str, Any], right: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[bool, float]:
    left_coverage = _coverage(left, evidence)
    right_coverage = _coverage(right, evidence)
    if left_coverage is None or right_coverage is None:
        return False, 0.0
    if left_coverage.get("kind") != right_coverage.get("kind"):
        return False, 0.0

    if left_coverage.get("kind") == _CLOCK:
        if left_coverage.get("outcome") != right_coverage.get("outcome"):
            return False, 0.0
        # The window is oriented to the higher observed outcome before it is
        # compared, so an item whose selected side was lower can still match
        # an item whose selected side was higher.
        if (left_coverage.get("orientation") != "higher_outcome"
                or right_coverage.get("orientation") != "higher_outcome"):
            return False, 0.0
        if not _window_overlap(left_coverage.get("higherWindow", {}), right_coverage.get("higherWindow", {})):
            return False, 0.0
        left_support = {str(value) for value in left_coverage.get("confirmationSupportDates", [])}
        right_support = {str(value) for value in right_coverage.get("confirmationSupportDates", [])}
        overlap = _jaccard(left_support, right_support)
        return overlap >= _MIN_GROUP_JACCARD, overlap

    if left_coverage.get("kind") == _ASSOCIATION:
        for key in ("episodeKind", "outcome", "direction"):
            if left_coverage.get(key) != right_coverage.get(key):
                return False, 0.0
        left_support = {str(value) for value in left_coverage.get("positiveEpisodeIds", [])}
        right_support = {str(value) for value in right_coverage.get("positiveEpisodeIds", [])}
        if not left_support or not right_support:
            return False, 0.0
        overlap = _jaccard(left_support, right_support)
        return overlap >= _MIN_GROUP_JACCARD, overlap
    return False, 0.0


def _group_findings(
    items: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Coalesce only findings with complete support provenance and high overlap."""
    ordered = [copy.deepcopy(dict(item)) for item in items]
    consumed: set[int] = set()
    output: list[dict[str, Any]] = []
    group_metadata: list[dict[str, Any]] = []
    for index, representative in enumerate(ordered):
        if index in consumed:
            continue
        members = [index]
        overlaps: list[float] = []
        for candidate_index in range(index + 1, len(ordered)):
            if candidate_index in consumed:
                continue
            compatible = True
            candidate_overlaps: list[float] = []
            for prior_index in members:
                can_group, overlap = _compatible(ordered[prior_index], ordered[candidate_index], evidence)
                if not can_group:
                    compatible = False
                    break
                candidate_overlaps.append(overlap)
            if compatible:
                members.append(candidate_index)
                overlaps.extend(candidate_overlaps)
        if len(members) == 1:
            output.append(representative)
            continue

        related = [copy.deepcopy(ordered[member]) for member in members[1:]]
        for member in members[1:]:
            consumed.add(member)
        source_ids = [str(ordered[member].get("id", "")) for member in members]
        evidence_ids = [str(ordered[member].get("evidenceId", "")) for member in members if ordered[member].get("evidenceId")]
        group_kind = str(_coverage(representative, evidence).get("kind"))  # type: ignore[union-attr]
        group_id = "presentation-" + hashlib.sha256("|".join(sorted(source_ids)).encode("utf-8")).hexdigest()[:16]
        representative["relatedFindings"] = related
        representative["presentationMetadata"] = {
            "groupId": group_id,
            "groupKind": group_kind,
            "relatedFindingIds": source_ids[1:],
            "relatedEvidenceIds": evidence_ids[1:],
            "sourceFindingCount": len(members),
            "minimumSupportOverlap": min(overlaps) if overlaps else 1.0,
            "representativeClaimPreserved": True,
        }
        output.append(representative)
        group_metadata.append({
            "groupId": group_id, "groupKind": group_kind,
            "representativeFindingId": source_ids[0], "sourceFindingIds": source_ids,
            "evidenceIds": evidence_ids,
            "sourceFindingCount": len(members),
            "minimumSupportOverlap": min(overlaps) if overlaps else 1.0,
            "representativeWindowAndComparatorPreserved": True,
        })
    return output, group_metadata


def _is_context(item: Mapping[str, Any]) -> bool:
    return item.get("id") == "brief-current-context" or item.get("kind") == "context"


def _is_analogue(item: Mapping[str, Any]) -> bool:
    return item.get("kind") == "analogue" or item.get("id") == "brief-analogue"


def _informative_analogue(item: Mapping[str, Any]) -> bool:
    facts = item.get("messageFacts")
    facts = facts if isinstance(facts, Mapping) else {}
    return int(facts.get("eventDays", 0) or 0) >= 2 and int(facts.get("eligibleDays", 0) or 0) >= 2


def _is_complex_or_variance(item: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    details = _evidence_for(item, evidence)
    comparison = details.get("comparison", {})
    comparison = comparison if isinstance(comparison, Mapping) else {}
    descriptor = comparison.get("descriptor", {})
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    target = str(comparison.get("outcome", ""))
    label = str(descriptor.get("label", ""))
    unit = str(comparison.get("unit", descriptor.get("unit", "")))
    if "variance" in target.lower() or "variance" in label.lower() or unit == "(mg/dL)^2":
        return True
    rules = details.get("contextRules", [])
    facts = item.get("messageFacts")
    facts = facts if isinstance(facts, Mapping) else {}
    fallback_rules = facts.get("ruleDescriptors", [])
    return max(len(rules) if isinstance(rules, list) else 0,
               len(fallback_rules) if isinstance(fallback_rules, list) else 0) >= 4


def _link_analogue_phrase(item: dict[str, Any], evidence: Mapping[str, Any]) -> None:
    if not _is_analogue(item):
        return
    evidence_id = item.get("evidenceId")
    details = evidence.get(evidence_id, {}) if evidence_id else {}
    similarity = details.get("similarity", {}) if isinstance(details, Mapping) else {}
    similarity = similarity if isinstance(similarity, Mapping) else {}
    glucose_only = bool(similarity.get("glucoseOnly"))
    phrase = "days with similar recent glucose" if glucose_only else "similar days"
    text = str(item.get("text", ""))
    if evidence_id:
        text = text.replace(f" [See recorded evidence](evidence://{evidence_id}).", "")
        text = text.replace(f" [See recorded evidence](evidence://{evidence_id})", "")
    link = f"[{phrase}](evidence://{evidence_id})" if evidence_id else phrase
    if f"[{phrase}](evidence://{evidence_id})" not in text and phrase in text:
        item["text"] = text.replace(phrase, link, 1)


def present_brief(
    brief: Mapping[str, Any], *, current_relevant_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Return a concise brief while keeping source findings and evidence reachable."""
    result = copy.deepcopy(dict(brief))
    if (result.get("presentationMetadata", {}) or {}).get("version") == "brief-presentation-v7":
        return result
    evidence = result.get("evidence", {})
    evidence = evidence if isinstance(evidence, Mapping) else {}
    source_items = [
        copy.deepcopy(dict(item))
        for key in ("items", "exploreItems")
        for item in (result.get(key, []) or [])
        if isinstance(item, Mapping)
    ]
    # Older stored briefs may not yet carry complete presentation coverage.
    # Do not infer full support from the capped examples shown in evidence.
    current_relevant_ids = {str(value) for value in (current_relevant_ids or set())}
    contexts = [item for item in source_items if _is_context(item)]
    analytical = [item for item in source_items if not _is_context(item)]
    grouped, groups = _group_findings(analytical, evidence)
    for item in grouped:
        _link_analogue_phrase(item, evidence)

    main_candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    explore_candidates: list[dict[str, Any]] = []
    for ordinal, item in enumerate(grouped):
        identifier = str(item.get("id", ""))
        relevant = identifier in current_relevant_ids
        analogue = _is_analogue(item)
        complex_or_variance = _is_complex_or_variance(item, evidence)
        informative_analogue = analogue and _informative_analogue(item)
        if complex_or_variance or (analogue and not informative_analogue):
            explore_candidates.append(item)
            continue
        # Historical outcomes from genuinely similar days answer the most
        # useful first-screen question. A currently relevant confirmed rule
        # follows it; neither is displaced by neutral current-context detail.
        if informative_analogue:
            priority = 0
        elif relevant:
            priority = 1
        elif not analogue:
            priority = 2
        else:
            priority = 3
        main_candidates.append(((priority, ordinal, 0), item))

    main_candidates.sort(key=lambda pair: pair[0])
    main = [item for _, item in main_candidates[:3]]
    selected_ids = {str(item.get("id", "")) for item in main}
    for _, item in main_candidates[3:]:
        explore_candidates.append(item)
    # Source-order stability for Explore; never drop an analytic candidate.
    explore_ids = {str(item.get("id", "")) for item in explore_candidates}
    explore_candidates.extend(
        item for item in grouped
        if str(item.get("id", "")) not in selected_ids | explore_ids
    )

    result["items"] = main
    result["exploreItems"] = explore_candidates
    result["contextDetails"] = contexts
    result["presentationMetadata"] = {
        "version": "brief-presentation-v7",
        "mainItemLimit": 3,
        "sourceFindingCount": len(source_items),
        "presentedMainCount": len(main),
        "exploreCount": len(explore_candidates),
        "contextDetailCount": len(contexts),
        "groupedSourceFindingCount": sum(group["sourceFindingCount"] - 1 for group in groups),
        "groups": groups,
        "groupingRequiresCompleteSupportProvenance": True,
        "engineClaimsAndEvidenceUnchanged": True,
    }
    return result


__all__ = ["present_brief"]
