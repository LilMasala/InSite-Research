"""Attach chart focus from measured outcome boundaries, never from generated prose."""
from datetime import datetime
from math import isfinite


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result if result.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def attach_evidence_focus(brief):
    facts = {}
    for key in ('items', 'exploreItems', 'contextDetails', 'evidenceOnlyItems'):
        for item in brief.get(key, []):
            facts.setdefault(item.get('evidenceId'), item.get('messageFacts', {}))
    for identifier, evidence in brief.get('evidence', {}).items():
        claim = facts.get(identifier, {})
        horizon = claim.get('horizon')
        for episode in evidence.get('episodes', []):
            episode['focusWindows'] = []
            origin = _time(episode.get('start'))
            window = episode.get('outcomeWindows', {}).get(horizon, {}) if horizon else {}
            start = _time(window.get('start', episode.get('outcomeStart')))
            end = _time(window.get('end', episode.get('outcomeEnd')))
            if origin is None or start is None or end is None or end <= start:
                continue
            lo, hi = (start-origin).total_seconds()/60, (end-origin).total_seconds()/60
            if not all(isfinite(v) for v in (lo, hi)):
                continue
            episode['focusWindows'] = [{'startMinute':lo, 'endMinute':hi,
                'label':'Window used in this finding', 'kind':'outcome'}]
    return brief
