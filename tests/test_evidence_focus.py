from insite_analytics.evidence_focus import attach_evidence_focus


def test_late_response_highlights_only_evaluated_interval():
    episode=dict(start='2026-01-01T12:00:00Z',outcomeStart='2026-01-01T15:00:00Z',outcomeEnd='2026-01-01T18:00:00Z')
    brief={'evidence':{'a':{'episodes':[episode]}}}
    attach_evidence_focus(brief)
    assert episode['focusWindows'][0]['startMinute']==180
    assert episode['focusWindows'][0]['endMinute']==360


def test_analogue_uses_claim_horizon_not_entire_day():
    episode=dict(start='2026-01-01T09:00:00-05:00',outcomeWindows={
        '3h':{'start':'2026-01-01T09:00:00-05:00','end':'2026-01-01T12:00:00-05:00'},
        'overnight':{'start':'2026-01-01T23:00:00-05:00','end':'2026-01-02T06:00:00-05:00'}})
    brief={'items':[{'evidenceId':'a','messageFacts':{'horizon':'overnight'}}],'evidence':{'a':{'episodes':[episode]}}}
    attach_evidence_focus(brief)
    assert episode['focusWindows'][0]['startMinute']==840
    assert episode['focusWindows'][0]['endMinute']==1260


def test_missing_reversed_or_naive_boundaries_do_not_invent_highlights():
    episodes=[{'start':'2026-01-01T09:00:00Z'},
        dict(start='2026-01-01T09:00:00Z',outcomeStart='2026-01-01T12:00:00Z',outcomeEnd='2026-01-01T11:00:00Z'),
        dict(start='2026-01-01T09:00:00',outcomeStart='2026-01-01T09:00:00Z',outcomeEnd='2026-01-01T11:00:00Z')]
    attach_evidence_focus({'evidence':{'a':{'episodes':episodes}}})
    assert all(e['focusWindows']==[] for e in episodes)
