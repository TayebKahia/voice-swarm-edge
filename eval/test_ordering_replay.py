"""eval/ordering_replay.py and eval/voice_onset.py: how a parse and a false hold are counted."""

from __future__ import annotations

from eval.ordering_replay import fate, tally
from eval.voice_onset import debounced, item_index


def _seg(intent=None, stale=False, applied=False, **extra) -> dict:
    seg = {"t0_s": 2.0, "speech_start_s": 1.0, "transcript": "t", "stale_discarded": stale, "applied": applied, **extra}
    return seg if intent is None else {**seg, "published_intent": intent}


def _row(take: str, *segments: dict, cls="swarm_hold", fired=True) -> dict:
    events = [{"intent": "hover", "seq": 5, "stream_s": 1.5}] if fired else []
    return {"take": take, "class": cls, "events": events, "segments": list(segments), "final_state": "FLYING"}


def test_a_parse_has_one_fate() -> None:
    assert fate(_seg(), "hover") == "no_command"
    assert fate(_seg("formation", stale=True), "hover") == "discarded"
    assert fate(_seg("formation"), "abort") == "rejected"
    assert fate(_seg("hover", applied=True), "hover") == "repeated"
    assert fate(_seg("formation", applied=True), "hover") == "changed"


def test_only_applied_parses_count_as_applied_after_the_stop() -> None:
    t = tally([_row("w1", _seg("hover", applied=True)),
               _row("w2", _seg("formation", applied=True)),
               _row("w3", _seg("hover", stale=True)),
               _row("w4", _seg())])
    assert (t["detected"], t["after_stop"]) == (4, 2)
    assert t["changed"] == [("w2", "t", "formation")]


def test_a_take_the_reflex_missed_is_not_in_the_denominator() -> None:
    t = tally([_row("w1", _seg("hover", applied=True), fired=False)])
    assert (t["detected"], t["after_stop"], len(t["missed"])) == (0, 0, 1)


def test_debounce_merges_as_the_spotter_counts() -> None:
    assert debounced([]) == 0
    assert debounced([1.0, 1.5, 3.0, 3.9, 5.0]) == 3


def test_an_onset_belongs_to_the_item_or_gap_it_falls_in() -> None:
    assert item_index([0.0, 2.0, 5.0], 0.0) == 0
    assert item_index([0.0, 2.0, 5.0], 4.9) == 1
    assert item_index([0.0, 2.0, 5.0], 5.5) == 2
