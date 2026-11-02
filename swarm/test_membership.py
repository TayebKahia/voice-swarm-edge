"""The article's membership property, checked against the state machine's legality table.

The reflex path may publish a command only if, in every state where it is legal, it stops lateral
motion and no other legal stop is strictly below it in kinetic consequence. Legality comes from
`swarm/fsm.py`; the consequences are declared here, read from `swarm/link.py`'s mapping. The abort
fails the property and stays on the reflex path as a stated exception, so that is asserted too: a
change to either side has to be a decision.
"""

from __future__ import annotations

import pytest

from runtime.branch_a import KEYWORD_TO_INTENT
from swarm.fsm import TABLE_9_LEGALITY, FlightState

#: intent -> (m, z, l): horizontal translation or rotation; change of height; a state no spoken
#: command can leave.
KAPPA = {
    "hover": (0, 0, 0),
    "abort": (0, 1, 1),
    "land": (0, 1, 0),
    "set_param": (0, 0, 0),
    "takeoff": (0, 1, 0),
    "move": (1, 0, 0),
    "altitude": (0, 1, 0),
    "rotate": (1, 0, 0),
    "formation": (1, 0, 0),
}
#: Commands after which every vehicle's commanded lateral velocity is zero.
STOPS = {"hover", "abort", "land"}
#: The reflex path's outputs that fail the property on purpose.
EXCEPTIONS = {"abort"}


def _below(a: str, b: str) -> bool:
    """`a` strictly below `b`, componentwise."""
    return KAPPA[a] != KAPPA[b] and all(x <= y for x, y in zip(KAPPA[a], KAPPA[b]))


def _legal(state: FlightState) -> set[str]:
    return {intent for intent, states in TABLE_9_LEGALITY.items() if state in states}


def satisfies_property(intent: str) -> bool:
    states = TABLE_9_LEGALITY[intent]
    if not states:
        return False
    for state in states:
        stopping = _legal(state) & STOPS
        if intent not in stopping or any(_below(other, intent) for other in stopping):
            return False
    return True


def test_every_dispatchable_intent_has_a_declared_consequence() -> None:
    assert set(KAPPA) == set(TABLE_9_LEGALITY) - {"unknown"}


@pytest.mark.parametrize("intent", sorted(set(TABLE_9_LEGALITY)))
def test_only_the_hold_satisfies_the_property(intent: str) -> None:
    assert satisfies_property(intent) is (intent == "hover")


def test_the_reflex_path_publishes_the_hold_and_the_stated_exception_only() -> None:
    published = set(KEYWORD_TO_INTENT.values())
    assert {i for i in published if not satisfies_property(i)} == EXCEPTIONS
    assert published - EXCEPTIONS == {"hover"}


def test_the_abort_fails_because_a_hold_is_below_it_where_a_hold_is_legal() -> None:
    assert _below("hover", "abort") and _below("hover", "land")
    assert _legal(FlightState.LANDING) & STOPS == {"abort"}    # the table's landing-state note
