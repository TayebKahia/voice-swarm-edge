"""runtime/pipeline.py -- the wiring, and PRD Sec. 4.3's correctness rule.

`Dispatcher` is tested against a real `CommandBus` (real UDP loopback
sockets) and a real `swarm.fsm.FlightStateMachine` -- both already exist and
are not reimplemented here, only wired together. `PipelineRuntime` is tested
with fakes standing in for the not-yet-trained wake model and the LLM decode,
which is exactly the seam `runtime/branch_a.py` and `runtime/parser.py` were
built to expose.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from runtime.branch_a import BranchA
from runtime.bus import CommandBus
from runtime.parser import ParseAborted, ParseResult
from runtime.pipeline import Dispatcher, PipelineRuntime
from runtime.stream import FRAME, StreamLoop, WavSource
from runtime.vad import Endpointer
from schema.schema import Abort, Hover
from swarm.fsm import FlightState, FlightStateMachine


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def bus():
    with CommandBus(port=_free_port(), bind=True) as b:
        yield b


def _flying_fsm() -> FlightStateMachine:
    fsm = FlightStateMachine()
    fsm.handle_command({"intent": "takeoff"})
    fsm.on_altitude_reached()
    assert fsm.state == FlightState.FLYING
    return fsm


# --------------------------------------------------------------------------
# Dispatcher / Sec. 4.3 correctness rule


def test_branch_a_message_is_applied_to_the_fsm(bus: CommandBus) -> None:
    fsm = _flying_fsm()
    dispatcher = Dispatcher(fsm)
    bus.publish("A", Abort(intent="abort"))

    applied = dispatcher.drain(bus, timeout=2.0)

    assert len(applied) == 1
    assert applied[0] == Abort(intent="abort")
    assert fsm.state == FlightState.ABORTED


def test_stale_branch_b_message_is_discarded_after_a_later_branch_a_trigger(bus: CommandBus) -> None:
    """The scenario Sec. 4.3 describes verbatim: Branch A triggers while
    Branch B is mid-inference. The sequence number Branch B's message carries
    was reserved *before* Branch A published (representing the moment Branch
    B started decoding), even though Branch B's message is sent *after*.
    """
    fsm = _flying_fsm()
    dispatcher = Dispatcher(fsm)

    # Branch B starts processing an utterance -- reserves its slot in the
    # sequence before it has anything to publish.
    branch_b_seq = bus.next_seq()

    # While Branch B is still decoding, the operator says "swarm abort".
    bus.publish("A", Abort(intent="abort"))

    # Branch B's decode finally finishes and publishes -- late, but still
    # carrying the sequence number it reserved before the abort.
    bus.publish("B", {"intent": "move", "dir": "north", "dist": 10.0}, seq=branch_b_seq)

    applied = dispatcher.drain(bus, timeout=2.0)

    assert applied[0] == Abort(intent="abort")
    assert applied[1] is None, "the stale move must never reach the FSM"
    assert fsm.state == FlightState.ABORTED, "the stale move must not undo the abort"


def test_branch_b_message_after_the_last_branch_a_trigger_is_applied(bus: CommandBus) -> None:
    """The discard rule must not be trigger-happy: a Branch B message that
    genuinely postdates the last Branch A trigger is legitimate and must
    reach the FSM."""
    fsm = _flying_fsm()
    dispatcher = Dispatcher(fsm)

    bus.publish("A", Hover(intent="hover"))
    bus.publish("B", {"intent": "rotate", "yaw": 45.0})

    applied = dispatcher.drain(bus, timeout=2.0)

    assert len(applied) == 2
    assert applied[1] is not None
    assert applied[1].intent == "rotate"


def test_drain_returns_empty_when_bus_is_idle(bus: CommandBus) -> None:
    fsm = FlightStateMachine()
    dispatcher = Dispatcher(fsm)
    assert dispatcher.drain(bus, timeout=0.05) == []


# --------------------------------------------------------------------------
# PipelineRuntime wiring


@dataclass
class FakeWakeDetector:
    scores: dict[str, float] = field(default_factory=dict)

    def predict(self, frame: object) -> dict[str, float]:
        return self.scores


@dataclass
class FakeParser:
    result: ParseResult | None = None
    abort_called: bool = False
    parse_calls: list[str] = field(default_factory=list)

    def abort(self) -> None:
        self.abort_called = True

    def parse(self, transcript: str) -> ParseResult:
        self.parse_calls.append(transcript)
        if self.result is None:
            raise ParseAborted("no result configured")
        return self.result


def test_on_wake_frame_publishes_and_preempts_the_parser(bus: CommandBus) -> None:
    parser = FakeParser()
    branch_a = BranchA(FakeWakeDetector({"swarm_abort": 0.9}))
    runtime = PipelineRuntime(bus=bus, branch_a=branch_a, parser=parser)

    msg = runtime.on_wake_frame(frame=None)

    assert msg is not None
    assert msg.branch == "A"
    assert msg.command == {"intent": "abort"}
    assert parser.abort_called, "Figure 1's preempt edge must fire on a Branch A trigger"


def test_on_wake_frame_returns_none_when_nothing_triggers(bus: CommandBus) -> None:
    parser = FakeParser()
    branch_a = BranchA(FakeWakeDetector({}))
    runtime = PipelineRuntime(bus=bus, branch_a=branch_a, parser=parser)

    assert runtime.on_wake_frame(frame=None) is None
    assert not parser.abort_called


def test_on_utterance_publishes_the_validated_command(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "go north ten metres")
    parser = FakeParser(result=ParseResult(raw='{"intent":"move","dir":"north","dist":10.0}',
                                            prefill_ms=12.0, decode_ms=34.0))
    branch_a = BranchA(FakeWakeDetector({}))
    runtime = PipelineRuntime(bus=bus, branch_a=branch_a, parser=parser)

    msg = runtime.on_utterance(Path("unused.wav"))

    assert msg is not None
    assert msg.branch == "B"
    assert msg.command["intent"] == "move"
    assert parser.parse_calls == ["go north ten metres"]


def test_on_utterance_reserves_its_sequence_before_transcription(
    bus: CommandBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sec. 4.3's ordering guarantee requires the sequence number to be
    reserved before the (possibly slow) transcribe/decode work runs, not
    after -- this is the property `test_stale_branch_b_message_is_discarded`
    relies on, checked here at the `PipelineRuntime` level."""
    seen_seq_at_transcribe_time = {}

    def fake_transcribe(path: Path, **_: object) -> str:
        seen_seq_at_transcribe_time["reserved"] = bus.next_seq() - 1
        return "hold position"

    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", fake_transcribe)
    parser = FakeParser(result=ParseResult(raw='{"intent":"hover"}', prefill_ms=1.0, decode_ms=1.0))
    runtime = PipelineRuntime(bus=bus, branch_a=BranchA(FakeWakeDetector({})), parser=parser)

    msg = runtime.on_utterance(Path("unused.wav"))

    assert msg is not None
    assert msg.seq == seen_seq_at_transcribe_time["reserved"]


def test_on_utterance_publishes_nothing_when_the_decode_is_aborted(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "irrelevant")
    parser = FakeParser(result=None)  # raises ParseAborted
    runtime = PipelineRuntime(bus=bus, branch_a=BranchA(FakeWakeDetector({})), parser=parser)

    assert runtime.on_utterance(Path("unused.wav")) is None
    assert bus.recv(timeout=0.05) is None


# --------------------------------------------------------------------------
# The parse of the stop phrase itself: the number is reserved at speech onset


class ScriptedVAD:
    """Speech from window `start` up to `stop`; the endpointer's arithmetic stays real."""

    def __init__(self, *spans: tuple[int, int]) -> None:
        self.spans = spans
        self.window = 0

    def reset(self) -> None:
        self.window = 0

    def __call__(self, window: np.ndarray) -> float:
        k, self.window = self.window, self.window + 1
        return 0.9 if any(start <= k < stop for start, stop in self.spans) else 0.0


@dataclass
class ScriptedSpotter:
    """A `swarm hold` trigger on one chosen frame."""

    fire_on: int
    frame: int = 0

    def predict(self, frame: object) -> dict[str, float]:
        k, self.frame = self.frame, self.frame + 1
        return {"swarm_hold": 0.9} if k == self.fire_on else {}


WALLED = ParseResult(raw='{"intent":"formation","shape":"wedge"}', prefill_ms=1.0, decode_ms=1.0)


def _stream(bus: CommandBus, fire_on: int, spans: tuple[tuple[int, int], ...], parser: FakeParser):
    runtime = PipelineRuntime(bus=bus, branch_a=BranchA(ScriptedSpotter(fire_on)), parser=parser)
    audio = np.zeros(FRAME * 60, dtype=np.int16)
    return StreamLoop(runtime, Endpointer(ScriptedVAD(*spans))).run(WavSource(audio, realtime=False))


def test_w19_a_misheard_stop_phrase_cannot_replace_the_hold(
    bus: CommandBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Take w19: "swarm hold", heard by the recogniser as "swarm, walled", parsed as a formation.

    The spotter fires before the endpointer closes the utterance, so the parse starts only after
    the hold is on the bus. A number reserved at that point is newer than the hold, the ordering
    rule lets the formation through, and it replaces the hold. Reserved at speech onset, it is
    older, and the rule discards it.
    """
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "swarm, walled.")
    fsm = _flying_fsm()
    dispatcher = Dispatcher(fsm)

    # Speech over windows 30-60 (0.96-1.92 s), spotter on frame 22 (ends 1.84 s), endpoint near 2.4 s.
    result = _stream(bus, fire_on=22, spans=((30, 60),), parser=FakeParser(result=WALLED))

    (event,) = result.branch_a
    (record,) = result.branch_b
    assert event.stream_s < record.utterance.t0_s            # the hold precedes the endpoint
    assert record.trace.message.command["intent"] == "formation"   # the parse did publish
    assert record.trace.seq < event.message.seq
    assert dispatcher.drain(bus, timeout=2.0) == [Hover(intent="hover"), None]


def test_a_command_with_no_stop_in_it_is_still_applied(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "form a wedge")
    dispatcher = Dispatcher(_flying_fsm())

    result = _stream(bus, fire_on=10_000, spans=((30, 60),), parser=FakeParser(result=WALLED))

    (record,) = result.branch_b
    (applied,) = dispatcher.drain(bus, timeout=2.0)
    assert applied is not None and applied.intent == "formation"
    assert record.trace.message is not None


def test_a_command_begun_after_the_stop_is_newer_and_is_applied(
    bus: CommandBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "form a wedge")
    dispatcher = Dispatcher(_flying_fsm())

    # The spotter fires on frame 5 (ends 0.48 s), before speech begins at 0.96 s.
    _stream(bus, fire_on=5, spans=((30, 60),), parser=FakeParser(result=WALLED))

    hold, formation = dispatcher.drain(bus, timeout=2.0)
    assert hold == Hover(intent="hover")
    assert formation is not None and formation.intent == "formation"


def test_each_utterance_keeps_the_number_of_its_own_onset(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "form a wedge")

    result = _stream(bus, fire_on=10_000, spans=((10, 20), (50, 60)), parser=FakeParser(result=WALLED))

    assert [r.trace.seq for r in result.branch_b] == [0, 1]


def test_a_direct_wav_entry_still_reserves_for_itself(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "form a wedge")
    runtime = PipelineRuntime(bus=bus, branch_a=BranchA(FakeWakeDetector({})), parser=FakeParser(result=WALLED))

    reserved = runtime.reserve_seq()

    assert runtime.process(Path("unused.wav"), seq=reserved).seq == reserved
    assert runtime.process(Path("unused.wav")).seq == reserved + 1


# --------------------------------------------------------------------------
# PipelineRuntime.process -- the stage timings Exp-2 reports (NFR-3)


def test_process_times_every_stage_and_publishes(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_transcribe(path: Path, **kwargs: object) -> str:
        seen.update(kwargs)
        return "hold position"

    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", fake_transcribe)
    parser = FakeParser(result=ParseResult(raw='{"intent":"hover"}', prefill_ms=12.0, decode_ms=34.0))
    runtime = PipelineRuntime(bus=bus, branch_a=BranchA(FakeWakeDetector({})), parser=parser)

    trace = runtime.process(Path("unused.wav"))

    assert seen == {"threads": 3}                  # Table 7: -t 3 on cores 1-3
    assert trace.transcript == "hold position"
    assert trace.raw == '{"intent":"hover"}'
    assert (trace.prefill_ms, trace.decode_ms) == (12.0, 34.0)
    assert not trace.aborted
    assert trace.message is not None and trace.message.seq == trace.seq
    assert trace.t_start <= trace.t_transcribed <= trace.t_parsed <= trace.t_published
    assert trace.stt_ms >= 0 and trace.parse_wall_ms >= 0 and trace.validate_ms >= 0


def test_process_records_an_aborted_decode(bus: CommandBus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.pipeline.transcribe_utterance", lambda path, **_: "irrelevant")
    runtime = PipelineRuntime(bus=bus, branch_a=BranchA(FakeWakeDetector({})), parser=FakeParser(result=None))

    trace = runtime.process(Path("unused.wav"))

    assert trace.aborted and trace.message is None and trace.raw is None
    assert trace.t_published is None
    assert bus.recv(timeout=0.05) is None
