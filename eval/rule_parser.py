#!/usr/bin/env python3
"""A hand-written keyword and regex parser: the rule-based baseline of A-M2(ii).

The corpus comes from deterministic templates, so a reviewer asks whether rules would
do as well as a fine-tuned language model. This module is that rule set, written to
be a fair opponent rather than a straw man, under one constraint that keeps it honest:

**It is built from the training split only.** Every keyword, synonym and pattern below
was read off `data/train.jsonl` (96 template families) and tuned against training
items alone. The generator's own vocabulary (`data/surface_forms.py`,
`data/template_families.json`) was not consulted, and no rule comes from a test,
validation or out-of-domain item, because the test families are disjoint from the
training families and using them would let the rules learn held-out phrasings the
language models never saw. The rules therefore face the same generalisation problem the
fine-tuned models face. Two disclosures: about six test items were seen in file heads
while checking formats, before any rule was written; and the rules were frozen before
the first test run, so one bug found afterwards is deliberately left in. The address
reader matches "one" inside "drone", so a repeated "drone" in an id list ("drone 3 and
drone 4") adds drone 1. It costs 2 golden items and 1 synthetic item on reference text.

Pipeline, in order:

1. **Drone addresses** ("units 1 and 3", "birds 2 through 5", "number four") are read
   off the raw lowercase text and removed, so their digits are not taken for slot
   values. Spoken ids are 1-based, the schema's 0-based.
2. **Numbers** go through `eval.norm.normalise`, the repository's one digit mapping
   (spelled-out numbers to digits), plus three repairs it does not make: "minus"
   before a digit, "and a half", and "100 and 68.1" written as two numbers.
3. **Intent** is the first rule in `INTENT_RULES` whose pattern matches. Order is
   the precedence: "stop" is abort only when nothing marks it as a hold.
4. **Slots** are read from the numbers left in the text, by intent.

The output is a canonical command string, scored exactly as a model's raw output
(`eval.metrics`). Anything no rule claims is `{"intent":"unknown"}`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python eval/rule_parser.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.norm import normalise
from schema.canon import canon

__all__ = ["parse"]

# --- drone addresses ---------------------------------------------------------

_ID_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
_ID_TOKEN = r"(?:[1-5]|one|two|three|four|five)"
_ADDRESS = r"(?:units?|drones?|birds?|numbers?|number ?s|uavs?|nos?\.?)"
#: One id, or three or more run together ("units 325"). Never the head of a decimal
#: or a longer number: "number 4, 42.0 to the east" keeps its 42.0.
_ID_ITEM = rf"(?:[1-5]{{3,5}}|{_ID_TOKEN})(?![\d.]\d|\w)"
#: A run of id tokens joined by "and", commas, "&" or a range word. Anything that
#: could be a slot value ("unit 1 heading 60") stops the run at the first non-id.
_ID_RUN = re.compile(
    rf"\b{_ADDRESS}\s*,?\s*(?:are\s+)?"
    rf"(?P<run>{_ID_ITEM}"
    rf"(?:(?:\s*(?:,|&|\band\b|\bthrough\b|\bthru\b|\bto\b|-)\s*)*(?:{_ADDRESS}\s*)?"
    rf"{_ID_ITEM})*)",
)
_RANGE = re.compile(rf"({_ID_TOKEN})\s*(?:through|thru|to|-)\s*({_ID_TOKEN})")


def _ids(text: str) -> tuple[list[int], str]:
    found: set[int] = set()

    def take(match: re.Match) -> str:
        run = match.group("run")
        for lo, hi in _RANGE.findall(run):
            a, b = sorted((_ID_WORDS[lo], _ID_WORDS[hi]))
            found.update(range(a, b + 1))
        run = _RANGE.sub(" ", run)
        for token in re.findall(rf"[1-5]{{3,5}}|{_ID_TOKEN}", run):
            if token.isdigit() and len(token) > 1:  # "units 325" -> 3, 2, 5
                found.update(int(c) for c in token)
            else:
                found.add(_ID_WORDS[token])
        return " , "

    stripped = _ID_RUN.sub(take, text)
    return sorted(i - 1 for i in found if 1 <= i <= 5), stripped


# --- numbers -------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


#: Recogniser homophones of a number before a spacing word, as the training split
#: shows them: "to apart" (two), "through your part" (three), "when a part" (one).
_HOMOPHONES = {"to": "2", "too": "2", "through": "3", "for": "4", "won": "1", "when": "1"}
_SPACING_HOMOPHONE = re.compile(
    r"\b(to|too|through|for|won|when)\s+(?:your\s+|a\s+|un)?(?=a ?part|apart|between|spacing|separation)")


def _numbers_text(text: str) -> str:
    text = normalise(text)
    text = re.sub(r"\ba part\b", "apart", text)
    text = _SPACING_HOMOPHONE.sub(lambda m: _HOMOPHONES[m.group(1)] + " ", text)
    text = re.sub(r"\b(\d+) and a half\b", r"\1.5", text)
    text = re.sub(r"\b(?:minus|negative)\s+(?=\d)", "-", text)
    # "100 and 68.1": a whole hundred and the rest, said as two numbers.
    text = re.sub(r"\b([1-9]00) and (\d{1,2}(?:\.\d)?)\b",
                  lambda m: str(int(m.group(1)) + float(m.group(2))).removesuffix(".0"), text)
    return text


def _nums(text: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(text)]


def _after(text: str, cue: str) -> float | None:
    """The first number after the last mention of `cue`, if any."""
    matches = list(re.finditer(cue, text))
    for match in reversed(matches):
        number = _NUMBER.search(text, match.end())
        if number:
            return float(number.group())
    return None


# --- intents -----------------------------------------------------------------

INTENT_RULES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (intent, re.compile(pattern)) for intent, pattern in (
        ("hover", r"\b(?:do not|don ?t|belay|negative|cancel|quit)\b|\bstop \w+ing\b|\bhover\b"
                  r"|\bhold\b(?! at)|position (?:right )?there|where you are"
                  r"|hold (?:position|station|still|it there)|\bstation\b|\bfreeze\b"
                  r"|stay (?:put|there|here|where)|keep (?:where|right there|still)"
                  r"|\bhold (?:there|here)\b(?!.*\d)|\bwait\b|\bhang\b|stop and hold|pause"),
        ("abort", r"\babort\b|\bkill\b|cut the motors|\bemergency\b|shut (?:it|them|everything) down"
                  r"|\ball stop\b|\bstop\b|motors off|\bdisarm\b"),
        ("land", r"\bland\b|\blanding\b|touch ?down|set (?:it |yourselves )?down|\brecover\b"
                 r"|come home|return home|bring (?:it|everything|them|yourselves) down"
                 r"|put (?:it|them) down|\bpad\b"),
        ("takeoff", r"take ?off|\blaunch\b|lift ?off|airborne|get (?:up )?in the air"
                    r"|spin up|\bin the air\b"),
        ("set_param", r"\bcruise\b|speed limit|\bgap\b|default height|\btransit\b"
                      r"|\bparameter\b|\bconfigure\b|\bcap\b|(?:speed|spacing|separation) to\b"),
        ("formation", r"\bformation\b|\bcircle\b|\bcircular\b|\bring\b|\bline\b|\babreast\b"
                      r"|\brow\b|\bwedge\b|\bvee\b|\bv\b|\barrow(?:head)?\b|\bgrid\b|\blattice\b"
                      r"|\bbox\b|\bsquare\b|\bcolumn\b|\bstack\b|\bflock(?:ing)?\b|\bcluster\b"
                      r"|\bradius\b"),
        ("rotate", r"\brotate\b|\bturn\b|\byaw\b|\bheading\b|\bpivot\b|\bspin\b|come round"
                   r"|\bswing\b|\bdegrees?\b|\bdeg\b|clockwise|\bday\b"),
        ("altitude", r"\baltitude\b|\bheight\b|\blevel\b|\bclimb\b|\brise\b|\bascend\b"
                     r"|\bdescen[dt]\b|\bdrop\b|come (?:down|up)|go (?:up|down) to|\bsit at\b"
                     r"|\bsettle\b|\bbe at\b|\bhold at\b"),
        ("move", r"\bnorth|\bsouth|\beast|\bwest|\bforward|\bahead\b|\bback(?:ward)?s?\b"
                 r"|\bleft\b|\bright\b|\bup(?:ward)?\b|\bdown(?:ward)?\b|\bhigher\b|\blower\b"
                 r"|\bmove\b|\bgo to\b|\bwaypoints?\b|\btranslate\b|\bshift\b|\bproceed\b"
                 r"|\bpush\b|\bhead\b|\bstarboard\b|\bport\b|\bstern\b|\bposition\b"
                 r"|ward\b|bound\b"),
    )
)

_SHAPES = (
    ("circle", r"\bcircle\b|\bcircular\b|\bring\b|\bradius\b"),
    ("wedge", r"\bwedge\b|\bvee\b|\bv\b|\barrow(?:head)?\b"),
    ("grid", r"\bgrid\b|\blattice\b|\bbox\b|\bsquare\b"),
    ("column", r"\bcolumn\b|\bstack\b|\bvertical\b|\bfile\b"),
    ("flock", r"\bflock(?:ing)?\b|\bcluster\b|\bloose\b"),
    ("line", r"\bline\b|\babreast\b|\brow\b"),
)

_DIRECTIONS = (
    ("north", r"\bnorth"), ("south", r"\bsouth"), ("east", r"\beast"), ("west", r"\bwest"),
    ("up", r"\bup(?:ward)?s?\b|\bhigher\b|\bclimb\b|\brise\b"),
    ("down", r"\bdown(?:ward)?s?\b|\blower\b|\bdescend\b"),
    ("forward", r"\bforwards?\b|\bahead\b"),
    ("back", r"\bback(?:ward)?s?\b|\breverse\b|\bstern\b"),
    ("left", r"\bleft(?:ward)?s?\b|\bport\b"), ("right", r"\bright(?:ward)?s?\b|\bstarboard\b"),
)

#: A correction ("make that", "scratch that", "correction", "I mean") replaces what
#: came before it, so only the text after the last one is read for slots.
_CORRECTION = re.compile(r"make that|scratch that|\bcorrection\b|\bi mean\b|\bno wait\b|\brather\b")


def _speed(text: str) -> float | None:
    for cue in (r"(?:speed|no more than|no faster than|at most|max(?:imum)?|keep it to|at)\b",):
        value = _after(text, cue)
        if value is not None:
            return value
    return None


def _slots(intent: str, text: str) -> dict:
    out: dict = {"intent": intent}
    corrected = _CORRECTION.split(text)
    tail = corrected[-1] if len(corrected) > 1 else text
    nums = _nums(tail) or _nums(text)

    if intent == "formation":
        out["shape"] = next((s for s, p in _SHAPES if re.search(p, text)), "line")
        if out["shape"] == "circle" and nums:
            # "six metres across" is a diameter.
            out["radius"] = nums[0] / 2 if re.search(r"\bacross\b", text) else nums[0]
        elif nums:
            out["spacing"] = nums[0]
    elif intent == "move":
        speed_cue = re.search(r"\b(?:speed|no more than|no faster than|at most|keep it to|at)\b"
                              r"|per second|m ?s\b", tail)
        speed = None
        if speed_cue:
            speed = _after(tail, r"\b(?:speed|no more than|no faster than|at most|keep it to|at)\b")
            if speed is None:  # "2 meters per second": the number before the unit
                before = _nums(tail[:speed_cue.start()])
                speed = before[-1] if before else None
        values = list(nums)
        if speed is not None and speed in values:
            values.remove(speed)
        if len(values) >= 3 and not any(re.search(p, tail) for _, p in _DIRECTIONS[:4]):
            out["pos"] = values[:3]
        else:
            hits = ([(m.start(), d) for d, p in _DIRECTIONS for m in re.finditer(p, tail)]
                    or [(m.start(), d) for d, p in _DIRECTIONS for m in re.finditer(p, text)])
            if not hits and len(values) == 1:  # "go to one point five metres"
                return {"intent": "altitude", "z": values[0]}
            out["dir"] = max(hits)[1] if hits else "forward"
            if values:
                out["dist"] = values[0]
        if speed is not None:
            out["speed"] = speed
    elif intent == "altitude":
        if nums:
            out["z"] = nums[0]
    elif intent == "takeoff":
        if nums:
            out["z"] = nums[-1]
    elif intent == "rotate":
        if nums:
            yaw = nums[0]
            if re.search(r"\bright\b|\bwrite\b|(?<!counter)(?<!counter )(?<!counter-)(?<!anti)"
                         r"(?<!anti )(?<!anti-)\bclockwise", tail) and yaw > 0:
                yaw = -yaw
            out["yaw"] = yaw
    elif intent == "set_param":
        for key, cue in (("speed", r"\b(?:cruise|speed(?: limit)?)\b(?! altitude)"),
                         ("spacing", r"\b(?:gap|spacing|separation)\b"),
                         ("alt", r"\b(?:altitude|height)\b")):
            value = _after(tail, cue)
            if value is not None:
                out[key] = value
    return out


#: Hesitations and cut-off words ("y— you"), and "right" as a discourse marker
#: ("right then", "all right,"), none of which says anything about the command.
_FILLERS = re.compile(r"\b(?:uh|um|er|erm|ah|hmm)\b,?|\b\w+[—–]\s*"
                      r"|^\s*(?:all ?right|right)\b(?: then)?,?|\bright then\b|\bright there\b")


def _intent(text: str) -> str:
    for name, pattern in INTENT_RULES:
        if pattern.search(text):
            # A hold with a number in it is a height to hold ("hold at 12 metres"),
            # not a hover: drone addresses are already gone, so any digit is a slot.
            if name == "hover" and _nums(text) and not re.search(r"\bhover\b", text):
                continue
            return name
    return "unknown"


def parse(transcript: str) -> str:
    """One transcript to one canonical command string."""
    lowered = _FILLERS.sub(" ", transcript.lower())
    ids, rest = _ids(lowered)
    text = _numbers_text(rest)
    intent = _intent(text)
    if intent == "unknown":
        return json.dumps({"intent": "unknown"})
    command = _slots(intent, text)
    intent = command["intent"]
    if ids and intent not in ("abort", "set_param"):
        command["ids"] = ids
    # A move, altitude or rotation with no number is not a command the schema can
    # hold ("higher", "up a bit"), so it is refused, as the grammar would force.
    required = {"move": ("dist", "pos"), "altitude": ("z",), "rotate": ("yaw",)}
    if intent in required and not any(k in command for k in required[intent]):
        return json.dumps({"intent": "unknown"})
    try:
        return canon(command)
    except Exception:  # a slot the schema rejects (an out-of-range value, say)
        return json.dumps(command)


if __name__ == "__main__":
    for line in sys.stdin:
        print(parse(line.strip()))
