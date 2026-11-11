#!/usr/bin/env python3
"""S1 (review A-M4b): a rotation-balanced copy of `train.jsonl` for a retrained variant.

    python data/s1_balance_rotation.py          # writes data/train_s1_balanced.jsonl
    python data/s1_balance_rotation.py --check  # verify only, write nothing

The deployed model was trained on `data/train.jsonl`, in which 63 of the 120
rotation pairs say counter-clockwise or left (all positive yaw) and 26 say
clockwise or right. That skew does not come from the surface realiser, which is
sign-symmetric (`data/surface_forms.py::_rotate_cores`). It comes from the value
draw: each rotation family has only two or three value samples, and under seed 42
nine of the fourteen came out positive.

`data/generate.py` cannot simply be re-run with a different sign draw. It uses one
global seed-42 stream, so a change in the rotation families would shift every
later family, `test_ood` and the 200 golden transcripts that were recorded as
audio. This script therefore leaves the generator alone and mirrors three whole
positive value-sample groups that are already in train:

    F099-s2 (+124.0, 7 rows)   F101-s1 (+113.0, 8 rows)   F104-s3 (+1.0, 4 rows)

For each row the yaw is negated and the direction word swapped (left -> right,
counter-clockwise -> clockwise). An unsigned bare angle, which under ADR-0003
means positive, gets "minus", as `_rotate_cores` renders a negative signed angle.
The 19 new transcripts are written out by hand below, not produced by regex, so
that filler words ("alright", "right then") are left alone and the ASR noise of
the original row (e.g. "twenty-for", "day" for "deg") is kept as it is. Every
other row of `train.jsonl` is copied byte-for-byte, so the size, families, order
and every non-rotation row are those of the deployed run. Each of the six train
rotation families then holds both signs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # `python data/s1_balance_rotation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.error_families import cue
from schema.canon import canon
from schema.validate import validate

DATA_DIR = Path(__file__).resolve().parent
SOURCE = DATA_DIR / "train.jsonl"
OUTPUT = DATA_DIR / "train_s1_balanced.jsonl"

#: row id -> (original transcript, mirrored transcript). The original is
#: asserted, so a regenerated `train.jsonl` cannot silently desynchronise this table.
MIRROR: dict[str, tuple[str, str]] = {
    "F099-s2p01": ("everybody rotate around 124 left",
                   "everybody rotate around 124 right"),
    "F099-s2p02": ("everybody let's turn left 124 deg",
                   "everybody let's turn right 124 deg"),
    "F099-s2p03": ("okay so, yaw left 124, everyone",
                   "okay so, yaw right 124, everyone"),
    "F099-s2p04": ("listen up all drones spin uh one hundred and twenty-for degrees counter-clockwise",
                   "listen up all drones spin uh one hundred and twenty-for degrees clockwise"),
    "F099-s2p05": ("everyone heading left one hundred and twenty-four",
                   "everyone heading right one hundred and twenty-four"),
    "F099-s2p06": ("all of you come round 124.0 left for me",
                   "all of you come round 124.0 right for me"),
    "F099-s2p07": ("Okay, the swarm, swing, round, left, call it 124",
                   "Okay, the swarm, swing, round, right, call it 124"),
    "F101-s1p01": ("units, 325, come round left 113.0",
                   "units, 325, come round right 113.0"),
    "F101-s1p02": ("Iwa, want you to bird 35 can you turn left 113?",
                   "Iwa, want you to bird 35 can you turn right 113?"),
    "F101-s1p03": ("so, numbers three to five heading one hundred and thirteen counter-clockwise",
                   "so, numbers three to five heading one hundred and thirteen clockwise"),
    "F101-s1p04": ("uh, alright, spin one hundred and thirteen degrees left, birds three to five",
                   "uh, alright, spin one hundred and thirteen degrees right, birds three to five"),
    "F101-s1p05": ("pivot by 113 day units, 335",
                   "pivot by minus 113 day units, 335"),
    "F101-s1p06": ("numbers 3 to 5 come round 100 and 13 for me",
                   "numbers 3 to 5 come round minus 100 and 13 for me"),
    "F101-s1p07": ("heading left 113.0, birds 3 thru 5",
                   "heading right 113.0, birds 3 thru 5"),
    "F101-s1p08": ("rotate 113 left, units 3 through 5",
                   "rotate 113 right, units 3 through 5"),
    "F104-s3p01": ("So, bird one, one, fitting one counterclockwise",
                   "So, bird one, one, fitting one clockwise"),
    "F104-s3p02": ("turn one degrees counter-clockwise... yeah, number one",
                   "turn one degrees clockwise... yeah, number one"),
    "F104-s3p03": ("unit erm won heading one left",
                   "unit erm won heading one right"),
    "F104-s3p04": ("drone, one spin by one day counter-clockwise",
                   "drone, one spin by one day clockwise"),
}


def rotation_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        target = json.loads(row["target"])
        if target.get("intent") != "rotate":
            continue
        sign = "+" if target["yaw"] > 0 else "-" if target["yaw"] < 0 else "0"
        counts[(cue(row["transcript"]), sign)] += 1
    return counts


def mirror_row(row: dict) -> dict:
    original, mirrored = MIRROR[row["id"]]
    if row["transcript"] != original:
        raise SystemExit(f"{row['id']}: transcript changed since the table was written: {row['transcript']!r}")
    target = json.loads(row["target"])
    if target.get("intent") != "rotate" or target["yaw"] <= 0:
        raise SystemExit(f"{row['id']}: expected a positive rotate target, got {target}")
    target["yaw"] = -target["yaw"]
    new_target = canon(validate(target, fallback=False))
    if json.loads(new_target) != target:
        raise SystemExit(f"{row['id']}: mirrored target is not canonical: {new_target}")
    out = dict(row)
    out["transcript"] = mirrored
    out["target"] = new_target
    out["s1_mirrored"] = True
    return out


def build() -> list[str]:
    lines = SOURCE.read_text().splitlines()
    out_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        row = json.loads(line)
        if row["id"] in MIRROR:
            seen.add(row["id"])
            out_lines.append(json.dumps(mirror_row(row), ensure_ascii=False))
        else:
            out_lines.append(line)
    missing = set(MIRROR) - seen
    if missing:
        raise SystemExit(f"rows not found in {SOURCE.name}: {sorted(missing)}")
    transcripts = [json.loads(l)["transcript"].lower().strip() for l in out_lines]
    dupes = [t for t, n in Counter(transcripts).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate transcripts after mirroring: {dupes}")
    return out_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify and print counts; write nothing")
    args = parser.parse_args()

    before = [json.loads(l) for l in SOURCE.read_text().splitlines()]
    out_lines = build()
    after = [json.loads(l) for l in out_lines]

    for name, rows in (("deployed train.jsonl", before), ("S1 balanced", after)):
        c = rotation_counts(rows)
        ccw_left = sum(n for (k, _), n in c.items() if k in ("counter-clockwise", "left"))
        cw_right = sum(n for (k, _), n in c.items() if k in ("clockwise", "right"))
        pos = sum(n for (_, s), n in c.items() if s == "+")
        neg = sum(n for (_, s), n in c.items() if s == "-")
        print(f"{name:<22} rows {len(rows)}  rotate {sum(c.values())}  "
              f"ccw/left {ccw_left}  cw/right {cw_right}  yaw +{pos} / -{neg}")
        print(f"{'':<22} {dict(sorted(c.items()))}")

    changed = sum(1 for a, b in zip(SOURCE.read_text().splitlines(), out_lines) if a != b)
    print(f"rows changed: {changed} (expected {len(MIRROR)})")
    if changed != len(MIRROR):
        return 1
    if not args.check:
        OUTPUT.write_text("\n".join(out_lines) + "\n")
        print(f"wrote {OUTPUT.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
