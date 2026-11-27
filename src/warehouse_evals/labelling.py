"""Blind human labelling in the terminal: ``gwm-eval label <sheet>``.

The sheet comes from ``gwm-eval label-sheet`` (see ``calibration.export_label_template``). This
shows one trajectory at a time with each criterion's pass, fail and not-applicable rules, takes
one key per criterion, and saves the sheet after every trajectory, so a session can stop at any
point and resume where it left off. Rows already labelled are skipped.

Answers: ``p`` pass, ``f`` fail, ``n`` not applicable, then optional words as a note
(``f no units in the answer``). ``s`` skips the trajectory for now, ``q`` saves and quits.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from warehouse_evals.judge import Criterion, Rubric

KEYS = {"p": "pass", "f": "fail", "n": "not_applicable"}

Ask = Callable[[str], str]
Say = Callable[[str], None]


def read_sheet(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic: a crash mid-write never leaves a half-written sheet."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Row indexes per trajectory, in sheet order."""
    order: list[str] = []
    by_id: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        tid = row["trajectory_id"]
        if tid not in by_id:
            order.append(tid)
            by_id[tid] = []
        by_id[tid].append(i)
    return [by_id[t] for t in order]


def _rules(criterion: Criterion) -> str:
    return (
        f"{criterion.id}: {criterion.question}\n"
        f"  pass  {criterion.pass_when}\n"
        f"  fail  {criterion.fail_when}\n"
        f"  n/a   {criterion.not_applicable_when}"
    )


def progress(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(trajectories fully labelled, trajectories on the sheet)."""
    groups = _groups(rows)
    done = sum(all(rows[i]["verdict"] for i in g) for g in groups)
    return done, len(groups)


def label_sheet(path: Path, rubric: Rubric, ask: Ask = input, say: Say = print) -> tuple[int, int]:
    """Label every blank row, saving after each trajectory. Returns ``progress`` at the end."""
    rows = read_sheet(path)
    criteria = {c.id: c for c in rubric.criteria}
    unknown = sorted({r["criterion"] for r in rows} - set(criteria))
    if unknown:
        raise ValueError(f"{path}: criteria not in {rubric.key}: {', '.join(unknown)}")
    groups = _groups(rows)
    say(f"Rubric {rubric.key}. What each role may see:\n{rubric.context}\n")
    for n, group in enumerate(groups, 1):
        blank = [i for i in group if not rows[i]["verdict"]]
        if not blank:
            continue
        done, total = progress(rows)
        say(f"\n{'=' * 78}\nTrajectory {n} of {total} ({done} done)  id {rows[group[0]]['trajectory_id']}\n")
        say(rows[group[0]]["trajectory"])
        for i in blank:
            say("\n" + _rules(criteria[rows[i]["criterion"]]))
            while True:
                answer = ask("[p]ass [f]ail [n]/a, optional note; [s]kip, [q]uit > ").strip()
                key, _, note = answer.partition(" ")
                key = key.lower()
                if key == "q":
                    write_sheet(path, rows)
                    return progress(rows)
                if key == "s":
                    break
                if key in KEYS:
                    rows[i]["verdict"] = KEYS[key]
                    rows[i]["note"] = note.strip()
                    break
                say("  answer p, f or n (a note may follow), s or q")
            if key == "s":
                for j in blank:  # leave the whole trajectory for later, not half of it
                    rows[j]["verdict"], rows[j]["note"] = "", ""
                break
        write_sheet(path, rows)
    return progress(rows)
