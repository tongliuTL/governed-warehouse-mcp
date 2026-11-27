"""Blind human labelling: sampling, the terminal labeller, and the rates the report gives."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.util import arun
from warehouse_evals import cli
from warehouse_evals.calibration import (
    Agreement,
    Run,
    export_label_template,
    load_labels,
    sample_runs,
    trajectories,
    trajectory_id,
    wilson,
)
from warehouse_evals.cases import load_agent_cases
from warehouse_evals.judge import Rubric, load_rubric
from warehouse_evals.labelling import label_sheet, progress, read_sheet
from warehouse_evals.runner import EvalContext


@pytest.fixture(scope="module")
def rubric() -> Rubric:
    return load_rubric()


@pytest.fixture(scope="module")
def runs(ctx: EvalContext) -> list[Run]:
    return arun(trajectories(ctx, load_agent_cases()))


def _live(runs: list[Run], n: int) -> list[Run]:
    """Stand-ins for replayed live runs: distinct trajectories with mode "replay"."""
    out = []
    for k, (case, result) in enumerate(runs[:n]):
        live = replace(result, mode="replay", policy="model-x", trial=k, final_text=f"live answer {k}")
        out.append((case, live))
    return out


# --- rates -----------------------------------------------------------------------------------


def test_wilson_known_values() -> None:
    assert wilson(0, 0) is None
    low, high = wilson(5, 5)  # type: ignore[misc]
    assert low == pytest.approx(0.566, abs=0.001) and high == 1.0
    low, high = wilson(50, 100)  # type: ignore[misc]
    assert low == pytest.approx(0.404, abs=0.001) and high == pytest.approx(0.596, abs=0.001)


def test_tpr_tnr_use_fail_as_positive() -> None:
    a = Agreement("c", "human")
    for judge, human in (
        [("fail", "fail")] * 8 + [("pass", "fail")] * 2 + [("pass", "pass")] * 9 + [("fail", "pass")]
    ):
        a.add("x", judge, human)  # type: ignore[arg-type]
    assert a.tpr == 0.8 and a.tnr == 0.9
    assert a.tpr_interval is not None and a.tpr_interval[0] < 0.8 < a.tpr_interval[1]
    assert Agreement("c", "human").tpr is None


# --- sampling --------------------------------------------------------------------------------


def test_sample_dedupes_and_is_seeded(runs: list[Run]) -> None:
    distinct = {trajectory_id(c, r) for c, r in runs}
    everything = sample_runs(runs + runs, None)
    assert len(everything) == len(distinct)  # identical trajectories are labelled once
    first, again = sample_runs(runs, 10, seed=3), sample_runs(runs, 10, seed=3)
    assert [trajectory_id(*r) for r in first] == [trajectory_id(*r) for r in again]
    assert [trajectory_id(*r) for r in sample_runs(runs, 10, seed=4)] != [trajectory_id(*r) for r in first]


def test_sample_draws_scripted_and_live_in_turn(runs: list[Run]) -> None:
    picked = sample_runs(runs + _live(runs, 30), 20)
    modes = [r.mode for _, r in picked]
    assert modes.count("replay") == 10 and modes.count("scripted") == 10


# --- the terminal labeller ------------------------------------------------------------------


def _sheet(tmp_path: Path, runs: list[Run], rubric: Rubric, n: int = 2) -> Path:
    path = tmp_path / "sheet.jsonl"
    export_label_template(runs[:n], rubric, path)
    return path


def _answers(*answers: str):  # type: ignore[no-untyped-def]
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def test_labeller_saves_verdicts_and_notes(tmp_path: Path, runs: list[Run], rubric: Rubric) -> None:
    path = _sheet(tmp_path, runs, rubric)
    k = len(rubric.criterion_ids)
    answers = ["f no figure in the answer", "x", *["p"] * (k - 1), *["n"] * k]
    done, total = label_sheet(path, rubric, ask=_answers(*answers), say=lambda _: None)
    assert (done, total) == (2, 2)
    rows = read_sheet(path)
    assert rows[0]["verdict"] == "fail" and rows[0]["note"] == "no figure in the answer"
    assert {r["verdict"] for r in rows[k:]} == {"not_applicable"}
    assert len(load_labels(path)) == 2 * k


def test_labeller_resumes_and_skips_whole_trajectories(
    tmp_path: Path, runs: list[Run], rubric: Rubric
) -> None:
    path = _sheet(tmp_path, runs, rubric)
    k = len(rubric.criterion_ids)
    # Label one criterion of the first trajectory, then skip: nothing of it is kept.
    # Then label the second fully, and quit before coming back.
    label_sheet(path, rubric, ask=_answers("p", "s", *["p"] * k), say=lambda _: None)
    rows = read_sheet(path)
    assert progress(rows) == (1, 2) and all(r["verdict"] == "" for r in rows[:k])
    assert label_sheet(path, rubric, ask=_answers("q"), say=lambda _: None) == (1, 2)
    assert label_sheet(path, rubric, ask=_answers(*["f"] * k), say=lambda _: None) == (2, 2)


def test_labeller_rejects_a_sheet_from_another_rubric(
    tmp_path: Path, runs: list[Run], rubric: Rubric
) -> None:
    path = _sheet(tmp_path, runs, rubric, n=1)
    rows = read_sheet(path)
    rows[0]["criterion"] = "not_a_criterion"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_criterion"):
        label_sheet(path, rubric, ask=_answers(), say=lambda _: None)


def test_cli_label_sheet_needs_no_judge_and_never_overwrites(db_path: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "labels" / "sheet.jsonl"
    base = ["--db", str(db_path), "--out", str(tmp_path / "out"), "label-sheet", str(sheet)]
    assert cli.main([*base, "--sample", "5"]) == 0
    assert progress(read_sheet(sheet)) == (0, 5)
    assert cli.main([*base, "--sample", "5"]) == 1  # labels in it would be lost
    assert cli.main([*base, "--sample", "7", "--force"]) == 0
    assert progress(read_sheet(sheet)) == (0, 7)
