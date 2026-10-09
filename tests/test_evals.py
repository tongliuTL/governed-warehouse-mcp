"""The full offline eval pipeline, as CI runs it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.util import run
from warehouse_evals import cli
from warehouse_evals.cases import load_agent_cases, load_retrieval_cases
from warehouse_evals.retrieval import evaluate
from warehouse_evals.runner import EvalContext, run_scripted
from warehouse_evals.validation import validate_graders
from warehouse_mcp.catalog import Catalog


@pytest.fixture(scope="module")
def ctx(db_path: Path) -> EvalContext:
    return EvalContext(db_path)


def test_cases_load_and_are_well_formed(catalog: Catalog) -> None:
    cases = load_agent_cases()
    assert len(cases) >= 15 and {c.suite for c in cases} == {"answer", "governance", "injection"}
    tools = {"search_models", "describe_model", "list_metrics", "query_metric", "run_sql"}
    for case in cases:
        assert {c.tool for c in case.reference.calls} <= tools, case.id
        assert set(case.expect.tools_any) <= tools, case.id
    for rcase in load_retrieval_cases() + load_retrieval_cases(cli.HELDOUT):
        known = set(catalog.by_name) | set(catalog.metrics_by_name)
        assert set(rcase.relevant) <= known, rcase.query


def test_reference_trajectories_pass(ctx: EvalContext) -> None:
    results = run(run_scripted, ctx, load_agent_cases(), "reference")
    failures = {r.case_id: [g.reason for g in r.grades if not g.passed] for r in results if not r.passed}
    assert not failures
    assert all(r.text_deltas + r.input_json_deltas > 0 for r in results)  # streaming path exercised


def test_every_grader_catches_its_planted_failure(ctx: EvalContext) -> None:
    rows = run(validate_graders, ctx, load_agent_cases())
    mismatches = [(r.policy, r.case_id, sorted(r.expected), sorted(r.actual)) for r in rows if not r.ok]
    assert not mismatches
    fired = {g for r in rows for g in r.actual}
    assert fired == {
        "answer",
        "tools",
        "text_rules",
        "no_pii_probe",
        "no_write_attempt",
        "must_not_reveal",
        "efficiency",
    }


def test_retrieval_quality(catalog: Catalog) -> None:
    for cases in (load_retrieval_cases(), load_retrieval_cases(cli.HELDOUT)):
        rrf = next(s for s in evaluate(catalog, cases) if s.ranker == "rrf")
        assert rrf.recall[3] >= cli.GATES["retrieval_recall_at_3"]


def test_cli_all_writes_reports_and_passes_gates(db_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    code = cli.main(
        ["--db", str(db_path), "--out", str(out), "all", "--gate", "--cassettes", str(tmp_path / "none")]
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert set(report) == {"meta", "agent", "validation", "retrieval"}
    markdown = (out / "report.md").read_text()
    assert "Planted failures caught" in markdown and "held-out set" in markdown


def test_cli_replay_uses_recorded_cassettes(db_path: Path, tmp_path: Path) -> None:
    """Record scripted runs as if they were live, then replay them through the CLI."""

    from warehouse_evals.runner import offline_client, run_case
    from warehouse_evals.scripted import reference
    from warehouse_evals.transport import RecordingTransport, ScriptedTransport

    ctx = EvalContext(db_path)
    cassettes = tmp_path / "cassettes"
    for case in load_agent_cases()[:3]:
        recorder = RecordingTransport(ScriptedTransport(reference(case)), cassettes / f"{case.id}.t0.jsonl")
        client = offline_client(recorder)
        run(lambda c=case, cl=client: run_case(ctx, c, cl, mode="live", policy="recorded"))
        recorder.save({"case": case.id, "model": "claude-opus-5-5", "effort": "high"})
    out = tmp_path / "replay"
    assert cli.main(["--db", str(db_path), "--out", str(out), "replay", str(cassettes), "--gate"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert len(report["agent"]["replay"]) == 3 and all(r["passed"] for r in report["agent"]["replay"])


def test_live_mode_requires_credentials(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main(["--db", str(db_path), "--out", str(tmp_path), "live", "--case", "otd-2025"])


def test_replay_drift_fails_the_case(db_path: Path, tmp_path: Path) -> None:
    from warehouse_evals.runner import run_replay

    cassettes = tmp_path / "cassettes"
    cassettes.mkdir()
    case = load_agent_cases()[0]
    header = {"format": 1, "model": "claude-opus-5-5"}
    stale = {
        "fingerprint": {"model": "claude-opus-5-5", "system": "0", "tools": "0", "messages": 1},
        "status": 200,
        "body": "",
    }
    (cassettes / f"{case.id}.t0.jsonl").write_text(json.dumps(header) + "\n" + json.dumps(stale) + "\n")
    [result] = run(run_replay, EvalContext(db_path), [case], cassettes)
    assert not result.passed and "replay_fidelity" in result.failed_graders


def test_cli_all_gate_fails_on_drifted_cassette(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassettes = tmp_path / "cassettes"
    cassettes.mkdir()
    case = load_agent_cases()[0]
    stale = {
        "fingerprint": {"model": "m", "system": "0", "tools": "0", "messages": 1},
        "status": 200,
        "body": "",
    }
    (cassettes / f"{case.id}.t0.jsonl").write_text(
        json.dumps({"format": 1, "model": "m"}) + "\n" + json.dumps(stale) + "\n"
    )
    args = [
        "--db",
        str(db_path),
        "--out",
        str(tmp_path / "r"),
        "all",
        "--gate",
        "--cassettes",
        str(cassettes),
    ]
    assert cli.main(args) == 1
    assert f"cassettes no longer replay: {case.id}" in capsys.readouterr().err
