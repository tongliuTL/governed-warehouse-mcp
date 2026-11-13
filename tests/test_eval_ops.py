"""Eval operations: the telemetry store, the 100% regression gate and real-query harvesting."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from warehouse_evals import baseline, cli, harvest, history
from warehouse_evals.cases import load_agent_cases, load_retrieval_cases
from warehouse_evals.graders import Grade
from warehouse_evals.report import Report
from warehouse_evals.retrieval import RankerScore
from warehouse_evals.runner import CaseResult
from warehouse_evals.sse import TextBlock, ToolUseBlock, Turn, encode_turn
from warehouse_evals.validation import ValidationRow
from warehouse_mcp.catalog import Catalog

ORIGIN = {"git_sha": "abc123", "git_ref": "main", "ci_url": None}


def _result(
    case_id: str, passed: bool, *, mode: str = "scripted", calls: list[dict] | None = None
) -> CaseResult:
    grades = [Grade("answer", passed, "" if passed else "wrong figure")]
    return CaseResult(
        case_id=case_id, suite="answer", role="analyst", mode=mode, policy="reference", trial=0,
        passed=passed, grades=grades, final_text="", stop="end_turn", tool_calls=calls or [], turns=1,
        input_tokens=0, output_tokens=0, text_deltas=1, input_json_deltas=0, elapsed_ms=1,
        first_token_ms=None,
    )  # fmt: skip


def _report(passed: dict[str, bool], recall3: float = 0.95) -> Report:
    report = Report(meta={"catalog_version": "v1-test", "seed": 7})
    report.agent["scripted reference"] = [_result(cid, ok) for cid, ok in passed.items()]
    report.validation = [ValidationRow("fabricator", "otd-2025", {"answer"}, {"answer"})]
    report.retrieval = {"dev set": [RankerScore("rrf", {1: 0.8, 3: recall3, 5: 1.0}, 0.9)]}
    return report


# --- telemetry -------------------------------------------------------------------------------


def test_history_records_runs_and_finds_flips(tmp_path: Path) -> None:
    store = tmp_path / "history.duckdb"
    t0 = datetime(2026, 11, 1, tzinfo=UTC)
    history.record(store, _report({"a": True, "b": True}), command="all", started_at=t0, origin=ORIGIN)
    history.record(
        store,
        _report({"a": True, "b": False}, recall3=0.9),
        command="all",
        gate_failures=["regression: b"],
        started_at=t0 + timedelta(hours=1),
        origin=ORIGIN,
    )
    runs = history.recent_runs(store)
    assert [(r.passed, r.cases) for r in runs] == [(1, 2), (2, 2)]  # newest first
    assert runs[0].rrf_recall_at_3 == {"dev set": 0.9} and runs[0].gate_failures == ["regression: b"]
    assert runs[0].git_sha == "abc123"
    (flip,) = history.flips(store)
    assert (flip.case_id, flip.now_passed, flip.failed_graders) == ("b", False, ["answer"])
    text = history.render(store)
    assert "b: now FAILS (answer)" in text and "**fail** (1)" in text


def test_history_compares_a_case_only_with_its_own_label(tmp_path: Path) -> None:
    store = tmp_path / "history.duckdb"
    t0 = datetime(2026, 11, 1, tzinfo=UTC)
    history.record(store, _report({"a": True}), command="all", started_at=t0, origin=ORIGIN)
    live = Report(meta={})
    live.agent["live model"] = [_result("a", False, mode="live")]
    history.record(store, live, command="live", started_at=t0 + timedelta(hours=1), origin=ORIGIN)
    assert history.flips(store) == []  # a live failure is not a flip of the scripted case


# --- regression gate -------------------------------------------------------------------------


def test_a_baseline_pass_that_now_fails_is_a_regression() -> None:
    accepted = baseline.snapshot(_report({"a": True, "b": False}))
    now = baseline.snapshot(_report({"a": False, "b": True, "c": True}))
    result = baseline.compare(accepted, now)
    assert result.regressions == ["scripted reference a.t0: passed in the baseline, fails now"]
    assert any("b.t0: fixed" in n for n in result.notes) and any("c.t0: new" in n for n in result.notes)


@pytest.mark.parametrize(("recall3", "regressed"), [(0.91, False), (0.89, True)])
def test_retrieval_metrics_may_drop_at_most_point_05(recall3: float, regressed: bool) -> None:
    accepted = baseline.snapshot(_report({"a": True}, recall3=0.95))
    result = baseline.compare(accepted, baseline.snapshot(_report({"a": True}, recall3=recall3)))
    assert bool(result.regressions) is regressed


def test_baseline_leaves_out_sets_too_small_to_gate() -> None:
    report = _report({"a": True})
    report.retrieval["real set"] = report.retrieval["dev set"]
    assert set(baseline.snapshot(report, {"dev set"})["retrieval"]) == {"dev set"}


def test_cli_gate_fails_on_a_regression_against_the_baseline(tmp_path: Path, db_path: Path) -> None:
    out, store, base = tmp_path / "reports", tmp_path / "h.duckdb", tmp_path / "baseline.json"
    common = ["--db", str(db_path), "--out", str(out), "--history", str(store)]
    assert cli.main([*common, "all", "--gate", "--baseline", str(base)]) == 1  # no baseline: fail closed
    assert cli.main([*common, "baseline", "--update", "--baseline", str(base)]) == 0
    assert cli.main([*common, "all", "--gate", "--baseline", str(base)]) == 0
    # A metric the baseline remembers as higher than it is now: the gate must catch the drop.
    tampered = json.loads(base.read_text())
    tampered["retrieval"]["dev set"]["rrf"]["mrr"] = 1.0
    base.write_text(json.dumps(tampered))
    assert cli.main([*common, "all", "--gate", "--baseline", str(base)]) == 1
    assert cli.main([*common, "baseline", "--baseline", str(base)]) == 1
    runs = history.recent_runs(store)
    assert len(runs) == 3 and any("mrr" in f for f in runs[0].gate_failures)
    assert cli.main([*common, "history"]) == 0


def test_committed_baseline_matches_the_suites() -> None:
    """Every case and validation row in the repo is in the committed baseline, passing."""
    data = baseline.load()
    assert data is not None, "evals/baseline.json is missing"
    ids = {f"{c.id}.t0" for c in load_agent_cases()}
    assert set(data["agent"]["scripted reference"]) == ids
    assert all(data["agent"]["scripted reference"].values()) and all(data["validation"].values())


# --- harvesting ------------------------------------------------------------------------------


def _cassette(path: Path, case: str, *queries: str) -> Path:
    turns = [Turn((ToolUseBlock("search_models", {"query": q, "limit": 5}),)) for q in queries]
    turns.append(Turn((TextBlock("done"),)))
    lines = [json.dumps({"format": 1, "case": case, "model": "m", "trial": 0})]
    lines += [
        json.dumps({"fingerprint": {}, "status": 200, "body": encode_turn(t, "m", chunk_size=7).decode()})
        for t in turns
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_harvest_reads_search_queries_from_cassettes_and_live_reports(tmp_path: Path) -> None:
    dev_query = load_retrieval_cases()[0].query
    _cassette(tmp_path / "otd-2025.t0.jsonl", "otd-2025", "late deliveries by border", dev_query)
    report = {
        "agent": {
            "live": [
                _as_dict(_result("x", True, mode="live", calls=[_search("Late deliveries by border?")]))
            ],
            "scripted": [_as_dict(_result("y", True, calls=[_search("authored, not real")]))],
        }
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    seen = harvest.collect([tmp_path], load_agent_cases())
    assert [s.query for s in seen] == ["late deliveries by border", dev_query, "Late deliveries by border?"]
    assert seen[0].role == "analyst"
    rows = harvest.dedupe(seen, harvest.existing_cases())
    assert [r.query for r in rows] == ["late deliveries by border"]  # dev query and duplicate dropped
    assert len(rows[0].sources) == 2


def _search(query: str) -> dict[str, object]:
    return {"name": "search_models", "input": {"query": query}, "is_error": False, "output": "[]"}


def _as_dict(result: CaseResult) -> dict[str, object]:
    from warehouse_evals.runner import result_dict

    return result_dict(result)


def test_template_then_promote_appends_only_labelled_known_rows(tmp_path: Path, catalog: Catalog) -> None:
    rows = harvest.dedupe(
        [harvest.Seen("trucks stuck at the border", "partner", "c"), harvest.Seen("vendor bills", None, "c")],
        [],
    )
    text = harvest.template(rows, catalog)
    items = yaml.safe_load(text)
    assert items[0]["role"] == "partner" and items[0]["relevant"] == []
    assert "fct_invoices" not in items[0]["candidates"]  # what this role is shown, not everything
    items[0]["relevant"] = ["mart_border_crossing_dwell"]
    items[1]["relevant"] = ["no_such_model"]
    labelled = tmp_path / "labelled.yaml"
    real = tmp_path / "retrieval_real.yaml"
    labelled.write_text(yaml.safe_dump(items), encoding="utf-8")
    assert harvest.promote(labelled, catalog, real).unknown_names  # nothing written on a bad name
    assert not real.exists()
    items[1]["relevant"] = []
    labelled.write_text(yaml.safe_dump(items), encoding="utf-8")
    result = harvest.promote(labelled, catalog, real)
    assert len(result.added) == 1 and result.skipped_unlabelled == 1
    assert [c.query for c in load_retrieval_cases(real)] == ["trucks stuck at the border"]
    assert harvest.promote(labelled, catalog, real).skipped_duplicate == 1  # idempotent


def test_cli_harvest_writes_a_template(tmp_path: Path) -> None:
    _cassette(tmp_path / "otd-2025.t0.jsonl", "otd-2025", "how late are we at the border")
    template = tmp_path / "real.yaml"
    args = ["harvest", str(tmp_path), "--template", str(template), "--real", str(tmp_path / "none.yaml")]
    assert cli.main(args) == 0
    assert [i["query"] for i in yaml.safe_load(template.read_text())] == ["how late are we at the border"]
    assert cli.main(["harvest"]) == 2
