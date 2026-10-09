"""Turn eval results into a JSON artifact and a Markdown report."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from warehouse_evals.retrieval import KS, RankerScore
from warehouse_evals.runner import CaseResult
from warehouse_evals.validation import ValidationRow


@dataclass
class Report:
    meta: dict[str, Any]
    agent: dict[str, list[CaseResult]] = field(default_factory=dict)  # label -> results
    validation: list[ValidationRow] = field(default_factory=list)
    retrieval: dict[str, list[RankerScore]] = field(default_factory=dict)  # set name -> scores


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _agent_section(label: str, results: Sequence[CaseResult]) -> list[str]:
    lines = [f"### Agent cases: {label}", ""]
    by_suite: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_suite[r.suite].append(r)
    lines += ["| suite | runs | passed | pass rate |", "|---|---:|---:|---:|"]
    for suite in ("answer", "governance", "injection"):
        rs = by_suite.get(suite, [])
        if rs:
            passed = sum(r.passed for r in rs)
            lines.append(f"| {suite} | {len(rs)} | {passed} | {_pct(passed / len(rs))} |")
    total = sum(r.passed for r in results)
    lines.append(f"| **all** | {len(results)} | {total} | **{_pct(total / len(results))}** |")
    trials: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        trials[r.case_id].append(r.passed)
    if any(len(v) > 1 for v in trials.values()):
        k = min(len(v) for v in trials.values())
        all_pass = sum(all(v) for v in trials.values()) / len(trials)
        lines += ["", f"pass^{k} (every trial of a case passes): {_pct(all_pass)} over {len(trials)} cases."]
    grader_fail = Counter(g.name for r in results for g in r.grades if not g.passed)
    grader_names = sorted({g.name for r in results for g in r.grades})
    lines += ["", "| grader | pass rate |", "|---|---:|"]
    for name in grader_names:
        lines.append(f"| {name} | {_pct(1 - grader_fail[name] / len(results))} |")
    lines += [
        "",
        "| case | suite | role | result | failed graders | tool calls | turns | streamed deltas |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        failed = ", ".join(sorted(r.failed_graders)) or "-"
        calls = " → ".join(
            c["name"] + (f" ✗{c['error_code']}" if c["is_error"] else "") for c in r.tool_calls
        )
        lines.append(
            f"| {r.case_id}{f' (t{r.trial})' if r.trial else ''} | {r.suite} | {r.role} | "
            f"{'pass' if r.passed else '**fail**'} | {failed} | {len(r.tool_calls)}: {calls or '-'} | "
            f"{r.turns} | {r.text_deltas + r.input_json_deltas} |"
        )
    tokens_in = sum(r.input_tokens for r in results)
    tokens_out = sum(r.output_tokens for r in results)
    if tokens_in or tokens_out:
        lines += ["", f"Tokens: {tokens_in:,} input, {tokens_out:,} output."]
    return [*lines, ""]


def _validation_section(rows: Sequence[ValidationRow]) -> list[str]:
    planted = [r for r in rows if r.expected]
    caught = sum(r.expected <= r.actual for r in planted)
    false_alarms = sum(len(r.actual - r.expected) for r in rows)
    ref = [r for r in rows if r.policy == "reference"]
    lines = [
        "### Grader validation",
        "",
        f"Planted failures caught: **{caught}/{len(planted)}**. Graders firing where they should not: "
        f"**{false_alarms}**. Reference trajectories passing every grader: **{sum(r.ok for r in ref)}/"
        f"{len(ref)}**.",
        "",
        "| policy | runs | exact verdict match | grader expected to fire |",
        "|---|---:|---:|---|",
    ]
    by_policy: dict[str, list[ValidationRow]] = defaultdict(list)
    for r in rows:
        by_policy[r.policy].append(r)
    for policy, rs in by_policy.items():
        expected = sorted({g for r in rs for g in r.expected}) or ["none"]
        lines.append(f"| {policy} | {len(rs)} | {sum(r.ok for r in rs)}/{len(rs)} | {', '.join(expected)} |")
    mismatches = [r for r in rows if not r.ok]
    if mismatches:
        lines += ["", "Mismatches:", ""]
        lines += [
            f"- {r.policy} / {r.case_id}: expected {sorted(r.expected)}, got {sorted(r.actual)}"
            for r in mismatches
        ]
    return [*lines, ""]


def _retrieval_section(name: str, scores: Sequence[RankerScore], n: int) -> list[str]:
    head = " | ".join(f"Recall@{k}" for k in KS)
    lines = [
        f"### Retrieval: {name} ({n} queries)",
        "",
        f"| ranker | {head} | MRR |",
        "|---|" + "---:|" * (len(KS) + 1),
    ]
    for s in scores:
        cells = " | ".join(_pct(s.recall[k]) for k in KS)
        lines.append(f"| {s.ranker} | {cells} | {s.mrr:.3f} |")
    return [*lines, ""]


def render_markdown(report: Report, retrieval_sizes: dict[str, int]) -> str:
    meta = report.meta
    lines = [
        "# Eval report",
        "",
        f"Catalog `{meta.get('catalog_version')}` · data seed `{meta.get('seed')}` · "
        f"{meta.get('generated_at')}",
        "",
    ]
    if meta.get("note"):
        lines += [str(meta["note"]), ""]
    for label, results in report.agent.items():
        lines += _agent_section(label, results)
    if report.validation:
        lines += _validation_section(report.validation)
    for name, scores in report.retrieval.items():
        lines += _retrieval_section(name, scores, retrieval_sizes.get(name, 0))
    return "\n".join(lines).rstrip() + "\n"
