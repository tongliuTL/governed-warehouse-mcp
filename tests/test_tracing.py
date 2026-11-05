"""Langfuse trace shape, scores and PII masking, captured in memory (nothing leaves the process)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.util import ReplyTransport, arun, message_reply
from warehouse_evals import tracing
from warehouse_evals.cases import load_agent_cases
from warehouse_evals.judge import ClaudeJudge, JudgeInput, load_rubric
from warehouse_evals.runner import CaseResult, EvalContext, offline_client, run_case, scripted_client
from warehouse_evals.scripted import POLICIES

TYPE = "langfuse.observation.type"


@pytest.fixture
def live(ctx: EvalContext, monkeypatch: pytest.MonkeyPatch) -> Iterator[Langfuse]:
    """A live client exporting to memory, installed as the process tracer that ``run_case``
    hands to live runs. ``live.exporter`` holds the spans and ``live.scores`` the scores."""
    exporter = InMemorySpanExporter()
    tracing.set_pii_values(ctx.pii_values)
    live = Langfuse(
        public_key=f"pk-lf-test-{id(exporter)}",  # the SDK keeps one pipeline per key
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        mask_otel_spans=tracing._mask_spans,
        should_export_span=tracing.SHOULD_EXPORT_SPAN,
    )
    scores: list[dict[str, Any]] = []
    monkeypatch.setattr(live, "create_score", lambda **kw: scores.append(kw))
    monkeypatch.setattr(tracing._state, "live", live)
    live.exporter, live.scores = exporter, scores  # type: ignore[attr-defined]
    yield live
    live.shutdown()


def _run(ctx: EvalContext, case_id: str, mode: str) -> CaseResult:
    case = next(c for c in load_agent_cases() if c.id == case_id)
    client = scripted_client(POLICIES["reference"](case, secret=ctx.gold(case.expect.must_not_reveal_sql)))
    result = arun(run_case(ctx, case, client, mode=mode, policy="reference"))
    tracing.flush()
    return result


def _spans(live: Langfuse) -> list[ReadableSpan]:
    return sorted(live.exporter.get_finished_spans(), key=lambda s: s.start_time or 0)  # type: ignore[attr-defined]


def test_live_run_is_one_agent_trace_with_generations_and_sibling_tools(
    ctx: EvalContext, live: Langfuse
) -> None:
    result = _run(ctx, "otd-2025", "live")
    spans = _spans(live)
    root = next(s for s in spans if s.parent is None)
    assert root.name == "answer-question" and root.attributes[TYPE] == "agent"
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    children = [s for s in spans if s is not root]
    # Generations and tool calls are siblings under the agent, not nested in each other.
    assert all(s.parent and s.parent.span_id == root.context.span_id for s in children)
    gens = [s for s in children if s.attributes[TYPE] == "generation"]
    tools = [s for s in children if s.attributes[TYPE] == "tool"]
    assert len(gens) == result.turns and len(tools) == len(result.tool_calls)
    assert [t.name for t in tools] == [c["name"] for c in result.tool_calls]
    gen = gens[0]
    assert gen.attributes["langfuse.observation.model.name"] == "claude-opus-5-5"
    usage = json.loads(str(gen.attributes["langfuse.observation.usage_details"]))
    assert {"input", "output"} <= set(usage)  # scripted turns report zero tokens
    first_input = json.loads(str(gen.attributes["langfuse.observation.input"]))
    assert first_input[0]["role"] == "system" and first_input[1]["role"] == "user"
    assert (
        root.attributes["langfuse.observation.output"] == result.final_text
    )  # plain text, readable in the UI
    assert root.attributes["langfuse.trace.name"] == "answer-question"
    assert root.attributes["user.id"] == "eval-analyst"
    names = {s["name"] for s in live.scores}  # type: ignore[attr-defined]
    assert "passed" in names and {f"grader.{g.name}" for g in result.grades} <= names


def test_pii_never_leaves_the_process(ctx: EvalContext, live: Langfuse) -> None:
    result = _run(ctx, "support-contact-email", "live")
    leaked = [v for v in ctx.pii_values if v in result.final_text]
    assert leaked, "the support role may read PII, so the answer itself contains it"
    exported = json.dumps([dict(s.attributes or {}) for s in _spans(live)])
    assert not [v for v in ctx.pii_values if v in exported]
    assert tracing.REDACTED in exported


@pytest.mark.parametrize("mode", ["scripted", "replay"])
def test_fixture_runs_are_not_traced(ctx: EvalContext, live: Langfuse, mode: str) -> None:
    _run(ctx, "otd-2025", mode)
    assert live.exporter.get_finished_spans() == ()
    assert live.scores == []  # type: ignore[attr-defined]


def test_judge_call_is_an_evaluator_wrapping_one_generation(live: Langfuse) -> None:
    rubric = load_rubric()
    verdicts = {cid: {"reason": "tool call 1", "verdict": "pass"} for cid in rubric.criterion_ids}
    usage = {"input_tokens": 900, "output_tokens": 80, "cache_read_input_tokens": 800}
    transport = ReplyTransport(message_reply(json.dumps(verdicts), usage=usage))
    item = JudgeInput("otd-2025", "reference", "analyst", "What was OTD?", "79.7%", ())
    arun(ClaudeJudge(offline_client(transport), rubric, tracer=live).judge(item))
    tracing.flush()
    spans = _spans(live)
    assert [(s.name, s.attributes[TYPE]) for s in spans] == [
        ("grade-trajectory", "evaluator"),
        ("judge-trajectory", "generation"),
    ]
    root, gen = spans
    assert gen.parent and gen.parent.span_id == root.context.span_id
    usage = json.loads(str(gen.attributes["langfuse.observation.usage_details"]))
    assert usage == {"input": 900, "output": 80, "cache_read_input_tokens": 800}
    assert set(json.loads(str(root.attributes["langfuse.observation.output"]))) == set(rubric.criterion_ids)
