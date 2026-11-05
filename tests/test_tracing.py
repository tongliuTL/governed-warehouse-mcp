"""Langfuse trace shape, scores and PII masking, captured in memory (nothing leaves the process)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from warehouse_evals import tracing
from warehouse_evals.cases import load_agent_cases
from warehouse_evals.runner import CaseResult, EvalContext, run_case, scripted_client
from warehouse_evals.scripted import POLICIES

TYPE = "langfuse.observation.type"


@pytest.fixture(scope="module")
def ctx(db_path: Path) -> Iterator[EvalContext]:
    ctx = EvalContext(db_path)
    yield ctx
    ctx.close()


@pytest.fixture
def exporter(ctx: EvalContext, monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
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
    exporter.scores = scores  # type: ignore[attr-defined]
    yield exporter
    live.shutdown()


def _run(ctx: EvalContext, case_id: str, mode: str, exporter: InMemorySpanExporter) -> CaseResult:
    case = next(c for c in load_agent_cases() if c.id == case_id)
    client = scripted_client(POLICIES["reference"](case, secret=ctx.gold(case.expect.must_not_reveal_sql)))

    async def go() -> CaseResult:
        return await run_case(ctx, case, client, mode=mode, policy="reference")

    result = anyio.run(go)
    tracing.flush()
    return result


def _spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return sorted(exporter.get_finished_spans(), key=lambda s: s.start_time or 0)


def test_live_run_is_one_agent_trace_with_generations_and_sibling_tools(
    ctx: EvalContext, exporter: InMemorySpanExporter
) -> None:
    result = _run(ctx, "otd-2025", "live", exporter)
    spans = _spans(exporter)
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
    names = {s["name"] for s in exporter.scores}  # type: ignore[attr-defined]
    assert "passed" in names and {f"grader.{g.name}" for g in result.grades} <= names


def test_pii_never_leaves_the_process(ctx: EvalContext, exporter: InMemorySpanExporter) -> None:
    result = _run(ctx, "support-contact-email", "live", exporter)
    leaked = [v for v in ctx.pii_values if v in result.final_text]
    assert leaked, "the support role may read PII, so the answer itself contains it"
    exported = json.dumps([dict(s.attributes or {}) for s in _spans(exporter)])
    assert not [v for v in ctx.pii_values if v in exported]
    assert tracing.REDACTED in exported


@pytest.mark.parametrize("mode", ["scripted", "replay"])
def test_fixture_runs_are_not_traced(ctx: EvalContext, exporter: InMemorySpanExporter, mode: str) -> None:
    _run(ctx, "otd-2025", mode, exporter)
    assert exporter.get_finished_spans() == ()
    assert exporter.scores == []  # type: ignore[attr-defined]


def test_judge_call_is_an_evaluator_wrapping_one_generation(exporter: InMemorySpanExporter) -> None:
    import httpx2

    from warehouse_evals.judge import ClaudeJudge, JudgeInput, load_rubric
    from warehouse_evals.runner import offline_client

    rubric = load_rubric()
    verdicts = {cid: {"reason": "tool call 1", "verdict": "pass"} for cid in rubric.criterion_ids}
    reply = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": json.dumps(verdicts)}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 80, "cache_read_input_tokens": 800},
    }

    class Reply(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=reply)

    item = JudgeInput("otd-2025", "reference", "analyst", "What was OTD?", "79.7%", ())

    async def go() -> None:
        await ClaudeJudge(offline_client(Reply()), rubric, trace=True).judge(item)

    anyio.run(go)
    tracing.flush()
    spans = _spans(exporter)
    assert [(s.name, s.attributes[TYPE]) for s in spans] == [
        ("grade-trajectory", "evaluator"),
        ("judge-trajectory", "generation"),
    ]
    root, gen = spans
    assert gen.parent and gen.parent.span_id == root.context.span_id
    usage = json.loads(str(gen.attributes["langfuse.observation.usage_details"]))
    assert usage == {"input": 900, "output": 80, "cache_read_input_tokens": 800}
    assert set(json.loads(str(root.attributes["langfuse.observation.output"]))) == set(rubric.criterion_ids)
