"""The agent loop against scripted SSE streams through the real Anthropic SDK."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import anthropic
import httpx2
import pytest

from tests.util import run_agent as run_with_client
from warehouse_evals.agent import FALLBACK_BETA, SYSTEM_PROMPT, Trajectory
from warehouse_evals.runner import offline_client
from warehouse_evals.sse import TextBlock, ToolUseBlock, Turn, encode_turn
from warehouse_evals.transport import ScriptedTransport
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse

Step = Turn | Callable[[dict[str, Any]], Turn]


def scripted(steps: Sequence[Step]) -> tuple[anthropic.AsyncAnthropic, ScriptedTransport]:
    def policy(body: dict[str, Any]) -> Turn:
        step = steps[min(len(transport.requests) - 1, len(steps) - 1)]
        return step(body) if callable(step) else step

    transport = ScriptedTransport(policy)
    return offline_client(transport), transport


def run_agent(
    gw: GovernedWarehouse, principal: Principal, steps: Sequence[Step], **kwargs: Any
) -> tuple[Trajectory, ScriptedTransport]:
    client, transport = scripted(steps)
    return run_with_client(gw, principal, client, **kwargs), transport


def last_tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in body["messages"][-1]["content"] if b["type"] == "tool_result"]


def test_request_shape(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, transport = run_agent(governed, analyst, [Turn((TextBlock("hi"),))], effort="medium")
    body = transport.requests[0]
    assert body["model"] == "claude-opus-5-5" and body["system"] == SYSTEM_PROMPT
    assert body["stream"] is True and body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"} and body["fallbacks"] == "default"
    assert all(t["eager_input_streaming"] for t in body["tools"])
    assert traj.final_text == "hi" and traj.stop == "end_turn" and traj.text_deltas >= 1


def test_beta_header_is_sent(governed: GovernedWarehouse, analyst: Principal) -> None:
    seen: list[str] = []

    class Spy(ScriptedTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            seen.append(request.headers.get("anthropic-beta", ""))
            return await super().handle_async_request(request)

    spy = Spy(lambda body: Turn((TextBlock("ok"),)))
    run_with_client(governed, analyst, offline_client(spy))
    assert FALLBACK_BETA in seen[0]


def test_tool_round_trip_and_parallel_results(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, transport = run_agent(
        governed,
        analyst,
        [
            Turn(
                (
                    TextBlock("Checking."),
                    ToolUseBlock("list_metrics", {}),
                    ToolUseBlock("query_metric", {"metric": "shipment_count"}),
                )
            ),
            Turn((TextBlock("Done."),)),
        ],
    )
    assert [c.name for c in traj.tool_calls] == ["list_metrics", "query_metric"]
    results = last_tool_results(transport.requests[1])
    assert len(results) == 2 and not any(r["is_error"] for r in results)  # one user turn carries both results
    assert json.loads(results[1]["content"])["rows"][0][0] > 0
    assistant = transport.requests[1]["messages"][1]
    assert assistant["role"] == "assistant" and assistant["content"][0]["text"] == "Checking."  # append-only
    assert traj.input_json_deltas >= 2 and traj.turns == 2


def test_tool_errors_are_returned_to_the_model(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, transport = run_agent(
        governed,
        analyst,
        [
            Turn((ToolUseBlock("run_sql", {"sql": "select contact_email from dim_customers"}),)),
            Turn((TextBlock("I can't access that."),)),
        ],
    )
    [result] = last_tool_results(transport.requests[1])
    assert result["is_error"] and "PII_DENIED" in result["content"]
    assert traj.tool_calls[0].error_code == "PII_DENIED"


def test_malformed_tool_json_is_never_executed(governed: GovernedWarehouse, analyst: Principal) -> None:
    # The SDK's tolerant parser would read this as {"sql": "select 1"}; the strict re-parse refuses it.
    traj, transport = run_agent(
        governed,
        analyst,
        [
            Turn((ToolUseBlock("run_sql", raw_json='{"sql": "select 1'),)),
            Turn((TextBlock("Retrying failed; stopping."),)),
        ],
    )
    [result] = last_tool_results(transport.requests[1])
    assert result["is_error"] and json.loads(result["content"]) == {"INVALID_JSON": '{"sql": "select 1'}
    assert traj.tool_calls == [] and traj.invalid_inputs == 1


def test_schema_invalid_and_unknown_tools(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, transport = run_agent(
        governed,
        analyst,
        [
            Turn((ToolUseBlock("run_sql", {"query": "select 1"}), ToolUseBlock("drop_everything", {}))),
            Turn((TextBlock("ok"),)),
        ],
    )
    first, second = last_tool_results(transport.requests[1])
    assert first["is_error"] and "'sql' is a required property" in first["content"]
    assert second["is_error"] and second["content"] == "UNKNOWN_TOOL: drop_everything"
    assert traj.invalid_inputs == 2 and traj.tool_calls == []


def test_truncated_tool_call_is_not_run(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, transport = run_agent(
        governed,
        analyst,
        [
            Turn(
                (ToolUseBlock("run_sql", {"sql": "select count(*) from dim_lanes"}),),
                stop_reason="max_tokens",
            ),
        ],
    )
    assert traj.stop == "max_tokens" and traj.tool_calls == [] and len(transport.requests) == 1


def test_refusal_stops_without_running_tools(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, _ = run_agent(
        governed,
        analyst,
        [
            Turn((TextBlock("I can't help"), ToolUseBlock("list_metrics", {})), stop_reason="refusal"),
        ],
    )
    assert traj.stop == "refusal" and traj.tool_calls == []


def test_pause_turn_continues(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, transport = run_agent(
        governed,
        analyst,
        [
            Turn((TextBlock("thinking..."),), stop_reason="pause_turn"),
            Turn((TextBlock("final"),)),
        ],
    )
    assert traj.final_text == "final" and len(transport.requests) == 2
    assert transport.requests[1]["messages"][-1]["role"] == "assistant"


def test_max_turns(governed: GovernedWarehouse, analyst: Principal) -> None:
    traj, _ = run_agent(governed, analyst, [Turn((ToolUseBlock("list_metrics", {}),))], max_turns=3)
    assert traj.stop == "max_turns" and len(traj.tool_calls) == 3


def test_unparseable_stream_is_retried_then_abandoned(
    governed: GovernedWarehouse, analyst: Principal, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anthropic.lib.streaming import _beta_messages

    def explode(*_: Any, **__: Any) -> Any:
        raise ValueError("Unable to parse tool parameter JSON from model")

    monkeypatch.setattr(_beta_messages, "accumulate_event", explode)
    traj, transport = run_agent(governed, analyst, [Turn((ToolUseBlock("list_metrics", {}),))])
    assert traj.stop == "invalid_json" and traj.json_retries == 3 and len(transport.requests) == 3


def test_api_errors_are_recorded(governed: GovernedWarehouse, analyst: Principal) -> None:
    class Failing(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
            )

    traj = run_with_client(governed, analyst, offline_client(Failing()))
    assert traj.stop == "api_error" and "busy" in (traj.error or "")


def test_sse_encoding_is_parseable() -> None:
    raw = encode_turn(Turn((TextBlock("a" * 50), ToolUseBlock("t", {"x": 1}, id="toolu_1"))), model="m")
    events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "message_start" and events[-1]["type"] == "message_stop"
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
    assert text == "a" * 50
