"""Record a run, replay it offline, and detect drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from tests.util import run_agent
from warehouse_evals.agent import Trajectory
from warehouse_evals.runner import offline_client
from warehouse_evals.sse import TextBlock, ToolUseBlock, Turn
from warehouse_evals.transport import CassetteMismatch, RecordingTransport, ReplayTransport, ScriptedTransport
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse


def policy(body: dict[str, Any]) -> Turn:
    if len(body["messages"]) == 1:
        return Turn((ToolUseBlock("query_metric", {"metric": "shipment_count"}),))
    rows = json.loads(body["messages"][-1]["content"][0]["content"])["rows"]
    return Turn((TextBlock(f"{rows[0][0]} shipments."),))


def agent_run(
    gw: GovernedWarehouse, principal: Principal, transport: httpx2.AsyncBaseTransport, **kwargs: Any
) -> Trajectory:
    client = offline_client(transport, api_key="sk-ant-secret-should-not-be-saved")
    return run_agent(gw, principal, client, "How many shipments?", **kwargs)


def test_record_then_replay(governed: GovernedWarehouse, analyst: Principal, tmp_path: Path) -> None:
    cassette = tmp_path / "case.t0.jsonl"
    recorder = RecordingTransport(ScriptedTransport(policy), cassette)
    live = agent_run(governed, analyst, recorder)
    recorder.save({"case": "demo", "model": "claude-opus-5-5"})
    text = cassette.read_text()
    assert "sk-ant" not in text and "x-api-key" not in text.lower()  # credentials never reach disk
    assert len(text.splitlines()) == 3  # header + two interactions

    replayed = agent_run(governed, analyst, ReplayTransport(cassette))
    assert replayed.final_text == live.final_text
    assert [c.name for c in replayed.tool_calls] == [c.name for c in live.tool_calls]


def test_replay_detects_prompt_drift(governed: GovernedWarehouse, analyst: Principal, tmp_path: Path) -> None:
    cassette = tmp_path / "case.t0.jsonl"
    recorder = RecordingTransport(ScriptedTransport(policy), cassette)
    agent_run(governed, analyst, recorder)
    recorder.save({})
    replay = ReplayTransport(cassette)
    traj = agent_run(governed, analyst, replay, system="A different prompt.")
    assert traj.stop == "api_error"  # the SDK surfaces transport errors as connection errors...
    assert replay.failure and "system" in replay.failure  # ...and the transport keeps the real reason


def test_replay_detects_extra_requests(
    governed: GovernedWarehouse, analyst: Principal, tmp_path: Path
) -> None:
    cassette = tmp_path / "short.jsonl"
    recorder = RecordingTransport(ScriptedTransport(lambda body: Turn((TextBlock("x"),))), cassette)
    agent_run(governed, analyst, recorder)
    recorder.save({})
    lines = cassette.read_text().splitlines()
    replay = ReplayTransport(cassette)
    replay.interactions = []  # simulate a run that needs more turns than were recorded
    agent_run(governed, analyst, replay)
    assert replay.failure and "more requests" in replay.failure
    assert len(lines) == 2


def test_rejects_unknown_cassette_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"format": 99}) + "\n")
    with pytest.raises(CassetteMismatch, match="format"):
        ReplayTransport(path)
