"""Run agent cases end to end: Claude API (scripted, replayed or live) -> agent -> MCP -> warehouse.

Each case gets a fresh server bound to the case's role and a fresh audit log. Gold answers are
computed from the warehouse at run time, so the suite can never drift from the data.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import anthropic
import httpx2
from mcp import Client
from mcp.server import MCPServer
from sqlglot import exp

from warehouse_evals.agent import DEFAULT_MODEL, Trajectory, WarehouseAgent
from warehouse_evals.cases import AgentCase
from warehouse_evals.graders import Grade, grade_all
from warehouse_evals.scripted import POLICIES
from warehouse_evals.transport import Policy, RecordingTransport, ReplayTransport, ScriptedTransport
from warehouse_mcp.catalog import Catalog, load_catalog
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse, ServerConfig
from warehouse_mcp.warehouse import Warehouse

Mode = Literal["scripted", "replay", "live"]


@dataclass
class CaseResult:
    case_id: str
    suite: str
    role: str
    mode: str
    policy: str
    trial: int
    passed: bool
    grades: list[Grade]
    final_text: str
    stop: str
    tool_calls: list[dict[str, Any]]
    turns: int
    input_tokens: int
    output_tokens: int
    text_deltas: int
    input_json_deltas: int
    elapsed_ms: int
    first_token_ms: int | None
    audit: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def failed_graders(self) -> set[str]:
        return {g.name for g in self.grades if not g.passed}


def _select_columns(model: str, columns: Sequence[str]) -> str:
    return (
        exp.select(*(exp.column(c, quoted=True) for c in columns)).from_(exp.table_(model, quoted=True)).sql()
    )


class EvalContext:
    """Shared, read-only state for a batch of runs."""

    def __init__(
        self, db: Path | str, catalog: Catalog | None = None, config: ServerConfig | None = None
    ) -> None:
        self.catalog = catalog or load_catalog()
        self.config = config or ServerConfig()
        self.warehouse = Warehouse(db, timeout_s=self.config.query_timeout_s)
        # Every PII value in the warehouse, from whatever columns the catalog marks as PII.
        self.pii_values = [
            str(v)
            for m in self.catalog.models
            if m.pii_columns
            for row in self.warehouse.execute_trusted(_select_columns(m.name, sorted(m.pii_columns)))
            for v in row
            if v
        ]
        self._gold: dict[str, tuple[Any, ...] | None] = {}
        self._servers: dict[str, tuple[GovernedWarehouse, MCPServer]] = {}

    def gold(self, sql: str | None) -> tuple[Any, ...] | None:
        if sql is None:
            return None
        if sql not in self._gold:
            rows = self.warehouse.execute_trusted(sql)
            self._gold[sql] = rows[0] if rows else None
        return self._gold[sql]

    def server(self, role: str) -> tuple[GovernedWarehouse, MCPServer]:
        """The server for a role, built once per role (tool schemas and the search index are
        not cheap) and reset to a fresh audit log and rate limit for each case."""
        if role not in self._servers:
            gw = GovernedWarehouse(warehouse=self.warehouse, catalog=self.catalog, config=self.config)
            principal = Principal.for_role(role, subject=f"eval-{role}")
            self._servers[role] = (gw, gw.build_mcp(lambda: principal))
        gw, server = self._servers[role]
        gw.reset()
        return gw, server

    def close(self) -> None:
        self.warehouse.close()


def offline_client(
    transport: httpx2.AsyncBaseTransport, api_key: str = "offline"
) -> anthropic.AsyncAnthropic:
    """The real SDK client over a transport that never touches the network."""
    return anthropic.AsyncAnthropic(
        api_key=api_key, max_retries=0, http_client=httpx2.AsyncClient(transport=transport)
    )


def scripted_client(policy: Policy) -> anthropic.AsyncAnthropic:
    return offline_client(ScriptedTransport(policy))


def replay_client(cassette: Path) -> tuple[anthropic.AsyncAnthropic, ReplayTransport]:
    transport = ReplayTransport(cassette)
    return offline_client(transport), transport


def live_client(record_to: Path | None) -> tuple[anthropic.AsyncAnthropic, RecordingTransport | None]:
    if record_to is None:
        return anthropic.AsyncAnthropic(), None
    recorder = RecordingTransport(httpx2.AsyncHTTPTransport(), record_to)
    return anthropic.AsyncAnthropic(
        http_client=anthropic.DefaultAsyncHttpxClient(transport=recorder)
    ), recorder


async def run_case(
    ctx: EvalContext,
    case: AgentCase,
    client: anthropic.AsyncAnthropic,
    *,
    mode: str,
    policy: str,
    trial: int = 0,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    on_text: Callable[[str], None] | None = None,
) -> CaseResult:
    gw, server = ctx.server(case.role)
    async with Client(server) as mcp:
        traj: Trajectory = await WarehouseAgent(client, mcp, model=model, effort=effort, on_text=on_text).run(
            case.question
        )
    gold = ctx.gold(case.reference_sql)
    secret = ctx.gold(case.expect.must_not_reveal_sql)
    grades = grade_all(
        case,
        traj,
        gold=gold,
        secret=secret,
        can_read_pii=Principal.for_role(case.role).can_read_pii,
        pii_values=ctx.pii_values,
        pii_columns=sorted(ctx.catalog.pii_column_names),
    )
    if traj.error:
        grades.append(Grade("completed", False, traj.error))
    return CaseResult(
        case_id=case.id,
        suite=case.suite,
        role=case.role,
        mode=mode,
        policy=policy,
        trial=trial,
        passed=all(g.passed for g in grades),
        grades=grades,
        final_text=traj.final_text,
        stop=traj.stop,
        tool_calls=[
            {
                "name": c.name,
                "input": c.input,
                "is_error": c.is_error,
                "error_code": c.error_code,
                "latency_ms": c.latency_ms,
            }
            for c in traj.tool_calls
        ],
        turns=traj.turns,
        input_tokens=traj.input_tokens,
        output_tokens=traj.output_tokens,
        text_deltas=traj.text_deltas,
        input_json_deltas=traj.input_json_deltas,
        elapsed_ms=traj.elapsed_ms,
        first_token_ms=traj.first_token_ms,
        error=traj.error,
        audit=[
            {"tool": r.tool, "decision": r.decision, "reason_code": r.reason_code, "models": r.models}
            for r in gw.audit.records()
        ],
    )


async def run_scripted(
    ctx: EvalContext, cases: Sequence[AgentCase], policy: str = "reference"
) -> list[CaseResult]:
    factory = POLICIES[policy]
    results = []
    for case in cases:
        client = scripted_client(factory(case, secret=ctx.gold(case.expect.must_not_reveal_sql)))
        results.append(await run_case(ctx, case, client, mode="scripted", policy=policy))
    return results


async def run_replay(ctx: EvalContext, cases: Sequence[AgentCase], cassette_dir: Path) -> list[CaseResult]:
    results = []
    for case in cases:
        for trial, cassette in enumerate(sorted(cassette_dir.glob(f"{case.id}.t*.jsonl"))):
            client, transport = replay_client(cassette)
            model = str(transport.meta.get("model", DEFAULT_MODEL))
            result = await run_case(
                ctx,
                case,
                client,
                mode="replay",
                policy=model,
                trial=trial,
                model=model,
                effort=str(transport.meta.get("effort", "high")),
            )
            if transport.failure or transport.position != len(transport.interactions):
                reason = transport.failure or f"{cassette.name}: run ended before the recording did"
                result.grades.append(Grade("replay_fidelity", False, reason))
                result.passed = False
            results.append(result)
    return results


async def run_live(
    ctx: EvalContext,
    cases: Sequence[AgentCase],
    *,
    model: str,
    effort: str,
    trials: int,
    record_dir: Path | None,
    on_text: Callable[[str], None] | None = None,
) -> list[CaseResult]:
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise SystemExit("live mode needs ANTHROPIC_API_KEY (or another configured Anthropic credential)")
    results = []
    for case in cases:
        for trial in range(trials):
            path = record_dir / f"{case.id}.t{trial}.jsonl" if record_dir else None
            client, recorder = live_client(path)
            result = await run_case(
                ctx,
                case,
                client,
                mode="live",
                policy=model,
                trial=trial,
                model=model,
                effort=effort,
                on_text=on_text,
            )
            if recorder is not None:
                recorder.save(
                    {
                        "case": case.id,
                        "model": model,
                        "effort": effort,
                        "trial": trial,
                        "sdk": anthropic.__version__,
                    }
                )
            results.append(result)
    return results


def result_dict(result: CaseResult) -> dict[str, Any]:
    return asdict(result)
