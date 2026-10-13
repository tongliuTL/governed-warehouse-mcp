"""Typed loaders for the eval case files."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from warehouse_mcp.policy import ROLE_SCOPES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerSpec(_Strict):
    kind: Literal["number", "ratio", "entity"]
    abs_tol: float | None = None
    rel_tol: float | None = None


class Expectations(_Strict):
    tools_any: tuple[str, ...] = ()
    tools_none: tuple[str, ...] = ()
    must_not_reveal_sql: str | None = None
    must_not_contain: tuple[str, ...] = ()
    must_mention_any: tuple[str, ...] = ()
    max_tool_calls: int = 10


class ReferenceCall(_Strict):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)


class Reference(_Strict):
    calls: tuple[ReferenceCall, ...]
    answer: str


class AgentCase(_Strict):
    id: str
    suite: Literal["answer", "governance", "injection"]
    role: str
    question: str
    reference_sql: str | None = None
    answer: AnswerSpec | None = None
    expect: Expectations = Expectations()
    reference: Reference

    def model_post_init(self, _context: Any) -> None:
        if self.role not in ROLE_SCOPES:
            raise ValueError(f"{self.id}: unknown role {self.role!r}")
        if (self.answer is None) != (self.reference_sql is None):
            raise ValueError(f"{self.id}: answer and reference_sql go together")


class RetrievalCase(_Strict):
    query: str
    relevant: tuple[str, ...]


def _read(name: str, path: Path | None) -> Any:
    if path is not None:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return yaml.safe_load(resources.files(__package__).joinpath("cases", name).read_text(encoding="utf-8"))


def load_agent_cases(path: Path | None = None) -> list[AgentCase]:
    cases = [AgentCase.model_validate(item) for item in _read("agent_cases.yaml", path)]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


def load_retrieval_cases(path: Path | None = None) -> list[RetrievalCase]:
    return [RetrievalCase.model_validate(item) for item in _read("retrieval_cases.yaml", path)]
