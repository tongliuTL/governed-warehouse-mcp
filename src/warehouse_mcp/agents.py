"""Which agents may call the server, and the most each one may reach.

Every token names its agent in the ``client_id`` claim. The registry maps each agent to the
scopes it is permitted; a token's effective scopes are the intersection of what it carries and
what its agent is permitted. That bounds an agent even when the identity provider mis-issues a
token, and it fails closed: a missing, unknown or disabled agent is rejected, never defaulted.
Onboarding an agent is one registry entry.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class UnknownAgent(Exception):
    """The token's agent is missing, not registered, or disabled."""


class Agent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: str = Field(min_length=1)
    description: str = ""
    permitted_scopes: frozenset[str]
    enabled: bool = True


class AgentRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agents: dict[str, Agent]

    def scopes_for(self, client_id: str | None, token_scopes: Iterable[str]) -> frozenset[str]:
        """The token's scopes that its agent is permitted to use."""
        agent = self.agents.get(client_id) if client_id else None
        if agent is None or not agent.enabled:
            raise UnknownAgent(
                "token has no client_id" if not client_id else f"agent {client_id!r} is not enabled"
            )
        return frozenset(token_scopes) & agent.permitted_scopes


def load_agents(path: Path | str | None = None) -> AgentRegistry:
    """The registry at ``path``, or the packaged one, which only registers local development."""
    if path is None:
        text = resources.files("warehouse_mcp").joinpath("agents.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return AgentRegistry.model_validate(yaml.safe_load(text))
