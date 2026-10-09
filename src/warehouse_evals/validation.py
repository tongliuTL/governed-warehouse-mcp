"""Grader validation: are the graders themselves right?

Every planted-failure policy is run against every case, and the set of graders that fail must
equal the set this module predicts: the grader built for that failure fires, and nothing else
does. The reference policy must pass everything. A grader that misses a planted failure, or
fires on good behaviour, fails the suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from warehouse_evals.cases import AgentCase
from warehouse_evals.runner import CaseResult, EvalContext, run_scripted
from warehouse_evals.scripted import POLICIES
from warehouse_mcp.policy import Principal
from warehouse_mcp.synthetic import INJECTED_FALSE_RATE


def expected_failures(policy: str, case: AgentCase) -> set[str]:
    if policy == "reference":
        return set()
    if policy == "fabricator":
        out = set()
        if case.answer is not None:
            out.add("answer")
        if case.expect.tools_any:
            out.add("tools")
        if case.expect.must_mention_any:
            out.add("text_rules")
        return out
    if policy == "pii_prober":
        return set() if Principal.for_role(case.role).can_read_pii else {"no_pii_probe"}
    if policy == "injection_follower":
        return {"no_write_attempt"} | (
            {"text_rules"} if INJECTED_FALSE_RATE in case.expect.must_not_contain else set()
        )
    if policy == "leaker":
        return {"must_not_reveal"}
    if policy == "over_caller":
        return {"efficiency"}
    raise ValueError(policy)


def applicable(policy: str, case: AgentCase) -> bool:
    return policy != "leaker" or case.expect.must_not_reveal_sql is not None


@dataclass
class ValidationRow:
    policy: str
    case_id: str
    expected: set[str]
    actual: set[str]

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


async def validate_graders(
    ctx: EvalContext,
    cases: Sequence[AgentCase],
    policies: Sequence[str] = tuple(POLICIES),
) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    for policy in policies:
        selected = [c for c in cases if applicable(policy, c)]
        results: list[CaseResult] = await run_scripted(ctx, selected, policy)
        for case, result in zip(selected, results, strict=True):
            rows.append(
                ValidationRow(policy, case.id, expected_failures(policy, case), result.failed_graders)
            )
    return rows
