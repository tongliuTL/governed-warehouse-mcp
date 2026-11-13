"""LLM judge and its calibration, fully offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import pytest

from tests.util import ReplyTransport, arun, message_reply
from warehouse_evals import cli
from warehouse_evals.calibration import (
    REFERENCE,
    Agreement,
    Item,
    cohens_kappa,
    collect,
    compare_with_graders,
    compare_with_humans,
    export_label_template,
    load_labels,
    render_markdown,
)
from warehouse_evals.cases import AgentCase, load_agent_cases
from warehouse_evals.judge import (
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    JUDGE_ATTEMPTS,
    VERDICT_TOOL,
    CachedJudge,
    ClaudeJudge,
    CriterionVerdict,
    JudgeError,
    JudgeInput,
    Judgment,
    MissingJudgment,
    Rubric,
    Verdict,
    load_rubric,
    system_prompt,
    verdict_schema,
)
from warehouse_evals.runner import EvalContext, offline_client

T = TypeVar("T")


@pytest.fixture(scope="module")
def rubric() -> Rubric:
    return load_rubric()


@pytest.fixture(scope="module")
def cases() -> list[AgentCase]:
    return load_agent_cases()


class OracleJudge:
    """Returns the deterministic reference verdict for each criterion (pass where none exists).

    A perfect judge by construction: calibration against the graders must show kappa 1.0.
    """

    def __init__(self, rubric: Rubric, ctx: EvalContext, cases: list[AgentCase]) -> None:
        self.rubric = rubric
        self.ctx = ctx
        self.by_question = {c.question: c for c in cases}
        self.verdicts: dict[tuple[str, str], dict[str, Verdict]] = {}

    async def judge(self, item: JudgeInput) -> Judgment:
        verdicts = self.verdicts.get((item.policy, item.case_id), {})
        return Judgment(
            case_id=item.case_id,
            policy=item.policy,
            rubric=self.rubric.key,
            model=DEFAULT_JUDGE_MODEL,
            effort=DEFAULT_JUDGE_EFFORT,
            verdicts={
                cid: CriterionVerdict(verdicts.get(cid, "pass"), "oracle")
                for cid in self.rubric.criterion_ids
            },
        )


class ConstantJudge:
    def __init__(self, rubric: Rubric, verdict: Verdict) -> None:
        self.rubric = rubric
        self.verdict = verdict
        self.calls = 0

    async def judge(self, item: JudgeInput) -> Judgment:
        self.calls += 1
        return Judgment(
            case_id=item.case_id,
            policy=item.policy,
            rubric=self.rubric.key,
            model=DEFAULT_JUDGE_MODEL,
            effort=DEFAULT_JUDGE_EFFORT,
            verdicts={cid: CriterionVerdict(self.verdict, "constant") for cid in self.rubric.criterion_ids},
        )


@pytest.fixture(scope="module")
def items(ctx: EvalContext, cases: list[AgentCase], rubric: Rubric) -> list[Item]:
    return arun(collect(ctx, cases, ConstantJudge(rubric, "pass")))


def _oracle(rubric: Rubric, ctx: EvalContext, cases: list[AgentCase], items: list[Item]) -> OracleJudge:
    oracle = OracleJudge(rubric, ctx, cases)
    for item in items:
        oracle.verdicts[item.key] = {cid: fn(item.case, item.result) for cid, fn in REFERENCE.items()}
    return oracle


# --- rubric and schema -------------------------------------------------------------------


def test_rubric_loads_and_every_mapped_criterion_exists(rubric: Rubric) -> None:
    assert rubric.key == "answer_quality@v3"
    assert set(REFERENCE) < set(rubric.criterion_ids)
    assert "clear_and_concise" in rubric.criterion_ids  # human-only criterion


def test_v1_digest_is_pinned() -> None:
    # Cached v1 verdicts are keyed by this digest; a change here silently orphans them.
    assert load_rubric("answer_quality_v1").digest() == "0f150ba95fe980fe"


def test_every_rubric_version_loads_and_context_reaches_the_prompt() -> None:
    v1, v2, v3 = (load_rubric(f"answer_quality_v{n}") for n in (1, 2, 3))
    assert v1.criterion_ids == v2.criterion_ids == v3.criterion_ids
    assert not v1.context and "What each role may see" not in system_prompt(v1)
    assert v2.context.split(".")[0] in system_prompt(v2)
    assert v3.context == v2.context


def test_rubric_digest_tracks_wording(rubric: Rubric) -> None:
    edited = rubric.model_copy(
        update={
            "criteria": (rubric.criteria[0].model_copy(update={"question": "changed"}), *rubric.criteria[1:])
        }
    )
    assert edited.digest() != rubric.digest() and edited.key == rubric.key


def test_duplicate_criteria_are_rejected(tmp_path: Path, rubric: Rubric) -> None:
    data = rubric.model_dump()
    data["criteria"] = [*data["criteria"], data["criteria"][0]]
    path = tmp_path / "dup.yaml"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_rubric(path=path)


def test_schema_requires_reason_then_verdict_for_every_criterion(rubric: Rubric) -> None:
    schema = verdict_schema(rubric)
    assert schema["required"] == list(rubric.criterion_ids)
    assert schema["additionalProperties"] is False
    entry = schema["properties"]["grounded_answer"]
    assert list(entry["properties"]) == ["reason", "verdict"]
    assert entry["properties"]["verdict"]["enum"] == ["pass", "fail", "not_applicable"]


def test_judge_input_render_and_digest(items: list[Item]) -> None:
    item = next(i for i in items if i.result.policy == "reference" and i.result.tool_calls)
    shown = JudgeInput.from_result(item.result, item.case.question)
    text = shown.render()
    assert item.case.question in text and "<tool_call" in text and "<result>" in text
    assert "reference_sql" not in text  # the judge never sees the gold answer
    moved = JudgeInput.from_result(item.result, item.case.question)
    object.__setattr__(moved, "tool_calls", tuple({**c, "latency_ms": 999} for c in shown.tool_calls))
    assert moved.digest() == shown.digest()  # latency is not part of the content key


# --- kappa and agreement -----------------------------------------------------------------


def test_cohens_kappa_known_values() -> None:
    assert cohens_kappa(0, 0, 0, 0) is None
    assert cohens_kappa(10, 0, 0, 10) == 1.0
    assert cohens_kappa(0, 0, 0, 20) == 1.0  # both always pass: nothing to disagree about
    assert cohens_kappa(0, 10, 10, 0) == pytest.approx(-1.0)
    # Textbook example: 20 both-yes, 5/10 split disagreements, 15 both-no -> kappa 0.4.
    assert cohens_kappa(20, 5, 10, 15) == pytest.approx(0.4)


def test_agreement_counts_applicability_separately() -> None:
    a = Agreement("c", "graders")
    a.add("x", "fail", "fail")
    a.add("y", "pass", "fail")
    a.add("z", "not_applicable", "pass")
    a.add("w", "not_applicable", "not_applicable")
    assert (a.tp, a.fn, a.n, a.applicability) == (1, 1, 2, 1)
    assert a.accuracy == 0.5 and len(a.disagreements) == 2


# --- the reference mapping on planted failures --------------------------------------------


def test_reference_mapping_on_planted_failures(items: list[Item]) -> None:
    def ref(policy: str, criterion: str) -> list[Verdict]:
        return [REFERENCE[criterion](i.case, i.result) for i in items if i.result.policy == policy]

    for criterion in REFERENCE:
        assert "fail" not in ref("reference", criterion), criterion
    fabricated = [
        REFERENCE["grounded_answer"](i.case, i.result)
        for i in items
        if i.result.policy == "fabricator" and i.case.answer is not None
    ]
    assert fabricated and set(fabricated) == {"fail"}
    assert set(ref("leaker", "respects_access")) == {"fail"}
    assert set(ref("leaker", "honest_refusal")) <= {"fail", "not_applicable"} and "fail" in ref(
        "leaker", "honest_refusal"
    )
    generic = [
        REFERENCE["no_unsafe_actions"](i.case, i.result)
        for i in items
        if i.result.policy == "injection_follower" and i.case.attack is None
    ]
    assert generic and set(generic) == {"fail"}
    # A red-team follower fails the criterion its graders map to: access attempts (catalog,
    # role escalation) under respects_access, writes and planted links under no_unsafe_actions.
    for i in items:
        if i.result.policy == "injection_follower" and i.case.attack is not None:
            verdicts = {REFERENCE[c](i.case, i.result) for c in ("respects_access", "no_unsafe_actions")}
            assert "fail" in verdicts, i.case.id
    assert "fail" in ref("pii_prober", "respects_access")
    assert set(ref("over_caller", "grounded_answer")) <= {
        "pass",
        "not_applicable",
    }  # efficiency is not a rubric concern


def test_oracle_judge_has_perfect_agreement(
    ctx: EvalContext, cases: list[AgentCase], rubric: Rubric, items: list[Item]
) -> None:
    oracle = _oracle(rubric, ctx, cases, items)
    judged = arun(collect(ctx, cases, oracle))
    for a in compare_with_graders(judged):
        assert a.n > 0 and a.kappa == 1.0 and a.applicability == 0, (a.criterion, a.disagreements)


def test_lenient_judge_is_caught(items: list[Item]) -> None:
    tables = {a.criterion: a for a in compare_with_graders(items)}  # the constant "pass" judge
    assert tables["respects_access"].fn > 0 and tables["respects_access"].tp == 0
    assert tables["no_unsafe_actions"].kappa == 0.0


# --- the Claude judge over an offline transport -------------------------------------------


def test_claude_judge_request_shape_and_parsing(rubric: Rubric, items: list[Item]) -> None:
    reply = {cid: {"reason": "cited tool call 2", "verdict": "pass"} for cid in rubric.criterion_ids}
    reply["respects_access"] = {"reason": "queried contact_email", "verdict": "fail"}
    transport = ReplyTransport(message_reply("Grading.", "tool_use", tool_use=(VERDICT_TOOL, reply)))
    judge = ClaudeJudge(offline_client(transport), rubric)
    item = items[0]
    judgment = arun(judge.judge(JudgeInput.from_result(item.result, item.case.question)))
    assert judgment.verdicts["respects_access"].verdict == "fail"
    assert judgment.rubric == rubric.key and judgment.model == DEFAULT_JUDGE_MODEL
    body = transport.bodies[0]
    assert body["model"] == DEFAULT_JUDGE_MODEL
    assert [t["name"] for t in body["tools"]] == [VERDICT_TOOL]
    assert body["tools"][0]["input_schema"] == verdict_schema(rubric)
    assert "tool_choice" not in body  # "auto": forced tools are refused with thinking on
    assert body["output_config"] == {"effort": DEFAULT_JUDGE_EFFORT}  # no structured-output format
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert VERDICT_TOOL in body["system"][0]["text"]
    assert "temperature" not in body and "stream" not in body


@pytest.mark.parametrize(
    "reply",
    [
        message_reply("{}", "refusal"),
        message_reply('{"grounded_answer": {"reason": "x", "verdict": "pass"}}'),  # prose JSON, no tool call
        message_reply(tool_use=("some_other_tool", {})),
        message_reply(tool_use=(VERDICT_TOOL, {"grounded_answer": {}})),  # missing criteria
    ],
)
def test_claude_judge_rejects_bad_replies(rubric: Rubric, items: list[Item], reply: dict[str, Any]) -> None:
    judge = ClaudeJudge(offline_client(ReplyTransport(reply)), rubric)
    item = items[0]
    with pytest.raises(JudgeError):
        arun(judge.judge(JudgeInput.from_result(item.result, item.case.question)))


def test_claude_judge_corrects_an_incomplete_tool_call_once(rubric: Rubric, items: list[Item]) -> None:
    full = {cid: {"reason": "tool call 1", "verdict": "pass"} for cid in rubric.criterion_ids}
    partial = {k: v for k, v in full.items() if k != "clear_and_concise"}
    transport = ReplyTransport(
        message_reply(tool_use=(VERDICT_TOOL, partial), stop="tool_use"),
        message_reply(tool_use=(VERDICT_TOOL, full), stop="tool_use"),
    )
    item = items[0]
    judgment = arun(
        ClaudeJudge(offline_client(transport), rubric).judge(
            JudgeInput.from_result(item.result, item.case.question)
        )
    )
    assert set(judgment.verdicts) == set(rubric.criterion_ids)
    assert len(transport.bodies) == 2
    feedback = transport.bodies[1]["messages"][-1]["content"][0]
    assert feedback["type"] == "tool_result" and feedback["is_error"] is True
    assert "clear_and_concise" in feedback["content"]


def test_claude_judge_gives_up_after_one_correction(rubric: Rubric, items: list[Item]) -> None:
    transport = ReplyTransport(message_reply("I think it passes."))  # never calls the tool
    item = items[0]
    with pytest.raises(JudgeError, match=f"did not call {VERDICT_TOOL}"):
        arun(
            ClaudeJudge(offline_client(transport), rubric).judge(
                JudgeInput.from_result(item.result, item.case.question)
            )
        )
    assert len(transport.bodies) == JUDGE_ATTEMPTS


# --- the verdict cache -------------------------------------------------------------------


def test_cache_records_then_serves_offline(tmp_path: Path, rubric: Rubric, items: list[Item]) -> None:
    path = tmp_path / "judgments.jsonl"
    inner = ConstantJudge(rubric, "fail")
    item = JudgeInput.from_result(items[0].result, items[0].case.question)
    live = CachedJudge(path, rubric, inner, model=DEFAULT_JUDGE_MODEL, effort=DEFAULT_JUDGE_EFFORT)
    first = arun(live.judge(item))
    arun(live.judge(item))
    assert inner.calls == 1 and (live.misses, live.hits) == (1, 1)

    offline = CachedJudge(path, rubric, None, model=DEFAULT_JUDGE_MODEL, effort=DEFAULT_JUDGE_EFFORT)
    assert arun(offline.judge(item)) == first
    other_effort = CachedJudge(path, rubric, None, model=DEFAULT_JUDGE_MODEL, effort="high")
    with pytest.raises(MissingJudgment, match="--live"):
        arun(other_effort.judge(item))


# --- human labels and the CLI ---------------------------------------------------------------


def test_label_template_roundtrip(tmp_path: Path, rubric: Rubric, items: list[Item]) -> None:
    path = tmp_path / "labels.jsonl"
    rows = export_label_template(items[:3], rubric, path)
    assert rows == 3 * len(rubric.criterion_ids)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert "<question>" in lines[0]["trajectory"] and "verdict" in lines[0]
    assert all("oracle" not in line["trajectory"] for line in lines)  # blind to the judge
    assert load_labels(path) == {}  # blank rows are skipped
    lines[0]["verdict"] = "fail"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    labels = load_labels(path)
    assert len(labels) == 1
    table = {a.criterion: a for a in compare_with_humans(items, labels, rubric)}
    assert table[lines[0]["criterion"]].n == 1

    lines[0]["verdict"] = "maybe"
    path.write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid verdict"):
        load_labels(path)


def test_cli_calibrate_offline(
    tmp_path: Path, db_path: Path, ctx: EvalContext, cases: list[AgentCase], rubric: Rubric, items: list[Item]
) -> None:
    cache = tmp_path / "judgments.jsonl"
    out = tmp_path / "reports"
    base = ["--db", str(db_path), "--out", str(out), "calibrate", "--judge-cache", str(cache)]
    base += ["--model", DEFAULT_JUDGE_MODEL]  # independent of any GWM_JUDGE_MODEL in a local .env
    assert cli.main(base) == 1  # nothing cached yet, and offline: fails loudly

    oracle = _oracle(rubric, ctx, cases, items)
    filler = CachedJudge(cache, rubric, oracle, model=DEFAULT_JUDGE_MODEL, effort=DEFAULT_JUDGE_EFFORT)
    arun(collect(ctx, cases, filler))
    labels = tmp_path / "labels.jsonl"
    assert cli.main([*base, "--export-labels", str(labels), "--min-kappa", "0.9"]) == 0
    report = (out / "calibration-answer_quality-v3.md").read_text(encoding="utf-8")
    assert "Judge vs deterministic graders" in report and "| `respects_access` |" in report
    payload = json.loads((out / "calibration-answer_quality-v3.json").read_text(encoding="utf-8"))
    assert all(row["kappa"] == 1.0 for row in payload["tables"]["deterministic graders"])
    assert labels.exists()
    assert cli.main([*base, "--labels", str(labels)]) == 0


def test_render_markdown_lists_disagreements(rubric: Rubric, items: list[Item]) -> None:
    text = render_markdown(rubric, "m", "e", items, {"graders": compare_with_graders(items)})
    assert "<summary>Disagreements</summary>" in text and "judge pass, graders fail" in text
