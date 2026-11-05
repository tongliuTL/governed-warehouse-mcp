"""``gwm-eval``: run the eval suites and write reports.

gwm-eval all                     # offline: retrieval + grader validation + scripted reference (+ replay)
gwm-eval live --trials 3 --record cassettes/   # real Claude API, records cassettes for CI replay
gwm-eval replay cassettes/       # re-run recorded live runs offline
gwm-eval calibrate --live        # judge every scripted trajectory, compare with graders (caches verdicts)
gwm-eval calibrate               # same, offline from the verdict cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from dotenv import dotenv_values, load_dotenv

from warehouse_evals import tracing
from warehouse_evals.agent import DEFAULT_MODEL
from warehouse_evals.cases import load_agent_cases, load_retrieval_cases
from warehouse_evals.judge import DEFAULT_JUDGE_EFFORT, DEFAULT_JUDGE_MODEL
from warehouse_evals.report import Report, render_markdown
from warehouse_evals.retrieval import evaluate
from warehouse_evals.runner import EvalContext, result_dict, run_live, run_replay, run_scripted
from warehouse_evals.validation import validate_graders
from warehouse_mcp.synthetic import DEFAULT_SEED, build_warehouse

HELDOUT = Path(__file__).parent / "cases" / "retrieval_heldout.yaml"
GATES = {"reference_pass_rate": 1.0, "grader_validation": 1.0, "retrieval_recall_at_3": 0.9}


def _context(args: argparse.Namespace) -> EvalContext:
    if not args.db.exists():
        build_warehouse(args.db, seed=DEFAULT_SEED)
    return EvalContext(args.db)


def _write(report: Report, out: Path, sizes: dict[str, int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": report.meta,
        "agent": {k: [result_dict(r) for r in v] for k, v in report.agent.items()},
        "validation": [
            {
                "policy": r.policy,
                "case": r.case_id,
                "expected": sorted(r.expected),
                "actual": sorted(r.actual),
                "ok": r.ok,
            }
            for r in report.validation
        ],
        "retrieval": {k: [asdict(s) for s in v] for k, v in report.retrieval.items()},
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report, sizes), encoding="utf-8")
    print(f"wrote {out / 'report.md'} and {out / 'report.json'}", file=sys.stderr)


def _meta(ctx: EvalContext, note: str) -> dict[str, Any]:
    return {
        "catalog_version": ctx.catalog.version,
        "seed": DEFAULT_SEED,
        "note": note,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }


async def _all(args: argparse.Namespace) -> int:
    ctx = _context(args)
    cases = load_agent_cases()
    note = (
        "Offline run. Agent cases use scripted model turns streamed through the real Anthropic SDK and "
        "MCP server: they verify the harness, the server and the graders, not a model's skill. "
        "Replayed sections, when present, are recorded live runs."
    )
    report = Report(meta=_meta(ctx, note))
    report.agent["scripted reference"] = await run_scripted(ctx, cases, "reference")
    if args.cassettes and args.cassettes.exists() and any(args.cassettes.glob("*.jsonl")):
        report.agent[f"replayed live runs ({args.cassettes})"] = await run_replay(ctx, cases, args.cassettes)
    report.validation = await validate_graders(ctx, cases)
    dev, heldout = load_retrieval_cases(), load_retrieval_cases(HELDOUT)
    report.retrieval = {"dev set": evaluate(ctx.catalog, dev), "held-out set": evaluate(ctx.catalog, heldout)}
    sizes = {"dev set": len(dev), "held-out set": len(heldout)}
    _write(report, args.out, sizes)
    ctx.close()

    failures = []
    ref = report.agent["scripted reference"]
    if sum(r.passed for r in ref) / len(ref) < GATES["reference_pass_rate"]:
        failures.append(
            "scripted reference cases failed: " + ", ".join(r.case_id for r in ref if not r.passed)
        )
    if not all(r.ok for r in report.validation):
        failures.append(
            "grader validation mismatches: "
            + ", ".join(f"{r.policy}/{r.case_id}" for r in report.validation if not r.ok)
        )
    for name, results in report.agent.items():
        # A recorded model's mistakes are reported, not gated; a cassette that no longer replays
        # means the server or agent changed behaviour since recording, and that is a regression.
        drifted = [r.case_id for r in results if "replay_fidelity" in r.failed_graders]
        if drifted:
            failures.append(f"{name}: cassettes no longer replay: " + ", ".join(drifted))
    for name, scores in report.retrieval.items():
        rrf = next(s for s in scores if s.ranker == "rrf")
        if rrf.recall[3] < GATES["retrieval_recall_at_3"]:
            failures.append(f"{name}: rrf Recall@3 {rrf.recall[3]:.3f} < {GATES['retrieval_recall_at_3']}")
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    if not failures:
        print("all gates passed", file=sys.stderr)
    return 1 if failures and args.gate else 0


ENV_FILE = Path(".env")


def _load_env() -> None:
    """Credentials from ./.env for the commands that call real APIs. Shell variables win."""
    if load_dotenv(ENV_FILE, override=False):
        print("loaded credentials from .env", file=sys.stderr)


def _setting(name: str, default: str) -> str:
    """A non-secret setting from the shell, else ./.env, else the default.

    Read without loading .env into the environment, so an offline ``calibrate`` finds the same
    model (and so the same verdict cache) as the live run that filled it, without its keys.
    """
    return os.environ.get(name) or dotenv_values(ENV_FILE).get(name) or default


def _start_tracing(ctx: EvalContext) -> None:
    tracing.set_pii_values(ctx.pii_values)
    if tracing.enable():
        print("tracing to Langfuse (PII values redacted)", file=sys.stderr)


async def _live(args: argparse.Namespace) -> int:
    _load_env()
    args.model = args.model or _setting("GWM_MODEL", DEFAULT_MODEL)
    ctx = _context(args)
    _start_tracing(ctx)
    cases = [c for c in load_agent_cases() if not args.case or c.id in args.case]
    stream = (lambda t: print(t, end="", flush=True)) if args.stream else None
    try:
        results = await run_live(
            ctx,
            cases,
            model=args.model,
            effort=args.effort,
            trials=args.trials,
            record_dir=args.record,
            on_text=stream,
        )
    finally:
        tracing.flush()
    report = Report(
        meta={**_meta(ctx, f"Live run against {args.model} (effort {args.effort})."), "model": args.model}
    )
    report.agent[f"live {args.model}"] = results
    _write(report, args.out, {})
    ctx.close()
    return 0


async def _replay(args: argparse.Namespace) -> int:
    ctx = _context(args)
    results = await run_replay(ctx, load_agent_cases(), args.cassettes)
    report = Report(meta=_meta(ctx, f"Replay of recorded live runs in {args.cassettes}."))
    report.agent["replay"] = results
    _write(report, args.out, {})
    ctx.close()
    return 0 if all(r.passed for r in results) or not args.gate else 1


async def _calibrate(args: argparse.Namespace) -> int:
    import anthropic

    from warehouse_evals.calibration import (
        collect,
        compare_with_graders,
        compare_with_humans,
        export_label_template,
        load_labels,
        render_markdown,
    )
    from warehouse_evals.judge import CachedJudge, ClaudeJudge, Judge, MissingJudgment, load_rubric

    rubric = load_rubric(args.rubric)
    args.model = args.model or _setting("GWM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    cache = args.judge_cache or Path("judgments") / f"{args.rubric}.{args.model}.{args.effort}.jsonl"
    ctx = _context(args)
    inner: Judge | None = None
    if args.live:
        _load_env()
        _start_tracing(ctx)  # judge calls only; the scripted runs being judged are not traced
        inner = ClaudeJudge(
            anthropic.AsyncAnthropic(), rubric, model=args.model, effort=args.effort, tracer=tracing.client()
        )
    judge = CachedJudge(cache, rubric, inner, model=args.model, effort=args.effort)
    try:
        items = await collect(ctx, load_agent_cases(), judge, concurrency=args.concurrency)
    except MissingJudgment as exc:
        print(f"{exc} (cache: {cache})", file=sys.stderr)
        return 1
    finally:
        ctx.close()
        tracing.flush()
    if args.live:
        print(f"judge cache {cache}: {judge.hits} hits, {judge.misses} new verdicts", file=sys.stderr)

    tables = {"deterministic graders": compare_with_graders(items)}
    if args.labels:
        tables["human labels"] = compare_with_humans(items, load_labels(args.labels), rubric)
    if args.export_labels:
        rows = export_label_template(items, rubric, args.export_labels)
        print(f"wrote {rows} label rows to {args.export_labels}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"calibration-{rubric.id}-v{rubric.version}"
    stem.with_suffix(".md").write_text(
        render_markdown(rubric, args.model, args.effort, items, tables), encoding="utf-8"
    )
    payload = {
        "rubric": rubric.key,
        "model": args.model,
        "effort": args.effort,
        "tables": {
            name: [
                {
                    "criterion": a.criterion,
                    "n": a.n,
                    "accuracy": a.accuracy,
                    "kappa": a.kappa,
                    "tp": a.tp,
                    "fp": a.fp,
                    "fn": a.fn,
                    "tn": a.tn,
                    "applicability_mismatches": a.applicability,
                    "disagreements": a.disagreements,
                }
                for a in agreements
            ]
            for name, agreements in tables.items()
        },
        "judgments": [i.judgment.to_dict() for i in items],
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {stem}.md and {stem}.json", file=sys.stderr)

    if args.min_kappa is not None:
        low = [
            f"{a.criterion} ({a.kappa:.2f})"
            for a in tables["deterministic graders"]
            if a.kappa is not None and a.kappa < args.min_kappa
        ]
        if low:
            print(f"GATE FAILED: judge kappa below {args.min_kappa}: " + ", ".join(low), file=sys.stderr)
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwm-eval", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=Path("data/warehouse.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    sub = parser.add_subparsers(dest="command", required=True)
    p_all = sub.add_parser("all", help="offline suites with gates")
    p_all.add_argument("--cassettes", type=Path, default=Path("cassettes"))
    p_all.add_argument("--gate", action="store_true", help="exit 1 when a gate fails")
    p_all.set_defaults(func=_all)
    p_live = sub.add_parser("live", help="run against the Claude API")
    p_live.add_argument("--model", help=f"default: $GWM_MODEL, else {DEFAULT_MODEL}")
    p_live.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    p_live.add_argument("--trials", type=int, default=1)
    p_live.add_argument("--case", action="append", help="limit to a case id; repeatable")
    p_live.add_argument("--record", type=Path, help="save cassettes here")
    p_live.add_argument("--stream", action="store_true", help="print model text as it streams")
    p_live.set_defaults(func=_live)
    p_replay = sub.add_parser("replay", help="replay recorded cassettes")
    p_replay.add_argument("cassettes", type=Path)
    p_replay.add_argument("--gate", action="store_true")
    p_replay.set_defaults(func=_replay)
    p_cal = sub.add_parser("calibrate", help="calibrate the LLM judge against graders and human labels")
    p_cal.add_argument("--rubric", default="answer_quality_v1")
    p_cal.add_argument("--model", help=f"default: $GWM_JUDGE_MODEL, else {DEFAULT_JUDGE_MODEL}")
    p_cal.add_argument(
        "--effort", default=DEFAULT_JUDGE_EFFORT, choices=("low", "medium", "high", "xhigh", "max")
    )
    p_cal.add_argument(
        "--judge-cache", type=Path, help="verdict cache (default judgments/<rubric>.<model>.<effort>.jsonl)"
    )
    p_cal.add_argument(
        "--live", action="store_true", help="call the Claude API for verdicts missing from the cache"
    )
    p_cal.add_argument("--concurrency", type=int, default=4)
    p_cal.add_argument("--labels", type=Path, help="human labels (JSONL) to compare against")
    p_cal.add_argument("--export-labels", type=Path, help="write a blank, blind labelling template here")
    p_cal.add_argument("--min-kappa", type=float, help="exit 1 if any grader-mapped criterion is below this")
    p_cal.set_defaults(func=_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(anyio.run(args.func, args))


if __name__ == "__main__":
    raise SystemExit(main())
