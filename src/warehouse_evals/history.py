"""Eval telemetry: every eval run appends its results to one DuckDB file.

A badge says whether the last run passed. This store says when a case started failing, on
which commit, and whether it has flipped before, so a red gate has a history an owner can act
on. ``gwm-eval all``, ``live`` and ``replay`` each append one run; ``gwm-eval history`` reads
it. CI restores the file from the previous run on ``main`` and uploads it again, so the table
grows with every run there.

Tables (one row per run, per case result, per retrieval metric):

    runs               run_id, started_at, command, git_sha, git_ref, ci_url, catalog_version,
                       seed, gate_failures
    case_results       run_id, label, suite, case_id, role, mode, policy, trial, passed,
                       failed_graders
    retrieval_metrics  run_id, set_name, ranker, metric, value
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from warehouse_evals.report import Report
from warehouse_evals.retrieval import KS, RankerScore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,  -- UTC
    command VARCHAR NOT NULL,
    git_sha VARCHAR,
    git_ref VARCHAR,
    ci_url VARCHAR,
    catalog_version VARCHAR,
    seed BIGINT,
    gate_failures VARCHAR[] NOT NULL
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    suite VARCHAR NOT NULL,
    case_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,
    policy VARCHAR NOT NULL,
    trial INTEGER NOT NULL,
    passed BOOLEAN NOT NULL,
    failed_graders VARCHAR[] NOT NULL
);
CREATE TABLE IF NOT EXISTS retrieval_metrics (
    run_id VARCHAR NOT NULL,
    set_name VARCHAR NOT NULL,
    ranker VARCHAR NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE NOT NULL
);
"""


def metric_values(score: RankerScore) -> dict[str, float]:
    """A ranker's numbers by name: ``recall@1``, ``recall@3``, ``recall@5``, ``mrr``."""
    return {**{f"recall@{k}": score.recall[k] for k in KS}, "mrr": score.mrr}


def _git(*args: str) -> str | None:
    try:
        # Fixed git subcommands, no user input; git from PATH is the developer's or CI's own.
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def provenance() -> dict[str, str | None]:
    """Which code produced a run: the commit, the ref, and the CI run when there is one."""
    server, repo, run = (
        os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    return {
        "git_sha": os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD"),
        "git_ref": os.environ.get("GITHUB_REF_NAME") or _git("rev-parse", "--abbrev-ref", "HEAD"),
        "ci_url": f"{server}/{repo}/actions/runs/{run}" if server and repo and run else None,
    }


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    return con


def record(
    path: Path,
    report: Report,
    *,
    command: str,
    gate_failures: Sequence[str] = (),
    started_at: datetime | None = None,
    origin: dict[str, str | None] | None = None,
) -> str:
    """Append one run to the store and return its id."""
    run_id = uuid.uuid4().hex[:12]
    origin = provenance() if origin is None else origin
    con = _connect(path)
    try:
        con.execute("BEGIN")
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                (started_at or datetime.now(UTC)).astimezone(UTC).replace(tzinfo=None),
                command,
                origin.get("git_sha"),
                origin.get("git_ref"),
                origin.get("ci_url"),
                report.meta.get("catalog_version"),
                report.meta.get("seed"),
                list(gate_failures),
            ],
        )
        rows: list[list[Any]] = [
            [
                run_id,
                label,
                r.suite,
                r.case_id,
                r.role,
                r.mode,
                r.policy,
                r.trial,
                r.passed,
                sorted(r.failed_graders),
            ]
            for label, results in report.agent.items()
            for r in results
        ]
        if rows:
            con.executemany("INSERT INTO case_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        metrics = [
            [run_id, set_name, s.ranker, metric, value]
            for set_name, scores in report.retrieval.items()
            for s in scores
            for metric, value in metric_values(s).items()
        ]
        if metrics:
            con.executemany("INSERT INTO retrieval_metrics VALUES (?, ?, ?, ?, ?)", metrics)
        con.execute("COMMIT")
    finally:
        con.close()
    return run_id


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    started_at: datetime
    command: str
    git_sha: str | None
    cases: int
    passed: int
    rrf_recall_at_3: dict[str, float]
    gate_failures: list[str]


@dataclass(frozen=True)
class Flip:
    label: str
    case_id: str
    trial: int
    now_passed: bool
    run_id: str
    previous_run_id: str
    failed_graders: list[str]


def recent_runs(path: Path, limit: int = 10) -> list[RunSummary]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        runs = con.execute(
            """
            SELECT r.run_id, r.started_at, r.command, r.git_sha, r.gate_failures,
                   count(c.case_id), coalesce(sum(c.passed::INT), 0)
            FROM runs r LEFT JOIN case_results c USING (run_id)
            GROUP BY ALL ORDER BY r.started_at DESC LIMIT ?
            """,
            [limit],
        ).fetchall()
        recall = con.execute(
            "SELECT run_id, set_name, value FROM retrieval_metrics WHERE ranker = 'rrf' AND metric = 'recall@3'"
        ).fetchall()
    finally:
        con.close()
    by_run: dict[str, dict[str, float]] = {}
    for run_id, set_name, value in recall:
        by_run.setdefault(run_id, {})[set_name] = value
    return [
        RunSummary(rid, started, command, sha, cases, passed, by_run.get(rid, {}), list(failures))
        for rid, started, command, sha, failures, cases, passed in runs
    ]


def flips(path: Path) -> list[Flip]:
    """Cases whose pass state changed between the two latest runs that produced them.

    Compared within a label (``scripted reference``, ``replay``...) and trial, so a live run
    and a scripted run of the same case are never compared with each other.
    """
    con = duckdb.connect(str(path), read_only=True)
    try:
        rows = con.execute(
            """
            WITH ordered AS (
                SELECT c.label, c.case_id, c.trial, c.passed, c.failed_graders, c.run_id,
                       row_number() OVER w AS recency,
                       lead(c.passed) OVER w AS previous_passed,
                       lead(c.run_id) OVER w AS previous_run_id
                FROM case_results c JOIN runs r USING (run_id)
                WINDOW w AS (PARTITION BY c.label, c.case_id, c.trial ORDER BY r.started_at DESC)
            )
            SELECT label, case_id, trial, passed, run_id, previous_run_id, failed_graders
            FROM ordered
            WHERE recency = 1 AND previous_passed IS NOT NULL AND passed <> previous_passed
            ORDER BY passed, label, case_id, trial
            """
        ).fetchall()
    finally:
        con.close()
    return [
        Flip(lab, cid, trial, passed, rid, prev, list(failed))
        for lab, cid, trial, passed, rid, prev, failed in rows
    ]


def render(path: Path, limit: int = 10) -> str:
    runs = recent_runs(path, limit)
    if not runs:
        return f"no runs recorded in {path}\n"
    sets = sorted({s for r in runs for s in r.rrf_recall_at_3})
    head = "".join(f" | rrf R@3 {s}" for s in sets)
    lines = [
        f"Last {len(runs)} runs in {path}",
        "",
        f"| run | started (UTC) | command | commit | cases passed{head} | gate |",
        "|---|---|---|---|---:|" + "---:|" * len(sets) + "---|",
    ]
    for r in runs:
        recall = "".join(f" | {r.rrf_recall_at_3[s]:.3f}" if s in r.rrf_recall_at_3 else " | -" for s in sets)
        gate = "pass" if not r.gate_failures else f"**fail** ({len(r.gate_failures)})"
        lines.append(
            f"| {r.run_id} | {r.started_at:%Y-%m-%d %H:%M} | {r.command} | "
            f"{(r.git_sha or '-')[:8]} | {r.passed}/{r.cases}{recall} | {gate} |"
        )
    changed = flips(path)
    lines += ["", f"Cases that changed state since their previous run: {len(changed)}"]
    for f in changed:
        state = "now passes" if f.now_passed else "now FAILS (" + ", ".join(f.failed_graders) + ")"
        trial = f" t{f.trial}" if f.trial else ""
        lines.append(f"- {f.label} / {f.case_id}{trial}: {state} in run {f.run_id}")
    return "\n".join(lines) + "\n"
