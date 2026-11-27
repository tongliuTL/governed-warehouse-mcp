# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Blind human labelling.** `gwm-eval label-sheet` writes a sheet of sampled trajectories,
  scripted and replayed live runs in turn, keyed by an opaque content id; `gwm-eval label`
  labels it in the terminal with the rubric's rules on screen, saving after every trajectory.
  `calibrate --labels` now also compares the graders with the labels, and `--cassettes` judges
  the live runs.
- **TPR and TNR with Wilson 95% intervals** in every calibration table, fail as the positive
  class.

### Changed

- The label template no longer carries the policy and case names, which gave the planted
  failure away; labels are keyed by `trajectory_id`.

## [0.2.0] - 2026-11-13

Addresses the known boundaries of 0.1.0: agent separation, injection coverage, eval
operations, retrieval and deployment.

### Added

- **Agent registry.** Each token's scopes are intersected with its agent's permitted scopes
  (`agents.yaml`, `gwm serve --agents`); a token with no, an unknown or a disabled `client_id`
  is rejected. Onboarding another agent is one registry entry.
- **One search index per visibility tier**, so entries a caller cannot see never shape its
  BM25 statistics, RRF ranks or returned scores.
- **Red-team injection cases** (3 to 8): a poisoned column description, a payload in an error
  message, disguised writes (comments, a CTE, full-width Unicode), an exfiltration link and a
  fake role escalation. New grader `no_planted_links`; `no_write_attempt` normalises with NFKC.
  All are judged and in the calibration gate (113 trajectories).
- **Eval telemetry and a 100% regression gate.** Every run is recorded in a DuckDB history;
  `evals/baseline.json` fails the gate when any previously passing case fails or a retrieval
  metric drops more than 0.05. A failing `main` opens one `eval-regression` issue for the code
  owner and closes it when green.
- **Real-query harvesting.** `gwm-eval harvest` turns the `search_models` queries of live runs
  into a label sheet; a labelled real-query set is reported and gated from 20 queries.
- **Optional dense ranker** (`dense` extra, `gwm serve --dense`): a local embedding model as a
  third RRF list, with per-catalog vector caching, a query cache and lexical failover. Measured
  and off by default: no held-out gain beyond noise.
- **Deployment.** A non-root Docker image with `/healthz` and `/readyz`, built and
  smoke-tested in CI with a read-only root filesystem.

### Changed

- **Statement limits are proven end to end:** the timeout interrupts a running query and
  returns a coded `TIMEOUT`, and spilling to disk is off so the memory limit is hard.

### Fixed

- **The judge answers every tool call in a correction turn**; MiniMax rejected a turn with an
  unanswered `tool_use` (error 2013).

## [0.1.0] - 2026-11-13

The first release.

### Added

- **Rubric `answer_quality_v3`**, the default. It states each role's access, has the judge
  decide applicability from the question and role, and counts any tool call that names a model,
  metric or column outside the role as an access attempt, a write included. v1 and v2 stay for
  comparison.
- **Graders for planted claims and restricted attempts.** `no_planted_claims` fails any answer
  that repeats a claim planted in the data. `no_restricted_attempt` fails any tool call that
  names a model or metric the role cannot see, even when the call is a blocked write. Judge
  calibration found both gaps.
- **Judge verdicts through a `submit_verdicts` tool**, so the judge works on
  Anthropic-compatible providers that ignore structured outputs, such as MiniMax. It asks again
  (up to twice) when a criterion is missing.
- **Committed judge verdicts and a CI calibration gate.** `judgments/` holds the
  MiniMax-M3.1-Flash verdicts for every rubric version. CI re-runs calibration from them with no
  API key and fails if any criterion drops below κ 0.6.
- **Langfuse tracing for live runs.** `gwm-eval live` and `gwm-eval calibrate --live` trace
  each agent run (generations with thinking and usage, tool calls, grader scores) and each
  judge call to Langfuse when its keys are set. Warehouse PII is redacted before export.
- **`.env` credentials and provider choice.** `gwm-eval live` and `calibrate --live` read a
  gitignored `.env` (template: `.env.example`, set up for MiniMax's Anthropic-compatible API);
  `GWM_MODEL` and `GWM_JUDGE_MODEL` set the default models.
- **Calibrated LLM judge.** Versioned rubrics, a judge with per-criterion verdicts and an
  offline verdict cache, and `gwm-eval calibrate`,
  which reports agreement, Cohen's kappa and confusion matrices against the validated graders
  and against blind human labels.
- **Governed MCP server.** It serves five tools (`search_models`, `describe_model`,
  `list_metrics`, `query_metric`, `run_sql`) and the `catalog://version` resource, over stdio,
  Streamable HTTP or in-process.
- **Semantic catalog** in YAML, with access tiers, PII and free-text columns, star-schema joins
  and metrics. It is validated at load time and versioned by content hash.
- **Scope-based access.** Contiguous tiers, `pii:read`, and identical errors for unknown and
  invisible models.
- **AST-level SQL guard** (sqlglot). It fails closed and regenerates the SQL that runs.
- **Read-only, sandboxed DuckDB execution**, with a timeout, a memory limit and locked
  configuration.
- **Request controls.** Ed25519 JWT authentication, a per-principal rate limit, and a JSONL audit
  log covering every allowed, denied and failed call.
- **Catalog search.** BM25 identifier and description rankers fused with RRF.
- **Synthetic warehouse.** A deterministic, seeded cross-border freight dataset with planted
  prompt-injection notes.
- **Eval harness:**
  - a streaming tool-calling agent on the Anthropic SDK;
  - scripted, record and replay transports, so CI runs with no API key;
  - 17 agent cases across answer, governance and injection;
  - deterministic graders, with grader validation against five planted-failure policies;
  - a retrieval eval on dev and held-out sets;
  - Markdown and JSON reports, and a CI gate.
- **CI.** Lint, strict mypy, tests on Python 3.11 to 3.13 with a coverage floor, and the offline
  eval gate.

[Unreleased]: https://github.com/tongliuTL/governed-warehouse-mcp/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tongliuTL/governed-warehouse-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tongliuTL/governed-warehouse-mcp/releases/tag/v0.1.0
