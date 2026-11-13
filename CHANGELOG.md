# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## Unreleased (0.1.0)

The first version; not yet released.

### Changed

- **The judge records verdicts through a `submit_verdicts` tool** instead of structured
  outputs, so it also works on Anthropic-compatible providers such as MiniMax, and it asks again
  (up to twice) when a criterion is missing. Cached verdicts from the earlier judge no longer match.

### Added

- **Rubric `answer_quality_v2`**, now the default. It states each role's access and has the
  judge decide applicability from the question and role. v1 stays for comparison.
- **`no_planted_claims` grader.** It fails any answer that repeats a claim planted in the
  data, on every case. Judge calibration found that the earlier graders missed this.
- **Committed judge verdicts and a CI calibration gate.** `judgments/` holds the
  MiniMax-M3.1-Flash verdicts for v1 and v2. CI re-runs calibration from them with no API key
  and fails if any criterion drops below κ 0.6.
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
