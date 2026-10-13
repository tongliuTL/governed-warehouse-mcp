# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## Unreleased (0.1.0)

The first version; not yet released.

### Added

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
