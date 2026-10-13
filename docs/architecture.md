# Architecture and design decisions

This document explains how the pieces fit together and why they are built the way they are.
The [README](../README.md) has the overview diagram. The [threat model](threat-model.md) covers
security properties, and [evals.md](evals.md) covers the measurement method.

## Components

| module | responsibility |
|---|---|
| `warehouse_mcp.synthetic` | Builds the DuckDB warehouse from a seed: 9 tables, 5 marts, planted injection notes. |
| `warehouse_mcp.catalog` | Loads and validates the YAML catalog: models, tiers, PII and free-text columns, joins, metrics. Its content hash is the catalog version. |
| `warehouse_mcp.policy` | Scopes, principals, the tier ladder, `visible_model`, and the `PolicyViolation` error type. |
| `warehouse_mcp.auth` | Ed25519 JWT verification. Implements the MCP SDK's `TokenVerifier` for HTTP; dev key and token helpers. |
| `warehouse_mcp.semantic` | Compiles a `MetricRequest` into a sqlglot `Select`. |
| `warehouse_mcp.sql_guard` | Checks any SQL against the catalog and the principal and returns regenerated SQL with a row limit. |
| `warehouse_mcp.warehouse` | Read-only, sandboxed DuckDB connection with timeout, row fetch limit and JSON-safe values. |
| `warehouse_mcp.discovery` | BM25 identifier and description rankers fused with RRF, filtered by what the caller may see. |
| `warehouse_mcp.audit`, `ratelimit` | JSONL audit sink, and a per-principal token bucket. |
| `warehouse_mcp.server` | `GovernedWarehouse`: the governance envelope, tool logic, and MCP wiring. |
| `warehouse_mcp.cli` | `gwm generate · keys · token · serve`. |
| `warehouse_evals.*` | The streaming agent, offline transports, cases, graders, grader validation, retrieval eval, reports, `gwm-eval`. |

## Life of a tool call

1. **Principal.** Over HTTP, the MCP SDK's bearer-auth middleware calls `JwtVerifier.verify_token`
   and `principal_from_request` builds the principal from the verified access token. Over stdio,
   the principal comes from the token verified at start-up. Missing → `UNAUTHENTICATED`.
2. **Envelope.** `_begin` opens an `AuditRecord` and applies the rate limit (`RATE_LIMITED`).
3. **Tool logic** runs in a worker thread, so DuckDB never blocks the event loop.
   - `query_metric` validates the request, compiles it from the catalog, then sends the compiled
     SQL through the guard again.
   - `run_sql` sends the agent's SQL through the guard.
4. **Execution.** The guard returns `GuardedQuery(sql, models, fetch_limit, max_rows)`. The
   warehouse runs it under a timer that interrupts the connection, fetches `max_rows + 1` rows,
   and reports truncation.
5. **Response.** The result carries rows, the SQL that ran, the models read and the catalog
   version. It is cut to the character budget, and it carries a warning when a free-text column
   is present.
6. **Outcome.** A `PolicyViolation` becomes an MCP tool error with the text `"CODE: message"`.
   Any other exception becomes `INTERNAL_ERROR`, with its details kept in the audit record only.
   Every outcome, allow, deny or error, writes one audit record.

## Design decisions

### A semantic layer and raw SQL, both governed

Metrics are the preferred path: the business logic is defined once, and the agent cannot get a
definition subtly wrong. Real questions outgrow any metric list, though, so `run_sql` exists too,
behind the same policy. Compiled metric SQL also goes through the SQL guard. A bug in the compiler
alone therefore cannot bypass the tier or PII checks, and there is one enforcement point to test.

### The guard works on the AST and fails closed

Regex and keyword filters lose to quoting, comments and dialect features. The guard parses with
sqlglot in the DuckDB dialect. It accepts only a SELECT or a set operation at the root, and it
walks the whole tree. Anything it does not recognise as a safe read of a visible model is
rejected. It then regenerates the SQL from the checked tree, so what runs is exactly what was
checked.

PII is checked by column name across every reference to a model that holds PII, not only in the
select list. A PII column in a `WHERE` clause leaks the column one predicate at a time just as
well as selecting it.

### The engine is a second, independent layer

DuckDB runs with these settings:
- `read_only=True`;
- `enable_external_access=False`;
- extension autoload and autoinstall off;
- a memory limit;
- `lock_configuration=True`, applied at connect time.

The tests for these settings run SQL that skipped the guard entirely. A guard bug alone therefore
still cannot write, read files, or switch the settings back on.

### Invisible means nonexistent

A principal that cannot see a model gets the same `UNKNOWN_MODEL` error as for a name that does
not exist. `search_models`, `describe_model` and `list_metrics` leave out what the caller cannot
see, and a metric is as sensitive as the most sensitive model it reads. The real reason goes to
the audit log as `internal_detail`, for operators only.

### Stable error codes

Agents need to tell "try a different metric" apart from "you are not allowed", and evals need to
grade that without parsing prose. Every denial starts with an uppercase code. `server.error_code`
is the single parser for that format, and both the eval agent and the tests use it.

### Lexical retrieval with fusion, no embeddings

Catalog search uses two BM25 rankers fused with RRF (k=60):
- one over identifiers, aliases, synonyms and column names, with snake_case and camelCase split
  and plurals folded;
- one over descriptions.

This keeps the server deterministic and free of model dependencies, so the retrieval eval is
reproducible. The ablation in the eval report shows why fusion is used: each single ranker wins
one question set and loses the other. A dense ranker would slot into the fusion without changing
the interface.

### A hand-written agent loop

The SDK's tool runner would work. The eval needs every step on the record, though, and the
streaming details matter:

- Tools are declared with `eager_input_streaming`, so long SQL streams as it is generated.
- Each tool input is re-parsed strictly from the raw `input_json_delta` text and validated
  against the tool's JSON Schema before it runs. The SDK's tolerant parser can turn malformed
  JSON into a silently truncated object.
- A turn that stopped on `max_tokens` or `refusal` never executes its tools.
- JSON the SDK cannot parse at all is retried a bounded number of times.
- History is append-only: assistant content is sent back exactly as received.
- Requests use adaptive thinking, an effort level, and server-side model fallback.

### Offline reproducibility at the transport layer

The SDK accepts any `httpx` transport, so the offline modes swap only that layer. Everything
above it, including SSE parsing, the agent loop, MCP and the warehouse, runs for real in CI.
A cassette stores response streams plus a fingerprint of each request: model, system prompt,
tool schemas and turn count. Replay fails when the fingerprints drift, which turns a silent
behaviour change into a red CI run.

### Deterministic graders that are themselves tested

The graders are plain functions, with no LLM judge. That makes them reproducible, and it makes it
possible to test them: five planted-failure policies must each trip exactly the intended grader,
and nothing else may fire. A grader that cannot pass this check is not trusted to grade a model.

## Catalog reference

```yaml
schema_version: 1
name: freight-warehouse
models:
  - name: example_model            # table or view name in the warehouse
    kind: fact                     # dimension | fact | mart
    tier: internal                 # public | internal | restricted
    grain: one row per ...
    description: ...               # used by search_models and shown to agents
    aliases: [...]                 # extra search terms
    time_column: event_date        # optional; enables time_grain and date filters on metrics
    columns:
      - {name: id, type: VARCHAR, description: ...}
      - {name: contact_email, type: VARCHAR, description: ..., pii: true}
      - {name: note_text, type: VARCHAR, description: ..., free_text: true}
    joins:
      - {model: dim_lanes, key: lane_id}   # one hop, star schema
metrics:
  - name: example_metric
    model: example_model           # base model
    description: ...
    expression: AVG(example_model.value)   # model-qualified columns only
    where: "example_model.status = 'done'" # optional
    format: ratio                  # count | ratio | days | hours | usd | number
    synonyms: [...]
```

Load-time validation:
- model and metric names are unique;
- every join points to an existing key;
- metric columns are model-qualified and reachable from the base model;
- no metric reads a PII column.

`tests/test_catalog.py` additionally checks the catalog against the warehouse schema in both
directions.

## Known limits

- Joins are one hop from the base model (star schema). Multi-hop joins would need a path search
  in the compiler, and the visibility rule would need to cover every model on the path.
- Access is per model and per column; there is no row-level security yet.
- Aggregation can still be used for inference (small groups, differencing). Minimum group sizes
  would be the next control to add. See the [threat model](threat-model.md).
