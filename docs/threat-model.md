# Threat model

The server's job is to let an LLM agent read an analytical warehouse on behalf of a principal,
and nothing more. This document lists what can go wrong, what stops it, and the test that
shows it is stopped.

## Assets and actors

- **Assets:** warehouse data at three sensitivity tiers (public, internal, restricted), PII
  columns inside otherwise readable models, the integrity of the warehouse, the host the server
  runs on, and the audit trail.
- **Actors:** an authenticated agent acting for a principal (trusted to be the principal, not
  trusted to behave), anyone who can reach the HTTP endpoint without a valid token, and
  untrusted text inside the data itself (free-text notes an attacker could have written).

The agent is treated as untrusted input. A model can be confused, over-eager or prompt-injected,
so every control below holds no matter what the agent sends.

## Threats and mitigations

| # | Threat | Mitigation | Proven by |
|---|---|---|---|
| 1 | Unauthenticated access over HTTP | Bearer JWT verified on every request by the MCP SDK auth middleware; 401 otherwise | `test_cli_and_transports.py::test_http_requires_a_valid_bearer_token` |
| 2 | Forged or replayed tokens | Ed25519 only (algorithm pinned, so `alg: none` and HS256-keyed-with-the-public-key fail); `exp`, `iat`, `sub`, `iss`, `aud` required; signature, audience and issuer checked | `test_auth.py::test_rejects_bad_tokens` (11 forgeries), `test_verifier_requires_ed25519` |
| 3 | stdio server started without an identity | `gwm serve` refuses to start unless `GWM_TOKEN` verifies | `test_cli_and_transports.py::test_stdio_refuses_to_start_without_a_valid_token` |
| 4 | Privilege escalation through scopes | Tiers are contiguous: a gap in the scope list caps access below it; no scopes means no access | `test_sql_guard.py::test_skipped_tier_scope_grants_nothing_above_the_gap`, `test_no_scopes_means_no_access` |
| 5 | Writes, DDL, attach, copy, install, settings changes | Guard: single statement, root must be a SELECT or set operation, forbidden node types anywhere in the tree. Engine: read-only connection, configuration locked | `test_sql_guard.py::test_denied` (`READ_ONLY`, `SINGLE_STATEMENT`), `test_warehouse.py::test_engine_blocks_unguarded_sql` |
| 6 | Reading files, the network or the host environment | Guard rejects table functions, file paths as tables, and file, network, settings and introspection functions. Engine: external access disabled, extension autoload and autoinstall off, locked | `test_sql_guard.py::test_denied` (`TABLE_FUNCTION`, `FORBIDDEN_FUNCTION`), `test_warehouse.py::test_engine_blocks_unguarded_sql` |
| 7 | Reading a model above the caller's tier | Every table reference must be a visible catalog model or a CTE in the same query, including inside subqueries; system schemas and other catalogs rejected | `test_sql_guard.py::test_denied` (`UNKNOWN_MODEL`), `test_tiers`, `test_semantic.py::test_dimension_on_invisible_model_is_refused` |
| 8 | Learning that a restricted model exists | Unknown and invisible models get identical errors; `search_models`, `describe_model` and `list_metrics` hide what the caller cannot see | `test_sql_guard.py::test_restricted_model_is_invisible_not_forbidden`, `test_server.py::test_search_hides_restricted_models`, `test_list_metrics_by_role` |
| 9 | Reading PII without `pii:read` | Guard rejects PII column names anywhere in the tree (select, filters, joins, order, function arguments, CTEs, subqueries, struct packing), `*` and `* EXCLUDE` over models with PII, whole-row references, `COLUMNS(...)` and positional references. The semantic layer never offers PII dimensions to such callers | `test_sql_guard.py::test_denied` (15 `PII_DENIED` and 2 `POSITIONAL_REFERENCE` cases), `test_semantic.py::test_available_dimensions_hide_pii_and_invisible_models` |
| 10 | SQL injection through metric parameters | `query_metric` builds a sqlglot AST from the catalog; user values only become quoted literals, identifiers are always quoted, and the compiled SQL goes through the guard again | `test_semantic.py` |
| 11 | Smuggling through comments or parser quirks | The SQL that runs is regenerated from the checked AST, without comments | `test_sql_guard.py::test_executed_sql_is_regenerated_without_comments` |
| 12 | Resource exhaustion | SQL size limit, row cap with `LIMIT max_rows + 1`, result character budget, query timeout that interrupts the engine, memory limit, per-principal token-bucket rate limit | `test_sql_guard.py::test_size_limit`, `test_row_limit`, `test_warehouse.py::test_timeout_interrupts_long_queries`, `test_server.py::test_large_results_are_truncated_to_budget`, `test_rate_limit` |
| 13 | Prompt injection through data | Free-text columns are marked in the catalog and results that contain them carry a warning that their content is data, not instructions; the agent's system prompt says the same; the server's controls hold even if the agent obeys | `test_server.py::test_free_text_columns_carry_a_warning`, eval suite `injection` (3 cases) and the `injection_follower` policy in grader validation |
| 14 | Leaking internals through errors | Policy denials carry a stable code and a safe message; detail for operators goes to the audit log only; unexpected exceptions become `INTERNAL_ERROR` | `test_server.py::test_denials_are_coded_and_audited`, `test_internal_errors_are_masked_and_audited` |
| 15 | Unaccountable access | Every tool call, allowed or denied, is written to the audit log with principal, tool, arguments, decision, reason code, models, row count and timing | `test_audit_ratelimit.py`, `test_server.py::test_denials_are_coded_and_audited` |
| 16 | DNS rebinding against a local HTTP server | Streamable HTTP runs with the SDK's DNS-rebinding protection and an allowed-hosts list | configuration in `cli.py`; exercised by the HTTP end-to-end tests |
| 17 | Secrets in eval recordings | Cassettes store response streams and a request fingerprint, never headers or request bodies | `test_transport.py` |

## Layering

The guard and the engine are independent: the engine settings in threat 5 and 6 are tested with
SQL that skipped the guard entirely, so a guard bug alone does not reach the file system or
write to the warehouse. The semantic compiler's output also goes through the guard, so a
compiler bug alone does not bypass PII or tier checks.

## Out of scope

- **Row-level security.** Access is per model and per column. A partner seeing only their own
  shipments would need row filters injected by the compiler and the guard.
- **Inference attacks.** A caller allowed to aggregate internal data can still learn things by
  aggregating cleverly (small-group counts, differencing). Minimum group sizes or differential
  privacy would address this.
- **Token issuance and revocation.** `gwm keys` and `gwm token` are for development. In
  production the verifier should point at an identity provider's public key, with short-lived
  tokens.
- **Denial of service at the network layer**, which belongs in front of the server.
- **Model-level jailbreaks** beyond what the injection cases cover. The controls above do not
  depend on the model behaving.
