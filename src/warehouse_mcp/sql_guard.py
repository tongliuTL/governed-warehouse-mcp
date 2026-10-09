"""Validate agent-written SQL before it reaches the warehouse.

The guard works on the parsed AST (sqlglot), never on the raw string, and it fails closed:
anything it does not positively recognise as a safe read of a visible model is rejected.

Checks, in order:

1. size and parse: exactly one statement that parses in the DuckDB dialect;
2. read-only: the statement is a SELECT or a set operation of SELECTs, and no write, DDL,
   session or file command appears anywhere in the tree;
3. sources: every table reference is a plain name of a catalog model (or a CTE defined in the
   same query) that the principal may see; table functions, other catalogs and system schemas
   are rejected; unknown and invisible models get the same answer, so the error does not reveal
   that a restricted model exists;
4. functions: file, network, settings and introspection functions are rejected;
5. positional references: ``#7`` and ``$1`` are rejected, since they name no column;
6. PII: without ``pii:read``, any column whose name is a PII column of a referenced model is
   rejected wherever it appears (select list, WHERE, JOIN, ORDER BY, function arguments), as
   are ``*`` over models with PII, ``COLUMNS(...)`` and whole-row references such as
   ``SELECT dim_customers FROM dim_customers``;
7. row limit: the outermost query gets ``LIMIT max_rows + 1`` unless it already asks for fewer
   rows, so the caller can tell a truncated result from a complete one.

The SQL that runs is regenerated from the checked AST, so comments and anything the parser
ignored never reach the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from warehouse_mcp.catalog import DIALECT, Catalog
from warehouse_mcp.policy import PolicyViolation, Principal

MAX_SQL_CHARS = 20_000

_FORBIDDEN_NODES: tuple[type[exp.Expr], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "AlterColumn",
        "TruncateTable",
        "Command",
        "Copy",
        "Set",
        "SetItem",
        "Pragma",
        "Use",
        "Transaction",
        "Commit",
        "Rollback",
        "Attach",
        "Detach",
        "Install",
        "LoadData",
        "Into",
        "Describe",
        "Show",
        "Grant",
        "Revoke",
        "Export",
        "Analyze",
        "Summarize",
    )
    if hasattr(exp, name)
)

# Lower-cased function names (or prefixes ending in "_") that touch files, the network,
# settings, extensions or system catalogs. External access is also disabled in the engine;
# this list is the first of two layers.
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_json_objects",
        "read_ndjson",
        "read_ndjson_auto",
        "read_text",
        "read_blob",
        "read_xlsx",
        "glob",
        "sniff_csv",
        "parquet_scan",
        "parquet_metadata",
        "parquet_schema",
        "parquet_file_metadata",
        "parquet_kv_metadata",
        "iceberg_scan",
        "delta_scan",
        "sqlite_scan",
        "postgres_scan",
        "mysql_scan",
        "getenv",
        "current_setting",
        "query",
        "query_table",
        "load_extension",
        "install_extension",
        "system",
        "shell",
        "columns",
        "unpivot",
        "pivot",
        "pragma_table_info",
        "pragma_storage_info",
        "pragma_show",
    }
)
_FORBIDDEN_PREFIXES = (
    "read_",
    "duckdb_",
    "pragma_",
    "parquet_",
    "iceberg_",
    "delta_",
    "sqlite_",
    "postgres_",
    "mysql_",
    "http",
    "s3_",
    "pg_",
)


@dataclass(frozen=True)
class GuardedQuery:
    sql: str
    models: frozenset[str]
    fetch_limit: int
    max_rows: int


def _function_name(node: exp.Expr) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    if isinstance(node, exp.Func):
        return str(node.sql_name()).lower()
    return ""


def _is_forbidden_function(name: str) -> bool:
    return name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_PREFIXES)


class SqlGuard:
    def __init__(self, catalog: Catalog, max_rows: int = 200) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.catalog = catalog
        self.max_rows = max_rows

    def check(self, sql: str, principal: Principal) -> GuardedQuery:
        if principal.max_tier is None:
            raise PolicyViolation("NO_DATA_ACCESS", "this token grants no warehouse access")
        if len(sql) > MAX_SQL_CHARS:
            raise PolicyViolation("SQL_TOO_LONG", f"SQL is limited to {MAX_SQL_CHARS} characters")
        root = self._parse_single(sql)
        if not isinstance(root, exp.Select | exp.SetOperation):
            raise PolicyViolation(
                "READ_ONLY", f"only SELECT queries are allowed, got {type(root).__name__.upper()}"
            )
        for node in root.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise PolicyViolation(
                    "READ_ONLY", f"{type(node).__name__.upper()} is not allowed in a read-only query"
                )
            if isinstance(node, exp.PositionalColumn | exp.Placeholder):
                # DuckDB's `SELECT #7 FROM t` reads a column by position, which would sidestep
                # every name-based check below.
                raise PolicyViolation("POSITIONAL_REFERENCE", "refer to columns by name, not by position")
        models, aliases = self._check_sources(root, principal)
        self._check_functions(root)
        if not principal.can_read_pii:
            self._check_pii(root, models, aliases)
        fetch_limit = self._apply_limit(root)
        return GuardedQuery(
            sql=root.sql(dialect=DIALECT, comments=False),
            models=frozenset(models),
            fetch_limit=fetch_limit,
            max_rows=self.max_rows,
        )

    # -- steps -----------------------------------------------------------------------

    @staticmethod
    def _parse_single(sql: str) -> exp.Expr:
        try:
            statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
        except (ParseError, TokenError) as err:
            first = str(err).splitlines()[0] if str(err) else "could not parse SQL"
            raise PolicyViolation("PARSE_ERROR", f"{first} (DuckDB dialect)") from None
        if len(statements) != 1:
            raise PolicyViolation("SINGLE_STATEMENT", f"send exactly one statement, got {len(statements)}")
        return statements[0]

    def _check_sources(self, root: exp.Expr, principal: Principal) -> tuple[set[str], set[str]]:
        cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
        models: set[str] = set()
        aliases: set[str] = set()
        for table in root.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                raise PolicyViolation(
                    "TABLE_FUNCTION", "table functions are not allowed; query catalog models"
                )
            if table.args.get("catalog") or (table.db and table.db.lower() != "main"):
                raise PolicyViolation("UNKNOWN_MODEL", f"{table.sql(dialect=DIALECT)} is not a catalog model")
            name = table.name.lower()
            if name in cte_names and not table.db:
                continue
            model = self.catalog.by_name.get(name)
            if model is None or not principal.can_see(model):
                raise PolicyViolation(
                    "UNKNOWN_MODEL",
                    f"{name!r} is not a model you can query; use search_models to find one",
                    internal_detail=None
                    if model is None
                    else f"{name} is {model.tier}; principal max tier is {principal.max_tier}",
                )
            models.add(name)
            aliases.add(name)
            if table.alias:
                aliases.add(table.alias.lower())
        return models, aliases | cte_names

    @staticmethod
    def _check_functions(root: exp.Expr) -> None:
        for node in root.find_all(exp.Func):
            name = _function_name(node)
            if _is_forbidden_function(name):
                raise PolicyViolation("FORBIDDEN_FUNCTION", f"function {name}() is not allowed")

    def _check_pii(self, root: exp.Expr, models: set[str], aliases: set[str]) -> None:
        pii = {c for m in models for c in self.catalog.model(m).pii_columns}
        if not pii:
            return
        for col in root.find_all(exp.Column):
            if isinstance(col.this, exp.Star):
                raise PolicyViolation(
                    "PII_DENIED", "'*' would include PII columns; list the columns you need"
                )
            name = col.name.lower()
            if name in pii:
                raise PolicyViolation("PII_DENIED", f"column {name!r} is PII and needs the pii:read scope")
            if not col.table and name in aliases:
                raise PolicyViolation(
                    "PII_DENIED", f"whole-row reference {name!r} is not allowed; name columns"
                )
        for star in root.find_all(exp.Star):
            if isinstance(star.parent, exp.Count):
                continue  # COUNT(*) reads no column values
            raise PolicyViolation("PII_DENIED", "'*' would include PII columns; list the columns you need")
        # Identifiers outside Column nodes (USING clauses, struct field access) can still name PII.
        for ident in root.find_all(exp.Identifier):
            if ident.name.lower() in pii and not isinstance(ident.parent, exp.TableAlias):
                raise PolicyViolation(
                    "PII_DENIED", f"identifier {ident.name.lower()!r} is PII and needs pii:read"
                )

    def _apply_limit(self, root: exp.Expr) -> int:
        limit = root.args.get("limit")
        requested: int | None = None
        if limit is not None:
            value = limit.expression
            if not (isinstance(value, exp.Literal) and value.is_int):
                raise PolicyViolation("LIMIT_NOT_LITERAL", "LIMIT must be an integer literal")
            requested = int(value.this)
        if requested is not None and requested <= self.max_rows:
            return requested
        root.set("limit", exp.Limit(expression=exp.Literal.number(self.max_rows + 1)))
        return self.max_rows + 1
