"""Compile a metric request into SQL deterministically.

Agents ask for a metric by name, optionally grouped by dimensions, filtered, and bucketed by
time. The compiler builds the query as a sqlglot AST from the catalog definition, so business
logic lives in one place and user-supplied values only ever become quoted literals, never SQL
text. Identifiers are always quoted to avoid collisions with function names.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp

from warehouse_mcp.catalog import Catalog, Metric, Model, Tier
from warehouse_mcp.policy import PolicyViolation, Principal

Scalar = str | int | float | bool
TimeGrain = Literal["day", "week", "month", "quarter", "year"]
FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"]

_COMPARISONS: dict[str, type[exp.Binary]] = {
    "eq": exp.EQ,
    "ne": exp.NEQ,
    "gt": exp.GT,
    "gte": exp.GTE,
    "lt": exp.LT,
    "lte": exp.LTE,
}
MAX_GROUP_BY = 4
MAX_FILTERS = 8
MAX_IN_VALUES = 50


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="Dimension to filter on, e.g. 'dim_lanes.border_crossing' or 'mode'.")
    op: FilterOp = Field(description="Comparison operator.")
    value: Scalar | list[Scalar] = Field(description="Value, or a list of values for 'in' and 'not_in'.")


class MetricRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    group_by: list[str] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    time_grain: TimeGrain | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None
    order_by: str | None = None
    descending: bool = True
    limit: int = Field(default=50, ge=1)


def _col(table: str, name: str) -> exp.Column:
    return exp.column(name, table=table, quoted=True)


def _literal(value: Scalar) -> exp.Expr:
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, int | float):
        return exp.Literal.number(value)
    return exp.Literal.string(value)


class SemanticCompiler:
    def __init__(self, catalog: Catalog, max_rows: int = 200) -> None:
        self.catalog = catalog
        self.max_rows = max_rows

    def available_dimensions(self, metric: Metric, principal: Principal) -> list[str]:
        out: list[str] = []
        for model in self.catalog.reachable_models(metric.model):
            if not principal.can_see(model):
                continue
            out.extend(f"{model.name}.{c.name}" for c in model.columns if principal.can_read_pii or not c.pii)
        return out

    def _resolve(self, ref: str, metric: Metric, principal: Principal) -> tuple[Model, str]:
        reachable = self.catalog.reachable_models(metric.model)
        if "." in ref:
            table, _, column = ref.partition(".")
            matches = [(m, column) for m in reachable if m.name == table and m.column(column)]
        else:
            matches = [(m, ref) for m in reachable if m.column(ref)]
        if not matches:
            raise PolicyViolation(
                "UNKNOWN_DIMENSION",
                f"{ref!r} is not a dimension of {metric.name}; call list_metrics for options",
            )
        if len(matches) > 1 and matches[0][0].name == metric.model:
            matches = matches[:1]  # a bare join key resolves to the metric's own model
        if len(matches) > 1:
            options = ", ".join(f"{m.name}.{c}" for m, c in matches)
            raise PolicyViolation("AMBIGUOUS_DIMENSION", f"{ref!r} is ambiguous; use one of {options}")
        model, column = matches[0]
        if not principal.can_see(model):
            raise PolicyViolation(
                "UNKNOWN_DIMENSION",
                f"{ref!r} is not a dimension of {metric.name}",
                internal_detail=f"{model.name} is {model.tier}",
            )
        if model.column(column).pii and not principal.can_read_pii:  # type: ignore[union-attr]
            raise PolicyViolation("PII_DENIED", f"{model.name}.{column} is PII and needs the pii:read scope")
        return model, column

    def compile(  # noqa: PLR0912, PLR0915 - one linear pass, each branch a validation step
        self, request: MetricRequest, principal: Principal
    ) -> tuple[exp.Select, Tier, set[str]]:
        """Return the query, the tier it reads at, and the models it touches."""
        try:
            metric = self.catalog.metric(request.metric)
        except KeyError:
            raise PolicyViolation(
                "UNKNOWN_METRIC", f"{request.metric!r} is not a metric; call list_metrics"
            ) from None
        tier = self.catalog.metric_tier(metric)
        if not principal.can_see_tier(tier):
            raise PolicyViolation(
                "UNKNOWN_METRIC",
                f"{request.metric!r} is not a metric; call list_metrics",
                internal_detail=f"{metric.name} is {tier}",
            )
        if len(request.group_by) > MAX_GROUP_BY:
            raise PolicyViolation("TOO_MANY_DIMENSIONS", f"group by at most {MAX_GROUP_BY} dimensions")
        if len(request.filters) > MAX_FILTERS:
            raise PolicyViolation("TOO_MANY_FILTERS", f"use at most {MAX_FILTERS} filters")
        if request.limit > self.max_rows:
            raise PolicyViolation("LIMIT_TOO_HIGH", f"limit must be at most {self.max_rows}")

        base = self.catalog.model(metric.model)
        used: dict[str, Model] = {base.name: base}
        for col in metric.expression_ast.find_all(exp.Column):
            used[col.table] = self.catalog.model(col.table)

        select = exp.Select()
        group_exprs: list[exp.Expr] = []
        output_names: list[str] = []
        if request.time_grain:
            if base.time_column is None:
                raise PolicyViolation("NO_TIME_DIMENSION", f"{metric.name} has no time column to bucket by")
            period = exp.Cast(
                this=exp.Anonymous(
                    this="date_trunc",
                    expressions=[exp.Literal.string(request.time_grain), _col(base.name, base.time_column)],
                ),
                to=exp.DataType.build("DATE"),
            )
            select = select.select(exp.alias_(period, "period", quoted=True))
            group_exprs.append(period)
            output_names.append("period")
        for ref in request.group_by:
            model, column = self._resolve(ref, metric, principal)
            used[model.name] = model
            name = column if column not in output_names else f"{model.name}__{column}"
            select = select.select(exp.alias_(_col(model.name, column), name, quoted=True))
            group_exprs.append(_col(model.name, column))
            output_names.append(name)
        select = select.select(exp.alias_(metric.expression_ast.copy(), metric.name, quoted=True))

        conditions: list[exp.Expr] = []
        if metric.where_ast is not None:
            conditions.append(metric.where_ast.copy())
        if (request.start_date or request.end_date) and base.time_column is None:
            raise PolicyViolation("NO_TIME_DIMENSION", f"{metric.name} has no time column to filter on")
        if request.start_date:
            conditions.append(
                exp.GTE(
                    this=_col(base.name, base.time_column or ""),
                    expression=exp.Cast(
                        this=exp.Literal.string(request.start_date.isoformat()), to=exp.DataType.build("DATE")
                    ),
                )
            )
        if request.end_date:
            conditions.append(
                exp.LTE(
                    this=_col(base.name, base.time_column or ""),
                    expression=exp.Cast(
                        this=exp.Literal.string(request.end_date.isoformat()), to=exp.DataType.build("DATE")
                    ),
                )
            )
        for flt in request.filters:
            model, column = self._resolve(flt.field, metric, principal)
            used[model.name] = model
            conditions.append(self._condition(_col(model.name, column), flt))

        select = select.from_(exp.Table(this=exp.to_identifier(base.name, quoted=True)))
        for join in base.joins:
            if join.model in used:
                select = select.join(
                    exp.Table(this=exp.to_identifier(join.model, quoted=True)),
                    on=exp.EQ(this=_col(base.name, join.key), expression=_col(join.model, join.key)),
                    join_type="left",
                )
        if conditions:
            select = select.where(exp.and_(*conditions))
        if group_exprs:
            select = select.group_by(*group_exprs)

        order_target = request.order_by or (
            metric.name if request.group_by else ("period" if request.time_grain else None)
        )
        if order_target is not None:
            if order_target not in (*output_names, metric.name):
                raise PolicyViolation(
                    "BAD_ORDER_BY", f"order_by must be one of {[*output_names, metric.name]}"
                )
            descending = request.descending if request.order_by or request.group_by else False
            select = select.order_by(exp.Ordered(this=exp.column(order_target, quoted=True), desc=descending))
        select = select.limit(request.limit)

        # Every model in `used` was checked for visibility above, so this is the tier the query reads at.
        return select, max(m.tier for m in used.values()), set(used)

    @staticmethod
    def _condition(column: exp.Column, flt: Filter) -> exp.Expr:
        if flt.op in ("in", "not_in"):
            values = flt.value if isinstance(flt.value, list) else [flt.value]
            if not values or len(values) > MAX_IN_VALUES:
                raise PolicyViolation("BAD_FILTER", f"'{flt.op}' needs 1 to {MAX_IN_VALUES} values")
            cond: exp.Expr = exp.In(this=column, expressions=[_literal(v) for v in values])
            return exp.Not(this=cond) if flt.op == "not_in" else cond
        if isinstance(flt.value, list):
            raise PolicyViolation("BAD_FILTER", f"'{flt.op}' takes a single value, not a list")
        return _COMPARISONS[flt.op](this=column, expression=_literal(flt.value))
