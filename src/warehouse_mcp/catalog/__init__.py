"""Typed, validated loader for the semantic catalog (``catalog.yaml``).

Loading fails fast on anything an agent could trip over later: unknown models in joins or
metrics, metric expressions that do not parse or that reference columns outside the metric's
reachable models, and metrics that would read PII columns.
"""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Literal

import sqlglot
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlglot import exp

DIALECT = "duckdb"


class Tier(IntEnum):
    """Access tiers, ordered so that a higher tier implies the lower ones."""

    PUBLIC = 0
    INTERNAL = 1
    RESTRICTED = 2

    @classmethod
    def parse(cls, value: str) -> Tier:
        return cls[value.upper()]

    def __str__(self) -> str:
        return self.name.lower()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Column(_Strict):
    name: str
    type: str
    description: str
    pii: bool = False
    free_text: bool = False  # user-written text: returned with an untrusted-content warning


class Join(_Strict):
    model: str
    key: str


class Model(_Strict):
    name: str
    kind: Literal["dimension", "fact", "mart"]
    tier_name: Literal["public", "internal", "restricted"] = Field(alias="tier")
    grain: str
    description: str
    aliases: tuple[str, ...] = ()
    time_column: str | None = None
    joins: tuple[Join, ...] = ()
    columns: tuple[Column, ...]

    @property
    def tier(self) -> Tier:
        return Tier.parse(self.tier_name)

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def pii_columns(self) -> frozenset[str]:
        return frozenset(c.name for c in self.columns if c.pii)

    @model_validator(mode="after")
    def _check(self) -> Model:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate column names")
        if self.time_column and self.time_column not in names:
            raise ValueError(f"{self.name}: time_column {self.time_column!r} is not a column")
        for join in self.joins:
            if join.key not in names:
                raise ValueError(f"{self.name}: join key {join.key!r} is not a column")
        return self


class Metric(_Strict):
    name: str
    model: str
    description: str
    expression: str
    where: str | None = None
    format: Literal["count", "ratio", "days", "hours", "usd", "number"]
    synonyms: tuple[str, ...] = ()

    @cached_property
    def expression_ast(self) -> exp.Expr:
        return sqlglot.parse_one(self.expression, dialect=DIALECT)

    @cached_property
    def where_ast(self) -> exp.Expr | None:
        return sqlglot.parse_one(self.where, dialect=DIALECT) if self.where else None


class Catalog(_Strict):
    schema_version: int
    name: str
    models: tuple[Model, ...]
    metrics: tuple[Metric, ...]

    @cached_property
    def by_name(self) -> dict[str, Model]:
        return {m.name: m for m in self.models}

    @cached_property
    def metrics_by_name(self) -> dict[str, Metric]:
        return {m.name: m for m in self.metrics}

    @cached_property
    def version(self) -> str:
        """Content hash: any change to the published contract changes the version."""
        payload = json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True)
        return f"v{self.schema_version}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    @cached_property
    def pii_column_names(self) -> frozenset[str]:
        return frozenset(c for m in self.models for c in m.pii_columns)

    def model(self, name: str) -> Model:
        try:
            return self.by_name[name]
        except KeyError:
            raise KeyError(f"unknown model {name!r}") from None

    def metric(self, name: str) -> Metric:
        try:
            return self.metrics_by_name[name]
        except KeyError:
            raise KeyError(f"unknown metric {name!r}") from None

    def reachable_models(self, base: str) -> tuple[Model, ...]:
        """The base model plus the models it declares joins to (one hop, star schema)."""
        model = self.model(base)
        return (model, *(self.model(j.model) for j in model.joins))

    def metric_models(self, metric: Metric) -> frozenset[str]:
        """The models a metric reads: its base model plus any its expression or filter names."""
        asts = (a for a in (metric.expression_ast, metric.where_ast) if a is not None)
        return frozenset({metric.model} | {c.table for a in asts for c in a.find_all(exp.Column)})

    def metric_tier(self, metric: Metric) -> Tier:
        """A metric is as sensitive as the most sensitive model it reads."""
        return max(self.model(t).tier for t in self.metric_models(metric))

    def replace(self, **changes: object) -> Catalog:
        """A validated copy with some fields replaced (cached indexes are rebuilt)."""
        return Catalog.model_validate({**self.model_dump(by_alias=True), **changes})

    @model_validator(mode="after")
    def _check(self) -> Catalog:  # noqa: PLR0912 - one branch per invariant
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError("duplicate model names")
        by_name = self.by_name
        for model in self.models:
            for join in model.joins:
                target = by_name.get(join.model)
                if target is None:
                    raise ValueError(f"{model.name}: join to unknown model {join.model!r}")
                if target.column(join.key) is None:
                    raise ValueError(f"{model.name}: {join.model} has no join key {join.key!r}")
        seen: set[str] = set()
        for metric in self.metrics:
            if metric.name in seen:
                raise ValueError(f"duplicate metric {metric.name!r}")
            seen.add(metric.name)
            base = by_name.get(metric.model)
            if base is None:
                raise ValueError(f"metric {metric.name}: unknown model {metric.model!r}")
            reachable = {m.name: m for m in self.reachable_models(base.name)}
            for ast in (metric.expression_ast, metric.where_ast):
                if ast is None:
                    continue
                for col in ast.find_all(exp.Column):
                    if not col.table:
                        raise ValueError(f"metric {metric.name}: column {col.name!r} must be model-qualified")
                    owner = reachable.get(col.table)
                    if owner is None or owner.column(col.name) is None:
                        raise ValueError(f"metric {metric.name}: {col.table}.{col.name} is not reachable")
                    if owner.column(col.name).pii:  # type: ignore[union-attr]
                        raise ValueError(f"metric {metric.name}: reads PII column {col.table}.{col.name}")
        return self


def load_catalog(path: Path | str | None = None) -> Catalog:
    """Load and validate a catalog. Defaults to the packaged ``catalog.yaml``."""
    if path is None:
        text = resources.files(__package__).joinpath("catalog.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return Catalog.model_validate(yaml.safe_load(text))


__all__ = ["DIALECT", "Catalog", "Column", "Join", "Metric", "Model", "Tier", "load_catalog"]
