"""Real retrieval queries: what the live agent actually sent to ``search_models``.

The dev and held-out sets were written by someone who could see the catalog, so they lean
towards catalog words. This module collects the queries a model wrote on its own during live
runs, so they can be labelled and evaluated as a third set, ``real``:

1. ``gwm-eval harvest cassettes/ reports/report.json`` reads recorded live runs and live or
   replayed reports, keeps each ``search_models`` query once, drops queries already in a
   retrieval set, and writes a label template with the candidates the server returns now.
2. A person fills in ``relevant`` (or deletes a row that is not a real retrieval need).
3. ``gwm-eval harvest --promote <template>`` appends the labelled rows to
   ``cases/retrieval_real.yaml`` after checking every name exists in the catalog.

Scripted runs are skipped: their queries are authored, not real. Labels are never guessed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from warehouse_evals.cases import AgentCase, RetrievalCase, load_retrieval_cases
from warehouse_mcp.catalog import Catalog
from warehouse_mcp.discovery import Discovery
from warehouse_mcp.policy import ROLE_SCOPES, Principal

REAL = Path(__file__).parent / "cases" / "retrieval_real.yaml"
SEARCH_TOOL = "search_models"
REAL_MODES = frozenset({"live", "replay"})
TEMPLATE_HEADER = """\
# Retrieval queries the live agent sent to search_models, for labelling.
# For each row, list in `relevant` the models or metrics that answer it (names from the
# catalog), or delete the row if it is not a real retrieval need. `candidates` is what the
# server returns for this role today: a hint, not the answer. Then run:
#   uv run gwm-eval harvest --promote <this file>
"""


@dataclass(frozen=True)
class Seen:
    query: str
    role: str | None
    source: str  # cassette or report path, with the case id


@dataclass
class Harvested:
    query: str
    roles: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)


def normalize(query: str) -> str:
    """Lower case, single spaces, no trailing punctuation: the key for deduplication."""
    return re.sub(r"\s+", " ", query).strip().lower().rstrip("?.!")


def _tool_uses(sse: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """(name, input) for every tool_use block in one recorded Messages API stream."""
    names: dict[int, str] = {}
    parts: dict[int, list[str]] = {}
    for line in sse.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        kind, index = event.get("type"), event.get("index")
        if kind == "content_block_start" and event["content_block"].get("type") == "tool_use":
            names[index], parts[index] = event["content_block"]["name"], []
        elif kind == "content_block_delta" and index in names:
            if event["delta"].get("type") == "input_json_delta":
                parts[index].append(event["delta"]["partial_json"])
        elif kind == "content_block_stop" and index in names:
            raw = "".join(parts.pop(index))
            name = names.pop(index)
            try:
                value = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                continue  # a malformed tool input never reached the server as a query
            if isinstance(value, dict):
                yield name, value


def from_cassette(path: Path, roles: dict[str, str]) -> list[Seen]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    meta = json.loads(lines[0])
    case = str(meta.get("case", ""))
    out = []
    for line in lines[1:]:
        for name, args in _tool_uses(json.loads(line).get("body", "")):
            if name == SEARCH_TOOL and isinstance(args.get("query"), str):
                out.append(Seen(args["query"], roles.get(case), f"{path.name} ({case})"))
    return out


def from_report(path: Path) -> list[Seen]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for results in data.get("agent", {}).values():
        for r in results:
            if r.get("mode") not in REAL_MODES:
                continue
            for call in r.get("tool_calls", []):
                query = call.get("input", {}).get("query")
                if call.get("name") == SEARCH_TOOL and isinstance(query, str):
                    out.append(Seen(query, r.get("role"), f"{path.name} ({r.get('case_id')})"))
    return out


def collect(paths: Iterable[Path], cases: Iterable[AgentCase]) -> list[Seen]:
    """Every search query in the given cassettes (``*.jsonl``) and reports (``*.json``);
    directories are searched one level deep."""
    roles = {c.id: c.role for c in cases}
    files: list[Path] = []
    for p in paths:
        files += sorted([*p.glob("*.jsonl"), *p.glob("*.json")]) if p.is_dir() else [p]
    seen: list[Seen] = []
    for f in files:
        seen += from_cassette(f, roles) if f.suffix == ".jsonl" else from_report(f)
    return seen


def dedupe(seen: Iterable[Seen], existing: Iterable[RetrievalCase]) -> list[Harvested]:
    """One row per distinct query, minus queries already in a retrieval set, in first-seen order."""
    known = {normalize(c.query) for c in existing}
    rows: dict[str, Harvested] = {}
    for s in seen:
        key = normalize(s.query)
        if not key or key in known:
            continue
        row = rows.setdefault(key, Harvested(s.query.strip()))
        if s.role:
            row.roles.add(s.role)
        row.sources.append(s.source)
    return list(rows.values())


def existing_cases(*paths: Path) -> list[RetrievalCase]:
    out = load_retrieval_cases()
    for p in paths:
        if p.exists():
            out += load_retrieval_cases(p)
    return out


def template(rows: Iterable[Harvested], catalog: Catalog, k: int = 5) -> str:
    discovery = Discovery(catalog)
    items = []
    for row in rows:
        # The role decides what the agent could see; with several roles, show the narrowest.
        role = min(row.roles, key=lambda r: len(ROLE_SCOPES[r])) if row.roles else None
        principal = Principal.for_role(role) if role else None
        items.append(
            {
                "query": row.query,
                "role": role,
                "seen": len(row.sources),
                "sources": row.sources,
                "candidates": [h.name for h in discovery.search(row.query, principal, k=k)],
                "relevant": [],
            }
        )
    return TEMPLATE_HEADER + yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=100)


@dataclass
class Promotion:
    added: list[RetrievalCase] = field(default_factory=list)
    skipped_unlabelled: int = 0
    skipped_duplicate: int = 0
    unknown_names: list[str] = field(default_factory=list)


def promote(labelled: Path, catalog: Catalog, real: Path = REAL) -> Promotion:
    """Append labelled rows to the real set. Nothing is written if any name is unknown."""
    rows = yaml.safe_load(labelled.read_text(encoding="utf-8")) or []
    known_names = set(catalog.by_name) | set(catalog.metrics_by_name)
    current = load_retrieval_cases(real) if real.exists() else []
    seen = {normalize(c.query) for c in [*existing_cases(), *current]}
    result = Promotion()
    for row in rows:
        relevant = tuple(row.get("relevant") or ())
        if not relevant:
            result.skipped_unlabelled += 1
            continue
        result.unknown_names += [f"{row['query']!r}: {n}" for n in relevant if n not in known_names]
        key = normalize(row["query"])
        if key in seen:
            result.skipped_duplicate += 1
            continue
        seen.add(key)
        result.added.append(RetrievalCase(query=row["query"], relevant=relevant))
    if result.unknown_names or not result.added:
        return result
    lines = [] if real.exists() else [_REAL_HEADER]
    lines += [
        yaml.safe_dump(
            [{"query": c.query, "relevant": list(c.relevant)}], default_flow_style=None, width=200
        ).rstrip()
        for c in result.added
    ]
    with real.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return result


_REAL_HEADER = """\
# Real retrieval queries: written by the agent during live runs, harvested with
# `gwm-eval harvest` and labelled by a person. Never edited to suit the rankers.
# Reported as the "real set"; gated once it has at least 20 queries."""
