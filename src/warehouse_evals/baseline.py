"""The regression gate: nothing that passed in the committed baseline may fail now.

The absolute gates in ``gwm-eval all`` check thresholds (every reference case passes, Recall@3
at least 0.9). This gate checks against the last state someone deliberately accepted
(``evals/baseline.json``):

- every agent case and grader-validation row that passed in the baseline still passes (100%,
  no tolerance);
- no retrieval metric of any ranker falls more than ``MAX_METRIC_DROP`` below its baseline.

Improvements never fail the gate. They show up as differences until someone runs
``gwm-eval baseline --update`` and commits the new file, which is the point: moving the
baseline is a reviewed change, not a side effect of a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from warehouse_evals.history import metric_values
from warehouse_evals.report import Report

BASELINE = Path("evals/baseline.json")
MAX_METRIC_DROP = 0.05
FORMAT = 1


def _label(label: str) -> str:
    """Stable section names: the replay label carries the cassette path, which may differ."""
    return "replay" if label.startswith("replay") else label


def snapshot(report: Report, gated_sets: set[str] | None = None) -> dict[str, Any]:
    """The parts of a report the gate compares. ``gated_sets`` limits retrieval to those sets
    (a set too small to gate is left out of the baseline)."""
    agent: dict[str, dict[str, bool]] = {}
    for label, results in report.agent.items():
        section = agent.setdefault(_label(label), {})
        for r in results:
            section[f"{r.case_id}.t{r.trial}"] = r.passed
    return {
        "format": FORMAT,
        "catalog_version": report.meta.get("catalog_version"),
        "seed": report.meta.get("seed"),
        "agent": {k: dict(sorted(v.items())) for k, v in sorted(agent.items())},
        "validation": dict(sorted((f"{r.policy}/{r.case_id}", r.ok) for r in report.validation)),
        "retrieval": {
            name: {s.ranker: {m: round(v, 4) for m, v in metric_values(s).items()} for s in scores}
            for name, scores in sorted(report.retrieval.items())
            if gated_sets is None or name in gated_sets
        },
    }


@dataclass
class Comparison:
    regressions: list[str] = field(default_factory=list)  # fail the gate
    notes: list[str] = field(default_factory=list)  # reported only

    @property
    def ok(self) -> bool:
        return not self.regressions


def _passes(section: str, baseline: dict[str, bool], current: dict[str, bool], out: Comparison) -> None:
    for key, was in baseline.items():
        if key not in current:
            out.notes.append(f"{section} {key}: in the baseline but not run")
        elif was and not current[key]:
            out.regressions.append(f"{section} {key}: passed in the baseline, fails now")
        elif not was and current[key]:
            out.notes.append(f"{section} {key}: fixed since the baseline")
    for key in sorted(set(current) - set(baseline)):
        out.notes.append(f"{section} {key}: new, not in the baseline")


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> Comparison:
    out = Comparison()
    if baseline.get("format") != FORMAT:
        out.regressions.append(
            f"baseline format {baseline.get('format')!r}, expected {FORMAT}; regenerate it"
        )
        return out
    for label, cases in baseline.get("agent", {}).items():
        if label not in current["agent"]:
            # Replays only run when cassettes are present; their absence is not a regression.
            out.notes.append(f"{label}: in the baseline but not run")
            continue
        _passes(label, cases, current["agent"][label], out)
    _passes("grader validation", baseline.get("validation", {}), current["validation"], out)
    for set_name, rankers in baseline.get("retrieval", {}).items():
        now_set = current["retrieval"].get(set_name)
        if now_set is None:
            out.regressions.append(f"retrieval {set_name}: in the baseline but not evaluated")
            continue
        for ranker, metrics in rankers.items():
            for metric, was in metrics.items():
                now = now_set.get(ranker, {}).get(metric)
                if now is None:
                    out.regressions.append(f"retrieval {set_name} {ranker} {metric}: missing")
                elif now < was - MAX_METRIC_DROP - 1e-9:
                    out.regressions.append(
                        f"retrieval {set_name} {ranker} {metric}: {now:.3f}, baseline {was:.3f} "
                        f"(more than {MAX_METRIC_DROP} below)"
                    )
                elif abs(now - was) > 1e-9:
                    out.notes.append(f"retrieval {set_name} {ranker} {metric}: {was:.3f} -> {now:.3f}")
    return out


def load(path: Path = BASELINE) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(data: dict[str, Any], path: Path = BASELINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
