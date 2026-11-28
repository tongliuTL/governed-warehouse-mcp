# Contributing

## Setup

```bash
uv sync                  # Python 3.11+; installs runtime and dev dependencies from uv.lock
uv run gwm generate      # build the synthetic warehouse at data/warehouse.duckdb
```

## Before you push

Run what CI runs:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest --cov      # coverage floor is 85%
uv run gwm-eval all --gate
```

All three must pass. Type checking is `mypy --strict` over `src/`.

## Ground rules

- **Synthetic data only.** Never add real names, emails, phone numbers, customers or carriers to
  the generator, cases or fixtures. Use reserved values: `.example` domains and `555-01xx`
  phone numbers.
- **Data changes are deliberate.** `tests/test_synthetic.py` pins a content fingerprint for each
  table. If you change the generator on purpose, update `PINNED` in the same commit and say why
  in the message. If the fingerprints change when you did not mean them to, that is a bug.
- **The catalog and the warehouse move together.** `tests/test_catalog.py` checks every catalog
  column against the warehouse, checks every warehouse column against the catalog, and compiles
  every metric. Add a column to both sides at once.
- **Keys and cassettes.** `.keys/` and `.env` are git-ignored; never commit a private key or an
  API key. Cassettes store no
  headers or API keys, but still review one before committing it.

## Adding things

- **A metric or model:** edit `src/warehouse_mcp/catalog/catalog.yaml`. Validation runs at load.
  Add a retrieval case if people would ask for it in words that differ from its name.
- **A guard rule:** add allowed and denied examples to `tests/test_sql_guard.py`. A rule that
  needs the engine as backup belongs in `tests/test_warehouse.py` as well.
- **An agent case:** add it to `src/warehouse_evals/cases/agent_cases.yaml`. Give it
  `reference_sql` (the gold answer is computed from it) and a `reference` trajectory. The
  scripted reference run must pass every grader, and grader validation must still predict
  exactly which graders each planted failure trips.
- **A grader or planted-failure policy:** update `validation.expected_failures` so the
  validation suite knows what the new policy should trip.
- **A retrieval case:** add it to the dev set (`retrieval_cases.yaml`). Keep the held-out set
  for reporting; do not tune against it. Queries the live agent sent go to the real set through
  `gwm-eval harvest`, never by hand (see [docs/evals.md](docs/evals.md#real-queries)).

## The regression baseline

`gwm-eval all --gate` fails if any case that passes in `evals/baseline.json` fails now, or if a
retrieval metric falls more than 0.05 below it. When a change is meant to move the numbers (a
new case, a ranker change, a case that now passes), update the baseline in the same pull request
and say why in the description:

```bash
uv run gwm-eval baseline            # what would change, without writing anything
uv run gwm-eval baseline --update   # rewrite evals/baseline.json; refuses while a gate fails
git diff evals/baseline.json        # review it like code
```

Never update the baseline to make a regression go away. If `main` fails the gate, CI opens an
issue labelled `eval-regression` for the code owner and closes it when `main` is green again;
`gwm-eval history` shows when each case last changed state.

## Live evals

```bash
cp .env.example .env     # model provider key, optional Langfuse keys; .env is git-ignored
uv run gwm-eval live --trials 3 --record cassettes/
```

Committing `cassettes/` makes CI replay those runs. A change to the system prompt, the tool
schemas or the model makes the replays fail, which is the signal to re-record them.

CI also re-runs judge calibration from the verdicts committed in `judgments/`. A change to an
agent case, a scripted policy, a rubric or the judge prompt changes the trajectories or the
cache key, and the offline run then stops with a missing-verdict error. Re-judge and commit the
updated cache:

```bash
uv run gwm-eval calibrate --live --model MiniMax-M3.1-Flash --min-kappa 0.6
```

A rubric change is a new file (`answer_quality_v<N+1>.yaml`); keep earlier versions so their
results stay reproducible.

A new agent case can land before it is judged: mark it `calibrated: false` and the offline run
skips it (and says so). Remove the flag in the change that commits its verdicts. To add a case
without touching the others' verdicts, append any data it needs after the existing rows, and
plant catalog text with a per-case `catalog_patch` rather than editing the shared catalog: every
tool result carries the catalog version, so a catalog edit changes every trajectory.

## Commits and pull requests

- Keep each commit about one thing, and say why in its message.
- In the pull request description, say what changed for someone using the server or the
  harness, and how you checked it.
- CI must be green before merge.

## Releases

[docs/releasing.md](docs/releasing.md) covers cutting a version, what the release workflow
publishes, and keeping the directory listings current.
