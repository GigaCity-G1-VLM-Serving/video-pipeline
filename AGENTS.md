# Agent guidance

## Context

Read `README.md`, `docs/architecture.md`, and `docs/contracts.md` before changing
shared interfaces. This is a collision-focused pipeline scaffold, not a running
VLM system. A/B own edge selection and transport; C owns serving/budgets; D owns
workload/replay/evaluation. Aditya coordinates integration and contract review.

For research rationale, read `docs/research/project-proposal.md` and
`docs/research/dataset-selection-and-pilot.md`. These are historical snapshots;
use architecture/contracts for current implementation decisions. Availability and
performance statements in the snapshots have not been revalidated.

## Development

- Use Python 3.12 and uv. Root `uv sync --locked` installs contracts and dev tools.
- Add runtime dependencies to their owning package; use root dev dependencies for tools.
- Keep contracts lightweight and independent of edge, serving, and GPU implementations.
- Use root Ruff settings: four-space Python indentation; two spaces for other text.
- Run `uv run poe format` and `uv run poe test` for relevant Python changes.
- Tests live in each package's `tests/`; use synthetic data and installed packages,
  not sys.path edits. Report checks performed and any unavailable checks.

## Shared contracts and research

- When proposing or reviewing handoff changes, read
  `.agents/skills/contract-review/SKILL.md` and check producers and consumers together.
- Update models, documentation, JSON examples, and affected tests/consumers together.
- Contracts are draft. Frames versus clips, engine, transport, result shape, and
  feedback policy remain unresolved; keep proposals labeled until team review.
- Preserve source-video identity, ordered capture times, and causal replay. Keep
  evaluation labels out of selection, budgeting, and inference inputs.
- Identify mock/synthetic results and separate them from measured inference results.
- TAR distractor No questions are not quiet-scene negatives for false-alarm claims.
- Keep downloaded media, model weights, credentials, and generated outputs out of Git.

## Commits

- Use Conventional Commits: `type(scope): summary` (scope is optional). For example,
  `feat(contracts): add draft window metadata` or `test: cover invalid windows`.
- When an agent creates a commit, include a `Co-authored-by: Name <email>` trailer
  after a blank line, attributing the contributing agent. Use the actual agent name
  and its tool/provider-documented attribution email, or an identity supplied by the
  user. Do not invent an attribution email or attribute work to a different agent.
  If no attribution identity is available, ask for it before creating the commit.
