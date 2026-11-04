# GigaCity video pipeline

A collision-focused research pipeline: replay recorded traffic video, select evidence
at the edge, send it to central VLM serving, and evaluate the complete flow.

**Status:** development foundation and draft contracts. There is no inference server
or end-to-end video demo yet. Frames versus clips and the serving engine remain open
for joint A/B and C/D review.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
this directory:

```bash
uv sync --locked
uv run poe format
uv run poe test
```

The root `.python-version` selects Python 3.12; uv can download it if needed.
The virtual workspace root installs all four lightweight package scaffolds and
development tools. The initial packages have no GPU dependencies, so these example
tests run on a normal laptop.

`format` sorts Python imports and runs Ruff's formatter. `test` discovers package-local
pytest tests. EditorConfig and editor settings handle basic text conventions.

## Layout and ownership

- `packages/contracts`: shared Pydantic types; Aditya coordinates cross-team review.
- `packages/edge`: A/B, Aditya and Aryan; evidence selection and transport.
- `packages/central`: C, owned by the C/D team; admission, serving, and budgeting.
- `packages/evaluation`: D, owned by the C/D team; workload, replay, and scoring.
- `docs`: [architecture](docs/architecture.md) and [draft contracts](docs/contracts.md).

Each package has its own `pyproject.toml`, `src/`, and `tests/` directory. The virtual
workspace root holds shared development tools and configuration, plus one `uv.lock`.
The existing `initial-research` directory stays outside this repo as reference material.

## Dependencies and deployment

Add shared tools from the root with `uv add --dev <tool>`. Add runtime dependencies
to their owning package with `uv add --package gigacity-edge <dependency>` (substitute
the appropriate package). Commit both manifest changes and `uv.lock`.

All consumers declare `gigacity-contracts`; uv resolves it from the workspace and
installs it editable during development. Import it as `gigacity_contracts`.
Keep GPU/model dependencies in their owning component, not in contracts or root tools.

Future Docker images can install an individual package and its runtime dependencies:

```bash
uv sync --locked --package gigacity-edge --no-dev --no-editable
```

Run that command in a build context containing the workspace manifests and required
source packages. The shared lockfile still resolves dependencies across the workspace;
package selection controls installation. Dockerfiles and CI are outside this milestone.

## Editors and coding agents

VS Code and Cursor users should install the recommended extensions and select this
repo's `.venv` interpreter. Python uses four spaces; other text defaults to two.

Codex and Cursor use root [AGENTS.md](AGENTS.md). Claude Code loads the same guidance
through [CLAUDE.md](CLAUDE.md). Ask your agent to summarize the project instructions
to verify that it loaded them. For contract work, it should read
[the contract-review skill](.agents/skills/contract-review/SKILL.md); tools without
native discovery can read that file directly. No matching slash-command menu is required.

## Contributing

Make focused changes in the owning component. Update draft models, examples, tests,
and affected consumers together when proposing a shared contract change. Review the
media handoff with both teams before implementing a frames-only or clips-only path.
Use Conventional Commit messages; agent-created commits must include the agent
co-author trailer described in `AGENTS.md`. Before handing off changes, run the two
commands above and report what was checked.
Use small synthetic test fixtures; keep datasets, weights, secrets, and generated
results out of Git. A project license has not yet been selected.
