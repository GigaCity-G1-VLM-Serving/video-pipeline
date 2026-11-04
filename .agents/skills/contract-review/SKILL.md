---
name: contract-review
description: Review GigaCity shared handoff changes across edge, central serving, and evaluation, including models, documentation, and example messages.
---

# Review a shared contract

Read `docs/contracts.md`, the relevant models in `packages/contracts`, and affected
producer/consumer code. Resolve paths from the repository root. If a component is
still a scaffold, identify the integration expectation without claiming it was tested.

Check that IDs, source-video times, units, ordering, media references, and optional
fields mean the same thing at both ends. Review invalid inputs and how failures,
rejections, drops, and late work remain visible to evaluation. Verify budget feedback
applies to the intended future window rather than silently changing past evidence.

Keep evaluation labels out of runtime messages. Distinguish requested budgets from
observed bytes/timing, and simulation metadata from measured serving outputs.
Do not silently settle the open frames-versus-clips or serving-engine decisions.

Check models, docs, fixtures, and affected callers together. Run `uv run poe test`
for model/example changes. Report concrete mismatches, affected components, checks
performed, and decisions still requiring team review; do not invent compatibility
or runtime guarantees for unimplemented handoffs.
