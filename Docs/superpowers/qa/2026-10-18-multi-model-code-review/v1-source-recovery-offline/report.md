# V1 source-range and completion recovery — offline only

Base: `2b544776f91a6b972784e9f387785b9a32b34f8d`. The prior
[live outcome](../final-handoff60/report.md) remains failed; AC24 stays open and
PR #3286 stays draft. This follow-up makes no provider requests or product child
launches. Cumulative actual usage remains **457 requests / 31 children**.

The retained verifier accepted four dispositions, then requested lines 1–10 from
captured worktree/base/worktree documents containing only seven/five lines. The
source owner correctly refused each range, but returned only
`invalid_source_range`. The runtime would also have discarded a detailed read
error body. Guidance made the live completion checkpoint conditional on history
reduction or uncertainty, rather than the normal next step after accepted batches.

The offline reproduction uses that exact three-call argument sequence through
the real source/service/provider and native/fenced agent loop. It reaches the
**consecutive failed-tool-call guard**, not the separate permission-denial guard
inferred in the historical outcome review. The later primary deadline/protocol
failure does not explain this earlier V1 stop.

The repair preserves strict range rejection and adds bounded feedback only after
source authority and reference checks: captured line count, same-document retry
without a guessed range, and a verifier-only live-checkpoint hint. The runtime
preserves this feedback for the model. Guidance uses bounded no-range source
reads and directs each accepted disposition batch through the existing checkpoint
to `verifier_complete` at the current revision, then return after acceptance.
It neither auto-completes verification nor changes accepted decisions, replay,
failure limits, permissions, accounting or terminal/capture sealing.

Tests use real captured files and production protocol/state owners; model turns
are scripted offline controls. The recovery path reads captured bytes, checks the
current revision, completes, and seals only after capture and terminal receipts.
Four originals expand exactly once. Ignoring recovery still stops after three
failed calls. Empty files, later file mutation, unknown references, permission
refusal and an insufficient result envelope remain covered.

See [checks](checks.json) and [scoped review](review.md). The red baseline had six
expected failures and five passing safety controls; all eleven new cases pass
after repair. All **205 targeted review regressions and 15 accounting tests pass**;
the scoped review found no actionable issues. The new test and tool provider pass
Ruff check/format. Across the other touched modules, 27 pre-existing Ruff warnings
and existing formatting debt are unchanged; no baseline cleanup was attempted.
Existing live evidence and its uncertainty are unchanged. Offline
success does not establish Haiku's future behavior, semantic correctness of its
dispositions, settled live usage, or AC24 completion. No qualification or budget
request is part of this follow-up.
