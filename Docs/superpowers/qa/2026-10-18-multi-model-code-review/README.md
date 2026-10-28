# TASK-35582 delivery evidence

AC24 remains open. The delivery candidate carries the implemented review recipe
onto dev `e3d07afa65927b9d477fe9f13af619550da9e005`; it does not claim that prior
live qualification passed. The next live work requires a new, bounded allowance.

## Current boundary

The latest historical run used43 requests and3 children, reaching a fresh Haiku
verifier under native [2,1]. All43 HTTP responses succeeded, but the verifier
ended incomplete with submission_conflict and no complete original-finding
mapping or readable unified report. Its accepted submission body was not retained.
See the [independent outcome review](ac24/fresh43-after-haiku-outcome/outcome-review.md).
Total historical spending is255requests and18accepted children; no allowance remains.
The local usage fix did launch the verifier. The Haiku capability correction is
retained; request acceptance alone does not prove a tool-result round trip.

## Existing evidence reused

- [Whole-branch review](final-fix/final-review.md) and [scoped final fixes](final-fix/final-fix-review.md) cover the original implementation. Integration changes receive only their relevant follow-up checks.
- [Resource repair review](ac23/ac23-resource-review.md) and [AC23 closure review](ac23/ac23-regression-review.md) record finite native ownership and deterministic controls.
- [Haiku capability review](ac24/fresh42-after-usage-outcome/haiku-fix/review.md) covers the exact model-family request correction.
- The native scripted journey has4originals/3blinded units (exact duplicate, distinct trigger and false positive); fault controls cover failed reviewers, truncation, drift, small contexts and storage overflow. These remain scripted evidence, not live judgments.

Prior tests/reviews retain their source revisions and limits. Inherited profile
admission failures, descriptor/static warnings, a Minor precedence-fixture gap,
and bounded recovery limits are not relabeled green. Current delivery integration passed97 focused tests and2 current-dev Stop/stall
controls. The [integration report](integration-report.md) and
[scoped integration review](integration-review.md) record exact boundaries; the
review found no integration findings. Live qualification remains pending.

## Preservation and review footprint

The PR retains required test fixtures (`seeded_average.py`,
`ac24/capacity-evidence.json`), concise reviews, and design documents imported by
the canonical ADRs. Redundant raw traces, payload snapshots, command logs and
private SDD reports are omitted from the delivery diff. They remain intact on
`codex/task-35582-multi-model-review` at3bce87f15c and in a verified6.1MB Git bundle.
The [retention manifest](retained-evidence.json) records paths, hashes, bundle
location and prerequisite. No original evidence or active worktree was deleted.

Historical artifacts call the feature ADR240. Current dev assigned that number
to MCP destination retirement, so delivery renames the feature decision to
[ADR-281](../../../../backlog/decisions/281-primary-agent-multi-model-code-review.md)
without changing its contracts. Verbatim historical reviews retain their numbering.
