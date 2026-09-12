---
id: TASK-31639
title: Chunking Lab - lossless draft and candidate state
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 23:11'
updated_date: '2026-09-05 00:24'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31638
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make controls, JSON, sample edits, and A/B snapshots share one recoverable authoring state without dropping advanced configuration or running stale valid data. Covers spec sections 4-5 and AC 3-5, 8-10, 18, 21, 23, 26. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: durable draft identity and editing authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Invalid raw JSON and incomplete control strings retain exact text and editing authority; switching views cannot discard pending edits or enable Run, Pin, or Save on an older valid document.
- [x] #2 Known control edits patch only their documented paths in ADR-078 flat bodies; unknown metadata, classifier rules, advanced options, and ordered operations survive Controls/JSON/import/export round trips.
- [x] #3 Stable candidate identities support editable B, deliberate pin or replacement of A from a current completed B result, correct staleness, and immutable captured run inputs; v1 rejects more than two candidates.
- [x] #4 Sample and template replacements and pinning are undoable; every recovery-relevant edit increments revision while profile, epoch, and immutable identities prevent cross-session mutation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements the approved draft identity and editing authority boundary. 1. Read the Task 2 brief and existing execution values. 2. Write failing lossless authoring, candidate, snapshot and epoch tests. 3. Implement detached serializable state and pure transitions, retaining invalid raw text and last-valid state separately. 4. Run targeted tests, lint and changed-code formatting; self-review and independent review. 5. Record implementation notes and acceptance evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Current task-level status: Done after independent review; earlier pending-review
statements below are historical. The final correction adds structurally validated
known presentation fields, raw `record_fields.tags_text`, canonical capture without
editor rewriting, and retention of the last successful Previous result across failed
reruns. Existing ADR-118 applies; no schema change. Final correction evidence and
remaining independent review: `.superpowers/sdd/2026-09-04-chunking-lab/final-fix-report.md`.

<!-- SECTION:NOTES:BEGIN -->
Implemented frozen publication models and pure copy-on-edit Lab transitions under the existing ADR-118 boundary. Raw invalid JSON now retains the last-valid parsed document separately; incomplete controls retain exact pending strings with sole editing authority, and explicit discard/Undo restores prior state without normalizing unknown data. Known control paths patch only their flat-body locations, preserving metadata, classifier rules, incompatible options, and operation order.

Added exact UTF-8 sample identities, stable editable-B/frozen-A candidates, deliberate baseline replacement, source/template/pin undo snapshots, and view-only revisions that do not consume content undo. Run capture freezes sample, prepared recipe/default/runtime identity, and loaded record ID/UUID/version plus authored fields. Because `capture_batch` is pure, the minimal `install_batch` transition publishes its immutable manifest before `accept_result` applies strict epoch/batch/request fences. Current/previous outcomes and sample/recipe staleness prevent a failed or old result from standing in for current input.

Publication boundaries detach and revalidate nested JSON values. Ordinary draft/view transitions use shallow session replacement and reuse immutable sample/result maps, avoiding copies of retained large reports on each edit. No Media DB schema, global validator, vendored engine, provider, or UI boundary changed. Targeted state/preflight/execution/runtime tests pass; independent review remains pending while the task stays In Progress.

ADR required: yes. ADR path: `backlog/decisions/118-chunking-lab-local-execution-and-recovery.md`. Reason: this directly implements the accepted authoring identity, immutable run-input, and recovery-undo contracts without making a new architectural choice.

Review fix: undoing a newly pinned A now invalidates any retained batch manifest that captured that candidate before removing it. This keeps the returned session publishable, fences both subsequent A/B completions as inactive, and preserves Undo even after a completed manifest remains retained. The later worker coordinator must observe this batch-to-`None` transition as a cancellation request and terminate active work/queued members; the pure state boundary already prevents their late results from being accepted.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Formerly TASK-31422; moved to TASK-31639 during the user-approved
2026-09-05 pre-push bookkeeping correction. Upstream dev independently uses
31421–31424; the complete Lab chain moved together to preserve dependency
ordering without changing upstream tasks. Original creation dates, acceptance
and implementation history are retained. Historical commits and ignored review
artifacts retain the old IDs; current references use the new IDs. See
Docs/Chunking_Lab_Verification.md for the complete mapping and provenance.
