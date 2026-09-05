---
id: TASK-31428
title: Chunking Lab - Library screen and recoverable authoring flow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 04:35'
labels:
  - chunking
  - chunking-lab
  - ui
dependencies:
  - TASK-31424
  - TASK-31425
  - TASK-31426
  - TASK-31427
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ship the Library-owned single-sample A/B authoring screen by composing the tested execution, editing, persistence, save, and comparison seams. Covers all user-facing spec sections and AC 1-5, 11, 13, 15-17, 20, 23-26. Reconcile the existing TASK-24404 Settings proposal under ADR-118 before landing; ship one authoring surface. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: route ownership, app lifecycle, and cross-module UI integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library, palette, and eligible local extracted-text actions open a lazy dedicated Chunking Lab route and return to the opener; direct entry bypasses starter filtering and never creates a Settings or global destination.
- [ ] #2 Users can paste, load UTF-8 text, or copy local extracted Library text; edit method controls or JSON; preview B, pin A, compare, and save either locally with no source mutation.
- [ ] #3 Reopen, autosave failures, conflict, cancellation, template import/export, recovery export/restore/Undo restore, and confirmed Clear have working visible actions and truthful status; profile switching never mixes sessions.
- [ ] #4 Keyboard-only flows work at 80x24, 120x40, and 160x50; typing r/p/s is safe in editors, F6/F1 follow global conventions, and the footer advertises only working actions.
- [ ] #5 Saved templates immediately refresh the ingest picker; TASK-24404 is reconciled without a duplicate editor, v2/v3 actions are absent, and real isolated-profile live verification plus targeted regression evidence is recorded.
- [ ] #6 Pure record-field edits and saved-record association preserve invalid JSON, pending controls, undo semantics and captured provenance; unsaved pinned records retain authored fields without fabricated catalog identity, and late saves cannot attach to an unrelated draft.
- [ ] #7 Fallback to a previous valid recovery checkpoint is visibly explained through a read-only coordinator warning without exposing private sample/path details.
- [ ] #8 Exclusive result preparation/inspection workers can be canceled before starting without leaking an unawaited coroutine.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: approved Library/profile/recovery UI integration. 1. Read Task8 brief/context, canonical route/Library/save/recovery contracts; re-read and archive unimplemented TASK24404 via CLI with ADR/design reference. 2. TDD with isolated temporary SQLite and mounted route/editor/recovery flows. 3. Async single-flight profile owner, Library/palette tool route and exact local-media handoff; preserve shell destinations. 4. Sample/Configure/Results/dialog composition; serialized off-loop deltas, validation and captured A/current B saves. Approved pure record-field/association extension preserves invalid/pending authority, undo, unsaved captured fields and lineage-fences delayed saves. 5. Recovery labels, retry/export/restore/undo/Clear, navigation/quit checkpoint through coordinator.cancel. Approved read-only coordinator.recovery_warning pass-through explains existing fallback without storage/lifecycle changes. 6. Real process/crash/cancel/failed-write/transfer tests, exact three-size flows and two bounded visual rounds; no full suite. 7. Approved narrow ResultsRegion callable worker submission with regression prevents pre-created unawaited coroutine on cancellation; no results behavior/layout change. 8. Task-derived CSS builder, targeted static/self-review, docs/privacy/runtime limits, exact baseline comparison and independent review; remain In Progress until review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Library-owned local recoverable authoring screen under ADR-118 (backlog/decisions/118-chunking-lab-local-execution-and-recovery.md), with async single-flight profile ownership, exact local text handoff, serialized lossless edits, captured A/current B saves, canonical ingest refresh preserving selection, explicit recovery transfer and guarded navigation/quit. Approved narrow extensions add pure record-field/save-lineage transitions, read-only fallback warning and lazy/teardown-fenced ResultsRegion workers. TASK24404 was re-read unimplemented and archived with ADR/design note, not marked Done. User workflow/privacy/runtime limits are in Docs/Chunking_Lab.md. Final affected UI gate:36 passed45.17s; prior release selection103 passed with the subsequently fixed result teardown failure. Full requested integration:294 passed33 failed; exact BASE replay reproduced30, while3 startup/navigation differentials remain unqualified review concerns (not baseline-proven). Exact commands, RED/GREEN chronology, real SQLite/child/crash/failed-write/fresh-profile evidence, two accepted three-size viewport rounds, static audits and AC mapping: .superpowers/sdd/2026-09-04-chunking-lab/task-8-report.md. Scoped Ruff/format/compile/whitespace checks passed; no full sweep or unrelated repair. Status and AC acceptance remain In Progress pending independent review.
<!-- SECTION:NOTES:END -->
