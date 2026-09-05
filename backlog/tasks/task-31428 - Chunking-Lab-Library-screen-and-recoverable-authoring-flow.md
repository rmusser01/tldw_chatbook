---
id: TASK-31428
title: Chunking Lab - Library screen and recoverable authoring flow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 05:12'
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
- [x] #1 Library, palette, and eligible local extracted-text actions open a lazy dedicated Chunking Lab route and return to the opener; direct entry bypasses starter filtering and never creates a Settings or global destination.
- [x] #2 Users can paste, load UTF-8 text, or copy local extracted Library text; edit method controls or JSON; preview B, pin A, compare, and save either locally with no source mutation.
- [x] #3 Reopen, autosave failures, conflict, cancellation, template import/export, recovery export/restore/Undo restore, and confirmed Clear have working visible actions and truthful status; profile switching never mixes sessions.
- [x] #4 Keyboard-only flows work at 80x24, 120x40, and 160x50; typing r/p/s is safe in editors, F6/F1 follow global conventions, and the footer advertises only working actions.
- [x] #5 Saved templates immediately refresh the ingest picker; TASK-24404 is reconciled without a duplicate editor, v2/v3 actions are absent, and real isolated-profile live verification plus targeted regression evidence is recorded.
- [x] #6 Pure record-field edits and saved-record association preserve invalid JSON, pending controls, undo semantics and captured provenance; unsaved pinned records retain authored fields without fabricated catalog identity, and late saves cannot attach to an unrelated draft.
- [x] #7 Fallback to a previous valid recovery checkpoint is visibly explained through a read-only coordinator warning without exposing private sample/path details.
- [x] #8 Exclusive result preparation/inspection workers can be canceled before starting without leaking an unawaited coroutine.
- [ ] #9 Catalog save acknowledgments identify exactly the revision written by that operation; an intervening peer update conflicts with the next unchanged local save.
- [ ] #10 Malformed known recovery presentation fields are refused before replacement, while valid historical unsupported results remain readable after restore and reopen.
- [ ] #11 Recovery import is inspectable after initial local-store failure and requires a validated summary plus explicit Replace current session confirmation before writable replacement.
- [ ] #12 Users can explicitly inspect current and retained Previous output per candidate, including after failed or pending reruns, without substituting old output into current comparisons.
- [ ] #13 Unfinished raw tag input survives ordinary renders and reopening, and deliberate save produces the intended separate tags.
- [ ] #14 Final review refinements cover lazy screen workers, builtin Save-as-new default, visible sample provenance, preservation-only template export, and accurate final documentation status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: direct correction of the approved canonical-save, recovery and authoring contracts; no new storage owner/runtime or policy. Original eight-step Task8 plan completed and reviewed. Final correction wave: 1. Read final-review.md, approved constraints and corresponding existing tasks/source. 2. TDD for exact catalog-write acknowledgment and post-commit peer interleaving; validate known recovery UI shapes while preserving opaque authored data. 3. Historical output readability and explicit per-candidate Previous inspection. 4. Read-only import inspection on failed local load, validated summary and explicit replacement confirmation. 5. Preserve unfinished raw tags, lazy screen workers, builtin copy default, sample provenance and export without runnable admission. 6. Focused amended-code service/state/recovery/UI tests, no full suite or new visual polish loop. 7. Reconcile docs/ADR/plan status, append exact evidence and self-review, one scoped final re-review; no merge/push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Library-owned local recoverable authoring screen under ADR-118 (backlog/decisions/118-chunking-lab-local-execution-and-recovery.md), with async single-flight profile ownership, exact local text handoff, serialized lossless edits, captured A/current B saves, canonical ingest refresh preserving selection, explicit recovery transfer and guarded navigation/quit. Approved narrow extensions add pure record-field/save-lineage transitions, read-only fallback warning and lazy/teardown-fenced ResultsRegion workers. TASK24404 was re-read unimplemented and archived with ADR/design note, not marked Done. User workflow/privacy/runtime limits are in Docs/Chunking_Lab.md. Final affected UI gate:36 passed45.17s; prior release selection103 passed with the subsequently fixed result teardown failure. Full requested integration:294 passed33 failed; exact BASE replay reproduced30, while3 startup/navigation differentials remain unqualified review concerns (not baseline-proven). Exact commands, RED/GREEN chronology, real SQLite/child/crash/failed-write/fresh-profile evidence, two accepted three-size viewport rounds, static audits and AC mapping: .superpowers/sdd/2026-09-04-chunking-lab/task-8-report.md. Scoped Ruff/format/compile/whitespace checks passed; no full sweep or unrelated repair. Status and AC acceptance remain In Progress pending independent review.

Task-level independent review complete after fix cd3e13926a: final-render edit ownership now rechecks the queue, with deterministic RED3/GREEN3 and final screen/recovery27 passing. Re-review found no new blocking issues. Controller explained the three startup differentials through active7-second splash versus readiness polling; exact3 no-splash intervention passes, untouched BASE reproduces same active-splash boundary. Original integration remains non-green; no blanket cold-start qualification. Durable qualifications and verification chronology: Docs/Chunking_Lab_Verification.md. Deferred Minor tag-entry normalization/eager screen workers go to the pending whole-branch review. All Task8 ACs accepted with documented existing-environment/platform limits; ADR118 unchanged.

Reopened for the single final whole-branch correction wave after review at462e5cc30e identified seven Important gaps and five Minor refinements. Task-level review history remains valid, but final completion awaits this correction and scoped re-review. Root startup diagnosis is documented separately and does not authorize unrelated startup repair.
<!-- SECTION:NOTES:END -->
