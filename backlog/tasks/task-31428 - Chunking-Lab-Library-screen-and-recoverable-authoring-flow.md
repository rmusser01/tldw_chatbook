---
id: TASK-31428
title: Chunking Lab - Library screen and recoverable authoring flow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 03:12'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements approved Library tool ownership, local profile lifecycle and recoverable authoring UI. 1. Read Task8 brief/context and canonical route/Library/save/recovery contracts; reconcile/archive unimplemented TASK24404 via CLI with ADR/design reference. 2. Write failing route/editor/Pilot/recovery flow tests using isolated real temporary SQLite. 3. Add async single-flight app coordinator ownership, Library tool/palette entry and local-media handoff, preserving global destinations. 4. Compose Sample/Configure/Results and dialogs with serialized off-loop editing, explicit full-config validation, pinned/current save semantics and ingest refresh. 5. Wire recovery status, retry/export/restore/undo/Clear and guarded navigation/quit. 6. Verify complete live isolated flows, crash/failed-write/cancel behavior and all three terminal sizes in bounded visual rounds; targeted regressions/static checks only. 7. Document workflow/privacy/limits/platform caveats, self-review and independent review before completion.
<!-- SECTION:PLAN:END -->
