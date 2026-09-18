---
id: TASK-32774
title: Keep imported profile review tied to the current workspace intent
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 06:57'
updated_date: '2026-09-18 07:22'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Imported Tool Profile approval must stay attached to the workspace and assistant selection the user chose, with readable keyboard review and truthful cancellation or stale-policy recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delayed first-bind review cannot reopen after navigation, suspension, workspace return or a newer staged assistant selection.
- [x] #2 Cancellation preserves the saved defaults and staged choices; exact confirmation saves the reviewed defaults once and rejects a changed policy or inventory with visible recovery.
- [x] #3 The first-bind dialog exposes target, policy details and complete keyboard actions in compact and wide dark and light layouts.
- [x] #4 Targeted regressions, existing binding and memory tests, and real private-profile native journeys verify persistence, focus, clean shutdown and unchanged default profiles.
- [x] #5 Persona labels remain literal in the selected workspace assistant status and picker, including bracketed names.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/107-portable-tool-use-packs.md, plus ADR-079/150/161. Reason: preserve existing exact first-bind authority and repair UI review lifetime and presentation only. 1. Preserve the four mounted failing delayed-review journeys and read TASK-28225/29229. 2. Capture Apply intent before worker dispatch and invalidate pending review on navigation or changed staging; exempt only the owned first-bind modal from suspension invalidation and preserve submitted local-save semantics. 3. Verify keyboard review, cancellation, exact persistence and changed-policy recovery with the real private permission store and registry; run affected binding/memory/UI checks and independent review. 4. Inspect native compact/wide dark/light captures, verify process/database/default-profile lifecycle, update the ledger and draft PR. Allocation refreshed all refs: maximum 32773 across 280 refs and 30 worktrees; CLI allocated 32774.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Captured Apply intent before worker dispatch and fenced delayed imported-profile review/token results after navigation, suspension or newer staging. Only the owned review modal preserves that intent; submitted local-save semantics and exact binding authority are unchanged. Failed Clear preserves existing memory acknowledgement and label. Persona names render literally in status and picker. Added ten mounted regressions; retained all existing assistant-default test functions/assertions with one bounded pane-ready wait. 70 distinct targeted cases pass across final/corrective runs. Four final real private-profile native cells produced 20 rendered and inspected captures; cancellation, stale policy refusal and exact persistence verified. Exit 0, absent PID, 11 healthy databases, reacquired lock and unchanged default fingerprints recorded. Scoped Ruff/format checks and independent review found no introduced blocker. No full suite or provider/tool execution. Existing ADR-107/079/150/161 apply; no new ADR. Production change is settings_screen.py; related tests, evidence gallery, completion ledger and async-intent lesson updated. QA: Docs/superpowers/qa/2026-09-18-settings-workspace-profile-review/README.md. Broader Tool Profiles management, Persona auto-creation and project-context interviewing remain separate reviews.
<!-- SECTION:NOTES:END -->
