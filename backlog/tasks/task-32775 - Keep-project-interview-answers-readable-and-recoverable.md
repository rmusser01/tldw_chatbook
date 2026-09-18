---
id: TASK-32775
title: Keep project interview answers readable and recoverable
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 07:27'
updated_date: '2026-09-18 07:57'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Project-context interviews must keep the current question and typed answer readable and retain failed submissions for correction without changing profile commit authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused answer entry paints the current question and entered value in compact and wide dark and light layouts.
- [x] #2 Rejected or failed submissions retain the draft answer for correction; accepted answers clear once and advance without duplicate submission.
- [x] #3 Keyboard cancellation and final review preserve the already-created workspace and commit only selected reviewed context.
- [x] #4 Targeted regressions and real private-profile native journeys verify focus, persistence, and clean shutdown with unchanged default profiles.
- [x] #5 Closing final review restores the current draft state and a usable Review action without committing or losing edits.
- [x] #6 Editable payload values in final review remain visible while focused.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md plus ADR-150/161. Reason: repair existing interview presentation and reversible answer handling without changing service, retention or commit authority. 1. Pin missing question/value paint and answer loss with failing mounted keyboard tests. 2. Correct token-backed interview geometry/focus and clear input only after accepted answer; retain failure recovery. 3. Verify existing interview/coordinator/handoff cases and review exact changes. 4. Run real private workspace creation and fixed interview cancellation/review/save in compact/wide dark/light, inspect captures and lifecycle, update ledger and draft PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept the question and focused answer/review values readable using existing tokens; failed answers remain editable and successful acceptance clears once. Successful review transitions are retained locally so Close/reopen preserves applied edits even when resume is unavailable. Added ten keyboard/state regressions, rebuilt modular CSS, and recorded independent review plus 139 distinct targeted cases (10 new, 84 related, 14 handoff, 31 governance; final nine callback cases overlap the related run). Four final native size/theme cells and 20 inspected captures verify real private workspace cancellation, failed-answer retry, edited selected-only encrypted context saves, runtime-disabled behavior and normal lifecycle. Twelve databases healthy; exact source hashes and unchanged default fingerprints recorded in Docs/superpowers/qa/2026-09-18-settings-project-interview/README.md. Scoped Ruff/format and backlog-ID checks pass; four legacy Ruff diagnostics unchanged. No full suite or provider execution. Existing ADR-102/150/161 apply, no new ADR. Native setup explicitly initializes Tool Profiles; the separately recorded Persona audit retains cold provisioning, valid-ID collisions and pagination defects as next repairs. Updated completion ledger; wider Personal Context and durable/adaptive interview gates remain open.
<!-- SECTION:NOTES:END -->
