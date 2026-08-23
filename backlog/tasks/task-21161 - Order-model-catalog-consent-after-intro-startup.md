---
id: TASK-21161
title: Order model catalog consent after intro startup
status: In Progress
assignee: []
created_date: '2026-08-23 15:09'
updated_date: '2026-08-23 15:16'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix startup ordering so an unconfigured model-catalog consent decision appears after the splash/intro, remains topmost and actionable, and reveals a usable initial app screen after Yes or No.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Splash/intro completes before model-catalog consent is shown.
- [ ] #2 Unrecorded consent produces one topmost Yes/No modal that cannot be buried by initial-screen startup.
- [ ] #3 Either consent choice dismisses to a usable initial app screen while preserving the existing persistence and refresh semantics.
- [ ] #4 Focused regression coverage drives the real splash-enabled screen stack and fails under the pre-fix ordering.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a splash-enabled full-app Textual Pilot regression that forces the real consent modal under run_test(), pins setup complete, suppresses project-skills discovery, verifies the ordered screen stack, and exercises Deny.
2. Confirm the regression fails because the current on_mount scheduling lets the initial screen cover consent.
3. Move the existing startup scheduler call from on_mount to the end of _push_initial_screen without changing ADR-020 gates, persistence, callbacks, or refresh workers.
4. Run the complete focused consent, catalog-wiring, and first-run startup tests; run Ruff and git diff --check; self-review the diff.
5. Check acceptance criteria, add implementation notes, and set the task Done only after verification.

ADR required: no
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md
Reason: This is a lifecycle ordering bug fix that preserves ADR-020 storage, consent, and network boundaries.
<!-- SECTION:PLAN:END -->
