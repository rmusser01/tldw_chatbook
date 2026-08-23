---
id: TASK-21161
title: Order model catalog consent after intro startup
status: In Progress
assignee: []
created_date: '2026-08-23 15:09'
updated_date: '2026-08-23 15:51'
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
1. Add a splash-enabled full-app Textual Pilot regression for intro -> Home -> consent, prove the old stack is Screen/Consent/Home, and exercise explicit Deny back to Home.
2. Move startup scheduling from on_mount to the shared post-intro _push_initial_screen path without changing ADR-020 consent persistence or refresh semantics.
3. Add red regressions for the reviewed competing paths: eligible project-skills startup discovery must defer when consent is unrecorded, and completed first-run exit navigation must settle before consent without any implicit None decision.
4. Record the unrecorded-consent startup branch so _push_initial_screen can skip the optional project-skills offer for that launch; sequence completed first-run exit navigation before scheduling consent while preserving no-route and same-Console behavior.
5. Run the complete focused consent, catalog-wiring, first-run startup, and live first-run contract tests; run Ruff and git diff --check; complete independent review.
6. Check acceptance criteria, add implementation notes, and set the task Done only after verification.

ADR required: no
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md
Reason: This is a lifecycle ordering bug fix that preserves ADR-020 storage, consent, and network boundaries; optional startup prompts are deferred rather than introducing a new modal coordinator.
<!-- SECTION:PLAN:END -->
