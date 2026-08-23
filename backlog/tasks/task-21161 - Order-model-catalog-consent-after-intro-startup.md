---
id: TASK-21161
title: Order model catalog consent after intro startup
status: In Progress
assignee: []
created_date: '2026-08-23 15:09'
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
