---
id: TASK-31669
title: Ignore Buddy overlay refresh after the screen stack closes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:02'
updated_date: '2026-09-05 18:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent queued app-owned Buddy refresh callbacks from raising during shutdown after the final screen has been removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A queued overlay refresh with no screens returns without querying a screen or creating presentation work.
- [x] #2 Mounted Buddy behavior and the Models test that exposed the teardown failure continue to pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the empty-stack callback with the real Textual harness for enabled and disabled preferences. 2. Return before controller or presentation work when the screen stack is empty. 3. Run Buddy lifecycle and complete Models adoption suites, scoped static checks and review. ADR required: no. ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md (existing). Reason: routine teardown guard preserving the app-owned presentation boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an early empty-screen-stack return before snapshot access or lazy owner allocation. New enabled/disabled real Textual harness cases both failed before the guard (ScreenStackError/attempted worker allocation), then passed; complete Buddy+Models matrix171passed308.43s (/private/tmp/tldw-review-buddy-models-final-20260905.xml), including original VAD teardown case. Full test-file Ruff and app.py Ruff excluding inherited E402, changed-range format, diff whitespace and self-review pass. ADR074 app-owned presentation boundary unchanged; no new ADR.
<!-- SECTION:NOTES:END -->
