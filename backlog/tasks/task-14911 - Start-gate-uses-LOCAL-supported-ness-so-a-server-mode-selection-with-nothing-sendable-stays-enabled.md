---
id: TASK-14911
title: >-
  Start gate uses LOCAL supported-ness, so a server-mode selection with nothing
  sendable stays enabled
status: To Do
assignee: []
created_date: '2026-08-11 02:00'
labels:
  - library
  - ingest
  - server
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while closing task-14827, and out of that task's scope (which was the forecast, not the gate).

task-14823 gates Start on a selection with nothing importable, but the predicate is 'the pre-flight found no supported type group' -- a LOCAL verdict. Since task-14827 the forecast knows the server refuses a different set (images have no server media type at all), so a folder of nothing but images now correctly forecasts '0 will be sent to the server - N will fail (unsupported by the server)' while Start stays enabled and every row lands as a failure. That is precisely the guaranteed-failure submit task-14823 exists to prevent, one backend over.

The forecast already carries the answer (will_import == 0 and every staged file refused), so the gate should read it rather than re-deriving supported-ness from type groups -- the same 'one computation' move task-14820 made for the commit and consent lines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing Start in server mode on a selection the server will refuse entirely is blocked, with a gate line naming the reason
- [ ] #2 The same selection in local mode is unaffected, because those files import fine on this machine
- [ ] #3 The gate reads the existing IngestForecast rather than deriving a second notion of what is importable
<!-- AC:END -->
