---
id: TASK-21103
title: >-
  Defer Persona_Buddy so it stops dragging Persona_Visual and PIL onto the boot path
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - imports
  - persona-buddy
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21103).

`Persona_Buddy` (eager at app.py:393) drags 93% of Persona_Visual (6,633 LOC) via
`controller.py:18,23` and `rendering.py:11-13` - the latter imports the tree for a single int
constant. This chain puts `PIL.Image`/`PIL._imaging` on the boot path: measured 1.276 s of the
3.10 s cold app import (41%). Both consumers already tolerate absence (`app.py:8582`,
`console_runtime.py:468,519`); the lazy-service house pattern is `_build_rag_admin_services`
(app.py:6054-6124).

## Acceptance Criteria

- [ ] `import tldw_chatbook.app` no longer imports Persona_Visual or PIL - pinned by a sys.modules assertion test
- [ ] The buddy controller is constructed lazily at first feature use; enabling/using Persona Buddy still works end to end
- [ ] Cold and warm importtime before/after numbers recorded in the task
