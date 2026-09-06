---
id: TASK-31823
title: Consume character Chat handoffs on cached Console resume
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-06 06:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A character-chat handoff staged while Console is hidden remains pending on return to the cached ChatScreen because only on_mount schedules its consumer. A scratch tracked-resume-timer probe confirms the missing lifecycle path. Production repair awaits design approval; the UAT also exposes a downstream trace provenance failure after this stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Returning to the same Console consumes and acknowledges one staged Chat handoff and creates exactly the intended character-bound session.
- [ ] #2 First mount and ordered saved-chat startup retain their existing behavior; suspending again stops pending resume timers before hidden consumption.
- [ ] #3 Focused real navigation regressions and relevant full lifecycle files pass without direct test-side handoff consumption.
<!-- AC:END -->
