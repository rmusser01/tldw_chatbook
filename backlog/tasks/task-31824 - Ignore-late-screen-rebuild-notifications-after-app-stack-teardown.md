---
id: TASK-31824
title: Ignore late screen rebuild notifications after app stack teardown
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-06 06:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Queued BaseAppScreen ContentsRebuilt events can arrive after app teardown empties screen_stack. The handler accesses self.screen before the existing overlay scheduler empty-stack guard, raising ScreenStackError. The intermittent inventory failure is deterministically reproduced by calling the real handler with an empty stack. Await bounded production guard approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A late ContentsRebuilt notification with an empty screen stack does not raise or schedule presentation work.
- [ ] #2 An active matching screen still schedules reconciliation, while stale nonmatching screen notifications remain ignored.
- [ ] #3 The exact negative regression fails before the guard and passes afterward, with relevant app lifecycle tests verified.
<!-- AC:END -->
