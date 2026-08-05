---
id: TASK-1330
title: Prove app_started is never emitted before the persistent sink installs
status: To Do
assignee: []
created_date: '2026-07-29 01:45'
labels:
  - logging
  - observability
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-1240 covers two disjoint halves: a synthetic caller reaching a real sink (Tests/test_persistent_log_is_not_empty.py), and monkeypatched production callers proving persist_event is invoked (Tests/App/test_app_lifecycle_events.py, Tests/Scheduling/test_scheduler_observability.py). No test composes a real production emitter with a real installed sink, so an ordering regression — app.py emitting app_started before Logging_Config installs the persistent sink — would pass both existing halves while the event is silently dropped, reproducing the original TASK-1240 failure mode (machinery works, caller calls, file stays empty) in a new form. Today the correct order holds only by entry-point accident (app.py sets _early_logging_initialized before run()), and that ordering is untested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test boots the real app (or an equivalent composed real-emitter-plus-real-sink harness) with the persistent sink installed via the normal startup path, and asserts app_started reaches the persistent log file on disk
- [ ] #2 The same test, run against a deliberately reordered startup where app_started fires before the persistent sink installs, fails — demonstrating the test actually discriminates the ordering rather than passing regardless
- [ ] #3 The existing two-half coverage (Tests/test_persistent_log_is_not_empty.py synthetic-caller guard and the Tests/App + Tests/Scheduling monkeypatched-caller guards) continues to pass unchanged
<!-- AC:END -->
