---
id: TASK-192
title: De-flake library note-conflict shell test under CPU contention
status: To Do
assignee: []
created_date: '2026-07-12 14:05'
labels:
  - follow-up
  - test-flake
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text intermittently fails under concurrent CPU load (passes in isolation). Widen its polling/timeout budget or replace the timing loop with condition-based waiting. Discovered during task 159; unrelated to that diff.
<!-- SECTION:DESCRIPTION:END -->
