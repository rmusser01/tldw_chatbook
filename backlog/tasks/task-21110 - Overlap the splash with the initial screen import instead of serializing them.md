---
id: TASK-21110
title: >-
  Overlap the splash with the initial screen import instead of serializing them
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21110).

With the splash enabled (default, 1.5 s), boot is strictly serial: `__init__` -> splash runs
1.5 s doing nothing else -> splash closes -> THEN chat_screen (~20k lines + closure) imports
and composes on the loop (app.py:8343-8386 -> 12611-12642 -> 11166-11222). The screen
pre-importer only starts after `_post_mount_setup`, so it cannot help the first screen. The
1.5 s window is pure wasted overlap on exactly the machines that hurt.

## Acceptance Criteria

- [ ] The resolved initial route's screen module import is kicked off (on a thread) when the splash mounts, so splash time overlaps the import
- [ ] Time-to-interactive with splash on, measured on the isolated-profile probe, improves by roughly the warm import cost of the initial screen; numbers recorded in the task
- [ ] Boot with splash disabled is unchanged; no import races introduced (the existing per-module import lock semantics are relied on, not fought)
