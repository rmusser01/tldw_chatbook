---
id: TASK-26037
title: Enable faulthandler for crash forensics
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
labels:
  - ops
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A hard crash or hang leaves no evidence. Verified on origin/dev: a named grep for excepthook and faulthandler across tldw_chatbook returns three hits, all comments - nothing is installed. Targeted crash guards exist (Utils/text_selection_crash_guard.py, Utils/fd_protection.py) but they catch known cases; an unexpected segfault, a C-extension crash or a deadlock produces nothing to diagnose from. Hermes enables faulthandler to a dedicated log with all-threads dumps plus a signal handler for on-demand dumps. The private log directory already resolves at config.py:7849.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 faulthandler is enabled at startup, writing to a file in the existing private log directory
- [ ] #2 Dumps include all threads, so a deadlock is diagnosable and not just a crash
- [ ] #3 A signal handler allows dumping stacks on demand from a hung process, on platforms that support it
- [ ] #4 The dump file is created with the same restrictive permissions as other private logs
- [ ] #5 The dump file is size-bounded or rotated so it cannot grow without limit
- [ ] #6 Enabling this adds no measurable startup cost - measured and recorded
- [ ] #7 Tracebacks are treated as potentially sensitive: the dump lives under the private log path and is not included in any shareable output
<!-- AC:END -->
