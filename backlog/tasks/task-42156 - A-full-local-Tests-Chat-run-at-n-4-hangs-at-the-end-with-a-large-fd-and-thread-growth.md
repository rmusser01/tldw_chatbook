---
id: TASK-42156
title: "A full local Tests/Chat run at -n 4 hangs at the end with a large fd and thread growth"
status: To Do
assignee: []
created_date: 2026-11-20 22:23
updated_date: 2026-11-20 22:23
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/conftest.py
- Tests/Chat/
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A consolerun follow-up (round 18). It is reported from that stream's full local runs and was not rerun here, because a full `Tests/Chat` run takes over 15 minutes.

A full local `Tests/Chat` run at `-n 4` hangs at the end. One worker sits idle for about 15 minutes, `fd_leak_sentinel` (`Tests/conftest.py:717`) reports +1,501 file descriptors, and each worker holds 50-180 threads.

The controllers the Console rigs never shut down (filed separately) are a likely contributor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The hang is reproduced, or shown absent, with the run's command, load and timings recorded
- [ ] #2 The cause of the end-of-run idle worker is named from stack or fd evidence, not inferred
- [ ] #3 Either the hang and the fd growth are fixed, with fd_leak_sentinel's growth recorded before and after, or a task is filed for each cause found
<!-- AC:END -->
