---
id: TASK-34928
title: Stop-replaying-the-whole-log-buffer-synchronously-at-Logs-mount
status: To Do
assignee: []
created_date: '2026-10-13 14:52'
labels:
  - performance
  - ui-perf
  - wave-3-screen-switch
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
logs_screen.on_mount calls logs_window.load_from_app() synchronously (logs_screen.py:73), replaying the entire in-memory log buffer into widgets on every Logs visit. The buffer grows for the whole session uptime, so the cost of opening Logs grows the longer the app has been running.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Logs mount is O(viewport) not O(buffer) — replay chunked or deferred to a worker,Latest log lines render within one refresh of mount (test)
<!-- AC:END -->
