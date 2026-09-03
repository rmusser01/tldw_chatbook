---
id: TASK-28232
title: Dockerfile for containerized chatbook
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - packaging
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C39, promoted by TASK-26041: genuinely missing, trivial to add. A slim image running the TUI (and textual-serve) with the config/data dirs as volumes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 docker build succeeds from the repo root and docker run reaches a working TUI via textual-serve
- [ ] #2 Config and user-data paths are documented volumes
- [ ] #3 The image installs only core dependencies (extras opt-in via build arg)
<!-- AC:END -->
