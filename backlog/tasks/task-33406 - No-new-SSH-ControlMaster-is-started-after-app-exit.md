---
id: TASK-33406
title: No new SSH ControlMaster is started after app exit
status: To Do
assignee: []
created_date: '2026-09-28 20:30'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After app exit has closed every session and master, a straggler tool thread's one-shot call still runs ensure_master and can start a detached ControlMaster that outlives the app until ControlPersist expires. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After the master manager is closed at app exit, no remote call starts a new ControlMaster or runs a master health check
- [ ] #2 A straggler call after app exit still gets a typed result (it connects directly)
- [ ] #3 Tests that exercise app exit do not leave a closed manager for later tests in the same process
<!-- AC:END -->
