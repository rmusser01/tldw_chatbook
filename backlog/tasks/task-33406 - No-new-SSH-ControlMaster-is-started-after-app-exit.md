---
id: TASK-33406
title: No new SSH ControlMaster is started after app exit
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 20:30'
updated_date: '2026-09-28 16:48'
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
- [x] #1 After the master manager is closed at app exit, no remote call starts a new ControlMaster or runs a master health check
- [x] #2 A straggler call after app exit still gets a typed result (it connects directly)
- [x] #3 Tests that exercise app exit do not leave a closed manager for later tests in the same process
<!-- AC:END -->

## Implementation Notes

Added `_closed` latch to `SshMasterManager`, set in `close_all()`. Entry guards in `ensure_master()` and `restart_if_dead()` return early when closed. Conftest reset fixture extends to reset the singleton between tests. Two new singleton tests verify the close behavior and manager isolation. Changes in `remote_workspace_transport.py`, `conftest.py` with 50 total tests passing.
