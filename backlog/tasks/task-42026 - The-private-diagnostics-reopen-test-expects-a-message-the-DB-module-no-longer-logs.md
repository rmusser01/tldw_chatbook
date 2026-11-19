---
id: TASK-42026
title: "The private-diagnostics reopen test expects a message the DB module no longer logs"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
labels:
- follow-up
- tests
priority: low
references:
- Tests/DB/test_client_media_debug_logging.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41714 (round 17), red at base. `test_closed_connection_reopens_without_private_diagnostics` expects a log message that no longer exists in the DB module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test still observes a reopen diagnostic (the message the module logs today, or a restored one) and still checks it leaks no path or client name, and it passes
<!-- AC:END -->
