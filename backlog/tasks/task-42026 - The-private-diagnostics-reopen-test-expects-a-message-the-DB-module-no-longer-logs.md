---
id: TASK-42026
title: "The private-diagnostics reopen test expects a message the DB module no longer logs"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:36
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
- [x] #1 The test still observes a reopen diagnostic (the message the module logs today, or a restored one) and still checks it leaks no path or client name, and it passes
<!-- AC:END -->

## Implementation Plan

1. Find what the Media DB logs on a reopen today (`_get_thread_connection`) and when the old warning went (b5251e9a6e).
2. Assert the message the module logs today on the reopen, keep the path and client-name leak checks, and show the reopen happened.
3. Show red under temporary source edits (message removed; path added), then green.

## Implementation Notes

Cause: test drift. b5251e9a6e (TASK-32628) rewrote the Media DB's liveness ping and dropped the `logging.warning("Media database connection was closed; reopening.")`; the reopen now logs only `logging.debug("Media database connection opened.")` from `_get_thread_connection`. No product change: the test asserts that message, keeps both leak checks (database path, `test-client`), and adds `db.get_connection() is not connection` so the open line it observes is the reopen's.

Red/green: red at base on the old message (`1 failed, 8 passed`). Under temporary source edits, run through pytest and reverted: renaming the open message failed the new assertion (`assert 'Media database connection opened.' in ...`), and appending `self.db_path_str` to it failed the path check. Green after: the file 9 passed at -n 4.

Files: `Tests/DB/test_client_media_debug_logging.py`.
