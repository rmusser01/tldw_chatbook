---
id: TASK-42251
title: "Unclosed SQLite databases in Tests/Chat leave hundreds of file descriptors per worker, and the controller-shutdown fixture runs refused shutdowns on a closed database"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
- database
priority: low
references:
- Tests/Chat/test_console_agent_bridge.py
- Tests/Chat/test_console_activity_receipts.py
- Tests/Chat/test_console_diff_feedback_delivery.py
- Tests/Chat/test_console_agent_swap.py
- Tests/Chat/test_console_chat_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (TASK-42155 and TASK-42156, round 19). A `Tests/Chat` run leaves SQLite files open: `runs.db` alone holds 170 to 495 file descriptors per worker. The sources named are `Tests/Chat/test_console_agent_bridge.py`, `test_console_activity_receipts.py`, `test_console_diff_feedback_delivery.py` and the `_controller` rig in `test_console_agent_swap.py`. Separately, the controller-shutdown fixture (`shut_down_console_controllers`, on `fu/r19-consoleb`) runs its refused shutdowns against a `CharactersRAGDB` the test body has already closed (see `Tests/Chat/test_console_chat_controller.py:8195`, where the body closes `chat_db` in its `finally`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tests and rigs that leave databases open are named, with their descriptor counts
- [ ] #2 A `Tests/Chat` run at `-n 4` holds no `runs.db` descriptors past each test, measured and recorded
- [ ] #3 The controller-shutdown fixture's shutdowns run before the test body's database is closed, or the order is documented in the fixture and pinned by a test
<!-- AC:END -->
