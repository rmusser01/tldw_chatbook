---
id: TASK-42653
title: "Fork children inherit the app's termination handler, so terminate() may not end Chunking's regex child"
status: To Do
assignee: []
created_date: 2026-11-28 17:29
updated_date: 2026-11-28 17:29
labels:
- follow-up
- shutdown
- chunking
priority: medium
references:
- tldw_chatbook/Utils/app_shutdown.py
- tldw_chatbook/Chunking/engine/strategies/ebook_chapters.py
- Tests/real_profile_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-42651 (round 29, 2026-11-28). That task showed that a fork child inherits a Python-level SIGTERM handler. CPython runs such a handler only at the child's next bytecode check, so a SIGTERM that lands just before a blocking call (`time.sleep`, for one) can sit unhandled until that call returns. The test guard now gives fork children the default SIGTERM back.

The app has the same shape:
- `Utils/app_shutdown.py::install_termination_handlers` installs `_handle_termination_signal` for SIGTERM and SIGINT at startup (`app_entry.py`). It is a graceful shutdown on the first signal: it arms a grace timer, reads configuration and starts the app's shutdown.
- `Chunking/engine/strategies/ebook_chapters.py::_run_finditer_process` forks a child (`mp.get_context("fork")`) to run a regex with a deadline. On timeout it calls `terminate()`, then `join(1)`, then raises.

So the regex child inherits the app's handler. If the SIGTERM is delayed, or the handler's graceful path runs in the child instead of the default action, the child may outlive the `join(1)` and keep running a possibly catastrophic regex until the app exits. This is not reproduced in the app yet; it is the same mechanism TASK-42651 measured in a test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reproduction (or a measured absence) shows whether a terminated `ebook_chapters` regex child outlives its `join(1)` with the app's handler installed
- [ ] #2 If it can, fork children of the app get the default SIGTERM and SIGINT disposition, and the owning process keeps its graceful handler
- [ ] #3 A test pins that a fork child of a process with the app's handlers ends by SIGTERM promptly
<!-- AC:END -->
