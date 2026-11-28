---
id: TASK-42653
title: "Fork children inherit the app's termination handler, so terminate() may not end Chunking's regex child"
status: To Do
assignee: []
created_date: 2026-11-28 17:29
updated_date: 2026-11-28 18:06
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
Found while fixing TASK-42651 (round 29, 2026-11-28), which showed that a fork child inherits the parent's Python-level SIGTERM handler. That task gives the test guard's fork children the default SIGTERM back, and blocks the signals across fork.

The app has the same shape:
- `Utils/app_shutdown.py::install_termination_handlers` installs `_handle_termination_signal` for SIGTERM and SIGINT at startup (`app_entry.py`). It is a graceful shutdown on the first signal: it arms a grace timer, reads configuration and starts the app's shutdown.
- `Chunking/engine/strategies/ebook_chapters.py::_run_finditer_process` forks a child (`mp.get_context("fork")`) to run a regex with a deadline. On timeout it calls `terminate()`, then `join(1)`, then raises.

So the regex child inherits the app's handler. The regex engine checks for signals while it matches, so the handler runs promptly in the child. But fork copied the parent's shutdown state with the app and its loop registered, so the handler takes its graceful path there instead of the default action. It:
- counts the signal;
- arms the child's own exit watchdog (the backstop, then `[general] shutdown_grace_seconds`);
- reads configuration;
- queues `_request_app_exit` on the copied loop with `call_soon_threadsafe` (nothing runs that loop in the child, and the call writes a wake byte into the self-pipe it shares with the parent);
- logs, and returns.

So `terminate()` does not end the child. It keeps running the regex past `join(1)` until its watchdog fires, or until a second SIGTERM (multiprocessing sends daemon children one at exit) takes the handler's hard-exit path. A stdlib-only model of this (a handler that returns, then `re.match(r"(a+)+$", "a"*40 + "b")`) stayed alive after `terminate()` and `join(1)`; one that raises `SystemExit` ended at once with 143. Taking locks in the fork of a multi-threaded app is a further, untested risk: a lock another thread held at fork never releases in the child. Not reproduced in the app itself yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reproduction (or a measured absence) shows whether a terminated `ebook_chapters` regex child outlives its `join(1)` with the app's handler installed
- [ ] #2 If it can, fork children of the app get the default SIGTERM and SIGINT disposition, and the owning process keeps its graceful handler
- [ ] #3 A test pins that a fork child of a process with the app's handlers ends by SIGTERM promptly
<!-- AC:END -->
