---
id: TASK-40720
title: Rewrite the native Console sync-admission test without profiler tracing, and gate it
status: Done
created_date: 2026-11-13 07:50
assignee:
- '@claude'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_sync_worker_admission.py
- scripts/ui_pr_gate_census.txt
- backlog/tasks/task-39435 - Fix-unawaited-native-Console-UI-synchronization-coroutine.md
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_sync_worker_admission.py` came with TASK-39435 (renumbered from TASK-38614 when PR #3480 was rebased). It pins that native Console sync requests create no coroutine before Textual admits their worker. It observed worker entry and publication scheduling with `sys.setprofile`, matching `f_code` against `ChatScreen._sync_native_console_chat_ui.__code__` and `run_worker.__func__.__code__`. Code-identity tracing like that breaks when a function is wrapped, and it cannot run under a coverage tracer.

On dev beab5c1e4e the file is also not in the UI PR gate, and neither of its tests passes reliably. Measured locally under an isolated profile:
- **The publication test fails in every run, both ways.** Its profiler matched only a `run_worker` call made by the snapshot itself. Since TASK-33620.15.4, `for_screen` always sets `publish`, so the publication goes through `publish_console_presentation` and the match can never happen.
- **The replacement test fails in 4 of 8 runs** with `StopIteration`. A pending whole-sync replay takes the resume's request, so no `console-sync` worker appears.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tests observe worker entry and publication scheduling through the screen's own seams, with no profiler or code-object matching.
- [x] #2 Each test still fails if its scheduler creates the sync coroutine before the worker starts.
- [x] #3 The publication test passes for both a cancelled-before-entry and an admitted publication.
- [x] #4 The file passes 10 consecutive full runs locally and runs in the UI PR gate.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`record_native_sync_entry` replaces the screen's `_sync_native_console_chat_ui` with an async wrapper that records the entering task. The publication test also wraps `screen.run_worker` to keep the `console-context-publication` worker and cancel it at once when asked. Both schedulers read these seams from the screen when they schedule. So a coroutine created before admission would be the wrapper's, and a refused or cancelled worker leaves its "never awaited" RuntimeWarning, which each test already asserts against.

The publication seams are installed after the snapshot exists, so its assertions that it holds the screen's real `run_worker` and sync still hold. The replacement test now waits for no sync and no pending whole-sync replay before each request. It also takes only workers created by its own request.

Verified locally on 2026-11-13 under the isolated profile:
- the file passes 10 consecutive full runs;
- dev's version fails the publication test in every run and the replacement test in 4 of 8;
- mutants that schedule an eager coroutine fail as expected. At `ChatScreen._console_sync_maintenance_resume` the replacement test fails, 2 of 2 runs. At `publish_console_presentation` the cancelled publication case fails, 2 of 2 runs; review saw the replacement test fail there too.

The file joins the UI PR-gate census tail. Review noted that the snapshot's own fallback `self.schedule(self.refresh, ...)` branch (`UI/Console_Modules/console_spend_projection.py:1636-1643`) is unreachable from `for_screen`, which always sets `publish`. It is production code, so it is left for its own change. It supersedes an unpushed 11-09 rewrite that swapped these mounted cases for a fake-screen unit test, losing the replacement, live-failure and publication coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
