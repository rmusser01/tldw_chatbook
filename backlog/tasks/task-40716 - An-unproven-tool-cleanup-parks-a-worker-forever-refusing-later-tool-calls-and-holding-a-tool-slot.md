---
id: TASK-40716
title: "An unproven tool cleanup parks a worker forever, refusing the run's later tool calls and holding an app-wide tool slot"
status: To Do
created_date: 2026-11-13 06:45
labels:
- follow-up
- agents
- console
- bug
priority: high
references:
- tldw_chatbook/Agents/agent_service.py
- tldw_chatbook/Agents/execution_capacity.py
- Tests/Chat/test_console_ssh_custody.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** When a tool operation's cleanup cannot be proven, the code keeps the worker alive on purpose by waiting on an event or future that nothing ever completes. The intent is that a resource that may still be live is never reused.
- Every later tool call in that run is then refused with `previous_tool_still_running`, because `Agents/execution_capacity.py:336-337` refuses while any tool operation is stopping.
- The parked operation keeps counting toward the app-wide `max_tool_workers` until the app restarts.

This happens on every platform with the workflow feature off. A probe on 2026-11-12 (dev e0186fa10a) confirmed both effects. The five `Event().wait()` parks arrived with #3480.

**The parks:**
- `Agents/agent_service.py:2254-2258`, `retain_unknown_return()`. It is reached from :2322, :2333, :2369, :2371 and :2373 when a source scope or `operation.finish()` exits other than through the tool's own error. Ordinary Console tool calls set that scope (`Chat/console_agent_preparation.py:2306`).
- `Chat/console_raw_cli_custody.py:299-305`, `retain_unproven()`, for Raw CLI.
- `Tools/raw_cli_executor.py:1210-1213`, in `activate()`'s finally when the physical stop is not proven.
- `Tools/workspace_tool_executor.py:1622-1630`, in SSH `activate()`'s finally.
- `Workspaces/change_revert_ownership.py:292-294`, `retain_unknown()`, for change-review revert.
- `Workspaces/native_git.py:63-71`, `while True: await asyncio.shield(never)` when native Git cleanup proof fails.

**Tests that pin the park:**
- `Tests/Chat/test_console_ssh_custody.py:1217-1220` asserts that the worker is alive and `tool_workers == 1`.
- `Tests/Chat/test_console_raw_cli_custody.py:552`.
- `Tests/Workspaces/test_native_change_review_git_custody.py:302,362`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One unproven cleanup no longer makes every later tool call in the same run fail with `previous_tool_still_running`. Later calls either run, or the run ends in a terminal state that names the unproven cleanup.
- [ ] #2 The app-wide tool-worker slot held for an unproven operation is returned once the operation is proven finished or the user explicitly abandons it, without restarting the app.
- [ ] #3 No thread or task waits on an event or future that nothing can complete, and the app can still exit while a cleanup is unproven.
- [ ] #4 A resource whose cleanup is unproven is still never reused while it may be live. The SSH, Raw CLI and native Git tests assert the new behaviour instead of a living parked worker.
<!-- AC:END -->
