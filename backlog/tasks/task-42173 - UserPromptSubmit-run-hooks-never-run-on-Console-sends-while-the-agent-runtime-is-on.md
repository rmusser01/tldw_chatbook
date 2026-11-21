---
id: TASK-42173
title: UserPromptSubmit run hooks never run on Console sends while the agent runtime is on
status: Done
assignee:
  - '@claude'
created_date: '2026-11-21 05:30'
labels:
  - console
  - hooks
  - agents
priority: high
references:
  - tldw_chatbook/Agents/run_hooks.py
  - Agents/hooks_v2/engine.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in a Linux UAT on dev 230205800c (2026-11-20): every Console send logs `run-hooks: event=UserPromptSubmit ... engine error (exception_type=AgentActivationRequired)`. It is not only log noise. A Console submit holds its agent admission (NativeAgentSources) for the whole submit; `RunHooksEngine.fire_async` runs `fire` on an `asyncio.to_thread` worker that inherits a copy of that admission, and the recovery admission guard refuses copied admissions. `fire` catches the error and returns an empty outcome, so with the agent runtime on (the default, POSIX, non-character chats) no UserPromptSubmit hook ran: none injected context and none could block a send, which the User Guide promises. Both halves arrived in PR #3480 (89ba88b61d, b942afc277); b942afc277 fixed the same hazard for hooks_v2 with `worker_guard()` but missed the legacy engine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A configured UserPromptSubmit hook injects its context on a send made while the submit holds its agent admission
- [x] #2 A blocking UserPromptSubmit hook (exit 2) blocks that send
- [x] #3 A test through the real native engine and admission guard fails on the pre-fix code
- [x] #4 No AgentActivationRequired hook warning is logged on an ordinary Console send in the running app
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red test in Tests/Agents/test_native_legacy_hooks.py: fire_async inside activation.execution(captured sources), context and deny variants.
2. fire_async: wrap `fire` in `activation.worker_guard()` when the native source is attached, as hooks_v2 does.
3. Check the app log on the Linux re-run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix (Agents/run_hooks.py `RunHooksEngine.fire_async`):** when the native source is attached, `fire` is wrapped in `activation.worker_guard()` before `asyncio.to_thread`. The guard captures the submit task's sources, clears the copied state in the worker and admits again there. This is the pattern b942afc277 applied to hooks_v2; the legacy engine was missed. It covers sends and resends, since both go through `fire_async`.

**Test:** `Tests/Agents/test_native_legacy_hooks.py::test_prompt_hook_runs_while_the_submit_task_holds_its_agent_admission` (context and deny) calls `fire_async` inside `activation.execution(sources=captured_sources(None))` with the real native engine. On base the outcome is empty: no context, and the deny variant does not block. The whole file passes (34/34).

**Live (Linux, 2026-11-20):** the UAT app log showed the `engine error (exception_type=AgentActivationRequired)` warning on every send before the fix, and none after (0 across 4 sends). Root cause traced by a read-only investigation (file:line chain in the task description).
<!-- SECTION:NOTES:END -->
