---
id: TASK-14912
title: >-
  Bound every "await a signal a background task must set" across Tests/UI
status: To Do
assignee: []
created_date: '2026-08-11 00:30'
labels:
  - tests
  - test-infrastructure
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing task-3316, and deliberately raised rather than absorbed.

**The incident.** `test_file_notes_collections_source_transition_blocks_mutation_through_recompose` launched `_select_library_rail_row` as a fire-and-forget `asyncio.create_task`, then waited on an `Event` only that coroutine could set. Its stub returned `None`, which matched `_flush_library_note_save`'s signature when the test was written (`eb036a6a1`). PR #1439 retyped that seam to return `NoteFlushOutcome`; the caller then read `note_flush.kind`, the awaited path died one line in on `AttributeError: 'NoneType' object has no attribute 'kind'`, and because nobody retrieves a `create_task` result **the exception was swallowed and the signal became unreachable** — so `await event.wait()` blocked forever.

Under this repo's configured `timeout_method = thread` a hung test cannot be cancelled: pytest-timeout dumps stacks and **terminates the whole process**, so every test after it in the file silently never runs. That file's real pass count was unknowable for as long as the hang existed, and repairing it revealed three further tests the hang had been hiding.

**Why this is a task and not a one-off fix.** The same shape is widespread in `Tests/UI/`: `test_personas_workbench.py` (20 `create_task` / 42 bare `.wait()`), `test_console_prompts_modal.py` (5/25), `test_console_native_chat_flow.py` (5/22), `test_library_file_notes_git_push.py` (2/21), and others. Each is a process-killing hang waiting for the next seam retype — the failure is not "a test breaks", it is "a whole suite's result becomes a lie".

**The fix already exists**: task-3316 added `_wait_for_background_signal` (returns when the Event is set, otherwise re-raises the background task's swallowed exception) and `_await_background_task` in `Tests/UI/test_screen_navigation.py`. The pattern turns an infinite hang into a 2-second failure naming the real `AttributeError`.

Note also the corollary this incident proved: **task-1466's advice that "the timeout stack dump names the hung test" does not hold here.** A coroutine suspended at an `await` has no frames on any thread stack, so the dump showed only `MainThread` idle in `selectors.select`. Diagnosis required inspecting the task object itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The `_wait_for_background_signal` / `_await_background_task` helpers live somewhere shared rather than in one test file, so new tests get the bound by default
- [ ] #2 Every `Tests/UI` site that awaits a signal a `create_task` coroutine must set is bounded, enumerated by a checkable method (AST, not grep) rather than an asserted sweep
- [ ] #3 A background task whose exception is swallowed surfaces that exception as the test failure, instead of the test hanging
- [ ] #4 Each file that contained a bounded site has its full pass count read and recorded — a file that has ever contained a hang has an unknown pass count until it is re-run whole
<!-- AC:END -->
