---
id: TASK-14912
title: Bound every "await a signal a background task must set" across Tests/UI
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 00:30'
updated_date: '2026-08-11 04:23'
labels:
  - tests
  - test-infrastructure
dependencies: []
priority: high
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
- [x] #1 The `_wait_for_background_signal` / `_await_background_task` helpers live somewhere shared rather than in one test file, so new tests get the bound by default
- [x] #2 Every `Tests/UI` site that awaits a signal a `create_task` coroutine must set is bounded, enumerated by a checkable method (AST, not grep) rather than an asserted sweep
- [x] #3 A background task whose exception is swallowed surfaces that exception as the test failure, instead of the test hanging
- [x] #4 Each file that contained a bounded site has its full pass count read and recorded — a file that has ever contained a hang has an unknown pass count until it is re-run whole
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the existing _wait_for_background_signal/_await_background_task in test_screen_navigation.py; find where Tests/UI keeps shared helpers (app_factory.py precedent).
2. Move the helpers to Tests/UI/background_signals.py, add a timeout-only wait_for_signal for product-owned background work, and re-export the old private names from test_screen_navigation.py.
3. Enumerate every at-risk site by AST (not grep): unbounded 'await <asyncio.Event>.wait()' that either follows a spawn in the same scope or sits in a module-level test body. Ship the enumerator as a guard test so new tests inherit the bound.
4. Classify each hit: at-risk (bind) vs structurally safe (Textual Worker handle / retained-operation handle / stub awaiting a release the test sets) and say why.
5. Bind every at-risk site, file by file, so a hang in one file does not cost the others' results.
6. Break-and-observe: deliberately break two stubbed seams and show the test fails naming the real exception in seconds rather than hanging.
7. Mutation-check the shared helper itself (break the re-raise; confirm a bound test stops naming the cause).
8. Run each touched file WHOLE and record the READ pass count; fix or report any previously-hidden failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the task-3316 bound out of one test file and made it the enforced default across
`Tests/UI`, then bound the 66 sites an AST sweep found.

**Where the helpers live.** `Tests/UI/background_signals.py`, following the
`Tests/UI/app_factory.py` precedent (task-1458 pulled `_build_test_app` out of this same
`test_screen_navigation.py` into a purpose-named module imported by 90+ test modules).
Rejected `Tests/UI/textual_test_helpers.py` -- it is topically right but has **zero
importers repo-wide**, so parking the bound there would not give new tests anything --
and `Tests/UI/conftest.py`, whose contents are fixtures, not call-site helpers.
`test_screen_navigation.py` now imports the shared versions under the original private
names (`_wait_for_background_signal` / `_await_background_task`), so in-flight branches
that import them from that module keep working.

A third helper, `wait_for_signal(event, what=...)`, was added for the case the original
pair did not cover: the coroutine is launched by the *product* (a Textual `run_worker`,
or a `create_task` inside the app), so the test holds no task handle to inspect. It
cannot name the underlying exception, but it turns a process-killing hang into a named
failure in seconds. 31 of the 66 sites are this shape.

**The enumeration (AC#2).** `Tests/UI/test_background_signal_bounds.py` is both the
checkable method and the guard: it ASTs every module under `Tests/UI` and fails on any
`await <x>.wait()` where `<x>` resolves to an `asyncio.Event` (matched by full expression
or by attribute name, so `self.started = asyncio.Event()` catches `service.started.wait()`)
that is neither wrapped in a bound nor structurally safe. A wait is reported when a
spawn (`create_task` / `ensure_future` / `run_worker`) precedes it in the same scope, or
when it sits in a module-level `test_*` body. Deliberately NOT reported, with tests
pinning each: a stub awaiting a release the test body sets (inverse shape -- the setter
is the test); a Textual `Worker.wait()` or `RetainedPushOperation.wait()` (both await the
task and re-raise, so neither can strand); a `test_`-prefixed *method* on a service fake;
anything already inside `asyncio.wait_for` / `asyncio.timeout` / the shared helpers.

Grep could not have done this: it cannot separate a bounded wait from an unbounded one,
cannot type the receiver, and cannot see scope. The reported lead counts were also wrong
in both directions -- `test_library_file_notes_git_push.py`'s 21 bare `.wait()`s are all
retained-operation handles (safe), while the sweep found two files the lead list never
mentioned, `test_speech_playground_pane_lifecycle.py` (16) and
`test_speech_playground_pane.py` (6), both invisible to a `create_task`-only search
because they use `run_worker`.

**What changed.** 66 sites across 8 files: personas_workbench 21, speech_playground_pane_
lifecycle 16, console_prompts_modal 11, speech_playground_pane 6, console_native_chat_flow
4, console_parallel_runs 4, library_prompt_collections 3, library_file_notes_git_push 1.
35 got the task-aware `wait_for_background_signal`; 31 got `wait_for_signal`.

**Proof (AC#3).** With the bound in place, breaking a stubbed seam the incident's way
(reading `.kind` off a value that is still `None`) fails
`test_console_workspace_switch_refresh_is_not_dropped_during_inflight_sync` in **3.1s**
with the real `AttributeError: 'NoneType' object has no attribute 'kind'`, re-raised
through `background_signals.py`. The same break with the bound removed hangs: at the
25s timeout pytest dumped stacks showing only `MainThread` in `selectors.select` -- it
never named the test -- and terminated the process with zero tests reported. Three more
sites broken the same way (personas, prompts_modal, prompt_collections) failed in
0.7-1.3s; in those the product caught the `AttributeError` internally, so the helper's
"finished without signalling" branch fired instead of the re-raise. Mutation checks:
disabling `task.result()` makes the naming demo degrade to the generic message; forcing
`_is_bounding` to `True` reds `test_rule_detects_the_incident_shape`; reverting one bound
site reds the sweep with that exact line.

**Pass counts, read whole (AC#4).** background_signal_bounds 7/7 (new);
console_parallel_runs 28/28; library_prompt_collections 47/47; git_push 59/59;
console_prompts_modal 75/75; speech_playground_pane 51/51;
speech_playground_pane_lifecycle 123/123; screen_navigation 126/126;
console_native_chat_flow **291 passed / 18 failed**; personas_workbench **310 passed /
2 failed**. All 20 failures were reproduced on the pristine HEAD (`eb9708cc4`) copy of
each file, so none is caused by this change and none was hidden by a hang. They are
stale-contract breakage from the screen-decomposition work -- e.g.
`AttributeError: 'ChatScreen' object has no attribute '_ensure_active_console_session_
settings'` (the seam moved to `ChatScreen._session`) -- reported rather than fixed here,
since repairing 20 unrelated console/persona contract failures is outside this task's ACs.

**Files:** added `Tests/UI/background_signals.py`,
`Tests/UI/test_background_signal_bounds.py`; modified `Tests/UI/test_screen_navigation.py`
(helpers removed, re-exported from the shared module) and the 8 files listed above;
added a lesson to `backlog/docs/lessons-testing-evidence.md` correcting TASK-1466's
"the timeout stack dump names it".
<!-- SECTION:NOTES:END -->
