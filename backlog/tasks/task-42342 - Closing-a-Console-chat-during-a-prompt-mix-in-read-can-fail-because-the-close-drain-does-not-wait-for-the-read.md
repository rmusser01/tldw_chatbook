---
id: TASK-42342
title: "Closing a Console chat during a prompt mix-in read can fail because the close drain does not wait for the read"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 19:43
labels:
- follow-up
- console
- bug
priority: high
references:
- tldw_chatbook/Chat/console_prompt_mixin_service.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Backup_Recovery/runtime_producer_lifetime.py
- tldw_chatbook/Chat/console_context_repository.py
- Tests/Chat/test_console_context_presentation_current.py
dependencies:
- TASK-34975
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of the PR #3493 port (TASK-34975, round 20, branch fu/r20-ctxclose). Read at 59cc99e61b (dev 8a55555798 plus the round-20 stack); line numbers are that tree's. Found by reading; not reproduced. The context-presentation fix this builds on (9470ee5528) exists only on the TASK-34975 port until that lands.

**Why it matters.** Closing a chat should end with "Session closed." If a prompt mix-in read's native callback is still running when the close finishes its drains, the close fails with `RuntimeError("A native preparation still owns this session.")`. The chat is then left fenced: it can neither send nor be closed again until the app restarts. TASK-34975's notes (2026-11-11 entry) record the same failure for its sibling, the context-presentation read, reproduced in a Windows UAT close and resume.

**Evidence.**
- Both prompt mix-in reads call `run_preparation_read` with `creator` and `reads` set to their `preparation_owner` and pass no `observers`: the snapshot read at `console_prompt_mixin_service.py:418-438` and the attempt-snapshot read at `:818-833`. `preparation_owner` is the controller (`console_chat_controller.py:21191`). Both reads are session-scoped.
- The runtime's `set_chat_controller` (`console_runtime.py:2255-2260`) calls `observe_preparation_reads`, which attaches only reads that already exist at binding time (`runtime_producer_lifetime.py:122-128`). Later mix-in reads live only in the controller's set.
- The close does wait for the read's issuing task: `session_shutdown_tasks` adds it through `preparation_tasks_for` (`console_chat_controller.py:15010-15017`), and the runtime's close waits for those tasks for up to the 2 s grace (`CONSOLE_SESSION_CLOSE_GRACE_SECONDS`, `console_runtime.py:401`). After the grace it cancels whatever is left without awaiting it, drains ordinary native commits, then drains only its own read set (`:6103-6105`) and calls `finalize_session_close`, which raises while the controller still holds a read for the session (`console_chat_controller.py:14909-14911`).
- Nothing bounds the callback itself. `asyncio.timeout(5.0)` around the mix-in reads does not: `run_preparation_read` shields the executor producer and retires the read only when the callback returns (`runtime_producer_lifetime.py:285-312`). So the failure needs a callback that is still running after the grace and the drains that follow, for example a slow database read.
- On that failure the close leaves `_admission_fenced_sessions` (`console_runtime.py:6027`), `_session_close_generations` (`console_chat_controller.py:14750`) and `_session_close_states` (popped only after the check, `:14914`) set. Admission is then refused with "Console session is closed." (`console_runtime.py:2933-2938`), and a second close is refused too (`console_chat_controller.py:14622-14625`).

TASK-34975 fixed the same gap for the context-presentation read: `console_context_repository.py` now passes `observers=(runtime._preparation_reads,)`. Two fixes were proposed: add the runtime observer at both mix-in sites, or fix the root by also draining the controller's `_preparation_reads` for the closing session in the runtime's close path, which would cover any other controller-owned read with the same gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A session close that starts while either prompt mix-in read (the snapshot read or the attempt-snapshot read) has a native callback still running past the close grace waits for that read to retire, then ends with the normal closed-session outcome instead of "A native preparation still owns this session."
- [x] #2 A test holds each mix-in read's native callback past the close grace, and is shown red on an assertion before the fix and green after
- [x] #3 The fix sits in the shared close path, or the implementation notes list every controller-owned `run_preparation_read` caller the runtime does not observe and why each one is safe
- [x] #4 The new tests run on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Regression test first, in `Tests/Chat/test_console_context_presentation_current.py` beside TASK-34975's sibling, parametrized over both mix-in reads (the snapshot read through `PromptScopeService.read_mixin_records`, the attempt-snapshot read through `get_conversation_mixin_attempt`). Hold the native callback on its worker thread, start `ConsoleRuntime.close_session` with a 0.01 s grace, and wait until the close is draining. Assert the close does not finish while the read is held, a new mix-in read for that session is refused before it registers, and after release the close returns, the session is gone and both read sets are empty. Show it red on dev for the real reason ("A native preparation still owns this session.").
2. Root fix in the shared close path (AC#3): `ConsoleRuntime._close_session_after_voice_drain` also drains the controller's own `_preparation_reads` for the session, immediately before `finalize_session_close`, so the drain checks exactly the set and predicate finalize checks and no await sits between them. This covers every controller-owned read, observed by the runtime or not.
3. Refusal: both mix-in `current()` checks also refuse once the controller has fenced the session for close (`_session_close_generations`), so no new mix-in read registers during a close.
4. Grep every `run_preparation_read(` caller with a controller-owned read set and record which ones the runtime did not observe.
5. Lane: add the test node just before the TASK-34975 node in console-p0-gate's admission-sensitive step, with pins in `Tests/CI/test_derived_artifacts_workflow.py` and `Tests/CI/test_response_rules_critical_gate.py`; measure it with the step's flags.
6. Docs: extend the TASK-34975 close-drain paragraph in `Docs/Architecture/console.md` to cover mix-in reads and every controller-owned read.
7. Run covering suites, preflight and size ratchets through the wrapper; record results.

## Implementation Notes

**Summary.** Root fix in the shared close path. `ConsoleRuntime._close_session_after_voice_drain` now drains the controller's own `_preparation_reads` for the closing session, as well as the runtime's own set, with the existing `drain_preparation_reads`. `finalize_session_close` refuses on exactly `preparation_reads_for(controller._preparation_reads, session_id)`, so the close now waits on the same set and the same predicate (session reads plus app-scoped reads). Close drains that set twice: early, next to the runtime's drain, and again just before finalize, with no await in between. Any controller-owned read is covered, whether or not the runtime observed it and whenever it started.

**Decisions.**
- Deviation from plan step 2. The plan put a single controller drain immediately before `finalize_session_close`. In practice there are two:
  - The early drain sits next to the runtime's drain, before `pending` is recomputed. A cancelled issuing task finishes when its read retires, so it leaves `pending`, and a clean drain still releases the close fences, as in TASK-34975. `test_session_close_drains_a_prompt_mixin_read_and_refuses_a_new_one` pins this with a spy on `release_session_close_fences` (one call). Negative control: with the early drain's controller set removed and only the pre-finalize drain left, both cases fail with `assert [] == ['session-1']`. The close succeeds, but its fences stay latched.
  - The pre-finalize drain (added after review) runs after the hook-review and `close_hooks_v2` awaits, with no await before `finalize_session_close`. A controller read that starts during those awaits is still drained. The live case is the app-scoped Console browser read (`UI/Console_Modules/console_browser_read.py:450`, `session_id=None`), which no session fence refuses. This is what the plan's single drain was meant to guarantee.
- Both drains only read the set. Reads still remove themselves from every set in `run_preparation_read`'s `finally` (success, refusal, cancel, exception). A read in both sets is gone from both once the first loop has drained it, so it is never registered twice or waited for twice.
- New mix-in reads are refused once Close has fenced the session. `ConsolePromptMixinService._closing` checks the controller's `_session_close_generations` (set by `begin_session_close`, under the runtime's admission fence) in both currentness checks. A new snapshot read raises `MixinSourceChangedError`, and a new attempt-snapshot read raises `MixinSnapshotUnavailableError`, before `run_preparation_read` registers anything. An in-flight read's own after-callback check also refuses. This closes the window between the drain and `finalize_session_close` for these reads.
- `run_preparation_read` callers with a controller-owned read set (AC#3 inventory, grep of every call at base ddd3c02052). Every one except the two mix-in reads also registers with the runtime's set when a runtime is bound. `observe_preparation_reads` attaches reads issued before binding. The callers: `console_context_repository.py:1427`, `console_chat_controller.py:10210`, `console_configuration_preparation.py:2451/2808` (and its skill-catalog helpers at :1320/:1425/:1722, whose callers in `console_ordinary_tree`, `console_received_dispatch` and `console_workflow_preparation` pass the runtime set), `console_received_dispatch.py:149/316/370/382`, `console_ordinary_tree.py:157/236/248/339/354/418`, `console_plugin_hook_preparation.py:292`, `console_workflow_preparation.py:951/1226/1469/1482/1514/2694`, `console_workflow_runs.py:353/409/1033/2012/2390`, `console_preparation_reads.py:89`, `console_automatic_submission.py:591`, `console_reference_preparation.py:471` (registered in both directly) and `UI/Console_Modules/console_browser_read.py:450` (app-scoped). The only unobserved ones were the snapshot read (`console_prompt_mixin_service.py:428`) and the attempt-snapshot read (`:827`), so both mix-in kinds are the complete unobserved set, and the test covers both. With the pre-finalize drain, any controller read is covered, whether added later, unobserved, or started while Close awaits its hooks, because nothing can register between that drain's last check and finalize.

**Red/green evidence.**
- Red on dev's source (test only), both cases: `AssertionError: RuntimeError('A native preparation still owns this session.')` at `assert not close.done()`.
- Drain fix without the refusal, both cases: `Failed: DID NOT RAISE <class '...MixinSourceChangedError'>` / `<class '...MixinSnapshotUnavailableError'>` at the new-read step.
- Hook-window case (`test_session_close_drains_a_controller_read_issued_during_its_hook_drain`). It patches `runtime.close_hooks_v2` to start a held app-scoped `run_preparation_read` on the controller's set, like the browser read. At ad2ed289cf, without the pre-finalize drain, it failed with `AssertionError: RuntimeError('A native preparation still owns this session.')` at `assert not close.done()`. It is green with the drain.
- Green: `test_console_context_presentation_current.py` 15 passed. Before review, the mix-in test plus the TASK-34975 sibling, repeated 6 times in separate processes: 18 of 18 passed at load 11-17. After review, the three lane nodes plus the sibling, serially with the step's flags, 5 times: 4 passed each run at load 11-13.
- Both new tests assert that the held callback ran off the event loop thread, so an inline read fails by name instead of hanging the loop.

**Lane.** The nodes `Tests/Chat/test_console_context_presentation_current.py::test_session_close_drains_a_prompt_mixin_read_and_refuses_a_new_one` (both cases) and `::test_session_close_drains_a_controller_read_issued_during_its_hook_drain` are in console-p0-gate's admission-sensitive step, just before TASK-34975's node. They are pinned by `_TASK42342_CLOSE_DRAIN_TARGETS` and `test_task42342_close_drain_target_runs_just_before_the_task34975_group` in `Tests/CI/test_derived_artifacts_workflow.py`, and by the INHERITED list in `Tests/CI/test_response_rules_critical_gate.py`. Negative control: removing the first node failed both pins. Measured serially with the step's flags: 3 passed in 2.8 s at load 12.

**Checks.**
- Covering run (the new file, the sibling, every Tests/ file naming `finalize_session_close`, `drain_preparation_reads` or `prompt_mixin`, the three close-path suites and the two CI pin files; 51 files at `-n 4`): 2105 passed, 12 failed, 1 xfailed, at load 12-30.
- After review, the same set plus the browser-read files (`test_console_browser_current_native_lifetime.py`, `test_console_browser_quit_fence.py`; 53 files at `-n 4`): 2122 passed, the same 12 failed with the same messages, 1 xfailed, at load 13-25. The test file plus both CI pin files: 155 passed.
- All 12 failures fail identically on dev's source with the same command: 4 in `test_console_chat_start.py` (`'not_started' == 'started'`, "required native initialization never ran"); 4 in `test_console_agent_progress.py` (3 `RecoveryRequired: raw_source_selection_changed`, 1 child `ConsoleSubmitResult` assertion); 3 in `test_console_staged_evidence_sessions.py`; and `test_console_tab_publication_read_barrier.py::...[same_session_drain]` ("released original held reader never returned").
- Preflight passed, before and after review. `Tests/lint/ruff_gate.py` passed. The size ratchets (`Tests/Architecture/test_module_size_ratchet.py`, `test_screen_size_ratchet.py`) passed: 234 passed.
- The `console_runtime.py` row went from 7089 to 7091 (+2), then to 7093 (+2) for the pre-finalize drain: the change's own lines each time. `console_prompt_mixin_service.py` grew from 874 to 881 lines; its first-send warm-up LOC row has slack.

**Files.** `tldw_chatbook/Chat/console_runtime.py`, `tldw_chatbook/Chat/console_prompt_mixin_service.py`, `Tests/Chat/test_console_context_presentation_current.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_derived_artifacts_workflow.py`, `Tests/CI/test_response_rules_critical_gate.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/Architecture/console.md`.
