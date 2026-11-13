---
id: TASK-40334
title: "test_console_refreezes_after_painting_a_conversation_and_after_a_run_settles times out before the provider is called on CI"
status: In Progress
created_date: 2026-11-12 21:05
labels:
- follow-up
- tests
- console
- flaky
- ci
priority: medium
references:
- Tests/UI/test_console_idle_heap_refreeze.py
- Tests/UI/test_console_send_acknowledgement.py
- https://github.com/rmusser01/tldw_chatbook/pull/3491
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found on 2026-11-12 while #3491 waited in the merge queue. Checked with read-only `gh` and the code at dev 4b90eb860b. Line numbers are that tree's.

**Why.** The test runs in UI Fast Lane shard 2, which feeds the required check. The merge queue retries a failed required check once and evicts the PR on the second failure (`scripts/merge_queue.py:224-226`), so this flake can evict an unrelated PR. Of the 21 shard-2 PR jobs that finished between 2026-11-12 05:15Z and 20:12Z (re-runs counted once), this test failed in 6, on 5 branches. Each time it was the shard's only failure. The times below are when each job finished:
- run 43165198830 (codex/send-feedback-dev-regressions, job 131012505672, 08:26Z);
- run 43172047439 (codex/workspace-wipe-guard, job 131039345585, 08:56Z);
- run 43148114307 (codex/task-39147-native-phase-rebuild, job 131085054099, 10:50Z);
- run 43242842744 (fu/hotfix-restyle, job 131272987686, 15:39Z);
- run 43245073678 attempt 2 (fu/census-nonliteral-lookups, job 131294344060, 17:04Z);
- run 43284372195 attempt 2 (fu/census-nonliteral-lookups, job 131429596853, 20:12Z).

It passed 3 of 3 runs locally on dev e0186fa10a under an isolated profile.

**Where it stops.** Every failure is the same wait, `test_console_idle_heap_refreeze.py:123`: `await until(gateway.validation_started.is_set)`, which times out after 15 s. Before it, the test has seen the Console register the send's flight (line 112) and enter the `HeldAdmission` hold on the hook-permission snapshot (line 115), and it has then released the hold (line 121).

The jobs' captured `console_send_stage` lines narrow it further.
- In 5 of the 6 failures, `controller_submit entered` is logged at 2.2–5.7 s and nothing follows until `provider_resolution entered` at about 15.4–15.7 s. In job 131429596853 that is 3.52 s, then 15.69 s. Job 131272987686 logs `controller_submit failed` at 15.64 s instead.
- So in those five the send stalls inside the controller's submit: after `controller_submit` is recorded (`console_chat_controller.py:3560`) and before `provider_resolution` (`:9285`).
- Job 131085054099 stops earlier: its `ui_submit entered` only comes at 15.17 s.

It is not yet known what the controller waits on. The test harness's hold thread is ruled out for those five. If the wait is in the product, it is user-visible: Send accepted, then no provider call for many seconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 What the controller's submit waits on between `controller_submit` and `provider_resolution` is identified and recorded with evidence (a CI log or a local reproduction under load), and so is the earlier stall before `ui_submit` in job 131085054099.
- [x] #2 If the wait is in the send path, a send released from the hook-permission snapshot reaches provider resolution without waiting on unrelated work, and a test pins that.
- [ ] #3 The test passes 30 consecutive runs of its shard-2 file set under xdist with a concurrent CPU stressor, and the next 10 CI shard-2 runs after the fix are green on it.
- [x] #4 The test still fails if the Console re-freezes its heap during a send: its busy assertions are not weakened, skipped or retried away.
<!-- AC:END -->

## Implementation Plan

ADR required: no. The change is to a test harness (how one test samples the Console's busy state); no product behaviour, data shape or contract moves.

1. Read the six CI logs and seven newer shard-2 failures; map the controller path from `controller_submit` to `provider_resolution` and its worker-thread hops.
2. Reproduce with a stall probe (asyncio task stacks, thread stacks, executor state) on macOS under a CPU stressor and in a local Linux 3.12.16 container with 4 CPUs, base code.
3. Name what the submit waits on. If it is the harness, fix the harness at its root; if it is the product, fix the product with a behaviour test.
4. Keep the mounted test's busy assertions as strict as they are; prove a busy-check mutation still fails it.
5. 30 consecutive runs of the shard-2 file set under xdist with a CPU stressor; preflight; Tests/Architecture.


## Implementation Notes

**Cause: the test harness, not the send path.** The mounted test's `BusySampler` re-queued itself with `await asyncio.sleep(0)` on every loop turn, so the event loop never blocked in its selector. On Linux `epoll_wait(0)` returns at once, so the loop thread stayed on-CPU (a bare `sleep(0)` loop: 112,810 turns and 0.200 s of CPU in 0.2 s inside a local Linux 3.12.16 container, against 457 turns and 0.007 s on macOS, where a zero-timeout `kqueue` still parks the thread). Every off-loop step of the send runs on a worker thread and needs the GIL back after each syscall: hook reads through `run_preparation_read` (`_submission_hook_admission`, `hold_legacy_stop`, which cold-imports the run-hooks engine), storage admission (`raw_participants`/`storage_admission` path checks), and the durable commit (`run_owned_db_call`). CPython escalates a waiting thread's 5 ms timeout to a forced hand-off only if no other thread took the GIL meanwhile; the loop thread re-takes it on every turn, so the workers only progressed by winning a race.

**Evidence (CI logs; the full stall did not reproduce locally).** 13 shard-2 failures: the six in the description plus 131599859132, 131622315157, 131724456937, 131741094903, 131808807330, 131821846776, 131941460802 (2026-11-13). In 11 of them the send logged nothing more until within about a second of the test's 15 s wait giving up, whose `finally` stops the sampler before the app shuts down: then came `provider_resolution` (six jobs), `ui_submit` (131085054099), or the submit's failure or refusal at shutdown (four jobs). In 131741094903 and 131821846776 the send crawled while the sampler ran: `provider_resolution` came 7.4-7.6 s after `controller_submit`, then the durable commit took 13.6 s and 12.0 s, against 0.2-0.3 s in the six jobs where it ran after the sampler stopped. In 131741094903 the commit's transaction sat 9.3 s after its `add_conversation` insert and committed within 15 ms after the wait gave up. The earlier stall before `ui_submit` (131085054099) is the same starvation on the dispatch's own worker hops after the hook-permission hold. Locally the old sampler passed about 20 macOS runs (some with a stressor) and 17 in the Linux container (4 or 2 CPUs, with and without a stressor); a stack probe there showed the submit parked in `run_preparation_read` while its worker was importing or resolving storage paths.

**Fix.** `BusySampler` now reads the busy check as each turn's selector poll returns (a wrapper on `loop._selector.select`, which `BaseProactorEventLoop` also aliases), once per turn, before that turn's callbacks, so an idle loop sleeps. The mounted test still samples every turn of the send window (275 samples locally, all busy) and keeps every assertion. New `test_the_busy_sampler_lets_an_idle_loop_sleep` pins that an idle loop with the sampler running makes at most 20 turns in 0.2 s; the old sampler made 728-1,658 there on macOS.

**AC#2:** not applicable; the wait is not in the send path, so no product change.

**AC#4:** with the fix in place, three temporary product mutations of `_console_heap_busy` each failed the mounted test: dropping the send-flight check ("a send being admitted did not hold the re-freeze off"), dropping the transcript-poll checks ("an accepted send did not hold the re-freeze off"), and a one-turn idle reading every 25th call during a send ("the Console read idle on 4 of 427 loop turns").

**AC#3 (local half met; the CI half is open).** 30 runs of the 57-entry shard-2 file set (`check_ui_pr_gate_census.py --shard 1 36`), each `-n 4` through the isolated wrapper on macOS (18 cores), three runs at a time beside an 8-process busy-loop stressor (host load 30-45): the test passed in all 30 and every run executed all 138 nodes. Unrelated failures in those runs: `test_settings_pending_save_drafts.py::test_staged_save_acknowledges_only_saved_values[later-appearance]` 14 of 30 (it also failed 3 of 3 base runs under a stressor) and `test_library_file_notes_workspace.py::test_delayed_old_workspace_cannot_replace_current_workspace_root` 1 of 30. Base never failed locally either, so these runs show no regression, not the fix; the next 10 CI shard-2 runs after merge are the check that counts.

**Verification.** New test fails with the old sampler and passes with the fix (mutation check). Covering files (`test_console_idle_heap_refreeze.py`, `test_console_send_acknowledgement.py`, `-n 4`): base 38 passed, head 39 passed. `./scripts/preflight.sh` passed. `Tests/Architecture` (serial): 900 passed, 3 skipped, 1 failed, `test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only` on `Agents/agent_service.py` ("on_child_settled consumer raised"), product code this branch does not touch, so it fails on base too.

**Files:** `Tests/UI/test_console_idle_heap_refreeze.py`, `backlog/docs/lessons-testing-evidence.md`.
