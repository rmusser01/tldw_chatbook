---
id: TASK-41317
title: "The served relay publisher's wait_idle spins on asyncio.sleep(0), which can starve worker threads on Linux"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- tests
- flaky
- web-server
priority: low
references:
- tldw_chatbook/Web_Server/served_relay_publisher.py
- Tests/Chat/test_console_relay_approval_integration.py
- backlog/docs/lessons-testing-evidence.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down, which applied TASK-40334's finding to other loop code. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-40334 showed that an event-loop wait that re-queues itself with `await asyncio.sleep(0)` keeps the loop thread on-CPU on Linux, where a zero-timeout `epoll_wait` returns at once. Worker threads that need the GIL back then stall for seconds. See the newest entry in `backlog/docs/lessons-testing-evidence.md`, "A `sleep(0)` observer starves worker threads on Linux, and the switch interval does not bound it". The relay publisher has the same shape. A thread that publishes through `call_soon_threadsafe` can be held off for the whole wait, so the wait gives up at its deadline with events still queued. That would fail a Linux test run that macOS passes. Not observed yet.

**Evidence.**
- `Web_Server/served_relay_publisher.py:178-193` (`ServedRelayPublisher.wait_idle`, documented as a test surface, default timeout 1 s): `while True: await asyncio.sleep(0)`, re-checking `queue.empty()` and `self._busy` until a loop-clock deadline.
- Its only caller is `Tests/Chat/test_console_relay_approval_integration.py:362` (`await publisher.wait_idle(timeout=2)`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `wait_idle` lets an idle loop sleep: a test running it against an empty-but-busy queue for 0.2 s sees a bounded number of loop turns (the TASK-40334 idle-loop check), and fails with the current spin
- [x] #2 `wait_idle` still returns as soon as every queued event is sent, and at its deadline otherwise (existing relay tests still pass)
<!-- AC:END -->

## Implementation Plan

ADR required: no. A test-surface wait changes how it blocks; no product behaviour, contract or architecture changes.

1. Add a test in `Tests/Web_Server/test_served_relay_publisher.py` that holds one send in flight (empty queue, drainer busy), counts real loop turns (`_run_once`, as `test_the_busy_sampler_lets_an_idle_loop_sleep` does) while `wait_idle(timeout=0.2)` runs, and asserts a bounded count plus a return at the deadline; then that a long-timeout `wait_idle` returns as soon as the send completes. Show it fails on the unmodified tree.
2. Have the drainer mark each item `task_done()` and make `wait_idle` yield once, then `asyncio.wait_for(queue.join(), timeout)`, suppressing the timeout (same return-never-raise contract). Drop the now-unused `_busy` flag.
3. Mutation check (revert the fix, test fails), run `Tests/Web_Server/test_served_relay_publisher.py` and `Tests/Chat/test_console_relay_approval_integration.py` on base and head, preflight, `Tests/Architecture`.

## Implementation Notes

`wait_idle` no longer polls. The drainer marks each item `task_done()` in its `finally` (sent, failed or cancelled mid-send). `wait_idle` yields once, so a put already scheduled through `call_soon_threadsafe` is counted, then waits on `queue.join()` under `asyncio.wait_for(timeout)` and suppresses the timeout. The return-never-raise contract is unchanged. `join()` covers both queued and in-flight items, so the old `_busy` flag was dead code and is removed.

- Test: `test_wait_idle_lets_an_idle_loop_sleep_until_the_send_finishes` (`Tests/Web_Server/test_served_relay_publisher.py`). It holds one send in flight, then counts real loop turns (a `_run_once` wrapper, the `test_the_busy_sampler_lets_an_idle_loop_sleep` check) while `wait_idle(timeout=0.2)` runs. It asserts at most 20 turns, a return at the deadline with nothing sent, and that a 60 s `wait_idle` returns within 5 s once the send completes. It measured 476-1705 turns on the unmodified code (macOS) and 3 after the fix; reverting only the product file brings the failure back.
- Covering files on base and head (`test_served_relay_publisher.py`, `Tests/Chat/test_console_relay_approval_integration.py`): base 29 passed, head 30 passed (one new). The other files that reference the publisher (`test_mobile_service.py`, `test_app_startup_delegation.py`, two `Tests/CI` contracts) pass on head (207).
- Files: `tldw_chatbook/Web_Server/served_relay_publisher.py`, `Tests/Web_Server/test_served_relay_publisher.py`.
- `./scripts/preflight.sh` passes. `Tests/Architecture` (serial, load 30-75): 8 failed, 1413 passed. The four head-only failures were 300 s timeouts in `test_persistent_diagnostic_inventory.py`'s TASK-15103 ledger tests, and they pass when run alone (4 passed). The other four also fail on base fbf1c9ea74: the two `test_timer_path_static_update_inventory.py` tests (`project_index_binding_card.py`), `test_reviewed_diagnostic_changes_are_metadata_only` (`agent_service.py`) and a timeout in `test_task_15103_review_ledger_canonical_schema_and_arithmetic`.
