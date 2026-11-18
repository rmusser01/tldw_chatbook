---
id: TASK-41317
title: "The served relay publisher's wait_idle spins on asyncio.sleep(0), which can starve worker threads on Linux"
status: To Do
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
- [ ] #1 `wait_idle` lets an idle loop sleep: a test running it against an empty-but-busy queue for 0.2 s sees a bounded number of loop turns (the TASK-40334 idle-loop check), and fails with the current spin
- [ ] #2 `wait_idle` still returns as soon as every queued event is sent, and at its deadline otherwise (existing relay tests still pass)
<!-- AC:END -->
