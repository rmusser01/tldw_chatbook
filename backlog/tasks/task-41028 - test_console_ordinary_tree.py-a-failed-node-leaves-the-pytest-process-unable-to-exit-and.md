---
id: TASK-41028
title: "test_console_ordinary_tree.py: a failed node leaves the pytest process unable to exit, and 32 of its 75 nodes are red on dev"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- ci
priority: high
references:
- Tests/Chat/test_console_ordinary_tree.py
- tldw_chatbook/Chat/console_ordinary_tree.py
- tldw_chatbook/Agents/ordinary_tree.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36950.1's covering run in Phase 4 of the 2026-10-25 architecture review: on base 26b2e677dd the run hung for 4 h at 97% in `Tests/Chat/test_console_ordinary_tree.py`, with the workers at 0% CPU. pytest-timeout never fired, and SIGINT did not stop the workers.

Re-checked on dev ecad404bd4 (2026-11-15):
- `-n 4 --timeout=120 --timeout-method=thread` on the file alone stops at 74 of 75 nodes, with all four workers at 0% CPU, until a 420 s watchdog kills it. The node that never reports is `test_original_stop_hook_holds_same_ordinary_tree[True-True]`.
- Run serially on its own, the node reports FAILED about 10 s into the test, and then the pytest process never exits (still alive at 300 s; killed by a watchdog). The hang comes after the test body, where pytest-timeout no longer runs, which is why the 120 s timeout never fires. The review of this filing saw the same shape in `Tests/Chat/test_native_v2_mcp_events.py`.
- The same run fails 32 nodes in 19 tests, none of them through the local `RecoveryRequired` trap. Among them are all 7 params of `test_original_unknown_library_factory_is_opaque_before_construction`, the nodes TASK-36950.1 saw hang. On this base they fail fast instead: six fail at `:507` because no provider call was made (`h.gateway.payloads` is empty), and `[swapped_native]` fails at `:505`.

Why it matters: a process that cannot exit after a failed test is a hang pytest-timeout cannot interrupt. It loses every other result in the run, and an xdist run never ends. The file came with #3480 (TASK-39147) and runs on no PR lane, so CI sees neither problem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pytest run of Tests/Chat/test_console_ordinary_tree.py alone, serially and at `-n 4`, and of `test_original_stop_hook_holds_same_ordinary_tree[True-True]` by itself, runs to completion and the pytest process exits on its own within the run's timeout, with no worker or interpreter left waiting
- [ ] #2 The task notes name the cause of the hang (a product deadlock or a test await) and why pytest-timeout could not interrupt it, and whether the same cause stops `Tests/Chat/test_native_v2_mcp_events.py::test_accepted_context_uses_positive_original_drain_without_reviving_dispatch[negative_drain]` from exiting
- [ ] #3 Every node of Tests/Chat/test_console_ordinary_tree.py passes on dev at `-n 4`, or is fixed or retargeted with its reason in the task notes; each red node first gets a named cause
- [ ] #4 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
