---
id: TASK-41028
title: "test_console_ordinary_tree.py hangs past its pytest timeout, and 32 of its 75 nodes are red on dev"
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
- Run serially on its own, that node also hangs past 400 s, and the 120 s timeout does not fire.
- The same run fails 32 nodes in 19 tests, none of them through the local `RecoveryRequired` trap. Among them are all 7 params of `test_original_unknown_library_factory_is_opaque_before_construction`, the nodes TASK-36950.1 saw hang. On this base they fail fast instead: six fail at `:507` because no provider call was made (`h.gateway.payloads` is empty), and `[swapped_native]` fails at `:505`.

Why it matters: a hang that pytest-timeout cannot interrupt loses every other result in the run, and an xdist run never ends. The file came with #3480 (TASK-39147) and runs on no PR lane, so CI sees neither problem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `test_original_stop_hook_holds_same_ordinary_tree[True-True]` finishes or fails within its pytest timeout, both alone and under `-n 4`
- [ ] #2 The task notes name the cause of the hang (a product deadlock or a test await) and why pytest-timeout could not interrupt it
- [ ] #3 Every node of Tests/Chat/test_console_ordinary_tree.py passes on dev at `-n 4`; each red node first gets a named cause
- [ ] #4 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
