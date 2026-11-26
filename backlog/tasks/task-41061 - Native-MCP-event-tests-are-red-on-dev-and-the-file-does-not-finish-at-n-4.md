---
id: TASK-41061
title: "Native MCP event tests are red on dev, and the file does not finish at -n 4"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:03
updated_date: '2026-11-26 07:05'
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- mcp
priority: high
references:
- Tests/Chat/test_native_v2_mcp_events.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15) (bootstrap profile, `-n 4`), five nodes of `Tests/Chat/test_native_v2_mcp_events.py` fail, and the file does not finish at `-n 4`:
- `test_unqualified_ordinary_mcp_keeps_real_existing_dispatch`: the submit is not accepted.
- `test_selected_deny_never_reserves_a_native_mcp_worker`: execution owners are still reserved after a selected deny.
- `test_revocation_during_actual_nested_capture_discards_without_host_leak`: raises `MixinSourceChangedError: mixin_session_changed`.
- `test_real_mcp_lifecycle_publish_and_operation_finish_return_are_joined`: times out.
- `test_accepted_context_uses_positive_original_drain_without_reviving_dispatch[negative_drain]` fails about 12 s in, run alone, and the pytest process then never exits (TASK-41028 has the same shape). At `-n 4` the file reports 42 of 43 nodes and stalls past the 120 s timeout; on 2026-11-15 the node that never reported was `test_passive_accepted_mcp_context_still_refuses_actual_saved_grant_revocation[mcp]`, which passes alone. These two findings come from the review of this filing (plain runs, no trap).

The architecture-review fleet runs on dev beab5c1e4e (2026-11-13) saw the first four red. If the product is wrong, an ordinary MCP dispatch is refused, or a denied MCP tool still holds a worker slot. A file that cannot exit is a hang pytest-timeout cannot interrupt, so it is filed high, like TASK-41028. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task notes say for each node whether the product or the test is wrong, and name the commit
- [x] #2 A pytest run of Tests/Chat/test_native_v2_mcp_events.py alone, serially and at `-n 4`, runs to completion and the pytest process exits on its own within the run's timeout
- [x] #3 Every node of Tests/Chat/test_native_v2_mcp_events.py passes on dev at `-n 4` (a node may first be deleted or retargeted, with its reason in the task notes); a fixed product defect is shown red first
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run the file at base 6e41919880 and at the #3480 merge (e10ce63fe8) to name each red node's cause and the commit that introduced it (scratch probes only, no product edits).
2. Decide per node whether the product or the test is wrong; fix a product defect with its test red first, repair a drifted test keeping its assertions, and record any deleted or retargeted node with its reason.
3. Find what keeps the process alive after its summary (a non-daemon thread or an un-awaited task) and make the file exit on its own, serially and at `-n 4`.
4. Lane the file in the recovered-suites step, timed with the step's flags.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base dev 6e41919880. The file came with #3480 (e10ce63fe8) and was red there; most nodes were written before 11d71e5db5 (ordinary Console physical ownership), which landed later on the same branch. Serially at base 5 nodes were red; `session_end_nested` and `distinct_actual_nested` (red in round 26 on dev 692acb43dc) failed only under load.

**Per node (AC #1):**
- `unqualified_ordinary_mcp_keeps_real_existing_dispatch`: **product.** 11d71e5db5 left a managed send's MCP capture pending when its metadata is not closed, and only the ordinary tree resolves it. `admit_ordinary_turn` admits no tree where native process groups are missing (Windows), so the turn's MCP maximum stayed empty, the hook's tool could not be resolved (`hook_mcp_tool_unavailable`), and the send was refused: "Required hook initialization failed.". **Fix:** `MCP/console_snapshot._capture_stays_pending`, which both capture functions share: the capture stays pending only where `supports_native_process_groups()`. Red at base, green after.
- `accepted_context..._without_reviving_dispatch[negative_drain]`: **product.** In `_CommandEvent._returned_context` (11d71e5db5), a borrowed ordinary-tree owner counted as returned even after it drained unproven, so accepted context outlived its registration (`check_memory` did not raise). **Fix:** that branch also requires `_drain_outcome is not False`; a live (None) tree owner still qualifies. Red at base (DID NOT RAISE), green after.
- `distinct_actual_nested_mcp_worker_and_command_share_one_original_event`: **product, then load timing.** Under 4 runs at once (load ~45), 11 of 12 failed "Required hook initialization failed.". A probe showed the outer MCP job's policy refusing with `hook_mcp_original_permission_changed`. The outer job checks every plan's permission, and the nested MCP job's `MCPHookPermission.refresh()` set the token to None before re-reading the store (37273cff7b). **Fix:** the token stays until the refresh succeeds or fails; a changed publication still fails other checks. New test `test_mcp_hook_permission_stays_current_to_other_jobs_while_it_refreshes` (`Tests/Agents/test_hooks_v2_execution.py`): red on base code with that PermissionError, green after. 7 of 20 then still failed, now with `handler_deadline`: the nested job and its leaf command (a private-profile interpreter) outlasted the default 10 s handler timeout. The test's three handlers now take the 60 s maximum. **24 of 24** after (6 at once, load 40-49).
- `session_end_nested_workers_consume_original_holds_after_capacity_close`: **load timing.** At `-n 4` and in serial runs at load ~50 it failed with `event_deadline`. A timeline probe (load 53) showed the MCP worker invoked 0.19 s after the seal, its nested PreToolUse command starting at 2.14 s and the deadline at 3.03 s: the 3 s SessionEnd allowance. A forced 3.5 s delay in the held worker reproduced it every time with 3 s and passed with 30 s. With 4 runs at once, `handler_deadline` (the default 10 s) and the 8 s cleanup wait failed next (12 of 12 with the 8 s wait). **Fix:** the engine's allowance is now named (`TEARDOWN_SECONDS`, `TEARDOWN_CLEANUP_SECONDS`; values unchanged), and the test sets it to 30 s, gives its handlers 60 s and waits up to 45 s for cleanup. **20 of 20** after (4 at once, load 45-52). In round 26 this node's failure also left an executor thread blocked, so the process could not exit (below).
- `selected_deny_never_reserves_a_native_mcp_worker`: **test.** The class-wide `reserve_tool` count caught the ordinary tree's admission (`console_ordinary_tree.py:285`) and MCP composition (`console_provider_composition.py:163`) reservations. 11d71e5db5 added both, and both return at once. Neither is a worker for the denied MCP hook. The count now takes only reservations made by `_CommandEvent._select`, the one place a native event reserves workers.
- `real_mcp_lifecycle_publish_and_operation_finish_return_are_joined`: **test.** The class-wide `OwnedOperation.finish` hold caught the tree admission's own finish (`console_ordinary_tree.py:351`) first and blocked the loop for 8 s. It now holds only the event's own operations (recorded at `_select`).
- `revocation_during_actual_nested_capture_discards_without_host_leak`: **test.** Its `conversation_binding_revision` bump also invalidates the send's prompt mix-in capture (7970fbcada, on dev before #3480). The mix-in check after hook preparation raises `MixinSourceChangedError("mixin_session_changed")` before the hook refusal returns, and `submit_draft` re-raises it by design. The node expects that raise; capture, no-MCP-call, cleanup, opaque and worker assertions are unchanged.
- No node was deleted or retargeted.

**The hang (AC #2).** Round 26's run (log `bootcmp/serial-dev.log`) dumped thread `asyncio_1`, the default-executor worker, blocked in `_EventCustodyRegistry.wait_closed`. That is an untimed `Condition.wait_for`, awaited by `HookEngine._drain` through `asyncio.to_thread` (2665a17673, `wait_closed` from 89ba88b61d). `session_end_nested` had failed with an event custody still open, and pytest-timeout cancelled its three teardown awaits after 300 s each. A cancelled `to_thread` await leaves its thread running, and pytest-asyncio's runner close (`shutdown_default_executor`, thread `_do_shutdown` in the dump) and then interpreter exit join it. **Fix:** `_EventCustodyRegistry.join_closed()` waits in 0.25 s slices, used by the engine drain and the legacy hook source drain (`ConsoleRuntime._drain_legacy_hooks`). The app's close no longer leaves a thread that keeps the process from exiting. New test `test_cancelled_close_leaves_no_thread_waiting_for_native_custody`: red with the untimed join (loop shutdown still waiting at 5 s), green after. On the merge tree `negative_drain`'s teardown also never ended (0% CPU in kqueue, killed after 7 min); pytest-timeout there still dropped its timer after a failed call, which TASK-41135 fixed.

**Runs.** The file at `-n 4` (with the hook, poll and hook-execution files): 139 passed twice in a row (241 s, load 44-50; 200 s, load 28-43), each process exiting 4-6 s after its summary. It also passed 43 of 43 alone at `-n 4` (131 s). Alone and serially on the final head: 43 passed (277 s, load 11-12), and the process exited 2 s after its summary. In the macOS step run: 43 passed (461 s, load 27-46).

**Covering tests.** The 60 other test files that name the changed code ran at `-n 4` under the soaks' load (53 min): 137 failed and 3 errored. 129 of those failed at base as well and 2 errored at base as well; 11 passed at base. On the branch at load 17, 10 of the 11 passed. `test_console_send_without_ready_runtime_keeps_setup_block_tooltip_only` failed 1 of 3 on the branch and 1 of 6 with base product code. `test_actual_console_registers_selected_managed_agent` had one teardown `plugin_owned_work_not_drained` at `-n 4` and passed 3 of 3 serially.

**Lane.** The file runs on no Linux lane (`native_console` calls `require_managed_runtime()`, Darwin only). It joins the macOS `plugins-recovery-ui` job's new step with the hook file, plus a check that both executed. With the step's flags (serial, `--timeout=300`) at load 27-46 the step took 600 s, this file 461 s of it. The job's last run took 1m49s against its 15-min cap.

**Owner decision (2026-11-26), verbatim.** Question: "TASK-41061 and TASK-42553: … They only run on macOS (`require_managed_runtime()`), so no Linux lane can cover them. The agent added a step to the macOS job `plugins-recovery-ui`, which has a 15-min cap and last ran in 1m49s. The new step took about 10 min locally under heavy load. Where should they run?" The owner chose "Existing job, cap 25 (Recommended)": keep the step in plugins-recovery-ui and raise the job's cap from 15 to 25 min. Done: the job's `timeout-minutes` is 25, and its lane contract test pins 25.

**Review fixes (2026-11-26).**
- **Lane.** The two new regression tests (`test_cancelled_close_leaves_no_thread_waiting_for_native_custody` and `test_mcp_hook_permission_stays_current_to_other_jobs_while_it_refreshes`) are now the last two targets of recovered-suites. Their file is on no PR lane in this workflow. With the step's flags (`-p no:xdist --storage-lease-guard --timeout=180`), at load 27-45, they took 1.1 s and 0.04 s.
- **Loop cleanup.** The join test's `asyncio.Runner` now takes `loop_factory=asyncio.new_event_loop`, so it no longer leaves a closed loop set on the main thread. It is still red on the untimed join (loop shutdown waiting at 5 s) and green on `join_closed`.
- **Wait budget.** This file now has its own `until` with a 30 s budget; the shared 8 s helper in `test_native_v2_command_events.py` is unchanged. At `-n 4` the file passed 43 of 43 four times at loads 23-60: alone (load 23-50, then 27-43), twice at once (25-45), and with 26 CPU burners (57-61). Serially it passed 43 of 43 in 478 s (load 23-42). The four touched files at `-n 4` passed 139 of 139.
- **Wording.** The `_capture_stays_pending` docstring now names all three native resolvers: the ordinary tree, workflow preparation and standalone compaction. ADR-285's Task8D2 paragraph now says the no-early-classification rule holds only where native process groups exist.
- **Reaper race (second review).** `test_original_reaper_and_done_callback_return_the_same_hold_once[done_first]` read `_callbacks == 2` at load 38-63. Adding `job.worker.done()` alone still failed (8 runs at once, load ~50: 2 of 40). A log of every retain and return showed start +1, worker +1, worker -1, plus `started=True`. The extra hold was the engine delivery's `target.activate()`, which is resumed after `job.task` finishes and returns its hold on exit. The wait now also requires that delivery to have left `engine._deliveries`. After that: 40 of 40 runs (both orderings, 8 at once, load 40-53) and 6 of 6 at `-n 4` (load 49-57).
- **Filed.** TASK-42575 (about 130 nodes red on dev in 18 unlaned files) and TASK-42576 (the Change Review initialization drain's untimed executor wait).

Files: `tldw_chatbook/MCP/console_snapshot.py`, `tldw_chatbook/Chat/console_configuration_capture.py`, `tldw_chatbook/Agents/hooks_v2/native_commands.py`, `tldw_chatbook/Agents/hooks_v2/mcp_executor.py`, `tldw_chatbook/Agents/hooks_v2/engine.py`, `tldw_chatbook/Agents/legacy_hook_ownership.py`, `tldw_chatbook/Chat/console_runtime.py`, `Tests/Chat/test_native_v2_mcp_events.py`, `Tests/Agents/test_hooks_v2_execution.py`, the lane files listed in TASK-42553, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
