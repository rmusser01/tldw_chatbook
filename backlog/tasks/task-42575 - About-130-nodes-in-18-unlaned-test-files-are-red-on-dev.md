---
id: TASK-42575
title: "About 130 nodes in 18 unlaned test files are red on dev"
status: To Do
assignee: []
created_date: 2026-11-26 15:35
labels:
- follow-up
- dev-red
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_async_mcp_snapshot.py
- Tests/Chat/test_console_workflow_scheduling.py
- Tests/Chat/test_console_archive_recovery_dispatch.py
- Tests/MCP/test_console_snapshot_source_contracts.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 27 by TASK-41061's covering run: the 60 test files that use the native hook, MCP capture and ordinary-tree code it changed. Measured on dev 6e41919880 (product and tests at that commit), on macOS, at `-n 4`: 131 nodes in 18 files are red, 129 failed and 2 with teardown errors. No PR lane runs any of them. Two files have other nodes in the UI census, and none of these red nodes is among them. If the product is wrong, MCP sources may be read on the UI loop, workflow scheduling may misroute, or archive-recovery dispatch may skip its confirmation. If the tests are wrong, they have drifted past refactors since #3480, as TASK-42553 and TASK-41061 found for their files.

Red nodes per file. Each note is the deepest frame from a `--tb=line` run of the same nodes on round 27's `fu/r27-native` branch, where they failed too (the dev run printed no tracebacks), so every cause still has to be confirmed on dev:
- `Tests/Chat/test_console_async_mcp_snapshot.py` (32): four nodes call the removed `ConsoleChatController._compose_local_provider_async` (AttributeError). Others assert "actual queue caller still blocks on native MCP source reads" and "actual async local composer blocks on native permission state". Many fail in helpers from `Tests/MCP/test_external_catalog_worker_ownership.py` (`assert False`, `RecoveryRequired: console_snapshot_owner_changed`).
- `Tests/Chat/test_console_workflow_scheduling.py` (26): `KeyError: 'result'` and "Regex pattern did not match".
- `Tests/Chat/test_console_archive_recovery_dispatch.py` (23): every red node is `test_real_console_native_dispatch_waits_for_exact_confirmation[...]`; `AssertionError: []`. The run also raised `OriginalSourceChanged("original agent storage source changed")` from `console_runtime.py` 22 times, in this file or the scheduling one.
- `Tests/MCP/test_console_snapshot_source_contracts.py` (16): one calls the removed `_compose_local_provider_async`. Others: count asserts (`0 == 1`, `1 == 0`) and reads still made on the main thread.
- `Tests/UI/test_console_live_work_handoffs.py` (5): the private-profile child fails (`Tests/private_profile.py: Failed`).
- `Tests/Chat/test_workflow_native_journeys.py` (4): not read.
- `Tests/Chat/test_hooks_v2_continuations.py` (4): `test_other_actual_root_still_occupies_capacity[preparing]` and `[streaming]` failed after about 5 min each. `test_consumption_before_foreground_or_uncertain_failure_is_not_reopened[stop]` and `test_lost_scheduler_response_never_returns_machine_work_to_queue` pass their calls and error in teardown.
- `Tests/Chat/test_console_ssh_custody.py` (4): "maximum missed the actual skills/runtime gate"; `'original SSH' in` the system prompt fails.
- `Tests/UI/test_console_internals_decomposition.py` (3): `'Incident Review' in ''`, `'Approvals: 1 pending' in 'Approvals: 0 pending'`, and a changed "Send blocked…" copy.
- `Tests/UI/test_console_custom_capture_worker_feedback.py` (3): not read.
- `Tests/Chat/test_console_shared_tool_preparation.py` (3): `['builtin:…'] == []`, `[True, True] == [True]`, and a read still made on the main thread.
- `Tests/UI/test_console_fallback_pressed_draft.py` (2): not read.
- One each, not read: `Tests/Plugins/test_native_components.py`, `Tests/Plugins/test_declared_variables.py`, `Tests/Chat/test_console_default_send_capture.py`, `Tests/Agents/test_unproven_tool_cleanup_run.py`.
- One each, read: `Tests/Chat/test_console_run_hooks_regressions.py` expects "Blocked by hook: policy" and gets a "Console send… config.toml." copy; `Tests/Chat/test_console_composition_tool_ceiling.py` gets an empty MCP maximum (`'local:one::first' in mappingproxy({})`).

Also seen, but load flakes rather than dev reds: in the same files, `test_console_send_without_ready_runtime_keeps_setup_block_tooltip_only` failed 1 of 6 on dev; `test_actual_console_registers_selected_managed_agent` had a teardown `plugin_owned_work_not_drained` once at `-n 4` and passed 3 of 3 serially. `test_late_subagent_stop_retains_original_tree_until_physical_return` and `test_real_mounted_console_stop_entry_emits_interrupt` passed on dev but failed in the heavily loaded run.

This task may be split per area (MCP snapshot and capture; workflow scheduling and native journeys; archive recovery dispatch; SSH custody; Console UI). Each part is filed as its own task when it is picked up and takes the criteria below; this task stays the umbrella until every part is done.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each red node, the notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [ ] #2 A product defect is fixed and its test shown red before and green after; a drifted test is repaired with every behaviour assertion kept
- [ ] #3 The files pass on dev at `-n 4`, twice in a row, and each pytest process exits on its own
- [ ] #4 The files run on a PR lane that blocks merges to dev, timed with that lane's flags
<!-- AC:END -->
