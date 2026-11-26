---
id: TASK-42553
title: "Eleven native-plugin and Console poll tests in two files no PR lane runs are red on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 04:50
updated_date: '2026-11-26 07:05'
labels:
- follow-up
- console
- plugins
- dev-red
priority: medium
references:
- Tests/Chat/test_native_plugin_hook_preparation.py
- Tests/UI/test_console_poll_reconciliation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 26 (TASK-42551's caller runs, 2026-11-25). Run serially on dev 692acb43dc, these 11 tests fail; no PR lane runs their files, so nothing reports them. If the product is wrong, a send can be refused with "Required hook initialization failed.", a plugin admission can run outside native activity, or a poll can fail to settle.

- `Tests/Chat/test_native_plugin_hook_preparation.py` (8): `test_activity_busy_is_original_visible_submission_refusal`, `test_actual_plugin_admission_has_one_native_operation`, `test_cancelled_preparation_keeps_original_admission_until_return`, `test_first_use_original_resource_replacement_refuses_before_plugin_worker[activity]` and `[capacity]`, `test_only_original_manual_submission_can_use_reserved_capacity[True]`, `test_original_host_check_cannot_accept_replaced_native_source`, `test_original_native_result_handoff_remains_owned`. Errors include "Required hook initialization failed." and "test failed to release actual plugin admission".
- `Tests/UI/test_console_poll_reconciliation.py` (3): `test_full_request_during_poll_await_replays_with_current_owner[original-demand]` and `[preparing-display]`, `test_poll_terminal_transition_settles_while_background_turn_remains_active`.

The seven red nodes of `Tests/Chat/test_native_v2_mcp_events.py` in the same run, and its teardown timeout, are TASK-41061. Three red tests in `Tests/UI/test_console_staged_evidence_sessions.py` are left to open PR #3707 (TASK-33620.7), which repairs that file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For each red test, the notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [x] #2 A product defect is fixed and its test shown red before and green after; a drifted test is repaired with every behaviour assertion kept
- [x] #3 Both files pass on dev at `-n 4`, twice in a row, and each pytest process exits on its own
- [x] #4 Both files run on a PR lane that blocks merges to dev, timed with that lane's flags
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run both files at base 6e41919880, at the #3480 merge (e10ce63fe8) and at the dev commits around each first red, to name each node's cause and the commit that introduced it (scratch probes only, no product edits).
2. Decide per node whether the product or the test is wrong. Fix a product defect with its test red before and green after; repair a drifted test with every behaviour assertion kept.
3. Run both files at `-n 4` twice in a row and serially, and check each pytest process exits on its own.
4. Lane the poll file at the UI census tail (set `MINIMUM_FILES`) and the hook file in the recovered-suites step, timed with each step's flags, and update the lane contract tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base dev 6e41919880. Each node was run at base, at the #3480 merge e10ce63fe8, at dev before it (7775edb567) and around #3536 (908e9840aa and its first parent), with the refusing Keychain probe on the old trees. Every red node here is a drifted test; the product behaves as designed.

**`Tests/Chat/test_native_plugin_hook_preparation.py` (8).** The file came with #3480 and was red there with the same eight errors.
- `actual_plugin_admission`, `cancelled_preparation`, `first_use[capacity]` and `[activity]`, `only_original_manual[True]`, `activity_busy`. The fixture held every `PluginAdmission.check`. A send first admits its plugin skill rows for model constraints (`console_plugin_context._native_skill_configuration`, since 0327cb5cf9, 2026-10-15, before the test was written). That admission is outside hook-preparation custody by design: `Docs/Architecture/console-run-hooks.md` puts the admission inside `PluginService.hook_configuration` under custody. So the held check was the skill one. That gave "plugin admission ran outside native activity", `owners[0]` None, a 10 s hold ("test failed to release actual plugin admission") and the `until` timeouts. **Fix:** a ContextVar set by a wrapper around `hook_configuration` marks the admissions to hold; the others pass through. The ContextVar propagates into the plugin worker's tasks.
- `first_use[capacity]` and `[activity]` then hit a second drift. Since 11d71e5db5 the first hook read is the ordinary tree's own maximum capture, after the tree pinned the runtime's capacity and activity. The replacement is refused there by `RuntimeError("original ordinary turn source changed")`, raised from `submit_draft` as the tree's other source changes are (`test_console_ordinary_tree.py` expects them raised). The node now expects that raise; the no-admission, no-worker and claimable-activity assertions are unchanged.
- `original_native_result_handoff`, `original_host_check`. Their wrappers of `hook_configuration(maximum)` refused the `scratch_manager`/`scratch_snapshot` keywords that `_Preparation.invoke` passes; both came in the same commit as the file (7b27e34b7e). The wrappers now pass `**kwargs`.

**`Tests/UI/test_console_poll_reconciliation.py` (3).** The file passed whole (6 passed) on dev 7775edb567.
- `full_request...[original-demand]` and `[preparing-display]` went red with #3480. d043e22438 (on its branch) hands `run_worker` the bound method instead of a coroutine, so the coroutine is only created once Textual admits the worker. The observer only accepted a coroutine whose `cr_code` was the FULL sync, so it saw no replay ("Pending demand reached neither original full-replay owner"). It now also accepts that bound method of this console.
- `poll_terminal_transition` was green at 908e9840aa^1 and red at 908e9840aa (#3536). 95ca4bf47b moved the receipt acknowledgement out of `ChatScreen._sync_native_console_transcript` (owner `self`, local `rendered_receipts`) into `console_background_effect.acknowledge_console_transcript` (owner `screen`, local `rendered`). The transcript sync calls it once its projection is published. `_transition_parent` now takes the owner argument's name, and the node requires both the acknowledgement frame and the transcript sync frame above the ack.
- **Load flake after the repair** (`[original-demand]` only): 1 of 12 runs failed in a first soak (load 40-50) and 1 of 1 at `-n 4`. The two messages were "No original natural poll reached the real tab lock" and "Unrelated full demand preceded the intervention". A probe of every `_console_sync_requested` setter showed the send's late follow-ups landing at load-dependent times: the acknowledgement release's `call_later` resync, image recovery and store listeners starting worker FULL passes, and the replay flag and its deferrals. One that reached the test's held tab lock first starved every poll tick of a visit pass; one that came while the poll was held left demand behind it. Waiting for a completed poll display first did not help (3 of 16). **Fix (test):** up to three holds. On either race the test releases the lock, waits until the in-progress, requested, tab, replay and pause flags settle, and holds the next natural poll. Both precondition assertions stay as final checks. With only the in-progress, requested and tab-count settle, 1 of 20 still failed. With the replay and pause flags added: **20 of 20** (4 runs at once, load 40-66). After review, the last try keeps its hold, so a race that survives all three tries is reported by its own assertion rather than as "no poll reached the lock".

**Runs.** The two files with the MCP and hook execution files at `-n 4`, twice in a row: 139 passed (241 s, load 44-50) and 139 passed (200 s, load 28-43). Each process exited 4-6 s after its summary. Serially the hook file passed 31 of 31 (139 s) in the macOS step run below, and the six poll nodes passed in 52 s (load 27-30).

**Lanes.**
- The poll file's six nodes are at the UI census tail; `MINIMUM_FILES` is the measured count. Times with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`, load 27-30), setup+call+teardown: `ordinary_preparing` 7.8 s, `in_place[disabled]` 7.3 s, `[enabled-control]` 11.4 s, `full_request[original-demand]` 5.8 s, `[preparing-display]` 5.6 s, `poll_terminal_transition` 10.4 s. The whole file took 52 s, over the 30 s entry budget.
- The hook file joins the macOS `plugins-recovery-ui` job as part of the new "Run managed native hook and MCP event suites" step, followed by a check that both suites executed. The brief's Linux homes would only skip it: its `native_console` fixture calls `require_managed_runtime()`, which is Darwin only. With the step's flags (`--timeout=300`, serial) at load 27-46 the step took 600 s; this file's share was 139 s.

Files: `Tests/Chat/test_native_plugin_hook_preparation.py`, `Tests/UI/test_console_poll_reconciliation.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
