---
id: TASK-38701
title: "Console archive-recovery flow tests: the plateau test is red, and a mounted send was refused in 2 of 2 runs"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 04:27
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_archive_recovery_flow.py
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
dependencies:
- TASK-38652
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down. Checked on dev 9bbd5c2515, 2026-11-03: two whole-file runs with `-n 4` (load 24-25) each gave 2 failed and 41 passed.
- `test_retained_native_recovery_resource_owners_plateau_and_close` failed in both runs at `assert stable[-1]["live_handles"]["chat"] >= 2` (`assert 1 >= 2`).
- In 2 of 2 runs, a different mounted-recovery test failed with `_ConsoleTurnRefusedError: Console turn was refused before durable acceptance.` (console_runtime.py:3388): `test_native_result_refresh_remains_manual_and_imported_json_is_inert` in run 1 and `test_native_date_search_labels_monday_chat_friday_workspace_and_unknown[False]` in run 2. A reviewer's own run of the file saw no such refusal, so this one is intermittent.

The file is in no PR lane (not in the UI census, no workflow target), so neither failure has been reported. PR #3366 (TASK-38652) edits archive_recovery.py's sync, which these tests drive.

Depends on TASK-38652 (in PR #3366).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_retained_native_recovery_resource_owners_plateau_and_close passes on dev, with the reason it saw one live chat handle where it expects two recorded
- [x] #2 The refused-turn failures are explained, and the file passes 5 of 5 whole-file -n 4 runs on dev
- [x] #3 The file runs on a PR lane, or the task notes record why it does not
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both failures on this base; probe who closes the plateau's worker handles and which path refuses the mounted turn.
2. Fix the test where its expectation is stale, the product where the product is wrong.
3. Show 5 of 5 whole-file `-n 4` runs green; put the file on a PR lane or record why not.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Neither failure came from the replay loop (TASK-38696). Both were test faults, so both are fixed in the test.**

**AC#1, the plateau (`assert 1 >= 2`).** Red on base c2cdb6b82e in every run (also when run alone). The sample dump shows a flat plateau: OS FDs 47, chat FDs 5, and `chat_registered` 1 across all five cycles. `threads_seen` chat=3 shows that both executor workers opened chat handles, but only the main-thread handle the test opened itself was still live. A profiler probe on `sqlite3.Connection.close` found the worker handles being closed by the operation that opened them. The close sites were operation-owned: `base_db.operation_owned_connection` via `console_library_policy_coordinator.owned_call`, the conversation browser's `read_in_worker` (`close_connection`), `character_context`, `console_context_repository`, `console_native_commit`, and for workspace handles the archive `search_service`. The test's `>= 2` assumed worker handles stay open; that was the 2026-10-18 qualification's observation. The test now asserts what it can observe: workers opened chat and workspace handles (`threads_seen >= 2`), and the plateau holds only the main-thread handle (`live_handles == 1`). I did not bisect which change stopped the retention (it would need a second worktree).

**AC#2, the refused turns.** A recorder on `ConsoleSubmitResult` gave three refusal messages: "Hooks unavailable; review or disable hooks before sending.", "Required hook initialization failed.", and "Couldn't save the prepared turn." (`Durable Console Library policy no longer matches acceptance.`). They came from 1 of 9 plain base runs and 8 of 12 runs with two at once (load 19-50). All three have one cause. After mount, `mounted_recovery` set `controller.app = host`. The Console resets `controller.app` to its `app_instance` on every `_ensure_console_chat_controller` (`wiring.attach_review_presenter`). Each turn's preparation reads record `controller.app` when they start and refuse if it changes before they finish: hook admission (`console_automatic_submission`), hook preparation (`console_hook_preparation`), and the Library-policy capture (`console_received_dispatch.preparation_native_runner`, which falls closed to an "unavailable" authority that the durable commit then rejects). A before/after diff around the hook read showed that only `controller.app` changed. Deterministic reproduction: setting `controller.app = host` just before the turn, with an ensure forced inside the hook read, refused 2 of 5 sends. With the harness line removed, the same forced ensure refused none (5 of 5). In production every writer assigns the same app object, so the product is not affected. Fix: the harness no longer reassigns `controller.app`.

**Evidence after both fixes** (branch, `-n 4`): 5 of 5 whole-file runs gave 43 passed (load 14-27), and 4 of 4 runs with two at once gave 42 passed (plateau deselected, load 16-20) with no refusal logged. Base at the same plain load also showed `test_receipt_refresh_refusal_keeps_native_source_safe[source_switch]` failing 2 of 9 runs at `assert not rig.console.query(".archive-recovery-receipt")`. It never failed in 19 branch runs, but I did not establish its mechanism; it is listed as a follow-up.

**AC#3, the lane.** The whole file is 42 mounted-Console cases, ~6 serial minutes here (call times 3-24 s), too heavy for one 20-minute UI shard. So two node ids are appended at the census tail: the plateau test, and `test_native_result_refresh_remains_manual_and_imported_json_is_inert` (a mounted send through the fixed harness, plus TASK-38688's imported-JSON copy). `MINIMUM_FILES` is 1373. Both are unverified on Linux CI; the plateau test skips without `lsof`. Review follow-up (2026-11-04): the plateau node's call time is 15-24 s (5 runs here, 15.4-19.8 s, under a concurrent whole-file run at load ~14; the reviewer measured 23.7-24.1 s). The census comment now says so. Its plateau-equality assert passed in all 5 runs.

**Files.** `Tests/UI/test_console_archive_recovery_flow.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md` (lesson: a harness that reassigns an owner the product keeps rebinding).

**Related (coordinator item).** The Perf Guard crash ("`archive_recovery.sync` calls `presentations()` after the runtime is disposed") is the same view sync firing after teardown. It is fixed at its source in a separate commit (sync returns when `ensure_archive_recovery()` is `None`). `Tests/UI/test_console_archive_recovery_cards.py::test_sync_whose_runtime_is_disposed_in_flight_stops_without_crashing` is red before the fix with the reported `AttributeError`.
<!-- SECTION:NOTES:END -->
