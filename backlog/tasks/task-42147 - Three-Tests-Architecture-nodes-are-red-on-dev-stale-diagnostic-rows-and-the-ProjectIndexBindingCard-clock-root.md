---
id: TASK-42147
title: "Three Tests/Architecture nodes are red on dev: stale diagnostic rows and the ProjectIndexBindingCard clock root"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:35
labels:
- follow-up
- tests
- architecture
priority: low
references:
- Tests/Architecture/test_persistent_diagnostic_inventory.py
- Tests/Architecture/test_timer_path_static_update_inventory.py
- tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py
- tldw_chatbook/Agents/agent_service.py
- tldw_chatbook/Chat/console_store_first_persistence.py
- tldw_chatbook/Agents/agent_native_execution.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the library stream (round 18). TASK-41307, TASK-41320, TASK-41637 and TASK-41640 recorded these as known base failures since at least fbf1c9ea74, and no task owns them. At d85e90bc9e (serial, through the test wrapper):
- `test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only` (line 620): two rows of `REVIEWED_METADATA_ONLY_DIAGNOSTICS` match nothing. `tldw_chatbook/Agents/agent_service.py` has no diagnostic containing "on_child_settled consumer raised" (6125fbc5b8 moved it to `Agents/agent_native_execution.py:1515`), and `Chat/console_chat_store.py` has none containing "Failed to flush Console roleplay context on first persist" (that message is now in `Chat/console_store_first_persistence.py`).
- `test_timer_path_static_update_inventory.py::test_clock_root_set_is_pinned` (line 2824): a new repeating-clock root, `set_interval` to `ProjectIndexBindingCard._poll` (`Widgets/Settings_Widgets/project_index_binding_card.py`).
- `test_timer_path_static_update_inventory.py::test_no_timer_path_update_defaults_to_layout_true` (line 2691): `.update(` calls reached from that root do not pass `layout=` (`_paint` line 167, `_feedback` line 194, `_launch` and `run` lines 356 and 365).

The card came with 2e9018b86d ("manage optional workspace keyword indexes").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two reviewed diagnostic rows name the diagnostics' current owners, or are removed with the reason, and `test_reviewed_diagnostic_changes_are_metadata_only` passes
- [x] #2 `ProjectIndexBindingCard._poll` is in the pinned clock roots, and each `.update(` it reaches either passes `layout=` (proved as the file's message requires) or has a `CLASSIFIED_SITES` row saying why it need not
- [x] #3 `test_persistent_diagnostic_inventory.py` and `test_timer_path_static_update_inventory.py` pass in a serial run
- [x] #4 These three nodes run in a PR lane (no PR lane runs them today), so a later drift fails CI
<!-- AC:END -->

## Implementation Plan

1. Move the two `REVIEWED_METADATA_ONLY_DIAGNOSTICS` rows to their current owners (`Agents/agent_native_execution.py`, `Chat/console_store_first_persistence.py`) with the same field pins, and map the moved TASK-15743 repair row through `_TASK_15743_CURRENT_OWNERS`.
2. Review `ProjectIndexBindingCard._poll` (what it repaints, how often), pin it in `EXPECTED_CLOCK_ROOTS`, and classify each `.update(` it reaches: `CLASSIFIED_SITES` rows for the method-level writes; explicit `layout=` for writes inside the nested `run` closure (the stale-row check keys a call by its innermost function, the census by every enclosing reachable function, so no row can cover them).
3. Lane: the diagnostic node beside its sibling node in pr-fast-lane (architecture-guards refuses a file another PR lane lists); the timer-path file whole in architecture-guards' sandboxed block. Update the Tests/CI target lists.
4. Run both files serially, Tests/CI, the whole Tests/Architecture suite serially, preflight and ruff gates.

## Implementation Notes

Red at base (dc371f0842, serial through the wrapper, load ~10): the two files gave 3 failed, 96 passed, 1 skipped in 280 s, exactly the three nodes. After: 99 passed, 1 skipped (the TASK-15743 archaeology skip, commits not fetchable here) in 429 s at load 17-35.

- **Diagnostic rows (AC #1).** Both rows moved, metadata-only pins unchanged: "on_child_settled consumer raised" (`type(exc).__name__`) to `Agents/agent_native_execution.py`, "Failed to flush Console roleplay context on first persist" (no fields) to `Chat/console_store_first_persistence.py`. 6125fbc5b8 moved both. The first is also a TASK-15743 repair row, so `_TASK_15743_CURRENT_OWNERS` maps it to its new owner the way the fleet.py moves are mapped; without that the archaeology delta test would fail on a machine that still has the commits.
- **Clock root review (AC #2).** `set_interval(2, ProjectIndexBindingCard._poll)` runs for the card's life but does nothing unless a build is running (state Building, no upgrade pending, no load in flight). During a build it reloads every 2 s on a thread and repaints the status copy, the feedback line and the action buttons. It is pinned in `EXPECTED_CLOCK_ROOTS`. `_paint`'s status write and `_feedback`'s write are NEEDS-LAYOUT rows: both are height:auto Statics in a Settings pane as narrow as 34 columns whose wrapped copy changes height (counts grow; guidance can turn into busy or error copy). `_feedback` is not equality-gated, so an unchanged Building guidance costs one layout pass per 2 s tick while a build runs; the row says so. The two "Keyword index unavailable" writes sit in the nested `run` closure. The census keys them under both `_launch` and `run`, while `test_classified_sites_are_not_stale` looks a call up by its innermost function only, so a `_launch` row would be reported stale. They now pass `layout=True` explicitly (the default, so no behaviour change), with a comment: one line replacing wrapped copy, and a None status stops the poll.
- **Lanes (AC #4).** `test_reviewed_diagnostic_changes_are_metadata_only` rides in pr-fast-lane next to its file's existing node (5.5 s serial at load 10). architecture-guards refuses a file that another PR lane lists. The timer-path file runs whole in architecture-guards' sandboxed block, before the Tests/DB entry (19 tests, ~37 s serial). `Tests/CI/test_ci_queue_pressure_contract.py`'s FAST_LANE_TARGETS and ARCHITECTURE_GUARD_TARGETS list both. Tests/CI at `-n 4`: 1811 passed, 2 skipped.

Files: `Tests/Architecture/test_persistent_diagnostic_inventory.py`, `Tests/Architecture/test_timer_path_static_update_inventory.py`, `tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
