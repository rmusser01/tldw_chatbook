---
id: TASK-34918
title: The Console and Library screen files are over their size budgets on dev
status: Done
created_date: 2026-10-12 22:38
labels:
- architecture
- tests
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-14 16:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The screen size ratchet pins a line budget per screen file that may only go down. Since the Dreams and Guardian merge on 2026-10-12 (PR #2921, 22:07Z) it fails on dev for two files: chat_screen.py at 25,209 lines against 25,204, and library_screen.py at 36,969 against 36,968. The ratchet is not in the required check, so the merge was not blocked, and every later change to either file inherits the red. PR #3091 (TASK-34789) adds 13 lines to chat_screen.py for a product fix and lands on top of it. The owner's rule is that a budget is never raised to fit growth; the files must shrink.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The screen size ratchet passes on dev for chat_screen.py and library_screen.py with their budgets unchanged or lowered
- [x] #2 The lines removed move into owner modules that tests already cover, with no behaviour change
- [x] #3 Code that no caller reaches (a shadowed duplicate definition, methods with no caller, imports left unused) is deleted rather than moved, each confirmed by an untruncated repo-wide search
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Measured at dev 1ef4fcf4ae: chat_screen.py 25,349 lines / 762 methods (budget 25,204 / 756); library_screen.py 37,103 / 1,345 (budget 36,968 / 1,343). Both the line and the method budgets are over, so lines AND methods must leave each class. Budgets stay unchanged; the 200-line slack test caps the shrink at 200 below budget.

1. Baseline: run Tests/Architecture serially on the unchanged branch and record every failure with its value.
2. chat_screen.py: delete code no caller reaches, each confirmed by an untruncated repo-wide search: the first of two on_button_pressed definitions (shadowed by the second since PR #2643's merge, so Python never binds it), and four methods with no caller (_reserve_console_default_intent, _publish_console_default_outcome, _clear_console_footer_shortcuts, _configure_console_copy_block).
3. chat_screen.py: move two pure staticmethods to the pure Chat modules that own their types, call sites retargeted: _console_badge_inspector_rows -> Chat/console_rail_state.py (beside the badge builder that consumes the rows), _launch_targets_chatbook_artifact -> Chat/console_live_work.py (beside ConsoleLiveWorkLaunch).
4. library_screen.py: delete five methods with no caller (_structural_records_for_comparison, _decorative_rail_counts_for_comparison, _safe_sync_scope_text, _hub_source_count_label, _library_workspace_scope_label) and the unused imports that are not part of the Library_Modules re-export contract or reached as library_screen.<name>.
5. library_screen.py: move the TASK-21243 ordinary-shell builders (_build_library_ordinary_shell_trio, _library_single_notes_stage) into Library_Modules/library_browse_route_swap.py, their co-owner, keeping its lazy import.
6. Measure; rerun Tests/Architecture serially and compare failures by value; run the UI/Chat tests that exercise every moved or deleted symbol on this branch and on a clean dev worktree; run the markup census, preflight; report the diagnostic-inventory drift without regenerating.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Both screens are back under their unchanged budgets. Measured on dev 1ef4fcf4ae, which had grown past the numbers in the description (both the line and the method budgets were over):

| file | before | after | budget |
|---|---|---|---|
| chat_screen.py (ChatScreen) | 25,349 lines / 762 methods | 25,012 / 755 | 25,204 / 756 |
| library_screen.py (LibraryScreen) | 37,103 / 1,345 | 36,915 / 1,338 | 36,968 / 1,343 |

Budgets are not lowered. Chat sits 192 lines under, inside the 200-line slack test, so the in-flight PRs that add lines keep room to land. AC #3 was added during the work: most of the removed lines were unreachable code, which is deleted, not moved.

**chat_screen.py**
- Deleted the first of two `on_button_pressed` definitions (204 lines). PR #2643's merge (2026-09-12) left an older copy above the live one, and Python binds only the last definition, so this copy never ran (ruff F811). An AST check confirms the live handler's text is byte-identical before and after.
- Deleted four methods nothing calls, each confirmed by an untruncated repo-wide search: `_reserve_console_default_intent` (never called), `_publish_console_default_outcome`, `_clear_console_footer_shortcuts` and `_configure_console_copy_block` (callers removed in July). Also removed the two imports that only the first two used.
- Moved two pure staticmethods into the pure Chat modules that own their types, and retargeted the call sites: `_console_badge_inspector_rows` became `console_rail_state.console_badge_inspector_rows`, and `_launch_targets_chatbook_artifact` became `console_live_work.launch_targets_chatbook_artifact`. Both modules are already on the boot path, so the import census is unchanged.

**library_screen.py**
- Deleted five methods nothing calls: `_structural_records_for_comparison` and `_decorative_rail_counts_for_comparison` (callers removed by 0dfadf4637), `_safe_sync_scope_text`, `_hub_source_count_label` and `_library_workspace_scope_label`.
- Removed 19 unused imports. None is in the Library_Modules re-export contract or reached as `library_screen.<name>` (AST scan). `set_mode` stays because a test reaches it through the module.
- Moved the TASK-21243 ordinary-shell builders (`_build_library_ordinary_shell_trio`, `_library_single_notes_stage`) into `Library_Modules/library_browse_route_swap.py`, which already calls them, and kept the lazy import used by `compose_content`'s other route-swap import.

**Evidence**
- Tests/Architecture was run serially before and after; failures are compared by name and value. The two screen-ratchet failures are gone.
- The one new red is expected drift: the diagnostic inventory row for chat_screen.py goes from 124 to 123 calls, because the deleted duplicate held one copy of the `logger.info` that the live handler still has. It is not regenerated here.
- `test_surface_swap_guard[#outgoing]` failed twice while the machine was overloaded (load average around 30), then passed 16 out of 16 runs. It uses a self-contained app.
- The Console and Library UI and Chat test files that exercise every moved or deleted symbol were run on this branch and on a clean dev worktree with the same arguments. They show no failure that is new on this branch; the differences are timing flakes that fail on both trees when rerun.
- Preflight passes except for the inventory drift above. The markup census and the textual worker contract checker pass.
- /verify (an app launch) was not run: this session forbids launching the app.
The status stays In Progress. Preflight's production diagnostic inventory check stays red until the inventory is regenerated for the chat_screen.py row (124 to 123 calls); move the task to Done after that.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
