# Console conversation UX implementation evidence

Implementation: TASK-32826, TASK-32827, TASK-32828; branch `codex/console-unread-inspector`.
Design authority: [ADR-171](../../../backlog/decisions/171-console-conversation-review-and-attention.md) and the approved [specification](../specs/2026-09-18-console-conversation-review-and-attention-design.md).

## Delivered behavior

- Durable local unread reminders use serialized compare-and-clear revisions. Current-chat refreshes, duplicate tabs, cancelled navigation and stale profile/paint callbacks preserve reminders; successful deliberate revisits clear them independently of operational receipts.
- Compact conversation rows and workspace children use one right-edge attention/action slot, meaningful Unicode and ASCII states, custom appearance restoration, and a combined menu. Existing ownership, paging, identity and subagent hierarchy remain in place.
- Conversation Inspector uses Context, Usage & cost, and Exchange history with lazy section/detail readers, narrow Back navigation, explicit target identity, freshness/budget information and retained capture/export safeguards. The Inspector sidebar is outside this change.

## Verification

Only targeted tests were run. The broad feature regression run produced **698 passed, 9 failed, 3 deselected** in 476.79 seconds. All nine failures below reproduce on the untouched approved-design baseline `fda0db1fbd`; none was suppressed by changing a budget, fixture checksum or unrelated implementation.

- `Tests/UI/test_console_workspace_tree_cursor_layout.py::test_non_boundary_cursor_move_arms_zero_screen_layout_passes`
- `Tests/UI/test_console_workspace_tree_cursor_layout.py::test_tree_tooltip_stays_correct_across_width_changes`
- `Tests/UI/test_console_workspace_tree_cursor_layout.py::test_tooltip_target_removed_by_projection_push_is_not_served_stale`
- `Tests/UI/test_console_modal_dismissal.py::test_console_modal_launch_declarations_match_runtime_construction`
- `Tests/UI/test_console_modal_dismissal.py::test_console_modal_inventory_matches_runtime_ast_and_transitive_launches`
- `Tests/UI/test_console_modal_dismissal.py::test_task2_modal_contract_table_is_complete_and_adopted`
- `Tests/UI/test_console_modal_dismissal.py::test_settings_clean_close_sources_restore_opener_focus[escape]`
- `Tests/UI/test_console_modal_dismissal.py::test_settings_redirected_select_click_uses_real_mro_dispatch`
- `Tests/UI/test_console_modal_dismissal.py::test_task2_contract_selector_exists_and_escape_returns_cancel_result[ConsoleModelPopover]`

The three explicitly deselected baseline failures were the frozen old tree projection checksum and the two cost-chip send-path tests (`test_keyboard_send_cancels_idle_refresh_before_run_starts`, `test_predispatch_echo_stays_in_context_but_not_current`). Each was separately reproduced on the same baseline. The optional preimport budget also fails identically at 515 modules against 500; its threshold was not widened.

The broad run covers local marks, attention/actions, conversation/tree projections, session and switcher navigation, activity receipts, costs, trace projection/export, modal lifecycle, design tokens, generated CSS, and the boot CSS budget. It emitted an open-file-descriptor growth warning; this warning has not been attributed to this change and is not claimed resolved.

The final review fix run passed 46 tests. A subsequent focused run passed 55 checks, including nine temporary screenshot captures; the screenshot helper was then removed. Rendered examples use synthetic content, production APP_STYLESHEETS, and 80x24, 120x40 and 160x48 viewports. Permanent regressions assert reader/frame/Close bounds and keyboard Back/focus, including normal `m` input in the composer. The user-guide illustration was refreshed from the production-style Context render.

Final clean qualification: **131 passed** in 84.83 seconds, covering the final modal workflow/readers, manual unread navigation, Inspector route factories, durable marks, pure presentation/attention, design-token governance, generated CSS synchronization and boot CSS budget.

```sh
PYTHONPATH="$PWD" python -m pytest Tests/UI/test_console_inspector_workflow.py Tests/UI/test_console_inspector_detail_pane.py Tests/UI/test_console_manual_unread.py Tests/UI/test_chat_screen_console_inspector_loader.py Tests/Chat/test_conversation_local_marks_service.py Tests/Chat/test_console_inspector_presentation.py Tests/Workspaces/test_conversation_attention.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Performance/test_boot_css_byte_budget.py -q
```

The flat-list owner and recompose guard qualification passed **58 tests** in 89.22 seconds. Five obsolete assertions for a left-side appearance button and three-line resting rows were updated to the approved single right action, compact height, saved colour, ASCII indicator, and menu-to-picker contract. The unsaved-row test now explicitly checks that the combined menu remains available rather than passing over a removed selector.

```sh
PYTHONPATH="$PWD" python -m pytest Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_workspace_tray_recompose_guard.py -q
```

Static checks: zero Ruff diagnostics on changed lines/new files, range formatting for modified legacy code, and `git diff --check`. Unrelated diagnostics in legacy modules remain. CSS was rebuilt from source; boot parsing is 765335/768000 bytes.

## Review fixes

A fresh read-only branch review found no Critical issues and two Important issues. Both were reproduced before fixing: durable `stuck` receipts now project intervention-required attention, and the cost-button entry supplies the same captured-session prepared-input estimator as direct Context entry. A keyboard regression also reproduced a hidden-list focus problem; entering narrow detail now focuses the visible reader. Obsolete prefetch/layout docstrings and the temporary render-only test were removed.

Ruling: shared unread, row and menu changes were implemented together because their projections and controls overlap. Legacy nested-expander DOM tests were replaced with section/detail behavior tests while retaining privacy, export, recovery, identity and accounting coverage. The existing session-close `session_id` use-before-assignment was repaired after baseline reproduction because it blocked navigation verification.

## Outstanding qualification

Native terminal verification is **not complete**. Computer-use access to iTerm was explicitly rejected as disallowed, and no Windows Terminal environment was available. Headless geometry/screenshots are not presented as native glyph, pointer, or restart evidence. The Backlog tasks remain In Progress under the approved plans until that qualification is completed. No merge or push was performed.
