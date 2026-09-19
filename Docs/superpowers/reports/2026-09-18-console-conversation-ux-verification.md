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

## Linux SSH verification

The user authorized verification on their Debian 13 host. Ran the real Textual
8.2.8 application under Python 3.12.8 in a dedicated tmux session, with committed
source from `b9106e2722`, a temporary virtual environment and a disposable profile.
The configuration parsed successfully and resolved its data path inside that
profile. No real conversations, provider credentials or model requests were used.

Terminal-driver keyboard and SGR pointer events exercised:

- Opening saved chats through Ctrl+K History; one-line rows and the right action
  target; the combined menu and its existing appearance picker.
- Marking the current conversation unread: the custom lightbulb became an envelope
  while the transcript remained open. SQLite confirmed the durable mark. A full
  process restart preserved both reminders; explicitly reopening one conversation
  cleared only its mark and restored its saved icon.
- Context, Usage & cost and Exchange history at 160x48, plus Context resizing at
  120x40 and 80x24. The narrow reader retained Back, payload controls and Close;
  Shift+Tab then Enter returned to the section list. Widening restored the split
  layout. Escape dismissed the modal.
- Estimated versus unpriced usage, and selection of a synthetic legacy capture
  with request/response details masked by the Safe viewer. Captures were seeded
  after opening the fixture because initial fixture captures were no longer
  present after the earlier startup/resume sequence; this does not qualify capture
  creation or retention across restart.
- ASCII row indicators, `m` menu access on both the flat list and workspace tree,
  and the workspace unread action.

Three live defects were corrected and rechecked:

1. The appearance menu label clipped to “Change icon and”. Shortened it to
   “Icon and colour…”, which fits the existing menu.
2. `str(ConsoleMessageRole.ASSISTANT)` exposed an internal enum name and counted
   estimated assistant tokens at the input rate. Normalize the enum value before
   estimation, pricing and display. The new enum regression failed before the fix;
   live rows now show `assistant`, zero estimated input and nonzero output.
3. Textual's non-CSS line padding clipped ASCII attention labels inside their
   nine-cell action target. Clear that padding without widening the shared token.
   A production-style painted-cell regression reproduced `[approv` and now checks
   every attention label fits in full.

Final targeted regression after all three fixes: **155 passed** in 132.64 seconds.
The suites cover cost accounting, conversation actions, Inspector presentation and
workflow, rendered menu indicators, and the flat-list owner. Changed-line Ruff
reported zero diagnostics; touched-range formatting and `git diff --check` passed.
The final three remote implementation files matched local SHA-256 hashes.

```sh
PYTHONPATH="$PWD" python -m pytest Tests/Chat/test_console_cost_tracker.py Tests/Chat/test_console_conversation_actions.py Tests/Chat/test_console_inspector_presentation.py Tests/UI/test_console_inspector_workflow.py Tests/UI/test_console_conversation_action_menu.py Tests/UI/test_console_workspace_context_rail.py -q
```

Plain and ANSI tmux captures and fixture database observations were saved in the
remote temporary evidence directory. These establish terminal-driver behavior and
server-side cell rendering; they are not screenshots of a terminal emulator and
cannot establish client font, pixel, hover or physical pointer behavior.

## Outstanding qualification

Native terminal verification is **not complete**. Computer-use access to iTerm was explicitly rejected as disallowed, and no Windows Terminal environment was available. Headless geometry/screenshots are not presented as native glyph, pointer, or restart evidence. The Backlog tasks remain In Progress under the approved plans until that qualification is completed. The subsequent PR follow-up is recorded below.


## Rebase and PR review follow-up

Rebased the seven Console-specific commits onto dev `cef6bd2a3e`, preserving its
new CSS source split and profile-recovery lifecycle. Full-app fixtures use the
existing private-profile subprocess helper, and cost arithmetic fixtures use a
seeded pricing catalog. Async menu/recompose tests wait for their actual targets.

PR #2725 Qodo review identified six issues, addressed together: public API
contracts, the unread target annotation, a shared narrow-layout breakpoint,
retained abandoned-call labels in individual readers, disabling unavailable usage
drill-in, and exact shared action widths in workspace trees. Regression coverage
selects missing/ambiguous usage identities, individual abandoned calls, and both
Unicode and ASCII action widths.

The post-review targeted run passed 318 cases and exposed one synthetic F1
projection fixture racing startup refresh. Waiting for the owning workers before
injecting the projection resolves that case; its focused rerun passed. The full
suite was not run. Changed-line Ruff and `git diff --check` pass.

CI follow-up defers Inspector imports to opening the modal, replaces broad
Inspector control selectors with a dedicated class, and records the off-loop
unread batch worker needed for restored reminder indicators in the boot census.
No boot module or CSS budget ceiling was raised. Final remote checks and review
completion are tracked on the PR. Native client rendering limitations above remain.

Final boot follow-up: reading an uninitialized receipt snapshot no longer loads
its coordinator; hydration/settlement retain that ownership. Capture-policy
bindings and their dialog also load only when invoked. The startup module census,
receipt service, affected controller binding, and Inspector loader checks passed:
**38 passed**. The other boot guard tests passed (**19 passed**); the module-census
failure from the earlier intermediate run is covered by this successful rerun.

Diagnostic inventory review: the Inspector redesign removed two old
`exchanges_loader` error statements (one per former turn view). Statement-level
comparison against inventory pin `149acda36be` confirmed exactly two removals,
zero additions, and no sink-topology change. Regenerated
`Docs/security/production-diagnostic-inventory.json` to reflect those removals.
The profile-owned path inventory also passes unchanged.
