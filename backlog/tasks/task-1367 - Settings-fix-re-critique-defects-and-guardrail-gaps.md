---
id: TASK-1367
title: Settings fix re-critique defects and guardrail gaps
status: Done
assignee: []
created_date: '2026-08-05 17:07'
updated_date: '2026-08-05 20:34'
labels:
  - settings
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique after the 1338-1346 fix round (.impeccable/critique/2026-08-05T16-56-50Z__tldw-chatbook-ui-screens-settings-screen-py.md, score 30/40) found 2 defects in the new work and 2 unguarded destructive actions: (1) model-catalog Automatic refresh group renders as an empty bordered box - all toggles clipped and unusable (height:1fr unresolved in auto-height card); (2) the s/r/t swallowed-key notice is unreachable by real keyboard (Textual Input consumes printable keys; notice only fires on direct action invocation; regression test bypasses key dispatch); (3) Run manual sync pushes pending Notes/Chat to a server with no confirmation (settings_screen.py:8642-8653); (4) theme Delete unlinks a user file with no confirmation or undo (settings_theme_editor.py:583-598).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model-catalog group and all its controls render and are usable at 120x35/100x30/80x24,Dead s/r/t notice machinery removed or repurposed with regression tests driving real key presses that document actual field behavior,Run manual sync requires a confirmation showing pending counts before any server push,Theme Delete of a user theme requires confirmation,Relevant suites green and backlog notes updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix 1: add height:auto to .settings-instant-apply-group in _agentic_terminal.tcss, rebuild modular css, Pilot probe at 120x35/100x30/80x24
2. Fix 2: remove _settings_shortcut_swallowed_by_text_entry + 3 call sites + dedup state; rewrite footer-hint tests with real pilot.press
3. Fix 3: ConfirmationDialog before Run manual sync, seeded with pending counts
4. Fix 4: ConfirmationDialog before user theme Delete
5. Run verification suites; update task-1340 notes; mark ACs; Implementation Notes
ADR required: no
ADR path: N/A
Reason: UI confirmation dialogs + CSS + test honesty only
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR required: no (UI confirmation dialogs + CSS + test honesty; no storage/sync-policy/provider-boundary decision). Implemented all four re-critique fixes:

1. Model-catalog group invisible: added `height: auto` to `.settings-instant-apply-group` in tldw_chatbook/css/components/_agentic_terminal.tcss (Textual's default Vertical 1fr never resolves inside the auto-height providers card, clipping the whole group) and regenerated tldw_chatbook/css/tldw_cli_modular.tcss via build_css.py. Verified with a new real-stylesheet Pilot suite Tests/UI/test_settings_model_catalog_layout.py (reuses _SettingsCssHarness/_scrolled_region_rows): at 120x35/100x30/80x24 the group sizes to content (20/22/21 rows) and the 'Automatic refresh' header plus 'applies immediately' hint render when scroll-stepping the detail pane. The compact <=90-col block only targets workbench/panes/search-help selectors and does not interact (80x24 probe passes).

2. Dead s/r/t swallowed-key notice (task-1340 follow-up): removed _settings_shortcut_swallowed_by_text_entry, _settings_text_entry_has_focus, the _swallowed_notice dedup state, the now-unused `import time`, and the allow_text_entry_focus escape hatch from the save/revert/test actions and their button callers. Printable keys legitimately yield to focused text entry (the field IS the feedback); the notice was unreachable by real keyboards. Tests rewritten to drive REAL pilot.press sequences: test_printable_shortcut_keys_type_into_focused_text_fields (s/r/t type into the field, no action fires) and test_shortcut_keys_fire_actions_outside_text_entry (s -> clean-save notice, r -> revert ConfirmationDialog, t -> stub toast). The capability-probe drift guard is intact (kwarg dropped from its calls). Also dropped the direct-action-call half of test_settings_provider_text_inputs_do_not_trigger_footer_shortcuts (it pinned the removed guard) and two obsolete _settings_text_entry_has_focus monkeypatches in test_settings_configuration_hub.py. Task-1340 Implementation Notes updated with a one-line follow-up.

3. Run manual sync: handle_manual_sync_run now pushes the existing ConfirmationDialog ('Push pending Notes/Chat changes to the server now? Pending outgoing: <counts>') seeded from the already-loaded preview rows; only the confirm callback sets the running rows and starts _manual_sync_run_worker. New Pilot test test_settings_manual_sync_run_requires_confirmation_with_pending_counts (fake control service, no network) asserts the dialog shows notes/chat counts, run_once is not called before confirm, cancel is a no-op, and confirm runs exactly once with the active server profile.

4. Theme Delete: on_delete_theme keeps the built-in/shipped/missing guards unchanged; the user-file path now pushes a ConfirmationDialog ('Delete the saved theme X? This cannot be undone.') and the unlink/tree-removal/reset moved to _delete_user_theme(theme_path, theme_name), invoked only on confirm with the name captured at dialog time. The two user-file delete tests in test_settings_theme_editor.py now drive the real dialog (cancel preserves, confirm deletes) via a new _isolated_editor_app_with_real_screens helper that restores the real push_screen/pop_screen the isolated harness mocks away.

Verification (all foreground): test_settings_configuration_hub.py 248 passed; test_settings_footer_hints.py 9; test_settings_theme_editor.py 10; test_settings_category_sweep.py + test_settings_save_commit_models.py 9; test_settings_narrow_layout.py + test_settings_model_catalog_layout.py 12 (22 with theme editor); test_screen_footer_hints.py 9. Ruff: no new findings (5 remaining are pre-existing at HEAD, confirmed by diffing against committed files).
<!-- SECTION:NOTES:END -->
