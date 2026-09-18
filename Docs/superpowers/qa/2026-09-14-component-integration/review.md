# Read-only Python/test merge integration review

Review inputs: branch `705a39bdb88ef4f64a926eeec169f0d9a02469e4`, dev `fd30614dcdc1e6cbd39b1532769d3e10be9b12b6`, merge-base `2ecf784d6ea41592a9d9b6b915fbe8c8c533470f`; inspected the in-progress merged worktree. Root owns all resolution and verification.

## Findings reported to root

1. **P2 — New upstream inline height violates the integrated style floor.** `tldw_chatbook/Widgets/Library/library_notes_canvas.py:2274` added `container.styles.height = "auto"` while packing selected-folder actions across rows. This fresh container should receive `h-auto`, matching its migrated sibling toolbar constructors. Static inspection establishes a covered fixed height write; no pytest execution was performed here. Verify with `test_component_pattern_governance.py::test_python_style_ratchet` and the two packed-folder-action mounted checks below. Root acknowledged and plans the replacement after its red governance run.

2. **P2 — New upstream test guard reads the deleted Console stylesheet.** `Tests/UI/test_consolidated_css_harness.py:286` still includes `screen_agentic_console.tcss` in `_SPLIT_SHEET_OWNERS`. Lines 510–513 unconditionally call `_selector_tokens()` for every mapping key, which calls `Path.read_text()`. Preserving ADR-161's Console-sheet deletion therefore makes `test_no_harness_composes_split_sheet_widgets_with_the_bundle_alone` raise FileNotFoundError. Remove that obsolete mapping entry; Console styles now ride the bundle. Run this guard after rebuild; it may then identify real bundle-only harnesses that need the existing APP_STYLESHEETS shape. Do not globally weaken the guard or add a blanket missing-file skip.

## Upstream behavior retained

Static method-level overlap inspection covered all 16 Python files changed on both sides. Only three existing methods changed on both sides: `ConsoleSettingsModal._sync_responsive_layout`, `LibraryNotesCanvas._compose_editor`, and `LibraryNotesCanvas._compose_list`. Added upstream methods remain present. Reviewing the dev-to-merged diff of overlapping production files shows the established design migration, alongside the upstream behavior; no lost upstream behavior was identified.

Specifically retained: Console custom-endpoint identity and selection corrections, the single combined base-URL-change handler, the measured Settings wide-tier class, pending worktree merge wiring, Notes toolbar packing/title display changes, and File Notes editor/navigation/status additions. These are source-review observations, not a claim of passing runtime verification. Root preserved APP_STYLESHEETS together with BUNDLED_STYLESHEET in the two conflict imports so both sides' harnesses remain defined.

## Focused checks recommended to root

- All four cases in `Tests/UI/test_library_style_transitions.py` (retained heading/row dimensions, emergency canvas width, navigation handle).
- `Tests/UI/test_library_notes_w4_layout.py::test_notes_toolbar_paints_whole_labels_with_a_note_open_at_235x52`
- `Tests/UI/test_library_notes_w4_layout.py::test_notes_folder_actions_wrap_rather_than_run_off_the_pane`
- `Tests/UI/test_library_notes_w4_layout.py::test_a_pane_that_widens_again_records_the_width_it_was_given`
- `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_shared_shell_retains_state_across_breakpoints`
- `Tests/UI/test_library_file_notes_workspace.py::test_file_notes_disclosed_actions_fit_wide_and_compact_layouts`
- `Tests/UI/test_library_file_notes_git.py::test_narrow_editor_actions_keep_complete_labels_at_40_by_20`
- `Tests/UI/test_library_file_notes_git.py::test_wide_prepare_session_quiets_and_restores_editor_toolbars_without_remount`
- `Tests/UI/test_console_settings_geometry.py` (particularly the live 150-column wide-tier transition).
- The two exact governance/guard tests named in the findings, alongside root's build/token checks.

No production source, test source, index or HEAD edits; no tests, app launches or builds executed by this reviewer during this pass. CSS conflict resolution and rebuilt cascade validation belong to root. This report records findings when inspected; root may already have corrected them afterward.


## Authorized harness corrections and verification

Root subsequently authorized correcting the 13 harnesses identified by the red guard run in `/tmp/component-integration-harness-green.log` (1 failed, 6 passed). Updated only those CSS_PATH declarations to `[str(path) for path in APP_STYLESHEETS]`, updating/removing imports as needed:

- `Tests/UI/test_library_file_notes_workspace.py`: `_WorkspaceHarness`, `_TwoWorkspaceHarness`, `_DynamicWorkspaceHarness`.
- `Tests/UI/test_library_honesty_accessibility.py`: `_DatabaseNoteEditorApp`.
- `Tests/UI/test_library_media_toolbar_adapt.py`: `_CanvasApp`.
- `Tests/UI/test_library_media_trash.py`: `_TrashCanvasApp`, `_ListApp`, `_ConfirmApp`, `_ViewerApp`.
- `Tests/UI/test_library_shell.py`: `_LibraryRailStyleContractHarness`.
- `Tests/UI/test_library_style_transitions.py`: `_NotesHost`, `_MediaHost`, `HandleHost`.

The complete seven-test `test_consolidated_css_harness.py` module and focused mounted checks now pass: **30 passed, 2 warnings in 27.93s**, no failures. Raw output: `harness-targeted.log`. This run includes every changed harness through direct cases or the existing transition hosts, preserving app-tier styling while adding the lazy screen-owned sheets. No blanket harness change or weakened guard was introduced.

Exact test selection:

```
.venv/bin/python -m pytest \
  Tests/UI/test_consolidated_css_harness.py \
  Tests/UI/test_library_style_transitions.py \
  Tests/UI/test_library_media_toolbar_adapt.py \
  Tests/UI/test_library_media_trash.py::test_trash_canvas_renders_heading_rows_and_enabled_restore \
  Tests/UI/test_library_media_trash.py::test_media_list_toolbar_offers_trash_outside_select_mode \
  Tests/UI/test_library_media_trash.py::test_confirm_copies_and_receipt_point_at_trash \
  Tests/UI/test_library_honesty_accessibility.py::test_database_save_remains_visible_focusable_and_in_normal_pane_order \
  Tests/UI/test_library_shell.py::test_library_rail_applies_reversible_ordinary_width_contracts \
  Tests/UI/test_library_shell.py::test_library_rail_skips_unchanged_matching_width_contract_writes \
  Tests/UI/test_library_file_notes_workspace.py::test_poll_and_narrow_navigation_retain_the_text_area \
  Tests/UI/test_library_file_notes_workspace.py::test_overlapping_root_persistence_only_winner_updates_config_and_owner \
  Tests/UI/test_library_file_notes_workspace.py::test_fresh_shared_workspace_follows_committed_owner_root -q
```

Fatal Ruff (`E9,F63,F7,F82`) and `git diff --check` passed for all six edited test files. No production changes, CSS builds, staging or commits performed by this reviewer. No full Library files/suite run. Root independently owns the source/governance/live integration checks and the stale Console mapping/comment-parser guard correction.
