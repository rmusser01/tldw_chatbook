---
id: TASK-38740
title: "Audit every Button rule that can leave a label zero content width in a grid"
status: Done
assignee: ['@claude']
created_date: 2026-11-04 03:09
updated_date: 2026-11-05 08:56
labels:
- follow-up
- ui
- crash
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-38710 and its review (2026-11-04). In Textual 8.2.8 a grid sizes an auto row with `get_content_height(..., column_width - gutter)`, which ignores `min-width`. `Content.get_height` then subtracts both of Button's `line-pad` cells. A Button styled `min-width: 0; padding: 0 1` in a 4-cell grid column therefore wraps its label at width 0, and Rich's `chop_cells(word, 0)` raises `ValueError: range() arg 3 must not be zero` during layout, which exits the app. Only a content width of exactly 0 raises; a negative width does not.

TASK-38710 fixed the File Notes action-button rule and the Session Git panel's button rule by making their labels truncate (`text-wrap: nowrap; text-overflow: ellipsis`). Other Button rules in the app with the same shape (`min-width: 0`, horizontal padding, inside a `1fr` grid or a narrow column) have not been checked, and the crash only shows at particular widths and heights, so a fixed-size test does not find it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Button rule in the app's stylesheets that sets `min-width: 0` (or a min-width below its padding plus line-pad) is listed in the task notes with its verdict: guarded, unreachable at width 0, or fixed
- [x] #2 No screen reachable at a terminal width of 40 or more raises during layout because a Button label reached zero content width; a test sweeps the fixed rules' screens across widths and heights and is red before the fix
- [x] #3 A check (test or preflight) fails when a new Button rule reintroduces the unguarded shape, or the notes record why a static check cannot express it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the failure condition in Textual 8.2.8 / Rich 14.3.3 directly: which `text-wrap` x `text-overflow` combinations raise for a Button-shaped Content (line-pad 1) at each content width, in both the height (layout) and render paths.
2. Enumerate every Button-targeting rule with a low `min-width` across the app's `.tcss` modules and the `DEFAULT_CSS` / `BUNDLED_CSS` / `BUNDLED_SCREEN_CSS` / `CSS` literals (scratch probe), and sweep every top-level destination of the real app across widths 40+ at two heights to find reachable crashes.
3. Decide between per-rule fixes and one app-wide Button rule; if app-wide, compare painted Button labels at wide sizes against base.
4. Fix, add a sweep test per fixed screen (red on base), and add a static check that fails on a new unguarded rule.
   (Deviation, see notes: the fix is one guard in the package's Textual shims, not per-rule or global CSS.)
5. Record the rule list with verdicts, the red/green evidence and the wide-size comparison in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix: one guard where every fold happens, not CSS.** `tldw_chatbook/__init__.py` `_install_textual_compatibility_shims` (run by `tldw_chatbook.UI` and `tldw_chatbook.Widgets` on import) now wraps `textual.content.Content._wrap_and_format`. Every `Content.get_height` (layout, grid auto rows included) and `render_strips` (paint) goes through it. When the fold width is 0 (`width == 0` with no wrap, `width == 2 * line_pad` with wrap) and overflow is `fold`, it renders as `text-wrap: nowrap; text-overflow: clip`: one line cropped to the box. Every other width calls upstream unchanged. The repo already patches Textual this way (`Utils/textual_css_fastpath.py`).

Why not CSS (measured, base b8608e4cdb, real `TldwCli` in `run_test`, all 15 destinations, plus every Settings category at 211x44/235x52):
- Per rule cannot hold. The two crashes found on base were not `min-width: 0` rules. `#settings-workspaces-list .settings-workspace-row` (`css/features/_settings.tcss:1146`, crashed at 44 columns, heights 20 and 30) and `#settings-console-context-memory-jump` (`:1422`, crashed at 46) keep `min-width: 16`. `max-width: 100%` of a 2-cell parent wins over it (max is applied after min); below about 60 columns the Settings detail pane is a few cells wide. A grid's auto row ignores `min-width` too. The Settings Theme `OptionList` (`#settings-theme-list`) raised the same `ValueError` at a zero-width box.
- `Button { text-overflow: ellipsis; }` stopped every raise. But Textual truncates a wrapped, unfolded line to the full content width and then adds line-pad. Overflow probe: 17 Buttons painted 1-2 cells past their region and shifted the rest of the row, among them the Research pane handles at 211x44 (`<---` in 4 cells) and 13 Settings buttons below 62 columns.
- Adding `text-wrap: nowrap` stopped that, but no-wrap drops line-pad: 2,013 of 4,610 painted labels moved at 211x44 and 235x52.
- With the guard, the wide comparison is 0 of 4,610 painted labels changed (base vs guard, both sizes). A second base run also differed by 0 there. At 100x30 and 80x24 two base runs differed in 1,163 of 4,563 labels (nav scroll offsets; Console rail, Research mode strip and Settings speech rows), so those sizes cannot show a difference; base2 vs guard differed in 108 there, in those same families.
- With the guard, no sweep of every Settings category (Theme included) or every Library rail view crashed, 100 down to 40 columns at heights 20 and 30. On base, the default views of all 15 destinations were already crash-free, 160 down to 40 at heights 16, 20, 24 and 44; the crashes were in Settings categories.
- Out of scope, pre-existing: below the cliff (content width 1), Buttons still paint 2 cells wide as on dev, and Settings > Tools `mcp-rail-row` paints 2 cells past its 10-cell row from 100 columns. Both are proposed as follow-ups, not fixed here.

**AC#1: Button rules with `min-width: 0`, or a min-width at or below padding plus line-pad.** Scratch probe: CSS parsed from every `.tcss` source module and every `DEFAULT_CSS`/`BUNDLED_CSS`/`BUNDLED_SCREEN_CSS`/`CSS` literal. A rule counts as a Button rule when its subject is `Button` or a subclass, or an id or class given to a Button constructor. Ids found only through f-string prefixes were checked by hand, and 16 non-Button matches were dropped. Result: 150 rules in 53 files. All 150 are **guarded** by the fold guard (no rule can make a fold width of 0 raise any more). Tags give more detail: `own nowrap+ellipsis/clip` (already safe on base), `inline line_pad 0`, and **fixed** (crash reached on base: the two File Notes rules TASK-38710 reached; #3403 also gives them nowrap+ellipsis). Format: file, then line and selector (`+N` = more selectors in the same rule):
  - `Third_Party/textual_fspicker/base_dialog.py`: 312 `FileSystemPickerScreen #path-breadcrumbs Button`
  - `UI/MCP_Modules/mcp_inspector.py`: 943 `MCPInspector Button`
  - `UI/MCP_Modules/mcp_tools_mode.py`: 225 `#mcp-tools-local-enabled`
  - `UI/Navigation/main_navigation.py`: 407 `.nav-overflow-hint`
  - `UI/Navigation/nav_overflow_menu.py`: 53 `.nav-overflow-destination`
  - `UI/Screens/artifacts_dreams_modal.py`: 333 `DreamsStoryModal #dsm-actions Button`
  - `UI/Screens/lab_mode_strip.py`: 69 `LabModeStrip Button.lab-mode-chip`
  - `UI/Screens/library_screen.py`: 1677 `#library-source-browser .library-source-action`; 1773 `#library-notes-source-strip Button`; 1954 `#library-shell-grid.library-notes-compact .library-canvas-action`
  - `UI/Screens/model_remote_view.py`: 506 `RemoteView.-narrow #remote-model-install`; 511 `RemoteView.-narrow .remote-completion-actions Button`
  - `UI/Screens/personas_screen.py`: 806 `Button.personas-mode-chip`; 903 `#personas-conversation-actions Button`
  - `UI/stts_profile_library.py`: 798 `#stts-clone-profile-actions Button`; 1785 `#stts-profile-toolbar Button +1`; 1854 `#stts-profile-dependency-actions Button`
  - `Widgets/Console/console_library_access_modal.py`: 128 `#console-library-access .console-library-access-actions Button`
  - `Widgets/Console/console_model_popover.py`: 459 `.console-popover-strip > Button`
  - `Widgets/Console/console_rag_settings_modal.py`: 214 `#console-rag-settings .console-rag-settings-actions Button`
  - `Widgets/Library/library_file_notes_git_panel.py`: 751 `LibraryFileNotesGitPanel Button` [fixed: reached in TASK-38710; #3403 also adds nowrap+ellipsis]; 4127 `#file-notes-push-auth-actions Button`
  - `Widgets/Library/library_file_notes_workspace.py`: 1140 `#file-notes-root-details +2`; 1155 `#file-notes-recovery-review`; 1383 `.file-notes-toolbar Button +2` [fixed: reached in TASK-38710; #3403 also adds nowrap+ellipsis]
  - `Widgets/Library/library_media_canvas.py`: 231 `.ds-toolbar > .library-canvas-action`; 242 `#library-media-filter-clear`
  - `Widgets/Persona_Widgets/buddy_character_review.py`: 52 `BuddyCharacterReviewDialog #buddy-actions Button`
  - `Widgets/Persona_Widgets/persona_profile_card_widget.py`: 44 `PersonaProfileCardWidget .ds-toolbar Button`
  - `Widgets/Persona_Widgets/persona_profile_editor_widget.py`: 55 `PersonaProfileEditorWidget .ds-toolbar Button`
  - `Widgets/Persona_Widgets/personas_character_card_widget.py`: 64 `PersonasCharacterCardWidget .ds-toolbar Button`
  - `Widgets/Persona_Widgets/personas_character_dictionaries.py`: 58 `PersonasCharacterDictionariesWidget #personas-char-dicts-toggle`
  - `Widgets/Persona_Widgets/personas_character_editor_widget.py`: 136 `PersonasCharacterEditorWidget #personas-char-editor-advanced-toggle`; 203 `PersonasCharacterEditorWidget #personas-char-editor-avatar-upload +2`; 255 `PersonasCharacterEditorWidget #personas-char-editor-expr-import +3`; 299 `PersonasCharacterEditorWidget .personas-char-editor-expr-upload +2`; 333 `PersonasCharacterEditorWidget .ds-toolbar Button`
  - `Widgets/Persona_Widgets/personas_character_tts_widget.py`: 209 `PersonasCharacterTTSWidget .personas-character-tts-actions Button`
  - `Widgets/Persona_Widgets/personas_character_world_books.py`: 57 `PersonasCharacterWorldBooksWidget #personas-char-worldbooks-toggle`
  - `Widgets/Persona_Widgets/personas_inspector_pane.py`: 132 `PersonasInspectorPane #personas-inspector-actions Button +1`
  - `Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py`: 117 `PersonaVisualCustomStateDialog Button`; 226 `PersonasPersonaVisualPackWidget #personas-persona-visual-actions Button`
  - `Widgets/Persona_Widgets/personas_preview_pane.py`: 52 `PersonasPreviewPane #personas-preview-toggle`; 87 `PersonasPreviewPane .ds-toolbar Button`
  - `Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py`: 100 `PersonasVisualIdentityPackWidget #personas-visual-identity-actions Button`
  - `Widgets/Persona_Widgets/petdex_import_review.py`: 48 `PetdexImportReviewDialog .petdex-actions Button`
  - `Widgets/Prompts/prompt_block_editor.py`: 412 `PromptBlockEditor.-narrow .prompt-block-action`
  - `Widgets/enhanced_file_picker.py`: 308 `PathBreadcrumbs .breadcrumb-button`
  - `css/components/_agentic_terminal.tcss`: 511 `ConsoleTerminalWorkspace Button`; 633 `.meetings-rail-button`; 989 `#lab-mode-strip Button.lab-mode-chip`; 1026 `#mcp-hub-rail Button.mcp-rail-row`; 1079 `#mcp-hub-inspector .console-action-primary +1`; 1391 `.mcp-compact #mcp-detail-toolbar .console-action-primary +1`; 1624 `.approval-compact Button#approval-approve-all +2`; 2067 `.notes-sync-receipt-actions > Button +1`
  - `css/components/_profile_interview.tcss`: 175 `.profile-interview-actions Button`; 188 `.profile-interview-cancel-actions Button`; 211 `.personal-context-review-row Button`
  - `css/components/_settings_personal_context.tcss`: 70 `.personal-context-toolbar Button`; 94 `.personal-context-record-row`
  - `css/components/_settings_splash_theme.tcss`: 149 `#settings-theme-card-column .theme-editor-action`
  - `css/features/_console.tcss`: 144 `.console-status-arrow` (min-width 2) [inline line_pad 0]; 865 `.console-pinned-terminal-action`; 1901 `.console-workspace-files-entry +1`; 1956 `ConsoleWorkspaceFilesModal.-compact #console-workspace-files-actions Button`
  - `css/features/_console_panels.tcss`: 338 `ConsoleSettingsModal.-conversation-settings-compact Button.console-settings-connection-action`; 695 `#console-inspector-more-toggle`; 835 `.console-character-group +4`; 1649 `.console-staged-source-action`; 2094 `.console-library-access-actions Button#library-access-save +4`; 2139 `.console-rag-settings-actions Button#console-rag-settings-run +1`; 2296 `.inspector-modal-control`
  - `css/features/_console_prompts.tcss`: 348 `.console-mixins-actions Button.console-mixins-action`
  - `css/features/_home.tcss`: 40 `.home-rail-row` [own nowrap+ellipsis]
  - `css/features/_lab.tcss`: 21 `#lab-mode-strip Button.lab-mode-chip`; 509 `#audio-cpp-runtime-actions Button.workbench-action`; 534 `#speech-playground-actions Button.workbench-action`; 613 `#speech-result-actions Button.workbench-action`; 968 `.speech-clone-reference-row Button +1`; 1100 `#speech-settings-actions Button.workbench-action +2`; 1188 `#tts-adopt-studio-preferences-btn`; 1239 `.stts-portability-actions Button`; 1282 `#audiobook-actions Button`; 1309 `CuratedView .curated-header Button +1`; 1344 `CuratedView .audio-cpp-actions Button +1`; 1503 `VllmSetupView.vllm-wide .vllm-mode-actions > Button.vllm-action-button +4`; 1539 `VllmSetupView.vllm-medium .vllm-mode-actions > Button.vllm-action-button +9`
  - `css/features/_library.tcss`: 219 `.library-conversation-reader-modes > Button +1`; 328 `#library-conversations-canvas > .ds-toolbar > .library-canvas-action`; 371 `.library-source-pager-controls > .library-canvas-action`; 396 `.library-rail-row` [own nowrap+ellipsis]; 772 `.library-source-action`; 938 `#library-shell-grid.library-notes-compact .library-canvas-action`
  - `css/features/_library_artifacts.tcss`: 24 `.library-artifacts-toolbar Button`; 75 `.library-artifacts-share-strip Button`
  - `css/features/_library_notes_panels.tcss`: 91 `.notes-sync-conflict-view`
  - `css/features/_library_panels.tcss`: 32 `.library-collections-scope-row`; 51 `#library-collections-more-saved-searches`; 107 `#library-collections-items-toolbar > Button +6`; 200 `.library-collections-item-row`; 337 `#library-collections-legacy-recovery +9`; 450 `#library-media-reader-primary-toolbar > Button +2`; 476 `.library-media-reader-primary-group > Button`; 487 `.library-media-trash-row` [own nowrap+ellipsis]; 610 `#library-notes-canvas .library-canvas-action +1`; 688 `.library-notes-tree-pager`; 731 `.library-notes-create-row` [own nowrap+ellipsis]; 915 `.library-skills-actions > .library-canvas-action`; 1429 `.library-prompt-history-row`; 1517 `#library-prompts-selection-management-actions > .library-canvas-action +1`; 1527 `#library-prompts-delete-receipt-actions > .library-canvas-action`; 1586 `#library-prompt-editor-actions .library-canvas-action`; 1858 `#library-media-viewer #library-media-link-undo`; 1877 `#library-media-type-filter`; 1918 `#library-media-canvas > .ds-toolbar > .library-canvas-action`; 2507 `.library-rag-scope-toggle`; 2822 `.library-rag-history-row` [own nowrap+ellipsis]; 2853 `.library-rag-history-clear`
  - `css/features/_logs.tcss`: 30 `#logs-filter-bar Button`
  - `css/features/_plugins.tcss`: 12 `#plugins-shell Button`; 59 `#plugin-review-shell Button`
  - `css/features/_research_workspace.tcss`: 61 `#research-mode-strip Button.research-mode-chip`; 124 `#research-authority-summary .research-data-source-button`; 175 `#research-manage-workspaces`; 203 `#research-pane-mode-strip .research-pane-mode-button`; 383 `ResearchSourcesRegion Button`; 574 `ResearchQuickNotesSection Button`; 737 `#research-add-source-dialog Button +1`
  - `css/features/_roleplay.tcss`: 225 `.personas-library-marks-action`
  - `css/features/_settings.tcss`: 704 `#settings-provider-api-key-clear +2`; 1233 `#settings-shell Button#settings-raw-cli-arm`; 1305 `#settings-advanced-config-actions Button`; 1530 `#settings-web-search-panel Button`; 1602 `.settings-speech-audio-cpp-package-remove`; 1680 `.settings-speech-stacked #settings-speech-actions Button`; 1701 `.settings-speech-stacked .settings-speech-voice-picker > Button`
  - `css/features/_watchlists.tcss`: 251 `#content-actions Button`; 448 `.watchlist-tree-actions Button.watchlist-tree-action`; 536 `#watchlist-add-source-options Button +1`; 662 `.watchlist-tree Button.watchlist-tree-expander`; 764 `#sources-new-button +2`; 790 `#sources-create-watchlist-selected`; 1009 `#items-refresh-button`; 1278 `#artifacts-state-region Button`; 1416 `#bpm-preset-list Button`; 1496 `.bpm-speaker-row Button`; 1585 `#kbm-kept-list Button`
  - `css/features/_wizards.tcss`: 1302 `.setup-tertiary-button`
  - `css/features/_workflows.tcss`: 90 `.workflow-compact`
- Not in the list above, but **fixed**: the two Settings rules above (`_settings.tcss:1146`, `:1422`, both `max-width: 100%`, not a low `min-width`) and the Theme `OptionList`.

**AC#2.** `Tests/UI/test_settings_category_sweep.py::test_narrow_settings_categories_lay_out_at_every_width_60_to_40[workspaces|console-behavior|theme-20|30]` (6 nodes). Settings uses the app's own stylesheets (`_BundledSettingsHarness`). The test opens the category, sweeps 60 down to 40 columns, and asserts that the app raised nothing. Red with the guard disabled (temporary early `return` in `_install_zero_width_fold_guard`): 6 of 6 failed with `ValueError: range() arg 3 must not be zero`, at 44 on `settings-workspace-row-workspace-default`, at 46 on `settings-console-context-memory-jump`, and on `settings-theme-list`. Green: 6 of 6, 11-16 s (call) each at load 10-15. Appended at the census tail; floor 1371 -> 1377.

**AC#3.** No stylesheet check can express the shape. Whether a box reaches 2 cells depends on runtime widths: grid columns, `max-width: 100%` of a squeezed parent, `1fr` shares. So the check guards the guard instead. `Tests/Architecture/test_zero_width_fold_guard.py` (5 tests, serial) asserts four things: the guard is installed; `Content.get_height` and `render_strips` still call `._wrap_and_format(` (a Textual upgrade that bypasses it fails); no width from -1 to 6 raises, with or without wrap, line-pad 0 or 1; and only the zero width differs from upstream, rendering one line inside the box. Red with the guard disabled: 5 of 5 failed (4 with `ValueError: range() arg 3 must not be zero`, 1 on the marker). This file is in no PR lane; the census'd sweep above is red on PRs if the guard goes.

Other checks: `Tests/UI/test_settings_narrow_layout.py` 7 of 7 green under a temporary `@private_profile_test` (TASK-38111 RecoveryRequired otherwise; marker removed). `Tests/UI/test_widget_css_consolidation.py::test_base_class_blocks_precede_their_subclasses` is red at base with the product change reverted (TASK-38697, filed by #3403).

No User Guide change: no copy or behaviour changes outside the zero width.

Files: `tldw_chatbook/__init__.py`, `Tests/Architecture/test_zero_width_fold_guard.py`, `Tests/UI/test_settings_category_sweep.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
