---
id: TASK-18960
title: 'Unscoped Checkbox height:2 rule blanks bordered checkboxes app-wide'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 00:00'
updated_date: '2026-08-20 16:28'
labels:
  - css
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during TASK-17961's painted-frame verification: the Settings ▸ Workspaces
"Show archived" Checkbox renders with zero content rows in EVERY state —
blurred included — so it is a different defect from the focus-outline family
17961 fixed. An unscoped `Checkbox { height: 2; }` rule in
`css/features/_conversations.tcss` applies app-wide; combined with a
`border: tall` (2 rows of chrome) the widget's content area is squeezed to
0 rows. Any bordered, non-compact Checkbox outside the conversations UI is
affected. TASK-17961's new painted-frame test file
(`Tests/UI/test_compact_focus_outline_render.py`) demonstrates the probe
technique; its Implementation Notes record the empirical evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The `height: 2` Checkbox rule is scoped to the conversations UI it was written for (or retired if unneeded there)
- [x] #2 Settings ▸ Workspaces "Show archived" renders its label and check glyph in blurred AND focused states (painted-frame test, production bundle)
- [x] #3 An app-wide sweep confirms no other unscoped bare-type height rules squeeze bordered widgets to zero content rows (grep `_*.tcss` for bare `Checkbox`/`Switch`/`RadioButton` type rules with height pins; each hit scoped or justified)
- [x] #4 Bundle rebuilt from module sources; `check_bundle_sync.py` green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: add Tests/UI/test_checkbox_height_render.py, mirroring test_compact_focus_outline_render.py's ProductionCssWidgetHarness/_rendered_text pattern, asserting a non-compact Checkbox("Show archived", True) shows its label blurred and focused against the production bundle.
2. Trace live consumers of features/_conversations.tcss's `.ccp-sidebar`/`.ccp-view-area`/`#conversations_characters_prompts-window` selectors and confirm whether any current screen's Checkbox depends on the bare `Checkbox { width:100%; height:2 }` rule before touching it.
3. Fix at the source module (not the generated bundle), rebuild via build_css.py, verify check_bundle_sync.py.
4. Sweep every `_*.tcss` source for other unscoped bare Checkbox/Switch/RadioButton/ToggleButton type rules with height pins.
5. Run the gate (new test + the 17961/non-obscuring-focus/settings-workspaces/CSS-cascade regression files that already reference this same rule) and the full-suite collect-only sanity check.
6. Update this task file and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: `Tests/UI/test_checkbox_height_render.py` (new) loads the production bundle on a bare `Checkbox("Show archived", True)` harness copied from `test_compact_focus_outline_render.py`'s `ProductionCssWidgetHarness`/`_rendered_text`. Pre-fix, both the blurred and focused painted frames were exactly the two-row border box (`┌...┐` / `└...┘`) with the label entirely absent — matching TASK-17961's empirical finding byte for byte.

Root cause confirmed unchanged from 17961's note: `ToggleButton`'s own `border: tall` DEFAULT_CSS is 2 rows, present even blurred; `_conversations.tcss`'s bare `Checkbox { height: 2; }` type selector gives the whole widget only 2 rows total, so the border consumes both and the label/glyph has 0 rows to paint in.

Scoping decision (AC#1): **retired**, not rescoped. Traced before touching it: `#conversations_characters_prompts-window` (the id this file is named after) has zero live Python consumers — `TAB_CCP`'s route now aliases entirely to `PersonasScreen` (`UI/Navigation/screen_registry.py`). The `.ccp-sidebar`/`.ccp-view-area` classes the file also defines are reused as generic form-utility classes by several current screens, but grepping every file referencing them found zero live `Checkbox` usage under any of those containers — there is nothing conversations-scoped to rescope the rule INTO. Retired `width: 100%`/`margin-bottom: 0` along with `height: 2` rather than leaving a narrower height-only fix: `margin-bottom: 0` already matches `ToggleButton`'s own DEFAULT_CSS default, and every one of five pre-existing escape hatches already fighting this exact rule elsewhere in the bundle (`.settings-imagegen-backend-row Checkbox`, `MCPSchemaForm Checkbox`, `PromptBlockEditor #prompt-editor-lane-options Checkbox`, `MCPToolsMode #mcp-tools-local-enabled`, `LibraryIngestCanvas .type-group-contents Checkbox` — each with its own comment citing this same rule) independently converged on `width: auto` + `height: auto`, so removing the rule entirely just lets `ToggleButton`'s own DEFAULT_CSS (`width: auto`, implicit `height: auto`) take over, which is exactly that established pattern.

Side benefit found while tracing: the bug was not scoped to the one named Settings ▸ Workspaces checkbox. At least three more unscoped, non-compact Checkboxes were equally blank pre-fix and had no escape hatch of their own: `library_prompts_canvas.py`'s "Include current text as starter content" (`#library-prompt-recipe-starter`), and `stts_profile_library.py`'s `#bundle-warning-ack` and `#stts-bundle-inactive-consent`. Retiring the rule fixes all of them at once.

Sweep (AC#3): `grep -rn -E "^(Checkbox|Switch|RadioButton|ToggleButton) \{|^(Checkbox|Switch|RadioButton|ToggleButton),"` across every `tldw_chatbook/css/**/*.tcss` source found exactly one unscoped bare-type hit with a height pin — `features/_conversations.tcss`'s `Checkbox { ... height: 2; }`, fixed here. Everything else matching the pattern is either class/id/container-scoped (already correctly scoped, e.g. the five escape hatches above) or a bare `ToggleButton:focus`/`ToggleButton.-textual-compact:focus`/`Switch:focus` pseudo-class rule (`components/_forms.tcss`, TASK-17961's fix) that sets no height at all — justified, no change needed.

GREEN: new test passes blurred+focused; `python3 tldw_chatbook/css/build_css.py` regenerated the bundle (only `tldw_cli_modular.tcss` changed — the widget_defaults/screen_css sheets were untouched, confirming this rule wasn't a class-level `BUNDLED_CSS` block); `check_bundle_sync.py` green.

Gate: new test file, `test_compact_focus_outline_render.py`, `test_non_obscuring_focus_contract.py`, `test_settings_workspaces_category.py`, and the three other files that already document/depend on this exact rule (`test_fspicker_keyboard_save.py`, `test_enhanced_file_dialog_bundle_css.py` — both about `_conversations.tcss`'s separate, untouched `Select { width: 100%; }` rule; `test_mcp_schema_form.py` — the `MCPSchemaForm Checkbox` escape hatch) — 191 passed, 0 failed on a clean run. An earlier run (with a concurrent full-suite `--collect-only` sweep competing for CPU) showed 3 unrelated failures in `test_settings_workspaces_category.py` (`event_loop_stall` diagnostics logged, up to 4.5s lag); reran isolated and 3x-repeated to confirm pre-existing timing flakiness (one passed 2/3 consecutive reruns with zero code changes between them), not a regression from this change. `Tests/ --collect-only -q` collected 51,472 tests with no new collection errors.

Files: `tldw_chatbook/css/features/_conversations.tcss` (+bundle), `Tests/UI/test_checkbox_height_render.py` (new).
<!-- SECTION:NOTES:END -->
