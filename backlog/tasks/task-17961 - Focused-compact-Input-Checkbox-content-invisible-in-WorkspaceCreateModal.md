---
id: TASK-17961
title: Focused compact Input/Checkbox content invisible in WorkspaceCreateModal
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18 03:09'
updated_date: '2026-08-20 15:18'
labels:
  - workspaces
  - ui
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-verifying task-18704 (shared workspace creation modal) found that the Name Input, folder-path Input, and the "Switch to this workspace" Checkbox in tldw_chatbook/Widgets/workspace_create_modal.py all render as an empty bordered box -- top and bottom border rows with zero content rows, hiding the value/label entirely -- whenever that specific widget has keyboard focus. Blurred, each renders correctly as a single-line compact row with its value/placeholder visible. The underlying data is unaffected (confirmed: typed values persist across focus changes, folder Add/Remove works, and Create/Cancel/toast behavior is functionally correct end to end), so this is a pure rendering defect, but a severe one: a user tabbing through the form or actively typing into the Name or folder-path field sees nothing on screen while doing so. A structurally similar collapsed-content-row appearance was also observed on the pre-existing, unrelated "Show archived" Checkbox in Settings ▸ Workspaces (which does not use compact=True), suggesting the root cause may be a broader pre-existing Textual/CSS rendering characteristic (possibly version 8.2.8's interaction between a widget's compact/tall border styles and its :focus state) rather than something newly introduced only by this modal -- needs root-causing across both cases before concluding scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce headlessly (Pilot test or compositor render_strips check) that a focused compact Input/Checkbox in WorkspaceCreateModal shows its label/value
- [x] #2 Root-cause whether this is scoped to WorkspaceCreateModal's CSS or a broader Textual/app-CSS interaction also affecting Settings' non-compact Show archived checkbox
- [x] #3 Fix the rendering so a focused field/checkbox in the modal always shows its current value or label
- [x] #4 Add a regression test asserting focused-state content is visible via Screen._compositor.render_strips(), not terminal-capture text alone
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the two prior family fixes (TASK-1160 DataTable, TASK-2300 compact-Select) to confirm the outline-over-content-row mechanism and house comment style.
2. Write a painted-frame RED test loading the production bundle (tldw_cli_modular.tcss), reading Screen._compositor.render_strips() for a compact Input and the WorkspaceCreateModal Checkbox.
3. Add Input.-textual-compact:focus and ToggleButton(.-textual-compact):focus outline:none opt-outs in components/_forms.tcss, verify compact-Select opt-out already exists, rebuild the bundle, confirm GREEN.
4. Extend Tests/UI/test_non_obscuring_focus_contract.py to pin the new rules in both source and bundle.
5. Update this task file and run the full gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: third member of the app-wide `*:focus { outline: solid $ds-focus-accent; }` (core/_reset.tcss) family, after TASK-1160 (DataTable:focus) and TASK-2300 (Select.-textual-compact:focus). Textual paints outline OVER a widget's outermost rendered lines. Compact Input is exactly one row (Textual pins border:none !important on .-textual-compact) so the outline erases its only content row. Compact ToggleButton/Checkbox/RadioButton lose the same row once focused, for a two-part reason: outline paints over it, AND this app's own pre-existing ToggleButton:focus{border:solid} rule re-adds a real border on focus that !important on the widget's own DEFAULT_CSS cannot block (app CSS always outranks a widget's DEFAULT_CSS in Textual's cascade regardless of !important -- Styles.extract_rules's default_rules tier). So both new rules restate border:none, not just outline:none, mirroring the same two-property need.

Fix (components/_forms.tcss): added Input.-textual-compact:focus {border:none; outline:none; background:$ds-focus-bg} and ToggleButton:focus{outline:none} + ToggleButton.-textual-compact:focus{outline:none; border:none; background:$surface}. The existing ToggleButton:focus > .toggle--label recolour contract (already pinned by test_non_obscuring_focus_contract.py) supplies the visible focus cue; not weakened. Select's compact opt-out (Select.-textual-compact:focus) already existed in components/_lists.tcss (TASK-2300 round 2) -- nothing to add there. Rebuilt the bundle; check_bundle_sync.py confirms it reproduces from source.

Root-caused AC#2's open question with a nuance the original report couldn't see: WorkspaceCreateModal's Name/folder-path Inputs and 'Switch to this workspace' Checkbox (all compact=True) ARE this outline-over-content-row family and are fixed here. Settings' 'Show archived' Checkbox (non-compact) is NOT the same bug -- investigation (measured against both unfixed and fixed bundle) found it is squeezed to zero content rows by a separate, pre-existing, focus-INDEPENDENT defect: an unscoped `Checkbox { height: 2; }` rule (features/_conversations.tcss) collides with ToggleButton's own border:tall (2 rows) even while blurred, leaving no row for the label in ANY state. Out of this fix's bounded scope; flagged for a follow-up task rather than expanded here.

Why pilot tests never caught this: ordinary widget-test harnesses never load the production CSS bundle (CSS_PATH), so the app-wide *:focus outline rule -- the actual cause -- is never in effect against them. Tests/UI/test_compact_focus_outline_render.py (new) explicitly loads tldw_cli_modular.tcss and reads Screen._compositor.render_strips(), the same pattern test_datatable_focus_outline_click.py already established for TASK-1160.

Files: tldw_chatbook/css/components/_forms.tcss (+bundle), Tests/UI/test_compact_focus_outline_render.py (new), Tests/UI/test_non_obscuring_focus_contract.py (+1 test).
<!-- SECTION:NOTES:END -->
