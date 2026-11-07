---
id: TASK-38719
title: "Move the File Notes history and workflow run wide-tier rules off the boot CSS bundle"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-05 22:16
labels:
- follow-up
- css
- performance
priority: low
references:
- tldw_chatbook/css/components/_agentic_terminal.tcss
- tldw_chatbook/css/build_css.py
- tldw_chatbook/css/screen_modal_file_notes_history.tcss
- tldw_chatbook/css/screen_feature_workflows.tcss
- Tests/UI/test_modal_wide_tier.py
- Tests/Performance/test_boot_css_byte_budget.py
dependencies:
- TASK-36522
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36522's notes, PR #3387). Checked on dev c2cdb6b82e, 2026-11-03.

TASK-36522 added nine `App.-wide-viewport` rules to the wide-tier block in `css/components/_agentic_terminal.tcss`, which is part of the boot bundle. Two of them style dialogs whose base rules already load later, from split sheets built by `build_css.py`:
- `App.-wide-viewport .workflow-run-dialog` (:2297). Its base rule is in `features/_workflows.tcss`, which is split into `screen_feature_workflows.tcss` and loaded on entry to the Workflows route.
- `App.-wide-viewport #file-notes-history-dialog` (:2383). Its base rules are split into `screen_modal_file_notes_history.tcss`, which `FileNotesHistoryDialog`'s own `CSS_PATH` loads on first open (`MODAL_OWNED_SPLIT_SHEETS`).

TASK-36522 measured these two rules at about 95 B and 80 B of boot CSS. They stay in the bundle because a `-wide-viewport` token pins a rule there under those splits. The comments say so: `screen_modal_file_notes_history.tcss:337-339` and the `FileNotesHistoryDialog` docstring in `library_file_notes_workspace.py`. There is a precedent for moving them. The Settings split claims the `-wide-viewport` token for its own modules (`build_css.py:511`), so its tier rules ship in `screen_agentic_settings.tcss` (:2121, :2177). One of those, `#console-hooks-review`, is for a modal whose own `CSS_PATH` loads that sheet when it opens from Console.

`Tests/UI/test_modal_wide_tier.py` pins the tier block 1:1 against `Tests/UI/modal_wide_tier_registry.py`. A move must keep both in step.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The wide-tier rules for the File Notes history dialog and the workflow run dialogs load with their split sheets, and the boot bundle no longer contains them
- [x] #2 The boot bundle's parsed bytes go down by those rules' size, as test_boot_css_byte_budget measures it
- [x] #3 At 150 columns or more both dialogs still open at their wide-tier width and cap, and below 150 columns at their base geometry; test_modal_wide_tier passes, with its registry updated to match
<!-- AC:END -->

## Implementation Plan

1. Measure the boot census on base with `test_boot_css_byte_budget`'s own `_boot_parsed_css_census`.
2. Move `App.-wide-viewport .workflow-run-dialog` into `features/_workflows.tcss`, beside its base rule, and move `App.-wide-viewport #file-notes-history-dialog` into `features/_notes.tcss`. Add `-wide-viewport` to those two splits' prefixes in `build_css.py`, as the Settings split already does. Then each rule moves with its owner's sheet.
3. Rebuild with `python tldw_chatbook/css/build_css.py`, and check that each rule is in its split sheet and gone from `tldw_cli_modular.tcss`.
4. Add the two source modules and the two built sheets to `test_modal_wide_tier.py`'s source and built lists, and update the registry docstring. Update the comments that said these rules ride the boot bundle: the `_notes.tcss` note, the `FileNotesHistoryDialog` docstring, and the tier-block banner.
5. Re-measure the census, then run `test_modal_wide_tier.py`, the boot budget, the CSS integrity tests and the bundle sync check.

## Implementation Notes

- The two tier rules now sit beside their base rules, in `features/_workflows.tcss` and `features/_notes.tcss`. Both splits in `build_css.py` claim the `-wide-viewport` token, as the Settings split does (`build_css.py:511`), so each rule moves into the sheet its modal already loads: `screen_feature_workflows.tcss` on entry to the Workflows route, and `screen_modal_file_notes_history.tcss` through `FileNotesHistoryDialog.CSS_PATH`. The tier block in `components/_agentic_terminal.tcss` no longer has them, and its banner says where they went. The `_notes.tcss` comment and the dialog docstring that said the rule rides the boot bundle are updated. Rebuilt with `python tldw_chatbook/css/build_css.py`; `check_bundle_sync.py` reports every sheet reproduces.
- AC#2, measured with `test_boot_css_byte_budget`'s own `_boot_parsed_css_census`: 598,492 B before, 598,293 B after, so -199 B. That is exactly the two rules as the bundle printed them (`tldw_cli_modular.tcss` 427,120 -> 426,921), 97 B for the workflow rule and 102 B for the history rule, blank line and newlines included. The limit stays at 608,090. The headroom (9,797 B) is under ADR-097's 25,000 B slack, so the tightening convention does not apply.
- `test_modal_wide_tier.py`: the source and built lists now come from one `WIDE_TIER_SPLITS` table (the hook review, File Notes history and workflow run). A new `test_split_modal_tier_rules_stay_off_the_boot_bundle[...]` checks that each split anchor's rule is in its sheet and not in the bundle. It was red before the move (`'#file-notes-history-dialog' in {}`, `'.workflow-run-dialog' in {}`) and is green after. The registry docstring now names the three split-sheet anchors. Whole file: 134 passed. That includes the spot geometry at 200x50, 211x44, 235x52 (wide cap), at 120x40 and 149x44 (base), and across live resizes, for both dialogs.
- Extra check (a probe, not committed): with only the boot bundle loaded and the dialog's own `CSS_PATH`, the history dialog measured 150 cells at 200x50 and 90 at 149x44.
- Covering run: `test_modal_wide_tier.py`, the boot budget, `test_css_build_integrity.py`, `test_css_staleness_manifest.py`, `test_widget_css_consolidation.py`, the workflows stylesheet, editor and projection files, `test_library_file_notes_history.py` and `test_settings_image_gen_panel.py`: 315 passed, 60 failed. All 60 are in `test_workflows_editor.py`, from the local `RecoveryRequired` trap, and the file fails the same 60 on the base CSS.
- The pin is appended at the census tail.
- Files: `tldw_chatbook/css/{build_css.py, components/_agentic_terminal.tcss, features/_notes.tcss, features/_workflows.tcss, tldw_cli_modular.tcss, screen_feature_workflows.tcss, screen_modal_file_notes_history.tcss}`, `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` (docstring), `Tests/UI/test_modal_wide_tier.py`, `Tests/UI/modal_wide_tier_registry.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

