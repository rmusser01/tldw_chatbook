---
id: TASK-38719
title: "Move the File Notes history and workflow run wide-tier rules off the boot CSS bundle"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
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
- [ ] #1 The wide-tier rules for the File Notes history dialog and the workflow run dialogs load with their split sheets, and the boot bundle no longer contains them
- [ ] #2 The boot bundle's parsed bytes go down by those rules' size, as test_boot_css_byte_budget measures it
- [ ] #3 At 150 columns or more both dialogs still open at their wide-tier width and cap, and below 150 columns at their base geometry; test_modal_wide_tier passes, with its registry updated to match
<!-- AC:END -->
