---
id: TASK-39171
title: "A Settings theme-picker test fails at 80x24 on every run, and the Try-then-clean category switch test fails intermittently while opening its Select"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- settings
- ui
priority: low
references:
- Tests/UI/test_settings_theme_picker_screen.py
dependencies:
- TASK-39135
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39135's covering runs, round-12 branch fu/r12-settings: they "fail the same way on `fu-base-r12`"). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08, and re-checked on dev 3a6498ae14, 2026-11-09 (run.sh). Both tests are `@private_profile_test`, so they run plain. Line numbers are dev 3a6498ae14's.

**`test_highlighted_launch_and_override_status_is_on_screen[size0]`** (80x24, `:2468`). It fails on every run.
- The private-profile child fails an `AssertionError` on the 80x24 screen text; `[size1]` and `[size2]` pass. TASK-39135's notes call it "(row width)".
- Red in every run: 3 of 3 runs that list failures by name on dev 3a6498ae14 (`-n 4`, load 23-37); five earlier runs of the same four nodes each had at least one failure (load 8-14); once each on dev b590a15afd and the stack; and every run in review r12fr.

**`test_try_then_clean_category_switch_restores_the_pre_editor_theme`** (`:1869`). It fails intermittently.
- Observed rates:
  - dev 3a6498ae14, alone and serial (`-p no:xdist`): 5 of 20 failed at load 14-30 (5 of the first 10, none of the next 10);
  - dev 3a6498ae14, beside the three `[size*]` cases at `-n 4`: 2 of 3 runs that list failures by name (load 23-37); five earlier runs failed 1, 2, 1, 1 and 1 of the four nodes (load 8-14), so it failed in at most one of them;
  - 2026-11-08: failed in its one run on dev b590a15afd and in its one run on the stack (load ~21-27);
  - review r12fr: 0 of 4 alone on dev 3a6498ae14, 0 of 2 on the stack.
- The failure is the private-profile child's `NoMatches` while it opens a Select:
  - on dev b590a15afd: "No nodes match '#label' on SelectCurrent(classes='-has-value')";
  - on the stack: "No nodes match 'SelectOverlay' on Select(id='settings-appearance-density', …)", from `select_overlay = self.query_one(SelectOverlay)`.

**Lanes.** Three other nodes of this file are UI census entries; these two are not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `test_highlighted_launch_and_override_status_is_on_screen[size0]` passes on dev, or is retired with the commit that changed what it pinned named; its red is shown first with the same command
- [ ] #2 It is classified as test drift or a product defect (a status row that does not fit at 80x24 would be one); a product defect gets a test that is red before its fix
- [ ] #3 The cause of `test_try_then_clean_category_switch_restores_the_pre_editor_theme`'s intermittent `NoMatches` is named from its own failure, and the test then passes 20 of 20 runs alone, with load recorded, at least ten of them at a one-minute load of 20 or more
- [ ] #4 Both nodes run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
