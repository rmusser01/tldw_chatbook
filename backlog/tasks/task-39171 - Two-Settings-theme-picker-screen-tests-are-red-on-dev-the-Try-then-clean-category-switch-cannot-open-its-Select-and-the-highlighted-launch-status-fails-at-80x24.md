---
id: TASK-39171
title: "Two Settings theme-picker screen tests are red on dev: the Try-then-clean category switch cannot open its Select, and the highlighted launch status fails at 80x24"
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
Found in round 12 of the 2026-10-23 burn-down (TASK-39135's covering runs, round-12 branch fu/r12-settings: they "fail the same way on `fu-base-r12`"). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08 (run.sh, `-n 4`, load ~21-27). Both tests are `@private_profile_test`, so they run plain.

**`test_try_then_clean_category_switch_restores_the_pre_editor_theme`.** The private-profile child fails with `NoMatches` while it opens a Select:
- on dev: "No nodes match '#label' on SelectCurrent(classes='-has-value')";
- on the stack: "No nodes match 'SelectOverlay' on Select(id='settings-appearance-density', …)", from `select_overlay = self.query_one(SelectOverlay)`.

**`test_highlighted_launch_and_override_status_is_on_screen[size0]`** (80x24). The child fails an `AssertionError` on the 80x24 screen text; `[size1]` and `[size2]` pass. TASK-39135's notes call it "(row width)".

**Lanes.** Three other nodes of this file are UI census entries; these two are not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node passes on dev, or is retired with the commit that changed what it pinned named; each red is shown first with the same command
- [ ] #2 Each is classified as test drift or a product defect (a status row that does not fit at 80x24 would be one); a product defect gets a test that is red before its fix
- [ ] #3 Both nodes run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
