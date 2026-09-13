---
id: TASK-23150
title: Console Behavior card grew past the test viewport and clicks land on nothing
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - settings
priority: medium
dependencies: []
---

## Description

All 3 tests in `Tests/UI/test_settings_console_rail_labels.py` fail. The production save path is
intact — the decisive evidence is inside the third test, where values set **programmatically**
stage correctly while the one reached by `pilot.click` does not, and the checkbox's own `.value`
stays `False`, so the widget never received the toggle.

The Console Behavior card grew roughly 46 lines above that checkbox across two 2026-08-26 commits,
at the test's 190x55 viewport.

**One caveat carried over from the diagnosis and deliberately not closed:** "below the fold" was
inferred from the two commits plus the zero-effect click, *not* asserted against the viewport. The
first step of this task is to verify it.

## Acceptance Criteria

- [ ] The checkbox's region is asserted to be inside the visible container **before** anything is
  changed, confirming or refuting the below-the-fold diagnosis
- [ ] The tests scroll or focus the control into view (or drive it by key) rather than clicking a
  fixed position
- [ ] A visibility assertion fails loudly on future layout growth instead of silently clicking air
- [ ] If the card genuinely no longer fits a realistic terminal, that is filed as a separate UX task
  rather than absorbed here

## Evidence

Production seams all present and unchanged: handler `settings_screen.py:19612`, key in
`CONSOLE_BEHAVIOR_SAVE_ORDER` (`:915`), save branch `:22520`, adapter call `:23110` (the
monkeypatched name). Checkbox at `settings_screen.py:13345`, now below an exchange-capture block and
a thinking-visibility block.

Growth blames to `c6218918d1` (32 lines, "Codex/full semantic capture (#2126)") and `4aa87159ee`
(14 lines, "feat: add Console thinking visibility and history controls"), both 2026-08-26. The test
file was last updated 2026-08-24 (`a0d61d9957`).
