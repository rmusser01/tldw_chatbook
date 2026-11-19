---
id: TASK-41986
title: 'Roleplay workbench test test_floating_buddy_close_refreshes_active_personas_inspector is red on dev'
status: To Do
assignee: []
created_date: '2026-11-19 13:30'
labels:
  - roleplay
  - tests
  - personas
dependencies: []
references:
  - Tests/UI/test_personas_workbench.py
  - tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_personas_workbench.py::test_floating_buddy_close_refreshes_active_personas_inspector` fails on dev today (`a6df4d8d48`) at `assert controller.current_preferences().open is False` (it is still `True` after the close click). This task's PR is the fix; no open PR fixes it (the three open PRs that touch the file, #3431, #3441 and #3518, do not touch this test).

**When it broke.** It passed at `121e5818aa` (2026-08-21) and failed at `461668df00` (2026-09-25). A `git bisect run` over that range (about 1,860 commits, no skips) names `ae9093a714` "fix(design): complete token-class migration and enforce visual style floors" (2026-09-14, a 135-file commit that landed through PR #2704, merge `e89f28d751`) as the first bad commit.

**What that commit changed.** In `PersonaBuddyWidget` it replaced the direct `collapse.styles.width` / `close.styles.width` assignments with utility classes (`w-3`, `w-6`, `w-7`), and the frame and widget sizing with `add_class("w-full")` and `set_styles(...)`. The close button is now a 3-cell `×` button carrying `w-3`.

**Attribution.** The `ae9093a714` attribution rests on the bisect alone. The diff changes the button sizing (w-3/w-6/w-7 classes, `w-full`, `set_styles`) and nothing about click handling, and the cause is not found.

**Probe on dev.** `press()` on `#persona-buddy-close` closes the Buddy (preferences `open` becomes `False`, one preference write), and `pilot.click("#persona-buddy-close", offset=(1, 0))` closes it too. The test's default click (the button's left-edge cell, 96,19 in the 100x30 run) does not, although `get_widget_at` reports the button at that cell. A real mouse click on the button's centre was not run against the live app.

**CI.** No PR-gate lane runs this test: `scripts/ui_pr_gate_census.txt` lists only `TestPersonaHumanIdentityRemoval` from this file, and nothing under `.github/` names the file. `nightly-deep.yml` has a cron, but `.github/workflows/test.yml` says it "remains disabled (ADR-366)". I did not check any nightly run. By the bisect the test has been red on dev since `ae9093a714` (2026-09-14).

**Related notes.** TASK-41800 (line ~53) already notes that this test and the resize-sync test "fail on dev too and are on no lane". This is the dedicated task for what that note mentions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev: the click target is corrected (or the edge-cell behaviour is fixed in the product), with the reason recorded in the task notes
- [ ] #2 The behaviour the test pins holds: closing the floating Buddy persists `open=False`, removes the widget, and refreshes the active Personas inspector (Show enabled, Close disabled)
- [ ] #3 A pilot test clicks each of the close button's three cells and the Buddy closes on each
- [ ] #4 If the test is kept, it runs somewhere a regression turns a check red (a PR-gate census entry, or a recorded reason it stays out)
<!-- AC:END -->
