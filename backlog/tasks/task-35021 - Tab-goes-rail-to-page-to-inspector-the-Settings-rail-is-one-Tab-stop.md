---
id: TASK-35021
title: "Tab goes rail to page to inspector: the Settings rail is one Tab stop"
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- keyboard
- accessibility
- ux-review-2026-10-14
dependencies:
- TASK-33007
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A keyboard user who opens a Settings category and presses Tab expects to reach the page. Instead, Tab walks every rail row, so it takes 16 to 33 presses to reach the first field, because each rail row is its own focusable button. The User Guide says Tab walks into the detail pane, but today only F6 does that, and F6 is easy to miss. Tab also stops on the page's scroll body before the first control, where the only focus cue is a one-row edge. Once the rail is a single stop, the arrow keys are its only keyboard path, so they must follow the order shown on screen. The code suggests that today they follow an internal list order and can land on a hidden Domain Defaults row. This task makes the rail one Tab stop, makes arrows and j/k follow the rail as shown, and stops Tab from landing on a scroll body that holds controls. F6, clicks and every existing rail id stay as they are.

Plan: `backlog/docs/plan-2026-10-14-settings-keyboard-search.md`. Findings closed: D8, B17, S5-N1 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For every Settings category at 211x44 and 235x52, one Tab from the open category's rail row reaches the first control of that page; a page with no control takes focus itself
- [ ] #2 Shift+Tab from a page's first control returns to the open category's rail row. Tab from the navigation bar, or with nothing focused, lands on that row, or on the first shown match while a filter hides it
- [ ] #3 Up/Down and j/k move through the rail rows in the order shown on screen, never land on a hidden row, and reach the Domain Defaults toggle. Enter opens the focused category or toggles the group. With a filter active they move only among the shown matches
- [ ] #4 After the arrow keys move along the rail, Tab leaves the rail from that row without visiting other rail rows. Clicking any rail row still opens and focuses it
- [ ] #5 Tab never stops on the detail pane's or the inspector's scroll body while that pane holds a control, and PageDown/PageUp pressed on a button or checkbox in a long page still scroll it
- [ ] #6 F6 and Shift+F6 cycle the rail, detail pane and inspector as before
- [ ] #7 The Settings User Guide describes the Tab, arrow and j/k path
<!-- AC:END -->
