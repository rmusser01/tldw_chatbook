---
id: TASK-17652
title: 'Console: status chips placement setting (above/below composer) + persistent collapse'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - console
  - ux
  - settings
dependencies:
  - task-17650
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The status-chip strip sits below the composer on dev (TASK-15704 moved it there and pinned the order). The owner wants a user-facing setting to place it either above or below the composer input, defaulting to below (current behavior).

The 2026-08-17 audit mapped what "above" must respect: the prompt-queue shelf is pinned immediately above the composer (`queue.y + queue.h == composer.y`), so chips-above means directly under the workspace grid, ABOVE the staged-evidence/prompt-queue cluster — not wedged between the shelf and the composer. `ConsoleCommandPopup.reposition`'s clearance loop deliberately excludes the chips (they are "below, out of reach") and would paint over them on every `/` in above mode. The F6/Tab region pairing maps the chips to the transcript surface and must follow the visual position. Two currently-green contract tests hard-assert chips-below and need parameterizing over both modes.

Also in scope: the Status collapse state (`_console_status_chips_collapsed`) is session-only screen state today — it resets every time the user leaves and re-enters Console. Since this task adds `[console]` persistence plumbing anyway, persist the collapse state alongside the position.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A `[console] status_chips_position` config key ("below"|"above", default "below") exists with a staged Settings > Console Behavior control using the 1-row toggle pattern, indexed for field search
- [ ] #2 In above mode the chips render directly under the workspace grid, above the staged-evidence/prompt-queue cluster; the prompt-queue shelf stays immediately adjacent to the composer in both modes
- [ ] #3 The command popup never paints over the chips in either mode
- [ ] #4 F6/Tab region pairing matches the chips' visual position in both modes
- [ ] #5 The order and popup contract tests are parameterized over both positions and green
- [ ] #6 The Status collapse state persists across Console re-entry and app restart
- [ ] #7 User Guide Console and Settings pages updated
<!-- AC:END -->
