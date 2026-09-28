---
id: TASK-33242
title: Settings dialogs open and close without a full-screen repaint stall
status: Done
assignee: []
created_date: '2026-09-28 08:00'
labels:
  - settings
  - theme
  - perf
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #2877's review: opening or closing any Settings dialog (Rename, Delete, Export, leave prompt) repaints the whole screen, ~250 ms at 211x44.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dominant cost of a Settings dialog open/close is measured and identified
- [x] #2 The stall is cut to under 100 ms, or a documented reason why not and the best achievable number
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Profile a Settings dialog push and pop at 211x44 (same harness, UI-thread CPU + stall meter).
2. Identify the dominant cost: Textual's modal compositing vs work the Settings screen does on suspend/resume.
3. Fix what is in this repo (e.g. resume work that a dialog close does not need); record Textual-owned costs with numbers.
4. Guard test asserting the avoided work is not invoked.
5. Record numbers in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured and identified the dialog cost. Removed the one part this repo owns: the sync-rows refresh after closing a Theme dialog. The rest is Textual's modal compositing, and it is already under 100 ms.

**Measured (AC#1).** Same harness as TASK-33241: 211x44, `TLDW_TEST_CSS_CACHE=0`, loaded machine, 8 interleaved rounds, medians. RagProfileNameModal, ConfirmationDialog and ThemeLeaveModal behave alike.
- There are no stylesheet reparses, and the restyles are tiny (the dialog's ~13 nodes on open, 1 on close).
- **Open.** The cost is ~59 ms of UI CPU, spread over 3 full-screen layout and render passes. It is all Textual's:
  - Each pass composites the dialog over the dimmed Settings screen. `BackgroundScreen.process_segments` blends every segment and takes about half of each render.
  - Textual also relays out the Settings screen once.
- **Close.** Textual's `App._prune` → `post_mount` → `App.refresh(layout=True)` relays out the whole Settings screen.
- **Close, repo-owned.** `SettingsScreen.on_screen_resume` re-ran `_refresh_sync_rows` on every dialog close. That is ~550 ms per close of backup-scoped DB work on a thread (private-sqlite helper spawns, storage admission), competing for the GIL, followed by an apply frame. The Theme category does not even show those rows.
- The "~250 ms" in the PR #2877 notes did not reproduce as a UI-thread stall. The same pushes read 60-120 ms under load average ~48, and 160-290 ms only under cProfile. It was most likely machine load.

**Fix (AC#2).**
- `on_screen_suspend` records when the covering screen is a `ModalScreen` raised while the Theme category is active. The next resume skips the sync-rows refresh (and the Providers-only default check, a no-op there).
- A resume from anywhere else, including a dialog over another category, still refreshes. The test's control half checks that.
- Leaving Theme through the leave prompt now shows the sync rows as of the last refresh. That matches leaving Theme without a prompt.

| Metric | Before | After |
|---|---|---|
| Close: UI CPU | 65.1 ms | 46.7 ms |
| Close: worst stall | 27 ms | 18 ms |
| Close: off-thread refresh | ~550 ms | none |
| Open: worst stall | 35 ms | 34 ms (Textual-owned, unchanged) |

Both are under 100 ms at 211x44. Making the open cheaper would mean an opaque (undimmed) dialog backdrop, which is a design change, or patching Textual. Neither was done.

**Files.**
- `UI/Screens/settings_screen.py`: `_suspended_under_theme_dialog`, `on_screen_suspend`, `on_screen_resume`
- `Tests/UI/test_settings_theme_switch_cost.py::test_closing_a_theme_dialog_skips_the_sync_rows_refresh`: fails on the base code
<!-- SECTION:NOTES:END -->
