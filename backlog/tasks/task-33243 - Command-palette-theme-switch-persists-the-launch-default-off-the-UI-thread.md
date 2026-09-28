---
id: TASK-33243
title: Command palette theme switch persists the launch default off the UI thread
status: Done
assignee: []
created_date: '2026-09-28 08:00'
updated_date: '2026-09-28 15:29'
labels:
  - settings
  - theme
  - perf
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #2877 (TASK-33121 moved the picker's Use write to a worker). The command palette's theme switch still writes the launch default synchronously on the UI thread (~80-150 ms).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A palette theme switch does not block the UI thread for the config write
- [x] #2 It shares the numbered launch-default write queue, so ordering with picker Use/Revert holds and quit waits for it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ThemeProvider.switch_theme (app.py): switch active theme via use_theme(persist=False) (immediate, on the UI thread), then queue_launch_default() + app.run_worker(_persist_switch(...), group='settings-theme-use') to await settle_launch_default off the UI thread and show the toast, mirroring ThemePicker._persist_use.\n2. Grep for other synchronous launch-default writers outside the picker (use_theme(persist=True), persist_launch_default()) to confirm the palette is the only one.\n3. Update the two mock-based ThemeProvider tests (test_command_palette_providers.py, test_command_palette_basic.py) that asserted a synchronous notify/config write, since the write is now queued.\n4. Add tests: UI-thread-free (StallMeter while a slow write is in flight), ordering (palette switch then picker Use lands the picker's theme, palette's own toast superseded), and quit-wait (wait_for_theme_quit_work blocks for a palette-queued write).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ThemeProvider.switch_theme (app.py) now mirrors ThemePicker._persist_use (TASK-33121): use_theme(persist=False) applies the active theme immediately on the UI thread, then queue_launch_default() submits the config write to the shared single-worker executor and app.run_worker(_persist_switch(...), group='settings-theme-use') awaits settle_launch_default off the UI thread before toasting. Only the latest queued write reports (review M-1 parity), so a picker Use/Revert started right after a palette switch supersedes it and the palette's toast is dropped, same as Use/Revert already do to each other. wait_for_theme_quit_work's existing wait_for_launch_default_writes() call covers the palette write too since queue_launch_default submits to the same executor -- no separate quit wiring needed. Grepped for other outside-picker synchronous writers (use_theme(persist=True), persist_launch_default()); the palette's switch_theme was the only one -- everything else already goes through persist_launch_default_async (editor) or use_theme(persist=False) (Try paths). Updated two mock-based tests (test_command_palette_providers.py, test_command_palette_basic.py) that asserted a synchronous notify/write to await/drain the queued worker. Added 3 tests to Tests/UI/test_settings_theme_file_actions_off_thread.py: UI-thread-free (thread-identity + non-blocking-return check, matching the existing TASK-33121 tail check rather than a StallMeter, which proved noisy against full-Settings-screen background work), ordering (palette switch then picker Use -- picker's write lands last and wins, palette's toast superseded), and quit-wait (wait_for_theme_quit_work blocks for a palette-queued write via the real switch_theme path). No .tcss/doc changes: behavior and toast wording are unchanged, only the write's timing moved off the UI thread; Docs/User_Guide/settings.md already described the palette as persisting 'like Use' generically.
<!-- SECTION:NOTES:END -->
