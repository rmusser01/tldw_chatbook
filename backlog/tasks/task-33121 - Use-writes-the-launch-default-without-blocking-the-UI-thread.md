---
id: TASK-33121
title: Use writes the launch default without blocking the UI thread
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 00:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-33075 profiling: the picker's Use persists the launch default synchronously, ~140 ms on the UI thread per Use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing Use does not block the UI thread for the config write
- [x] #2 A failed write is still reported to the user
- [x] #3 The launch default is persisted before the app can exit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Look for an existing queued/async config writer (none: other screens use asyncio.to_thread(apply_settings_mutation_to_cli_config)).
2. Add a single-thread FIFO writer in theme_catalog so writes land in the order asked and a quick Revert/second Use still wins; route the blocking persist through it too.
3. Picker Use: switch at once, queue the write in an app worker, toast the real outcome (failure -> "launch default was not saved").
4. Quit: the off-loop quit persistence waits for queued writes before the config shutdown pass and exit.
5. Tests: write thread, ordering, failure report, quit wait.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
theme_catalog now has one writer thread (`_LAUNCH_DEFAULT_WRITER`, a 1-worker ThreadPoolExecutor). `persist_launch_default_async` queues the write there and awaits it shielded (a cancelled waiter never cancels a queued write), then records the in-memory `app_config` on the loop. The blocking `_persist_launch_default` (Revert, the palette's switch) submits to the same thread and waits, so every launch-default write lands in the order it was asked -- a Revert right after a Use still wins. `wait_for_launch_default_writes(timeout)` returns once everything queued so far has run.

Picker Use: `use_theme(persist=False)` switches the theme at once; the pending Revert records the change as persisted (its own write queues behind Use's); an app-owned worker (`_persist_use`) awaits the write and shows `use_theme_toast` with the real outcome -- a failed or raising write is reported as "\<name\> applied; the launch default was not saved" (AC#2) -- then refreshes the launch marker.

Quit (AC#3): `TldwCli._run_blocking_quit_persistence` (already off the loop, before the config shutdown pass and `exit()`) first calls `wait_for_launch_default_writes(timeout=5.0)`. Other exits still finish the write: executor threads are joined at interpreter exit.

Measured: the config write took 78-147 ms on the UI thread per Use on dev; now it runs on the writer thread (asserted by thread identity).

Tests: Tests/UI/test_settings_theme_file_actions_off_thread.py (Use's write thread, FIFO ordering incl. a blocking write behind a pending one, quit waits before the config pass, a raising write is reported). Picker tests wait for the Use worker.

Files: tldw_chatbook/css/Themes/theme_catalog.py, tldw_chatbook/Widgets/settings_theme_picker.py (`_switch`, `_persist_use`), tldw_chatbook/app.py (`_run_blocking_quit_persistence`), Docs/User_Guide/settings.md.
<!-- SECTION:NOTES:END -->
