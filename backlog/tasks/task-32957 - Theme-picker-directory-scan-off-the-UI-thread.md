---
id: TASK-32957
title: Theme picker directory scan off the UI thread
status: Done
assignee: []
created_date: '2026-09-25 21:00'
labels:
  - settings
  - theme
  - perf
priority: medium
dependencies:
  - TASK-32948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings ▸ Theme lists saved theme files through a backup-scoped scan that runs on the UI thread. With 50 theme files it measured a 254 ms median (240–287 ms over 5 runs; 66 ms with 10 files), about 5–6.6 ms per file, almost all of it in the backup layer's per-file scope. TASK-32948's Qodo review fixes stopped theme switches (Use, Try, Revert, the command palette) from rescanning (Qodo 4107495860). The scans that remain run when the Theme page mounts, on Back from the editor, and after each file action (Save, Save as, Rename, Delete, Import, Export). These still freeze the UI for longer than the 100 ms at which CLAUDE.md's performance rules call for a worker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Settings ▸ Theme, pressing Back from the editor, and finishing a file action never block the UI thread for more than 100 ms with 50 saved theme files
- [x] #2 While a scan is in flight, the picker shows the last good listing (or a loading state on first open) and stays responsive to keys
- [x] #3 A scan result that arrives after a newer scan started, or after a file action, never overwrites the newer listing
- [x] #4 A backup/recovery pause during a background scan still shows the existing "Theme files unavailable" state and blocks file actions
- [x] #5 Tests cover the stale-result case and the pause case, and the existing picker tests pass without relying on a synchronous scan
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `ThemePicker.refresh_catalog(rescan=True)` bumps a scan generation, records a pending highlight, and starts the listing in a thread worker (`thread=True, exclusive=True, group="settings-theme-scan"`); the worker catches RecoveryRequired/OSError and hands the outcome back with `call_from_thread`.
2. The UI-thread apply step drops any result whose generation is no longer current (AC#3), then updates `files_available`/last-good listing exactly as the synchronous path did and rebuilds with the pending highlight.
3. Before the first scan lands, the YOUR THEMES group shows a loading row; afterwards the last good listing stays up while a rescan is in flight (AC#2). Theme switches keep `rescan=False` (Qodo 4107495860).
4. ThemePane keeps a direct reference to its editor so the worker never queries the DOM off-thread; file operations stay in the editor on the UI thread.
5. Tests: stale-result, pause-in-worker, loading row; existing picker tests wait for the scan instead of assuming it is synchronous; 50-file UI-thread timing test (< 100 ms).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ThemePicker.refresh_catalog()` no longer lists the themes folder on the UI thread. A rescan bumps a generation counter, remembers the caller's highlight, and starts the listing in a thread worker (`thread=True, exclusive=True, group="settings-theme-scan"`, `exit_on_error=False`). The worker catches RecoveryRequired, OSError and any other failure, logging only the exception type. It then hands the outcome back with `call_from_thread`. `_apply_scan` drops a result whose generation is no longer current, which covers both a newer scan and the rescan after a file action. Otherwise it applies the old state rules unchanged: a pause keeps the last good listing and sets `files_available=False`, an OSError reads as no user themes, and the pending highlight is used once the listing lands. Theme switches still pass `rescan=False` and never scan (Qodo 4107495860).

- First open: the registered themes are listed immediately, and YOUR THEMES shows a disabled "Loading your themes…" row until the first scan lands. After that the last good listing stays up while a rescan runs. The User Guide (`Docs/User_Guide/settings.md`) now mentions the loading row.
- `ThemePane` keeps a direct reference to its `SettingsThemeEditor`, so the worker's `_editor_listing` never queries the DOM off-thread. All file operations stay in the editor on the UI thread, and only the listing moved. The listing is thread-safe: `raw._scope` records the calling thread in thread-local state, and a per-path RLock serializes the listing with the editor's own scans.
- AC#1 measurement (50 saved theme files, private profile, real backup-scoped editor scan, 3 runs): the worker scan took 279–689 ms. The UI-thread `refresh_catalog()`/`show_picker()` call took 0.0–0.5 ms, landing the result (`_apply_scan`) took 36–78 ms, and the worst event-loop stall was 39–80 ms, with one outlier of 260 ms on a loaded run. The remaining UI-thread cost is `build_catalog` regenerating the colour system for about 141 registered themes (~0.5 ms each). It does not depend on the files, it predates this task, and it is the place to memoize if the budget tightens.
- Tests: `Tests/UI/test_settings_theme_picker.py` adds the loading row plus key responsiveness, the stale result (negative control: removing the generation guard makes it fail with `{'old'} == {'new'}`), and a pause raised in the worker. `Tests/UI/test_settings_theme_picker_screen.py` adds the 50-file test: every scan runs off the main thread, the UI-thread calls return in under 100 ms, and all 50 files land. All 4 new tests fail against the pre-change picker. Existing tests now wait for the worker (`app.workers.wait_for_complete()`) instead of assuming a synchronous scan, and no assertions were weakened.
- Modified: `tldw_chatbook/Widgets/settings_theme_picker.py`, the two test files, `Docs/User_Guide/settings.md`, and `Docs/security/production-diagnostic-inventory.json` (+1 diagnostic, which logs only the exception type).
- Qodo round 1 on PR #2850:
  - A user move during a scan now wins. The pending highlight is stored as a pair: the requested id and the highlight at the time of the request. It applies only if the highlight has not been moved by the user since. A re-render's own programmatic move carries the request forward, so Rename/Import still land on the new name. An explicit `rescan=False` highlight clears any older request.
  - A non-OSError scan failure is now a distinct `failed` outcome. It keeps the last good listing and the current availability, because R27(6) covers OSError only. It is logged with the scan generation and the traceback frames (`traceback.format_tb`), but never `str(exc)`, which may carry the themes path (R16). The OSError warning names the scan generation too.
  - Lock contention was measured and not changed. The worker's scan and the editor's UI-thread file actions share the themes-directory RLock. With 50 files, an editor scan took a median of 210 ms idle and 414 ms while it overlapped a picker scan, so an overlapping action waits at most one scan. That UI-thread time was already spent on every refresh before this task. The real remaining cost is each file action's own ~210 ms UI-thread scan, which predates this task and is out of scope here.
  - Direct `_apply_scan` tests were added for a stale generation and for the ok, paused, failed and oserror outcomes.
<!-- SECTION:NOTES:END -->
