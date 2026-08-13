---
id: TASK-15470
title: Config persistence: batch writes and take them off the click path
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/Dictation_Window_Improved.py:765-781` `_save_settings` performs 8-10 sequential `save_setting_to_cli_config` calls — each a full config.toml read + atomic rewrite + settings-cache reload, synchronous on the event loop — and the buffer-duration input triggers it PER KEYSTROKE whenever the typed value parses (`:502-513`). The batch API (`save_settings_to_cli_config`) already exists and is simply not used. Separately, every Chat sidebar collapsible toggle rewrites `ui_state.toml` synchronously (read+parse+serialize+write) via `watch_sidebar_state` (`chat_screen.py:19744-19807`). About ten more verified one-rewrite-per-click sites: Library notes-sync controls (`library_screen.py:19534-19657`), ingest backend/browse/submit/reset (`:20040/:20316/:21070/:21681`), `Widgets/enhanced_file_picker.py:1137`, `Widgets/settings_splash_screen_viewer.py:237`, `settings_screen.py:3581`, `UI/Console_Modules/console_settings_modal.py:1201`.

Fix direction: batch the dictation save into one call; debounce + `to_thread` the writes; keep parsed ui-state in memory instead of re-loading the file per save. Stability constraints: the config writer's lock + atomic-replace semantics must be preserved, and any debounced write must flush on unmount/quit so no setting is lost. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dictation save performs one batched config write; no config rewrite fires on a parsing keystroke
- [ ] #2 Sidebar toggles perform no synchronous file I/O on the event loop; state survives restart including quit-immediately-after-toggle (flush test)
- [ ] #3 Each listed per-click rewrite site is converted or explicitly justified in the notes
<!-- AC:END -->

## Implementation Plan

1. Verify each audit-cited site at HEAD (line numbers have shifted / some already partially fixed by other sessions since the audit).
2. Dictation window: batch `_save_settings` into one `save_settings_to_cli_config` call (`dictation` + `dictation.privacy` sections); make `_persist_settings` (the single mount-noise gate) debounce (0.6s) instead of writing synchronously; dispatch the write to a worker off the loop; snapshot `self.settings` on the main thread before handing to the worker; add `on_unmount` to force-flush a pending write.
3. Chat sidebar: convert `watch_sidebar_state` from a synchronous `_save_sidebar_state()` call to a debounce (0.5s) + worker dispatch (`asyncio.to_thread`), snapshotting `ui_state` on the main thread first; add `on_unmount` flush (waiting on an in-flight worker rather than double-writing); route `handle_reset_settings` through the same scheduling helper since reassigning an already-`{}` `sidebar_state` is a reactive no-op that never calls the watcher.
4. Library screen notes-sync cluster (6 call sites) and ingest backend/browse/submit/reset (4 call sites): wrap each write in a `@work(thread=True)` helper, matching the file's existing `_save_library_search_history`/`_save_library_rail_preferences` convention. Keep `_remember_library_ingest_location` itself synchronous (existing tests call it directly without a mounted app) with a thin `@work` wrapper (`_persist_library_ingest_location`) at its one production call site instead.
5. Remaining four listed sites: `enhanced_file_picker.py`'s `_save_last_directory` (`@work`, it's a `ModalScreen` so `self.app` is always available), `settings_splash_screen_viewer.py`'s `_save_config_value` (split the in-memory update from the deferred write), `settings_screen.py`'s remote-images toggle AND the model-catalog stale-hours `Input.Changed` handler (found in the same ADR-020 immediate-write cluster while fixing the toggle -- same per-keystroke defect class as dictation, not in the audit's literal citation but the same bug), `console_settings_modal.py`'s Save-as-default button (`asyncio.to_thread`, not fire-and-forget, since its success/failure UX contract needs the awaited result before deciding whether to dismiss or show an inline error).
6. Update every existing test that called a handler on a bare/unmounted screen and hit `run_worker`'s `NoActiveAppError` -- patch the new per-site worker method instead of the module-level config function it wraps, preserving each test's actual subject.
7. Write flush-on-quit tests for dictation and sidebar; mutation-verify by disabling the flush and confirming the test goes red, then restore via Edit.
8. Run the full affected test surface; write the report.
