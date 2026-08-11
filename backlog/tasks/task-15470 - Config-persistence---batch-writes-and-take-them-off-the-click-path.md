---
id: TASK-15470
title: Config persistence: batch writes and take them off the click path
status: To Do
assignee: []
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
