---
id: TASK-34388
title: Move config saves off the UI thread; batch the legacy general-settings save
status: Done
created_date: 2026-10-11 21:44
labels:
- performance
priority: medium
assignee:
- '@zcode'
updated_date: 2026-10-11 23:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
switch_theme (app.py) runs the full save_setting_to_cli_config cycle synchronously in a command-palette handler on the App message task, hitching the UI; the legacy Tools_Settings_Window _save_general_settings fires ~10 sequential full-cycle saves where the batch API save_settings_to_cli_config exists (already adopted by Dictation_Window_Improved and Logs_Window).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 switch_theme performs its config save via a worker/thread and does not block the event loop for the write,_save_general_settings persists all its keys through one save_settings_to_cli_config call,Theme switching and legacy general-settings Save behave identically (same keys, same values persisted),Settings-screen tests and app startup tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read app.py switch_theme and Tools_Settings_Window._save_general_settings
2. Move switch_theme's config save to a worker thread
3. Collapse the legacy general-settings sequential saves into one save_settings_to_cli_config call
4. Targeted tests + close
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
switch_theme saves via thread worker; legacy general settings batched through one atomic save_settings_to_cli_config call
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

- `app.py` `switch_theme`: the `save_setting_to_cli_config("general", "default_theme", ...)`
  write now runs on a thread via `self.app.run_worker(..., thread=True)`; failures are
  surfaced through `call_from_thread(self.app.notify, ...)` since `run_worker` has no
  `on_error` parameter. The theme applies instantly as before; only the persistence
  moved off the App message task.
- `UI/Tools_Settings_Window.py` `_save_general_settings`: the ~12 independent
  save_setting calls (general.default_tab/default_theme/users_name/log_level,
  media_ingestion.ui_style, splash_screen.enabled/duration, chat_defaults.
  provider/model/temperature, character_defaults.provider/model/temperature) are now
  collected into one dict and persisted through a single atomic
  `save_settings_to_cli_config` batch (import added to the module's top-level config
  import). Unparseable numeric fields keep the old warn-and-skip behavior; the
  IngestUiStyleChanged side-effect now fires after batch success; `_save_chat_api_key`
  and the encryption-toggle dialog flow are unchanged.
- Trade-off documented: the batch is atomic, so a failed save persists none of the
  form's keys where the old loop could persist a prefix.
- Verification: `Tests/UI/test_tools_settings_window.py`,
  `test_settings_tools_section.py`, `test_call_from_thread_guard.py`,
  `test_command_palette_basic.py`, `test_command_palette_providers.py` run; the only
  failures (tools-settings vacuum ×2, palette library-ingest ×3+7) reproduce with this
  task's changes stashed, i.e. pre-existing on the branch.
- Deviation: none from ACs.
