---
id: TASK-2775
title: 'Settings has no About pane — version/project info unreachable since TASK-1346'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 17:30'
labels:
  - settings
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing task-1995: the About section (project description, license, GitHub/docs/issues links) lives only in `UI/Tools_Settings_Window.py` (`#ts-view-about`), and that whole window is unrouted dead UI since TASK-1346 — the `tools_settings` route resolves to MCPScreen, and the canonical F9 Settings screen (`UI/Screens/settings_screen.py`) has no About category. There is currently no place in the app where a user can see what version they run, the license, or where to file an issue.

The About content itself was converted to real markdown in task-1995 (`ABOUT_MARKDOWN` constant) and is ready to mount; the work here is deciding its home on the Settings screen (a small "About" category or a footer block on Overview) and wiring the existing `Markdown.LinkClicked` → browser handler.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can reach an About section from the F9 Settings screen showing project description, license, and GitHub/docs/issues links
- [x] #2 Links open in the system browser with a notification
- [x] #3 The section shows the installed application version
- [x] #4 Live capture at 235x52 recorded
<!-- AC:END -->

## Implementation Plan (the how)

1. Move ABOUT_MARKDOWN to `Utils/about_text.py` (+ `get_app_version()` via importlib.metadata; deprecated window re-imports for back-compat).
2. New read-only `SettingsCategoryId.ABOUT` under Troubleshooting, taught to EVERY per-category surface: summaries, rail group, _INSPECTOR_GUIDANCE (compose-critical), ownership record (writes_allowed=False → automatic "(view)" badge), guided-edits hint, detail-pane branch (version row, license row, Markdown with open_links=False), link handler (http(s)→browser+notify, else warning — same policy as Console/HF links).
3. Update the pinned category-count guard (23→24) deliberately; tests; live capture; User Guide section + stamp.

## Implementation Notes

`settings_config_models.py` (enum), `settings_screen.py` (six per-category surfaces + render branch + `_handle_about_link`), `Utils/about_text.py` (new), `Tools_Settings_Window.py` (re-import), `Docs/User_Guide/settings.md` (category row, About section, stamp), tests: `Tests/UI/test_settings_about_2775.py` (4) + the count-pin update in `test_settings_configuration_hub.py`. Full hub suite green (315 passed) — including the exhaustiveness guards this addition was required to satisfy. Live tmux capture at 235x52: rail shows "About (view)" under Troubleshooting; pane shows Version: 0.1.8.0 (real installed metadata), License AGPLv3+, rendered markdown links; inspector shows the About guidance.
