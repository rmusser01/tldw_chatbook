---
id: TASK-2775
title: 'Settings has no About pane — version/project info unreachable since TASK-1346'
status: To Do
assignee: []
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
- [ ] #1 A user can reach an About section from the F9 Settings screen showing project description, license, and GitHub/docs/issues links
- [ ] #2 Links open in the system browser with a notification
- [ ] #3 The section shows the installed application version
- [ ] #4 Live capture at 235x52 recorded
<!-- AC:END -->
