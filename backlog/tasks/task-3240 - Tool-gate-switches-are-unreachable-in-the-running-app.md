---
id: TASK-3240
title: Tool gate switches are unreachable in the running app
status: To Do
assignee: []
created_date: '2026-08-07 20:15'
labels:
  - settings
  - ux
  - tools
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every [tools] gate switch — all _GATEABLE_BUILTINS rows plus the web_deep_search row task-3222 added — lives in UI/Tools_Settings_Window.py, which TASK-1346 deprecated: the "tools_settings" route in screen_registry.py resolves to MCPScreen, and the canonical settings_screen.py has no Tools category. The only live surface exposing tool gates is the FirstRunSetupWizard, which a user sees once. Net effect: after first run, no tool gate can be flipped from inside the app at all; users must hand-edit config.toml. Task-3222 satisfied its ACs inside the deprecated window (where the sibling machinery lives) and this task carries the real product gap it uncovered. Owner decision needed on the fix shape: add a Tools category to the canonical settings_screen (note: settings_screen.py is under active critique work on fix/settings-ux-critique-rounds — coordinate to avoid collision), or re-route "tools_settings" to a live wrapper of the existing window. Restart-to-apply semantics for construction-time gates (web_deep_search) must stay visible in whatever ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can reach a tool-gate settings surface through the running app's navigation after first run
- [ ] #2 All [tools] gate switches (gateable builtins + web_deep_search) are present there and round-trip to config.toml
- [ ] #3 Gates that need an app restart to apply state that where shown
<!-- AC:END -->
