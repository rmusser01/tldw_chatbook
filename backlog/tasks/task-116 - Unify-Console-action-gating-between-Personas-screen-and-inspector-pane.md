---
id: TASK-116
title: Unify Console-action gating between Personas screen and inspector pane
status: To Do
assignee: []
created_date: '2026-06-11 04:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
personas_screen._console_action_allowed and PersonasInspectorPane._apply_action_state derive enablement independently; when prompts/dictionaries/lore selection lands they will diverge. Push a single set_console_actions_enabled(bool) from the screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One source of truth for attach/start-chat enablement
<!-- AC:END -->
