---
id: TASK-33081
title: Remove the legacy Tools and Settings window
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [cleanup, ui, settings]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI/Tools_Settings_Window.py is navigation-unreachable (the tools_settings route resolves to MCPScreen in UI/Navigation/screen_registry.py) and fully superseded by the canonical Settings hub, which has strictly better persistence guarantees (snapshot-fenced TOML writes versus the window's unfenced parse-and-replace). Deleting the window and its test surface removes roughly 9k LOC of dead settings code and retires a parallel unfenced config write path that can never be reached through the UI but must still be read, reviewed, and kept green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tools_Settings_Window.py, its unrouted screen wrapper, and tools_settings_messages.py are deleted with no remaining importers.
- [ ] #2 Backup-manifest and character-card serialization helpers are relocated to a canonical home and their test importers pass.
- [ ] #3 The DB backup and restore capability is either migrated to the Settings hub or explicitly dropped, with the decision recorded in Implementation Notes.
- [ ] #4 The dead IngestUiStyleChanged handler in app.py, vestigial window IDs, and orphaned CSS constants are removed.
- [ ] #5 Targeted tests covering the relocated helpers pass.
<!-- AC:END -->
