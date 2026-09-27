---
id: TASK-33081
title: Remove the legacy Tools and Settings window
status: In Progress
assignee:
  - '@robert'
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
- [ ] #4 Dead handler and orphaned setting surface removed: the IngestUiStyleChanged handler is already gone on dev; the media_ingestion.ui_style setting and get_ingest_ui_style getter (whose only caller and editor were the dead window) are removed; the live, palette-tested "tools_settings" alias to MCP and TAB_TOOLS_SETTINGS are retained.
- [ ] #5 Targeted tests covering the relocated helpers pass.
<!-- AC:END -->

## Implementation Plan

Verified against origin/dev before implementation (2026-09-27):

1. Relocate the three test-facing helper clusters the window squats on into tldw_chatbook/Backup_Recovery (the superseding backup surface): SETTINGS_DATABASES, the character-card backup serializer, and the manifest publication/write/unlink helpers (zero production callers; subjects of the TTS/Character_Chat/DB-migration tests). Repoint the three test files.
2. Repoint test_markdown_hygiene_1995.py at the existing canonical ABOUT_MARKDOWN home (Utils/about_text.py) — no relocation needed.
3. Delete UI/Tools_Settings_Window.py, UI/Screens/tools_settings_screen.py, UI/tools_settings_messages.py, css/features/_tools-settings.tcss (rebuild the bundle), and the window's own tests (Tests/UI/test_tools_settings_window.py, Tests/ProductionApp/test_tools_settings_backup.py).
4. Remove the ToolsSettingsScreen entries from UI/Screens/__init__.py and the window rows from test_widget_css_consolidation.py.
5. Remove get_ingest_ui_style and the ui_style default from config.py (no remaining callers or tests).
6. Retain the "tools_settings" palette alias to MCP and TAB_TOOLS_SETTINGS (live, pinned by the command-palette tests).
7. Run the targeted suite: relocated-helper tests, css consolidation, palette pair, settings about, design-token governance after the bundle rebuild, plus a package import sanity check.

ADR required: no
ADR path: N/A
Reason: deleting a navigation-unreachable deprecated UI surface; the canonical settings hub is already the governed surface, and the supersession decision for the DB backup/restore UI (drop, superseded by the new Backup_Recovery UI) was made by the owner on 2026-09-27 and is recorded here.

