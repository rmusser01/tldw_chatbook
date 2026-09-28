---
id: TASK-33126
title: Remove the legacy Tools and Settings window
status: Done
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
- [x] #1 Tools_Settings_Window.py, its unrouted screen wrapper, and tools_settings_messages.py are deleted with no remaining importers.
- [x] #2 Backup-manifest and character-card serialization helpers are relocated to a canonical home and their test importers pass.
- [x] #3 The DB backup and restore capability is either migrated to the Settings hub or explicitly dropped, with the decision recorded in Implementation Notes.
- [x] #4 Dead handler and orphaned setting surface removed: the IngestUiStyleChanged handler is already gone on dev; the media_ingestion.ui_style setting and get_ingest_ui_style getter (whose only caller and editor were the dead window) are removed; the live, palette-tested "tools_settings" alias to MCP and TAB_TOOLS_SETTINGS are retained.
- [x] #5 Targeted tests covering the relocated helpers pass.
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

## Implementation Notes

Approach: premise-checked against origin/dev first, then deleted with the three test-facing helper clusters relocated rather than lost. Commit e31d4583a0 on chore/task-33126-remove-legacy-settings-window (branched from docs/cascade-tasks-33081-33094-filings).

- Deleted: UI/Tools_Settings_Window.py (6,928 LOC), UI/Screens/tools_settings_screen.py, UI/tools_settings_messages.py, css/features/_tools-settings.tcss (bundle rebuilt via build_css.py, manifest entry retired with a house-style comment), Tests/UI/test_tools_settings_window.py, Tests/ProductionApp/test_tools_settings_backup.py — ~9.5k LOC removed.
- Relocated to new tldw_chatbook/Backup_Recovery/settings_backup_helpers.py: SETTINGS_DATABASES, serialize_character_cards_for_backup (was _-prefixed), and the staged-manifest machinery (BackupManifestPublication, build/write/unlink, cancellation + control-flow helpers). These have zero production callers but are the live subjects of the TTS profile-backup integration tests, the character-card backup export tests, and the ChaChaNotes migration test; all three test files repointed and passing.
- Decision recorded (AC #3): the legacy DB backup/restore UI is dropped, superseded by the Backup_Recovery surface (owner decision 2026-09-27).
- ABOUT_MARKDOWN needed no relocation — Utils/about_text.py was already canonical; test_markdown_hygiene_1995.py repointed.
- Re-owned the settings.bulk_backup / settings.integrity / settings.vacuum SQLiteOwnerPolicy entries from the deleted module path to tldw_chatbook/Backup_Recovery (contract tests drive these keys; production_module is descriptive).
- Removed media_ingestion.ui_style default and get_ingest_ui_style (only caller/editor was the window; no test references). Deviation from the original AC #4 wording, amended before implementation: the "tools_settings" route alias to MCP and TAB_TOOLS_SETTINGS are retained — they are live navigation pinned by the command-palette tests.
- Screens/__init__.py ToolsSettingsScreen entries removed; css consolidation allowlist rows removed.
- Verification: 486 passed / 2 skipped across the targeted battery (TTS profile backup, character backup export, markdown hygiene, css consolidation, settings about, command palette pair, design-token governance, private_sqlite, ChaChaNotes migration). 8 failures total — settings_about x3, css consolidation x1, palette providers x4 — each reproduced identically on a clean origin/dev baseline worktree, so all are pre-existing on dev, none introduced here.
- Remaining references to the window in code are provenance comments only (app.py, settings_screen.py, about_text.py, local_tool_provider.py, etc.), which document where relocated helpers came from and stay accurate.



## Renumbering provenance

Filed 2026-09-27 as task-33081. origin/dev minted its own task-33081 before this branch merged, so per the landed-keeps-id rule this task moved to task-33126; every inbound reference moved with it.
