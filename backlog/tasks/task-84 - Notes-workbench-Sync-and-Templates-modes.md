---
id: TASK-84
title: Notes workbench Sync and Templates modes
status: Done
assignee: []
created_date: '2026-06-10 03:16'
labels: []
dependencies:
  - TASK-83
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements Sync + Templates modes from Docs/superpowers/specs/2026-06-09-notes-workbench-design.md. Sync profiles/history UI is a documented follow-up. ADR: none required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync mode embeds sync status/quick-sync/progress/activity sections and runs a sync
- [x] #2 notes-sync-button switches to Sync mode instead of pushing modal
- [x] #3 Templates mode lists templates with preview and creates a note returning to Notes mode
- [x] #4 Tests cover both modes
<!-- AC:END -->

## Implementation Plan

1. NotesSyncPane embedding SyncStatusCard/QuickSyncSection/SyncProgressSection/RecentActivitySection from notes_sync_widget_improved.py
2. Wire NotesSyncService correctly (the modal's relative import was broken and always fell back to demo mode); run real syncs via sync_folder
3. Rewire #notes-sync-button to switch to Sync mode instead of pushing the modal (modal stays for the legacy window)
4. NotesTemplatesPane: template list + preview + create, returning to Notes mode with the new note loaded
5. Tests for sync run, sync-button mode switch, and templates flow

## Implementation Notes

- `NotesSyncPane` reuses the embeddable section containers from the modal but fixes two latent modal bugs: the broken `from ..Notes.sync_service` relative import (which forced permanent demo mode) and the nonexistent `sync_service.sync()`/`get_session_results()` API — the pane calls the real `NotesSyncService.sync_folder(...)` and summarizes the returned `SyncProgress`. When the service can't be wired, the pane reports "Sync service unavailable" honestly instead of running a fake demo sync.
- Compacting CSS overrides keep Sync Now / auto-sync / activity visible without scrolling (the section containers' 1fr-height Verticals otherwise push them below the fold); a test pins Sync Now visibility.
- `#notes-sync-button` in the editor controls now switches to Sync mode; `NotesSyncWidgetImproved` remains in use only by the legacy `UI/Notes_Window.py`.
- `NotesTemplatesPane` lists `NOTE_TEMPLATES` with a content preview; creating routes through the shared `_create_local_note_from_template` (also used by the navigator button) and returns to Notes mode with the note loaded.
- Sync profiles/history UI remains a documented follow-up (lives only in the older `notes_sync_widget.py`).
- Tests: 12 workbench tests (3 new for B2) + 67 notes tests + navigation suite all pass.
- QA captures: `Docs/superpowers/qa/notes-workbench/pr-b2-*`.
- ADR: none required.
