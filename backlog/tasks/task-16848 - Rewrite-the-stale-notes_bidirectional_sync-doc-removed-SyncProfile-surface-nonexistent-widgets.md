---
id: TASK-16848
title: >-
  Rewrite the stale notes_bidirectional_sync doc (removed SyncProfile surface,
  nonexistent widgets)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-16 17:33'
labels:
  - docs
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Docs/Features/notes_bidirectional_sync.md` documents a feature surface that no longer
exists, confirmed by the TASK-15781 review (PR #1711) and re-verified at dev
`ee741cf10`:

- It documents the **SyncProfile CRUD surface** ("### 3. Sync Profiles" line 25,
  "Creating a Sync Profile" 105, "Using Sync Profiles" 112, "Auto-Sync" 118, and a
  `sync_profiles.json` config example ~275) — all removed from `Notes/sync_service.py`
  by task-15781, which verified the profile CRUD had zero callers and trimmed
  `NotesSyncService` down to the live `sync_folder` path.
- It cites **two widget files that do not exist anywhere in the repo**:
  `notes_sync_widget.py` (line 84) and `notes_sync_events.py` (line 90) —
  `find`-verified absent.

The live reality: `NotesSyncService(notes_service=, db=)` constructed in one place
(`UI/Screens/library_screen.py`, Library's notes-sync flow), `sync_folder` the only
method called; conflict handling and history are what remain. Rewrite the page against
that (or fold it into the relevant `Docs/User_Guide/` library page and retire this one),
with a current "Verified against" stamp. Flagged as out-of-scope-but-must-be-filed by
15781's own notes; no existing backlog task targets this doc.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The doc no longer describes SyncProfile CRUD, auto-sync profiles, or `sync_profiles.json`
- [x] #2 Every file the doc cites exists at the cited role, and the described flow matches the shipped `sync_folder` behavior (verified against the live code, stamp updated)
- [x] #3 If retired instead, User_Guide coverage of the surviving notes-sync flow is confirmed and any inbound links are repointed (N/A — kept as a page, not retired: `Docs/User_Guide/library/notes.md` already covers the user-facing flow and cross-links here; the two inbound links from User_Guide keep resolving since the path is unchanged)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the live surface: Notes/sync_service.py (post-trim NotesSyncService: sync_folder + get_sync_history/get_conflicts_for_session/resolve_conflict/get_notes_sync_status), Notes/sync_engine.py (NotesSyncEngine algorithm, SyncDirection, ConflictResolution, SyncProgress), Notes/sync_paths.py (PinnedSyncRoot path-safety boundary), Library/library_notes_sync_state.py (panel display-state, SYNC_CONFLICTS excludes ASK), and library_screen.py's actual sync entry point (#library-notes-sync-open button -> in-canvas panel -> _run_library_notes_sync -> NotesSyncService.sync_folder on a worker thread; auto-sync Timer).
2. Grep-verify every file/class/method the old doc named is either real or phantom; confirm notes_sync_widget.py / notes_sync_events.py do not exist anywhere and SyncProfile/sync_profiles.json machinery is gone from sync_service.py (task-15781/PR #1711).
3. Check Docs/User_Guide/library/notes.md for overlap -- confirmed it already accurately documents the user-facing panel (verified 2026-08-11) and already cross-links to this Features page as "deep dive on the sync engine." Decision: keep this page, defer the user-facing walkthrough to the User_Guide page (cross-link both directions), and rewrite this page to cover engine internals only (architecture, directions, conflict resolution incl. the ASK-excluded-from-UI nuance, path safety, DB schema, the four DB-backed history/conflict methods and that none are wired to any UI, and the real [notes] config keys replacing sync_profiles.json).
4. Rewrite Docs/Features/notes_bidirectional_sync.md against verified reality, add a "Verified against <sha>" stamp, shorten from 298 to ~188 lines since the live feature no longer supports the old doc's length.
5. Reference-check every path/class/method cited against HEAD; confirm inbound links (Docs/User_Guide/library.md, Docs/User_Guide/library/notes.md) still resolve since the file path is unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewrote Docs/Features/notes_bidirectional_sync.md from scratch against live code at dev 766b0ff9e. The prior version described a SyncProfile CRUD surface (save/name/reuse configs, sync_profiles.json, per-profile auto-sync) removed by task-15781/PR #1711, and cited two files (notes_sync_widget.py, notes_sync_events.py) that never exist in this repo.

New doc describes the real surface: NotesSyncEngine (sync_engine.py) + NotesSyncService (sync_service.py, sync_folder() is the only method the UI calls) + the path-safety boundary (PinnedSyncRoot/sync_paths.py) + the panel's own display-state module (Library/library_notes_sync_state.py) + library_screen.py, which owns the actual Notes-surface "Sync" button, in-canvas panel, worker dispatch (_run_library_notes_sync), and the fixed-300s auto-sync Timer. Also documents: the three directions; that the panel exposes only 3 of the engine's 4 ConflictResolution values (ASK is excluded -- the panel never prompts); the engine/service method default is actually ASK even though the panel's own persisted default and parse-fallback is newer_wins; that get_sync_history/get_conflicts_for_session/resolve_conflict/get_notes_sync_status exist but are called nowhere outside sync_service.py (no history/conflict UI exists); that resolve_conflict has a standing TODO and doesn't apply the resolution; and the real [notes] config keys (sync_directory, sync_direction, sync_conflict_resolution, auto_sync) replacing sync_profiles.json.

Checked Docs/User_Guide/library/notes.md first -- it already accurately documents the user-facing panel (last verified 2026-08-11) and already cross-links to this Features page as "deep dive on the sync engine behind the panel." Decision: keep this page (don't retire), defer the walkthrough/screenshots/common-tasks content to the User_Guide page, cross-link both directions, and keep this page scoped to engine internals a developer would need. Doc shrank 298 -> 188 lines (net -110) since the live feature is smaller than what the old doc described.

Verification: every file path (6), class/const name, and method cited was grep/Read-confirmed present at HEAD; every relative link resolves on disk. No other Features doc in the repo carries a "Verified against" stamp (checked all 9), but the task description asked for a current one, so this page carries one anyway, matching the User_Guide convention. Inbound links (Docs/User_Guide/library.md, Docs/User_Guide/library/notes.md) both already point at this unchanged path, so no repointing was needed.

Modified: Docs/Features/notes_bidirectional_sync.md.
<!-- SECTION:NOTES:END -->
