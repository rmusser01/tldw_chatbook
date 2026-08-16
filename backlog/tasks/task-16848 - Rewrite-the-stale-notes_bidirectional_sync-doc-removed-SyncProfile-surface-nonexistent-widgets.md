---
id: TASK-16848
title: 'Rewrite the stale notes_bidirectional_sync doc (removed SyncProfile surface, nonexistent widgets)'
status: To Do
assignee: []
created_date: '2026-08-16'
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
- [ ] #1 The doc no longer describes SyncProfile CRUD, auto-sync profiles, or `sync_profiles.json`
- [ ] #2 Every file the doc cites exists at the cited role, and the described flow matches the shipped `sync_folder` behavior (verified against the live code, stamp updated)
- [ ] #3 If retired instead, User_Guide coverage of the surviving notes-sync flow is confirmed and any inbound links are repointed
<!-- AC:END -->
