# Notes Bi-Directional Sync — engine internals

*Verified against dev @ `766b0ff9e` — 2026-08-16 (task-16848: rewritten from
scratch — the previous version described a SyncProfile CRUD surface removed
by task-15781/PR #1711, and cited `notes_sync_widget.py` /
`notes_sync_events.py`, which never exist in this repo).*

For the user-facing walkthrough (where the "Sync" button lives, what each
control does, screenshots) see
**[Library notes → Notes sync panel](../User_Guide/library/notes.md#notes-sync-panel)**.
This page covers the engine underneath that panel: what it does, how
conflicts are decided, and what's tracked in the database. Every path,
class, and method named below was read at the commit stamped above.

## What it does

The sync feature mirrors Markdown/text files in a folder on disk with
notes in the ChaChaNotes database. There is exactly one entry point in the
running app: the Library screen's Notes surface, notes list toolbar,
**Sync** button (`#library-notes-sync-open` in
`tldw_chatbook/UI/Screens/library_screen.py`), which opens an in-canvas
panel — not a separate screen, dialog, or file. There is no other way to
start a sync (no CLI command, no scheduled job outside the panel's own
auto-sync toggle described below).

There used to be a **Sync Profiles** system (save/name/reuse multiple
folder+direction+conflict configurations, one auto-sync interval per
profile, backed by `~/.config/tldw_cli/sync_profiles.json`). That CRUD
surface had zero callers and was removed from `NotesSyncService` by
task-15781 (PR #1711). What remains is a single, global configuration
(one folder, one direction, one conflict policy, one auto-sync toggle) —
see [Configuration](#configuration) below.

## Components

| File | Role |
|---|---|
| `tldw_chatbook/Notes/sync_engine.py` — `NotesSyncEngine` | The sync algorithm: scans the disk, reads the DB-synced notes for the folder, diffs by content hash, and applies one of the three directions. |
| `tldw_chatbook/Notes/sync_service.py` — `NotesSyncService` | Thin wrapper the UI calls: `sync_folder(...)` drives one engine pass; the rest of the class is DB-backed history/conflict lookups (see [Sync history & conflict tracking](#sync-history--conflict-tracking-db-only)). |
| `tldw_chatbook/Notes/sync_paths.py` — `PinnedSyncRoot`, `SafeSyncFile`, `SyncPathError` | The filesystem boundary: opens the sync root as a directory descriptor and rejects symlinked/reparse entries, so the engine can never read or write outside the folder the user picked. |
| `tldw_chatbook/Library/library_notes_sync_state.py` | Pure display-state helpers for the panel (label/status-line formatting, direction/conflict cycling) — no I/O. |
| `tldw_chatbook/UI/Screens/library_screen.py` | Owns the panel's widgets, config load/persist, the `Sync now` worker dispatch (`_run_library_notes_sync`), and the auto-sync `Timer`. |

There is no `notes_sync_widget.py` or `notes_sync_events.py` — the panel is
composed and event-handled directly inside `library_screen.py`, in the
"Notes sync panel" section of that file.

`NotesSyncService` is constructed fresh per run —
`NotesSyncService(notes_service=notes_service, db=db)` — inside
`_run_library_notes_sync`; it is not a long-lived singleton.

## Sync directions

`SyncDirection` (`sync_engine.py`), one of three, chosen in the panel's
"Direction" choice row:

- `disk_to_db` — files on disk win; new/changed files create or update
  notes, and a file that disappeared from disk is recorded as a conflict
  (see below), not silently deleted.
- `db_to_disk` — notes in the DB win; new/changed notes create or update
  files.
- `bidirectional` (default) — both sides are diffed against the
  last-synced hash; whichever side changed since the last sync wins for
  that item, and an item changed on **both** sides is a conflict.

## Conflict resolution

`ConflictResolution` (`sync_engine.py`) has four members —
`ASK`, `DISK_WINS`, `DB_WINS`, `NEWER_WINS` — but the Library panel only
ever offers three of them: `newer_wins` (default), `disk_wins`, `db_wins`.
`ASK` is deliberately excluded from the panel (`SYNC_CONFLICTS` in
`library_notes_sync_state.py`): nothing in the Library ever prompts on a
conflict, so the panel would never be able to act on it. In practice this
means **the panel never asks** — every conflict is resolved automatically
by whichever of the three policies is selected, and the resulting count is
reported afterward in the status line and activity log.

A conflict (both sides changed since the last sync, or a file the DB
still expects went missing from disk) is always recorded as a row in the
`sync_conflicts` table (see [Database schema](#database-schema)) before
any auto-resolution runs, regardless of which policy is active.

`newer_wins` compares the note's `last_modified` timestamp against the
file's mtime and keeps whichever is more recent — this is the
last-write-wins behavior CLAUDE.md's "Notes Sync" summary refers to. It is
the Library panel's persisted config default (`sync_conflict_resolution`,
see [Configuration](#configuration)) and its parse-fallback for an
unrecognized stored value, though not the engine method's own parameter
default — `NotesSyncEngine.sync()` and `NotesSyncService.sync_folder()`
both default to `ConflictResolution.ASK` when called with no explicit
value. The panel never relies on that default: it always resolves and
passes one of its three exposed policies explicitly.

## Path safety

Every read and write goes through `PinnedSyncRoot`
(`sync_paths.py`), which opens the sync root once as a directory file
descriptor (`O_NOFOLLOW`/`O_DIRECTORY`, POSIX-only; the guard degrades to
"unsupported" rather than silently trusting paths on platforms without
those flags) and re-verifies identity on every child open. A symlink, a
path that resolves outside the pinned root, or an entry on the wrong
device is rejected as a `SyncPathError` and recorded in
`SyncProgress.skipped_items` (reason only — no raw exception text) instead
of being read or written. This is why a sync run can report skipped items
distinct from either "no change" or "error."

## Database schema

Fields added to the `notes` table (`tldw_chatbook/DB/ChaChaNotes_DB.py`):

```sql
file_path_on_disk TEXT UNIQUE,           -- Absolute path to synced file
relative_file_path_on_disk TEXT,         -- Path relative to sync root
sync_root_folder TEXT,                   -- Root folder for this sync
last_synced_disk_file_hash TEXT,         -- SHA256 hash at last sync
last_synced_disk_file_mtime REAL,        -- File modification time at last sync
is_externally_synced BOOLEAN DEFAULT 0,  -- Whether note is synced
sync_strategy TEXT,                      -- Preferred sync direction
sync_excluded BOOLEAN DEFAULT 0,         -- Exclude from sync
file_extension TEXT DEFAULT '.md'        -- File extension to use
```

Plus two standalone tables the engine writes on every run:
`sync_sessions` (one row per sync pass — folder, direction, conflict
policy, status, file/conflict/error counts) and `sync_conflicts` (one row
per detected conflict, linked to its session).

None of `is_externally_synced`, `file_path_on_disk`, or the other sync
columns currently drive any per-note UI indicator — the Library notes list
shows no synced/conflict badge on individual rows (confirmed: nothing in
`library_screen.py` reads those columns for display). The only sync
feedback surfaced to the user is the panel's own status line and activity
log, described in the User Guide page linked above.

## Sync history & conflict tracking (DB-only)

`NotesSyncService` exposes four methods that read/write `sync_sessions`
and `sync_conflicts` directly, beyond the `sync_folder()` call the panel
uses:

- `get_sync_history(limit=50)` — recent sessions from `sync_sessions`.
- `get_conflicts_for_session(session_id)` — conflict rows for one session.
- `resolve_conflict(conflict_id, resolution, user_id)` — records a
  resolution string on a `sync_conflicts` row. **It does not currently
  apply the resolution** — the method body has a standing
  `# TODO: Implement actual resolution logic based on resolution type`;
  it only marks the row resolved.
- `get_notes_sync_status(root_folder=None)` — computes a per-note status
  (`synced` / `file_changed` / `db_changed` / `conflict` / `file_missing`
  / `file_error`) by re-hashing each synced note's current DB content and
  its file on disk.

None of these four are called anywhere outside `sync_service.py` itself —
there is no history browser, conflict-review list, or per-note status
view in the running app today. They exist as service-layer API surface;
the Library panel's own activity log (capped at 20 entries, in-memory,
not backed by these tables) is what a user actually sees.

## Configuration

`[notes]` in `config.toml`, read and written by the sync panel (defaults
shown):

- `sync_directory` — `~/Documents/Notes`
- `sync_direction` — `bidirectional`
- `sync_conflict_resolution` — `newer_wins`
- `auto_sync` — `false`

This is one global configuration, not a list of named profiles — there is
no `sync_profiles.json` and no per-profile settings anywhere in the repo.

When `auto_sync` is on, `library_screen.py` arms a single Textual
`Timer` (`self.set_interval(AUTO_SYNC_INTERVAL_SECONDS, ...)`,
`AUTO_SYNC_INTERVAL_SECONDS = 300` in `library_notes_sync_state.py` — a
fixed five minutes, not user-configurable) that reruns `sync_folder`
against the configured folder whenever no run is already in flight. The
timer is scoped to that Library screen instance's lifetime: it is
re-armed the next time the sync panel is entered while `auto_sync` is
true, not persisted or resumed as a background job across app restarts or
other screens.

## Related docs

- [Library notes — Notes sync panel](../User_Guide/library/notes.md#notes-sync-panel) —
  the user-facing walkthrough: controls, screenshots, common tasks.
- [Library overview](../User_Guide/library.md) — where Notes sits among
  the other Library sources.
