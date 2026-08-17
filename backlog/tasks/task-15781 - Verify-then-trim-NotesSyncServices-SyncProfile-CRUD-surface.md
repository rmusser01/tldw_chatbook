---
id: TASK-15781
title: Verify-then-trim NotesSyncService's SyncProfile CRUD surface
status: Done
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - cleanup
  - notes
priority: low
---

## Description

Verify-then-trim candidate surfaced by task-15481's disclosure (input-latency
burn-down's dead-code sweep). Task-15481 trimmed `Notes/sync_service.py`'s
dead auto-sync-loop machinery (`create_profile`, `start_auto_sync`,
`stop_auto_sync`, `stop_all_auto_syncs`) but explicitly left `SyncProfile`,
profile load/save, `get_profile`, `list_profiles`, and `sync_with_profile` in
place, on the grounds that they were "not named dead by the task and not a
crash-on-touch landmine" — i.e. task-15481 did not investigate whether
these remaining methods have production callers, only that they were out of
its own AC's scope.

Re-verified here by a repo-wide grep: `library_screen.py` is the sole
production importer of `NotesSyncService`, and it calls exactly one method
on the instance it constructs — `service.sync_folder(...)`. No production
code anywhere calls `NotesSyncService.get_profile`, `.list_profiles`,
`.sync_with_profile`, or constructs a `SyncProfile`. (Several *other*,
unrelated `get_profile`/`list_profiles` methods exist elsewhere in the
codebase — TTS profile managers, RAG profile managers, MCP local control —
none of which touch `Notes/sync_service.py`; they are not evidence against
this finding.)

## Acceptance Criteria

- [x] Re-verify at implementation time that `SyncProfile`, `get_profile`,
      `list_profiles`, and `sync_with_profile` in `Notes/sync_service.py`
      have zero production callers (grep, not trusting this task's finding
      blind)
- [x] If still dead: remove the CRUD surface (with git-log provenance
      recorded) and its now-orphaned test-only coverage, keeping
      `NotesSyncService.sync_folder` and everything it depends on untouched
- [x] If a live caller is found: the task closes as "not dead," with the
      caller documented, and no removal happens
- [x] `Tests/Notes/` and `Tests/Sync_Interop/` stay green; a final grep
      sweep for the removed names returns no production hits

## Implementation Plan

1. Re-verify reachability of `SyncProfile`, `get_profile`, `list_profiles`,
   `sync_with_profile` (and their unlisted support machinery --
   `delete_profile`, `_load_profiles`, `_save_profiles`, `self.profiles`,
   `config_path`, `sync_profiles.json`) via repo-wide grep across
   production code, tests, and docs. Confirm `library_screen.py` (the sole
   production importer of `NotesSyncService`) only calls the constructor
   and `.sync_folder(...)`. Confirm no test file calls the profile-CRUD
   methods either (dead-with-no-test, not dead-with-a-test).
2. Confirm the profile CRUD surface has no DB-layer counterpart (it is a
   flat `sync_profiles.json` file next to the app config, not a DB table)
   so there is no schema-orphan question to defer.
3. Baseline `Tests/Notes/` + `Tests/Sync_Interop/` before touching code.
4. Trim: remove `SyncProfile`, `get_profile`, `list_profiles`,
   `sync_with_profile`, `delete_profile`, `_load_profiles`,
   `_save_profiles` from `Notes/sync_service.py`; strip the now-dead
   `config_path`/`self.profiles` plumbing from `__init__` (including the
   local `_get_effective_config_path` import it existed only to feed).
   Keep `sync_folder`, `get_sync_history`, `get_conflicts_for_session`,
   `resolve_conflict`, `get_notes_sync_status`, and `sync_engine`
   untouched.
5. Re-run `Tests/Notes/` + `Tests/Sync_Interop/`; final grep sweep for the
   removed names; ruff check/format on the touched file.
6. Record git-log provenance and evidence table in Implementation Notes.

## Implementation Notes

Re-verified the task's own finding with a fresh repo-wide grep and found the
dead surface was actually *wider* than the four named symbols: the CRUD
surface's private support methods (`_load_profiles`, `_save_profiles`,
`config_path`) and one more public method the task didn't name
(`delete_profile`) are also unreachable, since their only callers were the
four already-dead public methods. All of it lived only in
`tldw_chatbook/Notes/sync_service.py`; no DB table backs it (`SyncProfile`
round-trips through a flat `sync_profiles.json` file next to the app config,
not `ChaChaNotes_DB.py`), so there is no schema-orphan question to defer to
a migration.

### Per-method verdict

| Symbol | Verdict | Evidence |
|---|---|---|
| `SyncProfile` (class) | **DEAD** | Zero constructions anywhere outside its own `from_dict`; zero external imports. Two unrelated classes elsewhere (`Sync_Interop.sync_state.SyncProfileState`, `Sync_Interop.sync_profile_status_state.SyncProfileStatusDisplay`) share the `SyncProfile` prefix but are distinct types with no relationship to `Notes/sync_service.py` -- confirmed via `\bSyncProfile\b` word-boundary grep, which returns zero hits post-trim. |
| `NotesSyncService.get_profile` | **DEAD** | Zero `.get_profile(`/`.list_profiles(`/etc. calls resolve to a `NotesSyncService`/`Notes.sync_service` instance anywhere in `tldw_chatbook/` or `Tests/`. Other `get_profile`/`list_profiles` hits repo-wide are unrelated TTS voice-profile managers, RAG config-profile managers, and the MCP local-control store -- verified by reading each hit's owning class. |
| `NotesSyncService.list_profiles` | **DEAD** | Same evidence as above. |
| `NotesSyncService.sync_with_profile` | **DEAD** | Zero callers anywhere, production or test (`sync_with_profile` is a unique-enough string that grep alone is conclusive -- no other symbol collides). |
| `NotesSyncService.delete_profile` | **DEAD** (not named in the task's AC, found during verification) | Zero callers pointing at `Notes.sync_service`; other `delete_profile` hits are TTS/RAG/MCP profile managers, confirmed unrelated the same way. |
| `NotesSyncService._load_profiles` | **DEAD-IN-EFFECT** | Called unconditionally from `__init__` (so technically reachable on every `library_screen.py` sync run), but its only effect -- populating `self.profiles` -- had zero remaining readers once the four public accessors above are gone. Kept, it would silently read a JSON file and log a line for a dict nothing ever consults. Removed together with its only consumers. |
| `NotesSyncService._save_profiles` | **DEAD** | Only called from `delete_profile` and `sync_with_profile`, both dead. |
| `NotesSyncService.config_path` / `self.profiles` / `__init__`'s `config_path=` kwarg | **DEAD** | Support-only state for the above. No caller anywhere passes `config_path=` to `NotesSyncService(...)` (both production and test construction sites use the 2-arg `notes_service=`/`db=` form), and nothing outside the class reads `.config_path` or `.profiles`. |
| `NotesSyncService.sync_folder` | **LIVE -- untouched** | Sole method `library_screen.py`'s `_run_library_notes_sync` calls on the `NotesSyncService` it constructs; `library_screen.py` is the sole production importer of the class. |
| `get_sync_history`, `get_conflicts_for_session`, `resolve_conflict`, `get_notes_sync_status` | **out of AC scope -- left untouched, not investigated** | Unrelated to the `SyncProfile` CRUD surface (they read/write `sync_sessions`/`sync_conflicts`/`notes` DB tables, not the profile JSON file); the task's AC names only the profile-CRUD symbols, so their reachability was not re-verified here. |

### What was trimmed vs preserved

Removed from `tldw_chatbook/Notes/sync_service.py` (138 lines, pure deletion,
no other file touched): the `SyncProfile` class in full (`to_dict`/
`from_dict`), and `NotesSyncService.get_profile`, `.list_profiles`,
`.sync_with_profile`, `.delete_profile`, `._load_profiles`,
`._save_profiles`. `__init__` lost its `config_path` parameter and the
`self.config_path` / `self.profiles` / `self._load_profiles()` lines, along
with the now-pointless local `from ..config import _get_effective_config_path`
import that existed only to compute the default `sync_profiles.json` path.

Preserved untouched: `sync_folder` (and everything it depends on --
`sync_engine`, `NotesSyncEngine`, `SyncDirection`, `ConflictResolution`,
`SyncProgress`), `get_sync_history`, `get_conflicts_for_session`,
`resolve_conflict`, `get_notes_sync_status`.

No orphaned test-only coverage was found to delete -- a full sweep of
`Tests/` for `SyncProfile`, `.get_profile(`/`.list_profiles(` against a sync
instance, `sync_with_profile`, and `delete_profile` (in a `Notes.sync_service`
context) turned up zero test references. This subset was dead-with-no-test,
not dead-with-a-test: `Tests/Notes/test_library_notes_sync_integration.py`
only exercises `sync_folder`, and `Tests/UI/test_library_shell.py`'s
`_RecordingNotesSyncService` mock only records `sync_folder` calls.

### Git-log provenance

`SyncProfile` and the CRUD methods were part of the file's original add
(`c137cd54b`). Task-15481 (`13a179c8f`, "chore: trim dead auto-sync loop
from Notes/sync_service.py") most recently touched this file, removing the
dead auto-sync scheduler (`create_profile`, `start_auto_sync`,
`stop_auto_sync`, `stop_all_auto_syncs`) but explicitly leaving `SyncProfile`
and the rest of the CRUD surface in place as out-of-scope rather than
verified-live. This task closes that gap.

### Schema implications

None -- `SyncProfile` was never DB-backed. It round-tripped through a flat
JSON file (`<config dir>/sync_profiles.json`), not a `ChaChaNotes_DB.py`
table, so there is no orphaned schema/migration to flag. A pre-existing
`sync_profiles.json` on a user's disk (if one was ever hand-created or
survived from an older build) simply becomes inert -- nothing reads or
writes it anymore.

### Documentation note (not acted on, out of AC scope)

`Docs/Features/notes_bidirectional_sync.md` documents the now-removed
`SyncProfile`/auto-sync surface (its "Sync Profiles" feature bullet, "Creating
a Sync Profile"/"Using Sync Profiles"/"Auto-Sync" usage walkthrough, and a
`sync_profiles.json` config example) and also references UI files
(`notes_sync_widget.py`, `notes_sync_events.py`) that no longer exist in the
repo -- this doc was already stale before this task (task-15481 removed the
auto-sync scheduler it also describes) and describes a pre-Library-embed
standalone Notes sync panel. Per CLAUDE.md's "do not implement anything not
in the AC" rule, this doc rewrite was left out of scope rather than done
piecemeal; flagging here in case the owner wants a dedicated doc-refresh
task.

### Tests

- Baseline (pre-trim, HEAD `edf357154`): `pytest Tests/Notes/ Tests/Sync_Interop/`
  -> 9 failed, 2656 passed, 5 skipped (209.51s). All 9 failures are in
  `Tests/Notes/test_note_import_planner.py` (`AttributeError: module 'os' has
  no attribute 'ScandirIterator'`), unrelated to `sync_service.py` and
  pre-existing on this environment/Python version.
- Post-trim: `pytest Tests/Notes/ Tests/Sync_Interop/` -> identical
  9 failed, 2656 passed, 5 skipped (192.60s) -- same 9 failures, same
  file/line, zero regressions.
- Targeted: `pytest Tests/Notes/test_library_notes_sync_integration.py
  Tests/Notes/test_sync_engine.py Tests/Notes/test_sync_containment.py` ->
  32 passed.
- `ruff check tldw_chatbook/Notes/sync_service.py` -> All checks passed
  (confirms no leftover unused imports from the trim).
- `ruff format --check tldw_chatbook/Notes/sync_service.py` -> already
  formatted.
- Final grep sweep for `SyncProfile`, `.get_profile(`, `.list_profiles(`,
  `sync_with_profile`, `.delete_profile(`, `_load_profiles`,
  `_save_profiles`, `sync_profiles.json`, and `NotesSyncService(...
  config_path=` across `tldw_chatbook/` and `Tests/` -> zero hits pointing at
  `Notes/sync_service.py`'s removed surface (remaining hits are the
  unrelated TTS/RAG/MCP profile managers and one unrelated
  `Voice_Cloning_Window._load_profiles` method, confirmed by owning class).

### Files changed

- `tldw_chatbook/Notes/sync_service.py` -- 138-line deletion (`SyncProfile`
  class + dead CRUD/support methods), no other lines touched.
