---
id: TASK-32242
title: >-
  Library ingest browser reuses a remembered directory without validation or
  write ordering
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 10:15'
updated_date: '2026-09-11 12:00'
labels:
  - library
  - ingest
  - critique-notes-2026-09
  - rider
  - library-screen
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32174 / PR #2554. That PR gave the three Notes pickers a
remembered start directory and, after review, routed every read through
`Utils/path_validation.py` and made every write generation-ordered, in the
shared `tldw_chatbook/Library/library_browse_location.py`.

The older Library ingest browser has the identical shape and was deliberately
left alone, because it is outside that PR's scope and its existing tests
monkeypatch `library_screen.get_cli_setting` directly. Two gaps remain there:

- `LibraryScreen._library_ingest_browse_location` validates a *typed* path
  fragment through `validate_path_simple`, but its `[library.ingest]
  last_directory` branch does a bare `Path(...).expanduser().is_dir()`. A
  relative stored value is therefore resolved against the process working
  directory and handed to the picker.
- `_persist_library_ingest_location` writes unconditionally from a worker, so
  two picks made inside one config write can land out of order and leave the
  browser reopening at the older directory (the same race Qodo raised as
  finding 4 on PR #2554).

Both are small and low-severity; the point of the rider is that the fix is now
a two-line reuse of an existing shared module rather than new code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A relative, traversing or vanished `[library.ingest] last_directory`
  never reaches the ingest `FileOpen`; the browser falls back to the shared
  start-directory chain instead -- the folder `[notes] sync_directory` names
  when it exists, else the user's home directory. (Amended: this criterion
  originally said "opens at the user's home directory". task-32251 AC#5,
  landed on the same branch, inserted the configured-notes-folder step for
  every picker, so the original wording became literally false for a
  configured profile while the guarantee it exists to state -- the unusable
  value never reaches the picker -- is unchanged.)
- [x] #2 Of two ingest selections made before the first config write finishes,
  the later one is the directory the browser reopens at
- [x] #3 Both are pinned by tests that fail without the fix, and the ingest
  browser's typed-path-fragment behaviour is unchanged
<!-- AC:END -->


## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Read `library_browse_location` (the shared module PR #2554 landed) and the ingest browser's two divergent spots.
2. RED tests: a relative/traversing/vanished `[library.ingest] last_directory` must not reach `FileOpen`; of two picks in flight the later wins.
3. Replace the bare `is_dir()` probe with `validated_browse_directory` (via the shared start-directory chain) and the unconditional worker write with claim-on-loop + `remember_browse_directory`.
4. Keep `_remember_library_ingest_location`'s name and its direct unit-test callers; keep typed-path-fragment behaviour untouched.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
A two-line reuse, as the rider said it would be. The read goes through
`browse_start_directory` (which runs `validated_browse_directory` on the
remembered value, then on `[notes] sync_directory`, then falls back to home),
and the write claims its generation on the event loop before dispatching the
worker, so the newest selection is the one that survives.

`_remember_library_ingest_location` keeps its name and its no-app,
direct-call unit tests; `generation` is optional there and claimed on the
spot when absent. The `@work(thread=True)` wrapper and its broad guard are
gone -- `remember_browse_directory` already owns both.

One existing test had to move its monkeypatch from
`library_screen.save_setting_to_cli_config` to the shared module's, which is
where the write now happens.

Files: `UI/Screens/library_screen.py`, `Library/library_browse_location.py`,
`Tests/UI/{test_library_screen,test_library_notes_wave_import_ux}.py`,
`Docs/User_Guide/library/import-and-export.md`.
<!-- SECTION:NOTES:END -->
