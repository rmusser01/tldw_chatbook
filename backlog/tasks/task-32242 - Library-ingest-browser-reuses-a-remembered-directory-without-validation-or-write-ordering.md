---
id: TASK-32242
title: >-
  Library ingest browser reuses a remembered directory without validation or
  write ordering
status: To Do
assignee: []
created_date: '2026-09-10 10:15'
updated_date: '2026-09-10 10:15'
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
- [ ] #1 A relative, traversing or vanished `[library.ingest] last_directory`
  never reaches the ingest `FileOpen`; the browser opens at the user's home
  directory instead
- [ ] #2 Of two ingest selections made before the first config write finishes,
  the later one is the directory the browser reopens at
- [ ] #3 Both are pinned by tests that fail without the fix, and the ingest
  browser's typed-path-fragment behaviour is unchanged
<!-- AC:END -->
