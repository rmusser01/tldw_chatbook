---
id: TASK-32174
title: 'Library Notes: vendored pickers do not remember a last-used start directory'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-09 09:12'
updated_date: '2026-09-09 17:39'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - pickers
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32122. task-32122 AC#4 fixed the vendored folder
pickers to resolve a typed-but-unsubmitted **Folder path** before Select,
but left the "remember where I was" gap partially open: Library ingest
already does this caller-side, via `_library_ingest_browse_location`, but
Notes' three folder-picker call sites — Import once, Keep a folder synced,
and Folder files — still always open at the same default location instead
of the directory last browsed in that context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import once's folder picker reopens at its own last-used directory,
  falling back to home when none is recorded yet
- [x] #2 Keep a folder synced's picker does the same, independently
- [x] #3 Folder files' picker does the same, independently
- [x] #4 Tests cover all three
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the ingest precedent (`_library_ingest_browse_location`/`_remember_library_ingest_location` in library_screen.py) and the three call sites: `_push_library_note_import_picker` (library_screen.py), the FolderRequested handler for Keep a folder synced (library_notes_controller.py), and `_open_root_picker`/`_root_selected` for Folder files (library_file_notes_workspace.py).
2. Add one browse-location resolver + one persist method per picker, each keyed to its own config section (library.notes_import, library.notes_sync, file_notes.browse), mirroring ingest's "remembered dir, else home" shape -- caller-side only, vendored pickers untouched.
3. Wire `location=` into each FileOpen/SelectDirectory call and persist on a successful pick, off the event loop (matching ingest's docstring rationale).
4. TDD: write failing tests per picker (in test_library_notes_wave_import_ux.py and test_library_notes_wave_file_notes.py), confirm red by reverting the prod diff, then green.
5. Live-verify all three pickers in the running app; update notes.md and file-notes.md verified-against stamps; backlog hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Caller-side last-used start directory for the three vendored Notes pickers, mirroring the ingest precedent (_library_ingest_browse_location/_remember_library_ingest_location); no changes to Third_Party/textual_fspicker or enhanced_file_picker.py.

- Import once (library_screen.py _push_library_note_import_picker): new _library_note_import_browse_location()/_remember_library_note_import_location(), persisted off the loop via @work(thread=True) _persist_library_note_import_location, key [library.notes_import] last_directory.
- Keep a folder synced (library_notes_controller.py FolderRequested handler): new _library_notes_sync_browse_location()/_remember_library_notes_sync_location(), persisted via self.run_worker(thread=True), key [library.notes_sync] last_directory.
- Folder files (library_file_notes_workspace.py _open_root_picker/_root_selected): new _file_notes_browse_location(), used only when no root is linked (a linked root still wins, unchanged); persisted via self.run_worker(thread=True), key [file_notes] browse.
- All three: remembered dir else home; a remembered dir that no longer exists falls back to home (matches ingest).

Tests: 8 new tests across Tests/UI/test_library_notes_wave_import_ux.py and Tests/UI/test_library_notes_wave_file_notes.py (resolver/persist unit tests via LibraryScreen(MagicMock())/its _notes_controller, plus two live-widget tests via _WorkspaceHarness for Folder files); confirmed red against the pre-change code (git checkout + rerun), confirmed green after. No changes to Tests/UI/test_file_picker_start_dir.py -- that file covers the unrelated EnhancedFileOpen wrapper's own built-in per-context memory, not these three call sites' plain vendored FileOpen/SelectDirectory.

Full test_library_file_notes_workspace.py run: 152 passed, 8 pre-existing failures verified identical against baseline d0ff40842f (contrast/focus tests, unrelated). test_library_screen.py, test_library_note_import_flow.py, and the notes/ingest Architecture wiring census tests (hardcoded method-name lists) all pass unchanged.

Live-verified in the running app (nw2-r-pickers, power profile): Import once opened at home on first use, remembered inbox/ after picking a file there and reopened there; Keep a folder synced opened at home, remembered vault/ after Ctrl+S selecting it and reopened there; Folder files opened at home with no root linked, and after linking vault/ persisted both file_notes.root and file_notes.browse.

Docs: Docs/User_Guide/library/notes.md and file-notes.md each got a new trailing Verified-against stamp naming the three config keys.
Review round (PR #2554, Qodo 5 findings -- 4 fixed, 1 partly rebutted):

The three copied resolver/persist bodies collapsed into one shared module, tldw_chatbook/Library/library_browse_location.py (validated_browse_directory / claim_browse_directory / remember_browse_directory), which is where findings 1, 3, 4 and 5 are answered at once. Each caller keeps its own get_cli_setting read (so the per-module monkeypatch seams stay) and its own home fallback; everything after that is shared.

1. Malformed remembered paths went unvalidated (Security). Real: the value is persisted user state read back from config.toml, and all three sites did Path(...).expanduser().is_dir() with no central validation -- a relative value resolved against the process cwd would have reached the picker. Now every read goes through validate_existing_absolute_directory (Utils/path_validation.py); anything relative, traversing, null-byte-bearing, non-existent or not a directory falls back to home. NOT extended to the older _library_ingest_browse_location, whose remembered branch has the same shape: its existing test monkeypatches library_screen.get_cli_setting and it is outside this PR -- called out here rather than silently changed.
2. Picker wiring could regress unseen (Testability). Real: every test called the private resolver/persist helpers, so deleting `location=` from the FileOpen call would have left them all green. The tests now drive _push_library_note_import_picker, handle_library_notes_lasting_folder_requested and _open_root_picker, inspect the pushed FileOpen/SelectDirectory instance's own location, and complete a selection through its registered callback. Verified by removing both the `location=` kwarg and the persistence call from the production code: 3 of the 4 Import once tests fail.
3. + 5. Save failures had no context, and the Folder files worker swallowed everything with a bare `except Exception: pass`. Both real. One handler now logs with logger.exception (type + traceback) naming the [section].key, and the documented False return of save_setting_to_cli_config is checked and reported instead of discarded.
4. An older selection could overwrite a newer one. Real but tiny in practice (two picks would have to land inside one config write). Note that exclusive=True would NOT have fixed it: cancelling a Textual thread worker does not stop the running thread. Instead each selection claims a generation on the event loop, and the worker re-checks it inside the same lock that serialises the write -- a superseded write is dropped, not reordered.

Tests: new Tests/Library/test_library_browse_location.py (14) for the shared validation/ordering/failure-logging contracts; Tests/UI/test_library_notes_wave_import_ux.py rewritten to the picker boundary (20 in file); Tests/UI/test_library_notes_wave_file_notes.py gains a SelectDirectory-instance test and a refused-value test (10 in file).

<!-- SECTION:NOTES:END -->
