---
id: TASK-32643
title: 'Library Notes: make the folder picker vault-aware'
status: Done
assignee: []
created_date: '2026-09-15 10:35'
updated_date: '2026-09-15 19:17'
labels:
  - library
  - notes
  - critique-4
  - idea
  - picker
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 9, ACCEPTED in task-32627. The folder picker is used for
three different Notes decisions and gives the user nothing to choose with:
folders and files interleave, no folder says how many notes it holds, and a
root chosen last week must be navigated to again from scratch.

Sits with the picker work (task-32606, task-32611) rather than on its own
branch — it is the same dialog.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Folders sort first, name-ascending, in every picker that can only return a folder.
- [x] #2 Each folder shows how many notes it holds, computed without walking the whole tree on open.
- [x] #3 Recently chosen roots are offered without navigation, and choosing one is a single keystroke away.
- [x] #4 A vault is marked as such in the listing, using the same detection task-32641 surfaces.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#1 lands with task-32611's folders-first default (the same sort key; every RETURNS_A_FOLDER dialog opens on it).
2. AC#2 the cheap source: reuse the existing VISIBLE-rows-only metadata hydration worker; one os.scandir of the folder itself, depth 1, capped at NOTE_COUNT_CEILING entries READ; render through size_text, the column a directory already leaves blank, so neither row renderer changes. Off unless the caller passed a notes_context.
3. AC#3 reuse RecentLocations ([filepicker] recent_<context>) rather than a second recents format; write it from remember_browse_directory, which all three doors already call on a worker, in the config write it was already making. ctrl+r focuses the list; Enter on a root dismisses with it on a folder-returning dialog.
4. AC#4 CALL Notes.note_import_discovery.folder_is_obsidian_vault -- the body lifted out of library_notes_sync_controller._carries_obsidian_marker, which now delegates to it. No second predicate.
5. RED->GREEN pins + both-sides name-set compare.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**AC#1 shipped with task-32611** in the same branch: the new `"folders"` sort
key (folders first, name-ascending, kept on top in both directions by a second
stable sort) is the default for every `RETURNS_A_FOLDER` dialog, which is
exactly "every picker that can only return a folder" plus Import once, which
can return either. "Discovery order" stays on the menu.

**AC#2 -- which cheap source, and the bound.** Considered and rejected: a
recursive count (the brief's named P0 shape -- this programme already lost a
session to an abandoned folder scan), and counting inside `project_records`
(runs over EVERY record on every sort and every keystroke of the filter). What
shipped rides the seam that already does bounded per-row filesystem work:
`_hydrate_visible`, the worker that stats only the rows in and near the
viewport. For a folder row it now also calls `read_folder_summary`, which is
ONE `os.scandir` of that folder, depth 1, no recursion, stopping at
`NOTE_COUNT_CEILING` (500) entries **read**, not merely displayed -- pinned
with a counting `scandir` wrapper, so the ceiling is a work bound and not a
formatting rule. Past it the badge reads "500+ notes". The result is rendered
through `FileRecord.size_text`, the column a directory has always left blank,
so both the vendored row and `EnhancedFileDialog`'s responsive one show it
with no renderer change and no new column.

Two things worth knowing. `show_folder_notes` is OFF unless the caller passed
a `notes_context`: "12 notes" is a useful badge when choosing where notes live
and noise in a model-file or character-card picker, so the other ~10 callers
of these dialogs pay nothing (pinned by a negative control). And
`_wants_hydration` asks the badge question independently of `metadata_loaded`
-- a metadata sort stats every record off-loop without ever counting notes, so
keying the queue on `metadata_loaded` alone left every folder that first became
visible under "Size" or "Last modified" permanently badge-less.

**AC#3 -- one recents store, not a second one.** `_get_recent_paths` was a
stub returning `[]` (which is why ctrl+r opened a permanently empty box on
every picker in the app). It now reads `RecentLocations(context=...)` -- the
`[filepicker] recent_<context>` store the enhanced picker family has always
used, with its own dedupe, newest-first ordering and trim. The write is folded
into `remember_browse_directory`, which all three doors already call on a
worker after a selection, as part of the config write it was already making:
one rewrite per pick, not two, and inside the existing lock because a list is
a read-modify-write. `picker_recent_context(section)` derives both ends from
the one string, so a caller cannot read one context and persist another, and
`recent_context` defaults to blank so a picker that never OFFERS recents (the
Library ingest browser) does not accumulate a key nothing shows. Every entry
read back goes through the same `validated_browse_directory` the start
directory does -- it is persisted user state.

"A single keystroke away" is `watch_show_recent` focusing the list (with focus
left on the path field, reaching an offered root still cost a Tab walk through
the listing -- the navigation the recents exist to replace) plus
`_on_recent_selected` DISMISSING with the root on a folder-returning dialog.
Navigating there and then requiring Select would have left it two keystrokes
away. A file picker still just moves the listing: there a recent directory is
a place to look, not a result.

**Reverted mid-task, recorded because the reasoning generalises.** I first
made ctrl+r refuse to open an empty panel. That broke
`test_fspicker_keyboard_save::test_file_picker_escape_peels_path_search_recent_in_order`
(x3), which opens all three transients on an EMPTY picker to pin the
Escape-peel order. The nicety was not in any AC and the contract is worth more,
so it is gone and the empty panel -- which predates this task -- is unchanged;
a pin now states that explicitly, including that focus must not move into an
empty list. Candidate rider, not fixed here: ctrl+r is a footer chip since
task-32606, so the empty box is more reachable than it was.

**AC#4 -- the predicate is CALLED.** `library_notes_sync_controller.
_carries_obsidian_marker` (task-32535, the detection task-32641 surfaces) was
the only callable form of "is this folder a vault"; its body moved to
`Notes.note_import_discovery.folder_is_obsidian_vault`, beside the
`OBSIDIAN_MARKER_DIRECTORY` constant it reads, and that function now delegates
to it. `read_folder_summary` calls the same function, lazily imported -- this
vendored package is deliberately absent from the app's boot import closure
(`Tests/Packaging/test_app_import_diet_closure`) and so is that module. A test
monkeypatches the shared function and asserts the LISTING's marker follows it,
which a second `.obsidian` test written inside the picker would not satisfy.

**A regression of my own, found by probe rather than by these pins.** With
the badge on, the picker opened at 100x30 with `..` highlighted in 4 runs out
of 4, against 0 of 4 without a `notes_context` -- so the first Enter inside
the listing went UP a directory. The cause is older than the badge: a
projection that started before the scan's first batch landed publishes an
EMPTY listing while `_scan_finished` has since become True,
`_settle_highlight` reads that as "empty directory", and its own first line
(`if self.highlighted is not None: return`) then makes the wrong answer
permanent. The badge's extra message-loop work only widened the window.
Guarded with the same "more is owed" fact `_settle_projection_highlight`
already consults one line above (`not self._projection_dirty`). Pinned
DETERMINISTICALLY against the production method with the five attributes it
reads, because the integration-shaped version of that test passed with the
fix reverted when run alone -- recorded in
`backlog/docs/lessons-testing-evidence.md`.

**Which extensions count as a note.** `NOTE_SUFFIXES` is deliberately local,
not a reused constant, because the three doors disagree: Folder files reads
`file_notes_service.SUPPORTED_EXTENSIONS` (adds `.text`), Keep a folder synced
reads `notes_sync_runtime._SYNC_FILE_EXTENSIONS`, Import once reads
`note_import_parsers.SUPPORTED_NOTE_EXTENSIONS` (adds `.rst`, `.json`,
`.yaml`, `.yml`, `.csv`). Verified at a prompt that `{.md, .markdown, .txt}`
is exactly their intersection, so the badge never over-promises on any door.

**RED->GREEN.** `Tests/UI/test_library_notes_w5_picker_followon.py`, 12 tests
for this task (18 in the file with task-32611's): 12 red at `origin/dev`
48d40df8ce -- `AttributeError: module '...progressive_directory_navigation' has
no attribute 'count_folder_notes'`, `ImportError: cannot import name
'folder_is_obsidian_vault'`, `TypeError: SelectDirectory.__init__() got an
unexpected keyword argument 'notes_context'` -- 12/12 green here. The negative
control (a picker with no `notes_context` drawing no badge) passes on both
sides.

**Files.** `Third_Party/textual_fspicker/parts/progressive_directory_navigation.py`,
`base_dialog.py`, `select_directory.py`, `file_open.py`, `file_dialog.py`,
`Library/library_browse_location.py`, `Notes/note_import_discovery.py`,
`UI/Library_Modules/library_notes_sync_controller.py`,
`library_notes_controller.py`, `UI/Screens/library_screen.py`,
`Widgets/Library/library_file_notes_workspace.py`,
`Tests/UI/test_library_notes_w5_picker_followon.py`. Guides and
`ENHANCEMENTS.md` §9 land in the same PR's docs commit, covering both tasks.
<!-- SECTION:NOTES:END -->
