# Library Notes Wide Browse Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Do not
> delegate in this session unless the user explicitly requests subagents.

**Goal:** Keep the Library rail available while browsing Notes on wide
terminals, give note editing and folder-backed Files the full workbench width,
and return to the exact prior Notes browse context.

**Architecture:** `LibraryScreen` derives a focused-task presentation from the
existing Notes source and view; it does not add another route or responsive
mode. One immutable browse-return receipt captures the existing
`LibraryNotesFocusIdentity` plus the rail scroll position before entering an
editor or Files. The new persistent `Library / Notes` return cue delegates to
the current guarded editor/File exit, while the existing compact one-stage
presentation remains authoritative below the 120-column breakpoint.

**Tech Stack:** Python 3.11+, Textual 8.x, existing Library screen/Notes canvas
and File Notes workspace, immutable dataclasses, pytest Pilot/compositor tests.

---

## ADR check

ADR required: no

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

Reason: ADR-076 already approves this responsive Notes structure and retains
the existing source, dirty, sync, conflict, and focus owners. This task changes
presentation and return continuity only; it introduces no storage, service,
sync-policy, or cross-screen contract.

## Spatial thesis

- Primary path: wide Notes users scan in the navigator with Library navigation
  visible, then temporarily give the whole workbench to the selected note or
  Files task.
- Supporting navigation: the Library rail remains mounted but hidden during a
  focused task so its collapse and scroll state survive without becoming task
  chrome.
- Return ownership: one top-level `‹ Library / Notes` action names where Back
  lands and uses the same guarded exit as Escape; inner compact Back controls
  remain the compact navigation owner.
- Adaptation: at 120 columns and wider, list = rail + navigator and task = cue +
  full-width canvas. Below 120, the current mutually exclusive rail/canvas
  stages, source strip, and in-canvas Back controls remain unchanged.

## File ownership

- `tldw_chatbook/UI/Screens/library_screen.py`: immutable browse receipt,
  focused-task presentation predicate, cue composition/handler, rail/canvas
  visibility, browse-origin capture, and guarded return restoration.
- `Tests/UI/test_library_shell.py`: Database Notes wide browse/editor geometry,
  exact context restoration, compact/wide resize continuity, guard authority,
  focus veto, and compositor evidence.
- `Tests/UI/test_library_file_notes_workspace.py`: Files focused-task geometry,
  persistent cue, retained workspace/draft across breakpoints, guarded return,
  and exact Database browse restoration.
- `Docs/User_Guide/library/notes.md`: ASCII-only wide browse/focused-task and
  compact behavior.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` and generated bundle:
  touch only if production geometry proves existing source-strip and workbench
  rules cannot contain the cue.

## Task 1: Pin the derived wide presentation

- [ ] Add RED mounted tests at 170x48 proving Database Notes list keeps the
  expanded rail, while opening a note hides both rail and rail handle, displays
  one painted `‹ Library / Notes` cue, hides the redundant wide in-canvas Back,
  and lets the canvas consume the workbench width.
- [ ] Add the smallest `_library_notes_focused_task_active()` predicate: true
  only for Database editor/loading/context/preview or active Files, never for
  the Database navigator/create/sync routes.
- [ ] Extend the existing stage-visibility choke point so a wide focused task
  behaves as a presentation-only single stage. Do not mutate
  `_library_rail_collapsed`, `_library_notes_stage`, the selected Library row,
  or any persisted preference.
- [ ] Compose one stable cue in the existing Notes source strip. On wide focused
  tasks, show the cue and hide Database/Files source choices; on wide browse or
  compact Notes, hide the cue and show the existing choices. Keep the compact
  in-canvas Back control; suppress only the duplicate wide editor Back.
- [ ] Run the two new nodes and confirm GREEN without CSS changes. If geometry
  fails, add only source-strip containment rules to the component stylesheet,
  regenerate the bundle, and rerun exact CSS parity.

## Task 2: Capture and restore the Database browse receipt

- [ ] Add RED tests with enough Notes and rail rows to produce real scrolling.
  Set a non-default filter/sort and placement selection, scroll the Notes list
  and rail, enter the note editor, then use the cue. Assert the exact filter,
  sort, placement/note identity, list scroll, rail scroll/collapse state, and
  semantic row focus return.
- [ ] Add one frozen `_LibraryNotesBrowseReturnReceipt` containing the existing
  `LibraryNotesFocusIdentity` and rail scroll offset. Capture it before an
  admitted note-row open or Database-to-Files transition. Preserve the prior
  receipt when switching from an already-open editor into Files.
- [ ] Reuse the receipt in `_exit_library_note_editor_guarded()` after its
  existing flush outcome permits exit. Keep the existing fallback identity for
  deep links/restored editors that have no browse receipt.
- [ ] Restore rail and navigator scroll only after the replacement canvas is
  mounted and only through the existing focus-generation authority. A newer
  user focus/scroll intent must veto the deferred restore.
- [ ] Prove the required inverse: replace the stored identity with a generic
  filter/first-row target; the exact selected-row/scroll test must fail. Restore
  immediately.

## Task 3: Route Files through the same truthful Back contract

- [ ] Amend the existing production-shell breakpoint test: at 120x40 and
  160x45 the Files workspace is retained and full-width with the rail hidden;
  at 40x20 it remains the current compact Notes canvas stage. At every size the
  same workspace/editor/draft/focus objects survive.
- [ ] Add a mounted RED return test starting from a scrolled, filtered Database
  Notes browse. Enter Files, edit a file, press the new cue, and prove the
  existing file flush/leave guard runs before Database Notes returns to the
  captured browse receipt.
- [ ] Route the cue to `_return_to_library_database_notes()` for Files and to
  `_exit_library_note_editor_guarded()` for Database editor. Do not implement a
  parallel flush, conflict, reload-confirmation, mutation, or Escape path.
- [ ] Replace Files' generic first-list-row return focus with receipt restoration
  when a receipt exists; retain the existing first-row fallback for deep-linked
  or restored Files sessions with no receipt.
- [ ] Prove the required inverse: bypass the Files flush/leave guard from the
  cue; the dirty/reload-confirmation test must fail. Restore immediately.

## Task 4: Resize, focus, and supported-size UAT

- [ ] Add a 170x48 -> 100x30 -> 170x48 editor test with the exact
  `TldwCli.CSS_PATH` production hierarchy. Assert the same coordinator,
  TextArea, draft, selection, scroll, selected note, and browse receipt survive;
  wide uses the top cue/full width and compact uses the existing in-canvas Back
  and one-stage canvas.
- [ ] Add the same breakpoint traversal for Files, preserving the same
  workspace, editor, path, text, conflict/dirty state, and focused control.
- [ ] Move focus after an admitted return but before deferred restoration and
  prove the newer live focus wins. Mutation-test by removing the existing
  focus-generation veto; restore immediately after the node fails.
- [ ] Assert the cue and active task content are compositor-painted and
  contained at 100x30 and 170x48; assert visual order and Tab order agree. Do
  not claim the cue at compact size, where the in-canvas Back owns return.
- [ ] Run only the touched Notes screen/File Notes owner selectors. Do not run
  repository-wide pytest.

## Task 5: Documentation, review, and closeout

- [ ] Update the Notes guide with ASCII diagrams for wide browse, wide focused
  task, and compact navigation-first layouts. Document that the cue and Escape
  share the same dirty/sync/conflict guard and that Back restores the prior
  Database browse context.
- [ ] Run Ruff on the exact changed Python inventory, CSS build/parity only if
  CSS changed, the Impeccable layout detector once on the final UI files, and
  `git diff --check`.
- [ ] Review the diff for duplicate navigation state, synchronous I/O, stale
  focus/scroll callbacks, accidental source resets, and copy/keyboard drift.
- [ ] Check TASK-19026 ACs, add concise implementation notes with exact test and
  inverse evidence, mark Done through Backlog CLI, and commit the exact task,
  guide, tests, and production files. Explicitly state that repository-wide
  pytest was not run.

## Required inverse checks

Apply one mutation at a time and restore it immediately:

1. Return to a generic first row instead of the captured placement/note; exact
   identity/scroll restoration fails.
2. Let the wide task follow `_library_rail_collapsed` rather than its derived
   focused-task state; focused full-width geometry fails.
3. Bypass the existing Files flush/leave guard; dirty/reload-confirmation return
   test fails.
4. Remove the focus-generation/newer-user veto; deferred return steals focus.
5. Recreate either note editor or Files workspace on breakpoint crossing;
   identity/draft/undo continuity test fails.

## Focused verification boundary

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_file_notes_workspace.py \
  -k 'library and notes and (wide or browse or focused or task or return or back or breakpoint or resize or focus or scroll or dirty or conflict or reload or geometry)'
```

No repository-wide pytest claim is permitted for this task.
