---
id: TASK-14823
title: Gate Start on a selection with nothing importable
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-10 21:41'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique. Staging a folder containing no importable files leaves Start ENABLED with an empty gate line, and pressing it manufactures a permanent failure receipt the preflight had already diagnosed.

Observed live: an empty directory produced `0 files` in the preflight summary, `0 will import` in the commit line, an EMPTY start-quiet-line, and an enabled `#library-ingest-start`. Pressing it produced `✗ failed · emptydir · No files to import were found in this folder.` plus a toast `Import finished — 1 failed`, permanently moving the queue tally and polluting Recent imports with a failure that was predictable before the click.

DESIGN.md's dense-form convention says an inert action carries its reason at the control. Here the action is live and the reason arrives as a failure receipt. The surface already has the right pattern one branch away: the not-found case gates Start with a named reason and offers `Choose a file…`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A staged selection with nothing importable gates Start with a stated reason instead of allowing a doomed run
- [x] #2 The gate distinguishes an empty folder from a folder whose files are all unsupported, since the recovery differs
- [x] #3 No failure receipt is created for a selection the preflight already knew could not import anything
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `library_ingest_state.py`: `nothing_importable` currently requires `total_files > 0`, so a folder with NO files sailed through the gate. Extend it to the zero-file selection and word the gate line by KIND -- an empty folder ('This folder is empty') vs a folder whose files are all unsupported/empty (existing 'Nothing in this selection can be imported -- N unsupported files') -- because the recovery differs.
2. `library_screen.py`: `_submit_library_ingest_form` refuses to submit while the gate is closed (quiet warning carrying the gate's own reason), so no entry point can manufacture the receipt the pre-flight already ruled out.
3. Tests: the empty-folder gate + its distinct reason, the all-unsupported reason still intact, and an end-to-end assertion that pressing Start on an empty folder creates NO job at all.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gated the one selection the gate let through, and enforced it at the submit seam.

**The hole** — `nothing_importable` required `total_files > 0`, so a folder holding NO files failed every clause: Start stayed enabled, the gate line was empty, and the press manufactured '✗ failed · emptydir · No files to import were found in this folder.' plus a toast, permanently moving the queue tally and polluting Recent imports.

**Gate + distinct reasons (AC#1/#2)** — an `empty_selection` term (pre-flight present, no errors, `total_files == 0`) now feeds `nothing_importable`, and it gets its OWN sentence because the recovery differs: `INGEST_EMPTY_SELECTION_COPY` = 'This folder is empty — there's nothing to import. Choose a folder with files, or a single file.' (mirroring the not-found branch's shape: named reason at the control plus the way out), while an all-unsupported folder keeps 'Nothing in this selection can be imported — N unsupported files.'

**No receipt (AC#3)** — the state exposes `selection_has_nothing_importable`, and `_submit_library_ingest_form` refuses on it with the gate's own reason, so no entry point (Start button, Enter in the path field, an accelerator, a future caller) can route around a disabled button. Deliberately NOT gated on `not start_enabled`: that is also False for transient/environmental blockers (blank path, missing media DB) where a submit is premature rather than doomed, and those keep their existing explanatory receipts.

Verified end-to-end: `Tests/integration/test_library_ingest_flow.py::test_empty_folder_creates_no_job_at_all` stages a real empty directory, runs the real pre-flight, and asserts the registry holds NO job after a submit. Mutating the guard to a no-op puts the manufactured job straight back.

Modified: `Library/library_ingest_state.py`, `UI/Screens/library_screen.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/integration/test_library_ingest_flow.py`, `Docs/User_Guide/library/import-and-export.md`.
**xhigh review round (2026-08-10)** — the gate was right; one of its two sentences was not, and this task's own hard block is what made that expensive.

`INGEST_EMPTY_SELECTION_COPY` ("This folder is empty…") fired for ANY folder the pre-flight measured at `total_files == 0` — including every folder whose entries `_collect_files` deliberately passes over: symlinks, dot-entries, unreadable subfolders. A folder of symlinked media was therefore told it was empty AND, since this task added the submit-side gate, refused outright: a wrong diagnosis turned into a dead end with no way forward stated.

The pre-flight is the only layer that can tell the two apart, so `_collect_files` now returns how many entries it skipped and `PreflightResult` carries `skipped_entries`. `total_files == 0 and skipped_entries` gets its own sentence and its own recovery — "Nothing in this folder could be scanned — N entries were skipped: folder imports pass over hidden files, links, and folders they can't read. Import a file directly, or choose another folder." — while a genuinely empty folder keeps AC#2's original wording.

The BLOCK itself is correct and stays: `submit_library_ingest_job` expands a folder with `collect_directory_files`, the public seam over the very same `_collect_files`, so the pipeline would collect nothing and manufacture exactly the "✗ failed · <folder>" receipt this task exists to prevent. The test asserts that equivalence directly (it calls `collect_directory_files` on the fixture and asserts it returns no files) rather than assuming it — the gate is only allowed to refuse what the pipeline genuinely cannot process.

Mutation-checked: forcing the empty sentence back for all zero-file folders turns the new test red. Modified: `Library/ingest_preflight.py`, `Library/ingest_types.py`, `Library/library_ingest_state.py`, `Tests/Library/test_ingest_preflight.py`, `Tests/Library/test_library_ingest_state.py`, `Docs/User_Guide/library/import-and-export.md`.
<!-- SECTION:NOTES:END -->
