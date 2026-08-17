---
id: TASK-15767
title: File Notes back-navigation broke when the destructive-reload confirm shipped
status: Done
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - notes
  - filesystem
  - regression
priority: high
---

## Description

`Tests/UI/test_screen_navigation.py::
test_action_library_notes_files_back_returns_to_database` is red on dev.
Confirmed live: `AttributeError: 'WorkspaceProbe' object has no attribute
'cancel_reload_confirmation'`.

Controller bisect (per the reviewing session's finding): first-bad commit is
`062c3ee30` ("fix: confirm destructive File Notes reload"), task-15503's PR
which added the retained inline confirmation before a conflict/error reload
can replace the File Notes draft (see task-15503's Implementation Notes —
the new opener, "Discard draft and reload", and its Cancel/Confirm/Escape
flow). That PR's own regression matrix passed 17 tests at the time, so this
is a drift introduced afterward or missed by that matrix's scope, not a
defect in the confirmation feature itself.

Two things need reconciling:
1. `test_action_library_notes_files_back_returns_to_database`'s own test
   double (`WorkspaceProbe`, defined multiple times in
   `test_screen_navigation.py`) needs a `cancel_reload_confirmation` method to
   match whatever the production back-navigation path now calls on it.
2. The production back-navigation path itself needs auditing against
   task-15503's confirmation state: if a user presses "back" while the new
   confirmation dialog is open (or in a conflict/error state that the
   confirmation now gates), the current behavior needs to be intentional, not
   an unhandled attribute lookup.

## Acceptance Criteria

- [x] `test_action_library_notes_files_back_returns_to_database` passes on
      dev without weakening what it originally asserted (back navigation from
      File Notes returns to the database view)
- [x] Every `WorkspaceProbe` double in `test_screen_navigation.py` that the
      back-navigation path can reach implements whatever contract production
      now expects, verified against the real
      `library_file_notes_workspace.py` shape from task-15503 (not a stub
      that merely silences the error)
- [x] Pressing "back" while File Notes is mid-confirmation (or in the
      conflict/error state task-15503 added) has explicit, tested behavior —
      not a crash
- [x] `Tests/UI/test_library_file_notes_workspace.py` (task-15503's own
      regression matrix) stays green

## Implementation Plan

1. Re-verify the named failure at HEAD (`8727a2861`) before touching anything —
   dev commit `e917b9076` (2026-08-14, "test: reconcile phase release harness
   contracts") already added `cancel_reload_confirmation -> False` to the one
   `WorkspaceProbe` the back action reaches, so the named test may already be
   green (AC1 partially dissolved by drift-forward).
2. Reproduce the original break born-red anyway: temporarily remove the probe
   method (exact pre-`e917b9076` shape), run the named test, capture the
   `AttributeError: 'WorkspaceProbe' object has no attribute
   'cancel_reload_confirmation'`, restore via Edit.
3. AC2: pin the production-side workspace contract structurally — a drift-guard
   test that AST-scans the Files-mode leave seams in `library_screen.py`
   (`action_library_notes_files_back`, `_return_to_library_database_notes`,
   `_flush_active_file_notes`, `_acquire_file_notes_transition`) for attributes
   accessed on the workspace, asserts the set is exactly
   {`flush_pending_work`, `acquire_transition`, `cancel_reload_confirmation`},
   and asserts each exists on the REAL `LibraryFileNotesWorkspace` with matching
   async-ness. Future contract widening then fails THIS test with a message
   naming the probes to update, instead of an opaque AttributeError.
4. AC3: add an explicit behavior pin — back pressed while a reload confirmation
   is open cancels the confirmation and stays on Files (no flush, no
   transition, no recompose; footer re-registered); the SECOND back navigates
   to Database. Mutation-test it by temporarily reverting production's
   cancel-first branch (test must go red), then restore via Edit.
5. AC4: run the full task-15503 matrix
   (`Tests/UI/test_library_file_notes_workspace.py`) plus the full
   `Tests/UI/test_screen_navigation.py`; ruff check + format on touched files.

## Implementation Notes

**Regression mechanism.** `062c3ee30` (task-15503) widened the workspace
contract that `action_library_notes_files_back` calls: it now calls
`workspace.cancel_reload_confirmation()` (cancel-first, stay on Files) before
the shared guarded return. The nav suite's `WorkspaceProbe` double stayed on
the old two-method shape (`flush_pending_work` + `acquire_transition`), so the
back action crashed with `AttributeError ... 'cancel_reload_confirmation'` at
`library_screen.py:14853` before ever reaching
`_return_to_library_database_notes`. The break lived at the harness seam —
the real widget grew the method in the same commit, so live back-navigation
never crashed; 15503's own matrix passed because it never runs the nav suite.

**State found at HEAD (`8727a2861`).** The named test was ALREADY green:
dev commit `e917b9076` (2026-08-14, "test: reconcile phase release harness
contracts") had added `cancel_reload_confirmation -> False` to that one probe.
Re-verified the mechanism born-red anyway by removing the probe method (exact
pre-`e917b9076` shape) — the named test failed with the task's exact
AttributeError at `library_screen.py:14853` — then restored via Edit.

**What this task adds** (production untouched — its behavior was already
intentional; the gap was coverage):

- `test_action_library_notes_files_back_cancels_open_reload_confirmation_first`
  (AC3): back pressed mid-confirmation cancels the pending decision and stays
  on Files — the probe's `flush_pending_work`/`acquire_transition` assert they
  are NOT reached while the confirmation is open — and the second back takes
  the normal guarded return to Database. Mutation-verified: temporarily
  deleting production's cancel-first branch turned the test red.
- `test_files_back_navigation_workspace_contract_matches_real_workspace`
  (AC2): AST-scans the four Files-mode back seams
  (`action_library_notes_files_back`, `_return_to_library_database_notes`,
  `_flush_active_file_notes`, `_acquire_file_notes_transition`) for attributes
  accessed on the workspace, pins the set to exactly
  {`flush_pending_work`, `acquire_transition`, `cancel_reload_confirmation`},
  and verifies each name exists on the REAL `LibraryFileNotesWorkspace` with
  the async-ness the screen assumes (plus `reload_confirmation_active` as a
  property for the footer chooser). Mutation-verified both directions: a
  widened contract (a bogus `workspace.dismiss_new_guard_probe()` call — the
  original regression's failure mode) and a narrowed one each fail THIS test
  with a message naming the probes to update, instead of an opaque
  AttributeError deep in an unrelated test.
- Annotated the existing probe's `cancel_reload_confirmation -> False` as
  faithful to the real widget for its state (no confirmation pending ->
  `_dismiss_reload_confirmation`'s None guard returns False), so it is not a
  bare error-silencer.

**Evidence.** Named test: 1 passed at HEAD; born-red AttributeError reproduced
and restored. Full `Tests/UI/test_screen_navigation.py`: 129 passed. Full
`Tests/UI/test_library_file_notes_workspace.py` (15503 matrix): 88 passed.
ruff check clean; ruff format diff on the file is dev's pre-existing 61 lines
(baseline-verified against `HEAD`'s copy) — my additions are format-clean;
did not reformat unrelated pre-existing code (smallest honest diff).

**Files.** `Tests/UI/test_screen_navigation.py` (only code change; two new
tests + one comment), this task file.
