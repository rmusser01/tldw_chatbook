---
id: TASK-3022
title: 'Ambient Library test debt: fixture bypass and two standing failures'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 12:20'
updated_date: '2026-08-07 20:25'
labels:
  - tests
  - library
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ambient test debt on dev, repeatedly re-confirmed (A/B against clean HEAD) by every task of the
`fix/library-uat-p1s` arc — none of it caused by that branch:

1. **Fixture-bypass cluster (~7 failures)**: several test fixtures construct `LibraryScreen`
   bypassing `__init__`, so `_library_ingest_preflight_generation` is never set and the tests fail
   on attribute access. Traced by git archaeology to `6fdde2e68` (2026-08-02, task-2011 generation
   stamp) — the fixtures predate the attribute. Fix the fixtures (construct properly or set the
   attribute), not the production code.
2. `Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story`
   fails on unmodified dev — adjacent to task-2520 (landing footer keyboard story) and task-2860
   (F6 hint stripped by `_RESERVED_GLOBAL_KEYS` in `AppFooterStatus.py`); fixing those may fix
   this. Coordinate rather than patch the assertion.
3. `test_shared_form_and_native_inputs_use_thin_non_semantic_focus` (`_forms.tcss`-adjacent) fails
   on unmodified dev.

4. Newly confirmed ambient during the P2-batch arc (A/B'd on clean dev `6b38a13b8`):
   `test_library_shell_note_save_result_after_switch_is_discarded` and
   `test_library_shell_note_conflict_reload_discards_local_edits` in
   `Tests/UI/test_library_shell.py` (the latter is the long-documented order-dependent
   notes tail). Fold into the same fixture/ordering repair pass.

A green run of the Library-adjacent suites currently requires knowing which failures are ambient;
that knowledge should live in fixed tests, not session notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fixture-bypass cluster is fixed at the fixture level; the ~7 affected tests pass on dev
- [x] #2 The two named standing failures either pass or are traced to their owning open tasks with a note in this task
- [x] #3 Targeted Library suites run green on dev with no known-ambient exclusion list
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Housekeeping: fix task-2860's Implementation Notes to reference AC items by name, not stale pre-renumbering numbers (done first, separately).
2. Fixture-bypass cluster: locate every LibraryScreen construction that bypasses __init__ (object.__new__) within the exit-bar suites; fix at the fixture level (set the missing _library_ingest_preflight_generation attribute), never touch production code.
3. Landing-footer test: consume task-2860's recorded before/after delta and update the literal-string assertion to the contract-correct (always-present-globals, ADR-031) expectation -- the sanctioned coordination point.
4. Focus/forms test: trace test_shared_form_and_native_inputs_use_thin_non_semantic_focus; grep backlog for an owning open task; if the underlying CSS behavior is intentional (already landed, reviewed), fix the stale assertion (test-only); otherwise document the trace.
5. Order-dependent notes-tail tests: attempt to characterize/fix the ordering dependency for test_library_shell_note_save_result_after_switch_is_discarded and test_library_shell_note_conflict_reload_discards_local_edits at the fixture/isolation level; if not tractable, document the precise trigger.
6. Run the full targeted sweep (Tests/Library, Tests/UI/test_library_screen.py, Tests/UI/test_library_shell.py, Tests/Widgets/Library) and paste honest before/after counts with no known-ambient exclusion list.
7. Backlog hygiene + commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Housekeeping (done first): fixed task-2860's Implementation Notes, which referenced ACs
by stale pre-renumbering numbers ("AC#1 and AC#3: met. AC#2 ... NOT met" and "Audit
(AC#3)") that no longer matched that file's final AC checklist. Rewrote those references
by NAME so they cannot rot again.

BEFORE (baseline sweep, this branch's HEAD after task-2860, foreground, real counts):
Tests/Library + Tests/UI/test_library_screen.py + Tests/UI/test_library_shell.py +
Tests/Widgets/Library: 7 failed, 1512 passed.
  1. Tests/UI/test_library_screen.py::test_do_submit_ingest_persists_options
  2. Tests/UI/test_library_screen.py::test_faster_whisper_recovery_handler_uses_explicit_provider
  3. Tests/UI/test_library_screen.py::test_switch_is_not_offered_when_the_server_seam_cannot_submit
  4. Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story
  5. Tests/UI/test_library_shell.py::test_library_shell_note_save_result_after_switch_is_discarded
  6. Tests/UI/test_library_shell.py::test_library_shell_note_conflict_reload_discards_local_edits
  7. Tests/UI/test_library_shell.py::test_library_shell_notes_sync_now_calls_recording_service_with_chosen_enums
    (a THIRD "order-dependent notes-tail" test, not named in this task's original
    description, empirically found by this task's own sweep -- see item 4 below)

Item 1 (fixture-bypass cluster): empirically only 3 tests, not ~7 -- the original estimate
counted every one of `_minimal_ingest_screen()`'s 23 call sites in Tests/UI/
test_library_screen.py as at-risk; only 3 actually exercise a code path touching an
attribute `__init__` sets that the bypass skipped. Root cause was broader than the single
named attribute (`_library_ingest_preflight_generation`): different call paths also hit
`_library_ingest_clear_finished_armed` and `_library_selected_row_id`, both also `__init__`-
only. Rather than whack-a-mole more missing attributes, fixed at the fixture level per the
task's own "construct properly" option: `_minimal_ingest_screen()` now calls
`LibraryScreen(MagicMock())` for real instead of `object.__new__(LibraryScreen)`.
`LibraryScreen.__init__` is pure attribute setup (no I/O, no compose(), no worker starts),
so this is both cheap (25 tests in test_library_screen.py: 4.7s) and immune to whatever
future attribute `__init__` adds. Every test still replaces `app_instance` with its own
MagicMock afterward, unchanged. No production code touched. The other three
`object.__new__(LibraryScreen)` sites this task's own grep found (test_library_ingest_
guardrail_modal.py, test_parakeet_v2_install_ui.py, Tests/App/test_submit_library_ingest_
job.py) are outside this task's exit-bar suites and were not observed failing -- not
touched, per the task's explicit AC scope.

Item 2 (landing-footer test): consumed task-2860's recorded before/after delta verbatim.
Updated the assertion to the contract-correct expectation ("... F6 next pane | F1 help ·
Ctrl+P palette · Ctrl+Q quit" -- the always-present-globals suffix ADR-031 and ~10 other
tests already enforce), matching the identical, already-updated pin in
Tests/UI/test_screen_footer_hints.py (task-2860's own file). This is the sanctioned
"coordinate rather than patch" moment task-3022's own original description called for.

Item 3 (test_shared_form_and_native_inputs_use_thin_non_semantic_focus, Tests/UI/
test_non_obscuring_focus_contract.py -- outside this task's exit-bar suites but named in
the debt inventory): traced, not merely patched. `Select:focus` in tldw_chatbook/css/
components/_forms.tcss carries NO border declaration by design -- task-2300 (Done,
"Watchlists Selects render empty option lists") went through two rounds on this exact
rule: f86151cdc first re-homed a border onto `Select:focus > SelectCurrent`, then
6385771b2 REMOVED that rule again (its specificity outranked `.settings-compact-select
> SelectCurrent`'s deliberate `border: none`), landing on today's final, reviewed,
documented shape (only background/color recolouring on `Select:focus`; the border is
Textual's own unmodified `&:focus > SelectCurrent` default). The test predated that final
shape and still asserted a border presence never true after 6385771b2. Fixed the
assertion (test-only) to match the current, intentional, already-landed contract; left
the Input/TextArea/.form-input/.form-textarea assertions in the same test untouched (they
were already correct). File's full suite: 97 passed.

Item 4 (order-dependent notes-tail, three tests): NOT actually order-dependent. All three
failed in isolation, run alone, repeatedly (3/3, 3/3, 2/3 samples) -- real cross-test
pollution would present intermittently depending on run order, not reproduce standalone.
Root cause: the identical "state-then-DOM race" shape task-699 (2026-07-26) already
diagnosed and fixed ONE instance of in this same file -- a poll loop watches a plain/
reactive Python attribute (`_library_note_detail`, `_library_notes_view` +
`_library_note_autosave_state`, `_library_notes_sync_running`), the loop exits the instant
the attribute flips, and the very next line does a ONE-SHOT `screen.query_one(...)` on a
widget the SAME state transition is supposed to (re)mount -- one event-loop tick before
Textual's recompose has actually run. These three are new instances of that exact bug
shape, introduced by later test additions (P2-batch arc) that never saw task-699's
diagnosis. Fixed all three the same way task-699 did: replaced the bare `query_one` with
`_wait_for_selector` (this file's existing wall-clock-bounded helper -- polls via
`screen.query`, a list, so a transiently-absent widget is just "not yet", never an
exception) immediately before the read.
  - test_library_shell_note_save_result_after_switch_is_discarded: waits for
    `#library-note-meta` (both reads) instead of a bare query_one.
  - test_library_shell_note_conflict_reload_discards_local_edits: waits for
    `#library-note-body` before reading `.text`/title.
  - test_library_shell_notes_sync_now_calls_recording_service_with_chosen_enums: waits for
    `#library-notes-sync-status` after run-completion instead of a bare query_one.
Verified each 3-5x standalone post-fix: 100% green (was 3/3, 3/3, 2/3 failing pre-fix).
Recorded as a lesson (backlog/docs/lessons-testing-evidence.md, new entry) since the
backlog itself mislabeled this "order-dependent" -- worth flagging that the label was a
hypothesis, never actually verified, and a test failing alone even once already disproves
it.

AFTER (full foreground sweep, split into two Bash calls per timeout headroom):
Tests/UI/test_library_screen.py + Tests/UI/test_library_shell.py: 391 passed, 0 failed
(589.88s). Tests/Library + Tests/Widgets/Library: 1128 passed, 0 failed (138.04s).
Combined: 1519 passed, 0 failed -- exactly the 1512+7=1519 collected before, now all
green, no known-ambient exclusion list required. Tests/Library --collect-only -q: 1110
collected, 0 errors (matches baseline).

Files changed: Tests/UI/test_library_screen.py (_minimal_ingest_screen fixture),
Tests/UI/test_library_shell.py (landing-footer assertion + 3 state-then-DOM-race fixes),
Tests/UI/test_non_obscuring_focus_contract.py (Select:focus assertion),
backlog/docs/lessons-testing-evidence.md (new entry),
backlog/tasks/task-2860 (Notes housekeeping), backlog/tasks/task-3022 (this file).
No production code touched anywhere in this task.
<!-- SECTION:NOTES:END -->
