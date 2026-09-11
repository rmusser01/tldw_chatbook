---
id: TASK-32245
title: >-
  Library Notes wizard hand-off: Write your first note lands with no focus, an
  Items pane that never loads, and an Enter that leaves for Home
status: Done
assignee: []
created_date: '2026-09-10 18:05'
updated_date: '2026-09-10 20:01'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - wizard
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Regression, new since `c4a7b1911f`: the button was added by `0b1f67aef7` (task-32140, PR #2538). It is the right idea -- a wizard that hands a first-timer a destination instead of a dashboard -- executed wrong, and it fails on the first keystroke at the single highest-intent moment in onboarding.

Reproduced by the reconciling parent on a clean fresh profile: after the click the middle ~130 columns read "Loading local Library sources..." and still did at t = 113 s (`R/caps/52`, `53`), focus is parked on nothing, the footer reads `esc back to notes`, and the first Enter navigates to Home (`R/caps/54`). The same view reached via `Ctrl+N` parks focus on `[]Blank note`, reads `enter create note`, and does not stall (`R/caps/57`). Both assessors saw it independently.

Cause PROVEN in part. The focus step exists only on the rail-press route: `library_screen.py:20561-20570` (`if row_id == LIBRARY_ROW_CREATE_NOTE and self.is_mounted: call_after_refresh(self._focus_library_note_control, "#library-notes-create-blank")`), which `action_library_notes_new` reaches through `_select_library_rail_row`; the wizard route resolves only a target row id at `library_screen.py:10042-10044`. The stall is PROVEN at `library_browse_route_swap.py:177-183`: the list child stays the "Loading local Library sources..." Static while `screen._library_loaded` is False and no lookup error is set, and this route never flips it. The exact divergence point between the two routes is INFERRED.

Why the pinning test did not catch it: `test_wizard_exit_route_notes_navigates_to_library_new_note` asserts only the `NavigateToScreen` message against a **mocked receiver**, so neither focus nor the Items pane is exercised. A replacement must drive the real route through to the mounted canvas.

Fix: route the wizard exit through the same `_select_library_rail_row` entry the `Ctrl+N` binding uses.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Arriving at Notes from the wizard's 'Write your first note' parks focus on Blank note and shows `enter create note` in the footer, exactly as `Ctrl+N` does
- [x] #2 The Items pane resolves on that route: no 'Loading local Library sources...' remains once the screen has settled
- [x] #3 The first Enter on arrival creates a note and never navigates to Home
- [x] #4 Covered by a test that drives the real wizard exit route through to the mounted Notes canvas -- not a mocked receiver -- and asserts both the focused control and the loaded list
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on `fix/library-notes-wizard-handoff-32245` (PR #2578). Both halves
traced to route divergence, and both fixed at the shared seam rather than on
the wizard's path -- every other caller of these routes was equally broken.

**Root cause 1 -- focus (`library_screen.py:20593`).** The destination row's
entry arm (`row_id == LIBRARY_ROW_CREATE_NOTE` -> `call_after_refresh(self.
_focus_library_note_control, "#library-notes-create-blank")`, plus its
Ingest/Prompt/Skill/ordinary-canvas siblings) lived INLINE in `_select_library
_rail_row_after_source_admission`, so only a rail press ran it. `apply_
navigation_context` applies the same row selection and recomposes from it but
ran no arm at all. Unfocused, `_library_focus_enter_label()` returns `""`, the
Notes footer drops to its bare `("esc", "back to notes")` tier, and Enter falls
through to the app's own binding -- Home. The task's guess (a missing
`_select_library_rail_row` call on the wizard route) was directionally right;
the actual divergence is that the rail route owns POST-selection work the deep
link never had.

Extracted verbatim to `LibraryScreen._arm_library_row_entry_focus(row_id)` and
run from both routes:
- mounted deep link: end of `_apply_navigation_context_after_source_admission`
  (`library_screen.py:10188`), after the recompose;
- pre-mount context (the wizard's case -- `handle_screen_navigation` applies
  the context to a freshly constructed screen, so `apply_navigation_context`
  takes its sync branch): parked in `_pending_library_entry_focus_row` by
  `library_navigation_controller.py:134` and consumed in `on_mount`
  (`library_screen.py:8612`), beside the note/media/collections deep-link loads
  already deferred there.

Routing the wizard through `_select_library_rail_row` itself (the task's
suggested fix) is not possible pre-mount -- it awaits a recompose on a screen
that has no canvas yet -- so the shared-arm extraction is the same fix at the
only seam both routes reach.

**Root cause 2 -- the Items pane (`library_screen.py:11417`).** The task
located the SYMPTOM at `library_browse_route_swap.py:177-183`; the cause is one
level up. `compose_content` mounts the shared "Loading local Library
sources..." placeholder for the `notes` AND `notes-create` canvas kinds (they
share one retained Items pane, `library_screen.py:13534`), but `_reconcile
_library_entry_state` matched only `LIBRARY_CANVAS_KIND_NOTES`. On
`notes-create` no branch matched, `sync_kind`/`replacement` stayed `None`, and
nothing ever replaced the placeholder -- for the life of the visit, which is
why the rail could read `Notes (0)` (snapshot landed, `_library_loaded` True)
above a pane still claiming to load. The reconciler now pairs the two kinds,
exactly as `_build_library_entry_active_child` and `canvas_sync._LIBRARY
_RESIDENT_CANVAS_OWNER_ROWS` (already `"notes" -> {BROWSE_NOTES, CREATE_NOTE}`)
do. Not wizard-specific: the Console setup card's "Write a note in Library"
action and a cold command-palette `new_note` hit the same stall.

**Tests (AC#4).** `test_wizard_exit_route_notes_navigates_to_library_new_note`
-- the mocked-receiver pin -- is REPLACED. Its message-shape assertions moved
into `_wizard_notes_handoff_navigation()`, which now runs the real
`TldwCli._handle_first_run_wizard_result` -> `EXIT_ROUTE_LIBRARY_NOTES` ->
app.py rewrite and hands the resulting `NavigateToScreen` to a real
`LibraryScreen` mounted through the production pre-mount ordering. Two new
tests in `Tests/UI/test_library_notes_wave_onboarding.py`, both proven RED
before the fix and GREEN after:
- `test_wizard_notes_handoff_parks_focus_on_blank_note` -- RED:
  `AssertionError: The wizard hand-off never focused Blank note; focus is
  'nav-home'.` (the live Enter-goes-Home symptom, in a test). Also asserts
  `("enter", "create note") in screen._library_notes_footer_shortcuts()`.
- `test_wizard_notes_handoff_items_pane_leaves_the_loading_state` -- RED:
  `#library-notes-canvas never mounted within 30.0s (1054 polls). Visible
  text: ... Loading local Library sources... ...`.
File total: 6 passed.

**Live verification** (fresh scratch profile per run, wizard walked to Summary,
"Write your first note" pressed; captures in the session scratchpad
`fix/caps/`):
- `broken-235x52-landing-BEFORE-FIX.txt` -- the defect reproduced first.
- `fixed-235x52-landing.txt` -- Items pane loaded ("No notes yet. Create your
  first note."), focus on `[]Blank note`, footer `enter create note | esc back
  to notes` (AC#1, AC#2).
- `fixed-235x52-enter.txt` -- the first Enter creates the note (`Notes (1)`,
  "Draft -- not saved yet") and stays in Library (AC#3).
- `fixed-100x30-landing.txt` -- same at 100x30, compact footer `enter create |
  esc notes`.

**Counts vs dev.** notes-wave onboarding 6 passed; notes-wave list + the four
`test_library_phase_c_*` 123 passed; crit8-keyboard, canvas-sync-defects,
first-run-profile-interview, wizard-stray-navigation passed.
`test_library_shell.py -k "nav_context or deeplink or notes_create or
entry_focus"` 7 passed / 2 failed with an IDENTICAL FAILED name set on a
detached `origin/dev` worktree (pre-existing; filed as its own rider).
`test_first_run_wizard_live_contract.py` reds are load-dependent flakes on both
trees and not nested (dev 4 failed/2 passed vs branch 3 failed/3 passed, with
different names each side, host load average ~30).
`scripts/check_persistent_diagnostic_inventory.py` verified, no re-pin (no new
log calls); no CSS touched.

**Docs.** `Docs/User_Guide/library/notes.md:434` already documents the fixed
behaviour, so this restores the documented contract rather than changing it --
no guide edit, and the stamp is left alone to avoid conflicting with the
in-flight critique-2 wave.

**Lesson recorded.** `backlog/docs/lessons-testing-evidence.md` -- "A
navigation pin taken at the message boundary is not a pin on the landing".

**Files.** `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Library_Modules/library_navigation_controller.py`,
`Tests/UI/test_library_notes_wave_onboarding.py`,
`backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
