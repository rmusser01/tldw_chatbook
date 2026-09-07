---
id: TASK-19047
title: >-
  stts_profile_library view-switch dismissal test still flakes under CPU load
  (outside 16842's family fix)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 08:40'
labels:
  - test-health
  - flake
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16842 fixed a five-test flake family in
`Tests/UI/test_stts_profile_library.py` (settle on the asserted state, bound
by wall clock — see its 2026-08-16 entry in
`backlog/docs/lessons-testing-evidence.md`). Its reviewer then reproduced a
sixth, pre-existing failure OUTSIDE 16842's diff:
`test_switching_stts_view_dismisses_owned_profile_modal_and_worker` fails
under CPU-burner load at BOTH the wave base `cef56efaf` and head.

Verified still present at dev `1bf7f234e` (:2840): the test uses the file's
`_wait_until` idiom for the modal/unmount/settings-pane/worker-finish waits,
but still contains unsettled one-shot samples in the same shapes 16842's
lessons entry catalogues (e.g. the `voice_profile_action` worker census and
its `not is_finished` assert taken immediately after the modal wait, and a
one-shot `pilot.click` before it). No backlog task covers this test (grepped
backlog/tasks at dev). Likely another settle-on-asserted-state candidate; the
diagnosis must come from a load reproduction, not from reading the test.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce first, no speculative patch: run the target test (both
   parametrizations) and the full file repeatedly under 14 parallel CPU
   burners (the 16842 lessons recipe: one busy-loop process per logical
   core), output captured to files; identify the exact failing assertion
   and save its signature.
2. Root-cause the mechanism from the reproduction against the test's
   remaining one-shot shapes (the `pilot.click` before the modal wait; the
   `voice_profile_action` worker census + `not is_finished` assert sampled
   immediately after the modal wait).
3. Fix strictly per the 16842 idiom: wall-clock-bounded `_wait_until`
   settles polling the actual asserted condition; no fixed pauses, no
   attempt-count waits. Preserve the pinned contract exactly: switching
   the S/TT/S view dismisses the owned modal AND finishes its worker —
   do not weaken the census (exactly one live worker at modal-open) or
   the finished-after-switch assertion.
4. If the reproduction shows a product bug (e.g. the worker genuinely
   survives the view switch under load), capture evidence, do NOT patch
   the product (sibling task 19043 owns Event_Handlers/TTS_Events +
   app.py), and report for routing.
5. Post-fix evidence at 16842's bar: repeated full-file runs green
   including runs under the same 14-burner load, plus repeated standalone
   runs of this test; record exact counts. ruff check + format on the
   touched file. Kill all burners when done.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failure is reproduced under load first (the 16842 lessons entry's CPU-burner recipe) and the exact failing assertion identified — no speculative patch
- [x] #2 The fix follows the established 16842 idiom: wall-clock-bounded waits polling the asserted condition itself; no new fixed pauses or attempt-count waits
- [x] #3 Post-fix evidence meets 16842's bar: repeated full-file runs green including runs under the same load, plus standalone runs of this test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Reproduction first (AC #1).** CPU-burner recipe (busy-loop python
processes; 14-28 burners, plus at times extreme ambient machine load).
The target test failed 11/11 standalone runs pre-fix (13 failing
parametrized instances), every one with the SAME signature — not the
catalogued worker-census/one-shot-click suspicions:

    Tests/UI/test_stts_profile_library.py:937: in _select_action_profile
    Tests/UI/test_stts_profile_library.py:939: in <lambda>
    E  textual.css.query.NoMatches: No nodes match '#stts-profile-table'

**Mechanism 1 (fixed).** `STTSWindow.watch_current_view` swaps the body in
a `speech-view-mount` worker, so after `current_view = "profiles"` the
table's *existence* is part of the condition being settled on — but the
predicate sampled it as a precondition: the unguarded `query_one` raised
NoMatches out of `_wait_until`'s first poll. Fix: `_profile_table_row_count`
helper returning None while the swap is mid-flight; predicate
`_profile_table_row_count(app) == 1` (shared `_select_action_profile`
helper, so every caller is repaired).

**Mechanism 2 (fixed, reproduced only after fix 1 unmasked it).** 2/15
loaded runs then failed at the test's settings-pane settle:
`IndexError: list index out of range` from
`app.query_one(".stts-content").children[0]` — polled inside the
observable empty window between `await remove_children()` and
`await mount(...)` in `_mount_view_unlocked`. Fix:
`_stts_content_first_child` helper (None while empty/mid-swap).

**Mechanism 3 (sibling, reproduced in the AC-3 full-file load runs; same
file, so in-boundary).** 1/3 loaded full-file runs failed
`test_audiobook_kokoro_blend_group_is_not_a_keyboard_select_option` at
`assert narrator.value == 'blend:duet'` — got `'shimmer'`, the LAST
OPENAI voice. The test settled on `provider_select.value == "openai"`,
but that flips inside the mount timer callback while the narrator-options
rewrite rides the queued Select.Changed message; under load the queued
rewrite landed AFTER the test's kokoro switch and silently restored the
openai options (the keyboard walk itself worked). The openai list is
byte-identical to the compose-time options, so the settle targets the
sibling @on handler's `provider` attribute write on the faked
character-voice widget — an observable of the same dispatch step. Its
attempt-count `for _ in range(100)` wait also became `_wait_until` (the
exhaustible-budget shape 16842 retired).

**Contract preserved (AC #2).** The worker census (`len == 1`,
`not is_finished` at modal-open) and the finished-after-switch settle are
untouched — analysis plus 41 loaded standalone runs show the census is
safe by construction (the worker sits inside `push_screen_wait` while the
modal is `app.screen`). No fixed pauses, no attempt-count waits added;
all new settles are wall-clock-bounded `_wait_until` polls of the actual
asserted condition.

**Evidence (AC #3), final file state.** Target test standalone: 15/15 @
20 burners, 15/15 @ 28 burners, 10/10 @ 20 burners, 15/15 unloaded — zero
failures. Sibling standalone: 10/10 @ 20 burners + 1/1 unloaded.
Full file (163 tests): 4/4 green @ 14 burners (117-147s), 3/3 green
unloaded (~64s), plus 3/3 unloaded at the intermediate state. ruff check
+ format clean.

**Known residue (not patched — no reproduction):** the view-cycling test
at ~:1104 carries five one-shot `children[0]` asserts of the same
mechanism-2 class after single-pause view switches; never fired across
all runs here. Left for a routed task if it ever flakes.

**Files:** `Tests/UI/test_stts_profile_library.py` (only code change),
this task file, `backlog/docs/lessons-testing-evidence.md` (new entry:
raising predicates are one-shot samples; value flips are not their
message cascade).
<!-- SECTION:NOTES:END -->
