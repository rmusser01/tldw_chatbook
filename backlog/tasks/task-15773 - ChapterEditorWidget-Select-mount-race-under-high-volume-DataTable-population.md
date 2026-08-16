---
id: TASK-15773
title: ChapterEditorWidget/Select mount race under high-volume DataTable population
status: Done
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - bug
  - stts
  - flake
priority: low
---

## Description

Found and documented as an out-of-scope dodge in task-15478's Review round 3
(input-latency burn-down). While testing chapter-detection on very large
pastes, an unrelated, pre-existing race in `ChapterEditorWidget`/`Select`'s
mount sequence tripped when the chapter table populated a very large number
of rows in one reactive update — observed once in a full-file test run with
`_make_large_book` producing ~999 chapters for a 3M-word book (2000
words/chapter density).

Task-15478 worked around it by reducing `_make_large_book`'s chapter density
(2000 -> 60,000 words per chapter), which made the flake reproduce 0/4
afterward in its own suite — a real dodge, not a fix. The race itself (what
in `ChapterEditorWidget`'s mount sequence loses ordering when `Select` is
populated with a very large row count in one shot) is still there and
unowned.

## Acceptance Criteria

- [x] The `ChapterEditorWidget`/`Select` mount-sequence race is reproduced
      deterministically (e.g. via the original higher-density
      `_make_large_book` shape, or a targeted stress test that populates the
      table with hundreds+ of rows in one reactive update)
- [x] Root cause is identified (an ordering assumption between the table
      populate and the Select's own mount/options-set) and fixed at the
      source, not worked around by capping row counts in tests
- [x] A regression test pins the fix at a row count that reproduced the race
      before the fix
- [x] `Tests/UI/test_speech_audiobook_chapter_detection.py` stays green,
      including at its original (pre-workaround) chapter density if restored

## Implementation Plan

1. Characterize what `set_chapters(N)` actually does on a mounted widget at
   HEAD (`feba3b080`): trace the ordering between `watch_chapters`'s table
   populate, the `recompose=True` teardown/remount it schedules, and the new
   `Select`'s Compose->Mount window. Probe in the scratchpad, not the repo.
2. Reproduce: prefer a deterministic interleave (gate the remount window
   open — e.g. block a child's `_on_mount` on an `asyncio.Event` — and land
   `_apply_detected_chapters`/`set_chapters` inside it, exactly as
   `call_from_thread` can in production, since it runs on the app loop, not
   the widget's serial pump). Fall back to bounded repetition of the
   original-density `_make_large_book` (999 chapters) full-file run if the
   deterministic shape refuses to trip.
3. Root-cause and fix at the source with the repo's boring deferral pattern:
   population must not run against a tree that is about to be (or is being)
   recomposed away; the Select/table must be ready before data lands.
4. Born-red regression test at a row count that reproduced the race; pin
   populated-table content identical (row count + cell text) so behavior is
   unchanged; restore the original chapter density in
   `test_speech_audiobook_chapter_detection.py` if it stays green.
5. Baseline vs origin/dev, pytest to a file, ruff check + format on touched
   files only, hand-edit this task file with notes, commit locally.

## Implementation Notes

**Root cause.** `chapters = reactive([], recompose=True)` on a widget whose
`compose()` is entirely static (no child reads `self.chapters`). Every
`set_chapters` therefore did two broken things, in order:

1. `watch_chapters` populated the CURRENT DataTable synchronously (N
   `add_row`s in one reactive update), THEN the scheduled recompose removed
   that whole subtree and mounted a fresh, empty one. The population always
   landed on the doomed tree: the settled table had **0 rows after every
   update, at any row count** — verified in the real STTS host too (pristine
   HEAD: `detected=13 table_rows=0`; the audiobook chapter table has been
   empty in production ever since the recompose reactive was introduced).
2. The remount re-ran `#chapter-voice-select`'s Compose→Mount sequence on
   every data arrival. Textual's `App._prune` snapshots `walk_children` at
   call time and `Widget.mount()` silently no-ops on a `_pruning` widget, so
   a teardown (app/test shutdown, Speech view transition, ancestor removal)
   landing between the fresh Select's *registration* and its *Compose*
   dispatch leaves the Select mounting no children while its `Mount` event
   still fires — `Select._on_mount → _setup_options_renderables →
   query_one(SelectOverlay)` then raises **`NoMatches: No nodes match
   'SelectOverlay' on Select(id='chapter-voice-select')`** through
   `app._handle_exception`. High row volume matters because the N-row
   populate + teardown of the old populated table clog the loop and widen
   that registration→Compose window — which is why task-15478 saw it once in
   a full-file run (surfacing at `run_test` teardown) and 0/4 after halving
   the row count.

**Reproduction.** Deterministic interleave, not repetition: a probe gated
`Select._on_compose` (park the freshly recomposed Select right before its
Compose dispatch), landed `editor.remove()` inside the window, released —
crash reproduced verbatim on the first run, every run. Un-gated stress
(populate+remove ticks, populate+immediate-exit, double-apply interleaves;
34 iterations) never tripped it — the organic window is real but narrow,
which matches the original 1-in-a-full-file-run sighting.

**Fix (boring, at the source).** Drop `recompose=True` — population updates
the persistent, already-mounted children in place; no teardown, no remount,
no reopened mount window. Pre-mount data stays queued in the reactives (the
existing `is_mounted` guards) and `on_mount` now replays both the table AND
the selected chapter's preview (previously only the table). No change to
`_refresh_chapter_table` itself, so populated content is byte-identical —
pinned in the regression test.

**Tests** (`Tests/Widgets/test_chapter_editor_widget_population_race.py`,
all three born red against the pre-fix widget via edit-based revert):
- `test_high_volume_population_lands_and_children_stay_mounted` — 999 rows
  (the original flake's chapter count) must survive settle with pinned cell
  content, and the table/Select must be the same mounted instances. Red:
  "settled table has 0 rows for 999 chapters".
- `test_teardown_racing_a_population_cannot_break_select_mount` — the gated
  deterministic interleave. Red: `NoMatches: No nodes match 'SelectOverlay'`
  raised out of the test's app teardown, plus the gate-engaged assert.
- `test_chapters_set_before_mount_are_replayed_when_ready` — pre-mount
  `set_chapters` replays table + preview at mount. Red: empty preview
  (review measured 3/3: the row-count assert PASSED pre-fix on this path —
  a pre-mount recompose is a no-op via `Widget.recompose()`'s
  not-attached early return, so only the preview replay was missing).

**Density restored** in `Tests/UI/test_speech_audiobook_chapter_detection.py`
(`words_per_chapter` 60_000 → 2000, the pre-workaround default: 999 chapters
for the 3M-word book), docstring updated; file green 5×5 consecutive
full-file runs (8 passed each).

**Verification.** New+tuple-order tests green 5× consecutive (4 passed
each). STTS/Speech batch (15 files): 565 passed, 3 failed — all 3 failures
reproduced bit-for-bit on a pristine extract of base HEAD `feba3b080`
(diagnostic-inventory pair pinning `Chat/console_runtime.py` + a speech
playground local-deps status test), i.e. pre-existing on the base, not from
this change. `ruff check` clean on all touched files; `ruff format --check`
clean on the two fully-owned files (the detection test file has pre-existing
format drift at HEAD in hunks this task does not touch — left alone).

**Files.**
- `tldw_chatbook/Widgets/TTS/chapter_editor_widget.py` — drop
  `recompose=True`, `on_mount` replay, rationale comment.
- `Tests/Widgets/test_chapter_editor_widget_population_race.py` — new, 3
  born-red regression tests.
- `Tests/UI/test_speech_audiobook_chapter_detection.py` — original density
  restored.
