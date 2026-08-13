---
id: TASK-15459
title: Library: compose once per visit
status: Done
assignee:
  - claude
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: a warm Library revisit composes 2-3 times — the initial compose with pre-cache state, an explicit `refresh(recompose=True)` after applying the app-scoped snapshot cache (`library_screen.py` on_mount, `:4992`), and again when `_refresh_local_source_snapshot` reconciles; worker chains (trust-posture etc.) each end in another full-screen recompose (`:7577-7598`). Since screens are rebuilt on every tab switch, this cost is paid on every visit.

Fix direction: seed the cached snapshot in `__init__`/`restore_state` (both run before mount, `app.py:7899/:7922`) so the FIRST compose already renders cached data, then make the reconcile a targeted update. Stability constraint: the on_mount comment explains why it currently recomposes — the restore ordering is subtle; pin the cached-then-fresh data behavior with tests before reordering. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A warm revisit composes exactly once before data reconcile, and reconcile updates in place (or at most one scoped recompose only when data actually changed) — evidence
- [x] #2 Cached-then-fresh rendering behavior preserved (tests)
- [x] #3 Library visit latency before/after recorded
<!-- AC:END -->

## Implementation Plan

1. Re-verify at HEAD (dev `ebf56a763`, includes task-15457's reconciliation) whether the audit's "2-3x" claim still holds -- read `on_mount`/`restore_state`/`compose_content`/`_apply_local_source_snapshot` fresh rather than trusting the audit's line numbers, which have drifted.
2. If still true, seed `_local_source_records` (and siblings) from the app-scoped `_library_source_snapshot_cache` in `__init__` AND `restore_state` (both run pre-mount) via a shared helper, so the FIRST `compose_content` on a warm revisit already renders cached data.
3. Once the first compose is pre-seeded, make `on_mount`'s own cache-check-and-recompose a fallback gated on `not self._library_loaded` (its only other setter) instead of an unconditional re-apply, so it no longer forces a second recompose when the pre-mount seed already succeeded.
4. Make the reconcile (`_apply_local_source_snapshot`, called when `_refresh_local_source_snapshot`'s worker lands) skip its recompose when the freshly-fetched snapshot is byte-for-byte identical to what is already rendered, while never suppressing a genuine "Loading..." -> real-data transition or the notes-editor-back focus-handoff's own recompose dependency.
5. Pin cached-then-fresh behavior with tests (pre-mount seeding in isolation, compose-count-across-a-warm-revisit, existing recompose/focus/carry-forward regressions), mutation-test each new guard, and record before/after compose counts + a synthetic latency sample.

## Implementation Notes

**Re-verification finding:** the audit's claim was still current at this branch's HEAD.
task-15457's reconciliation (merged as `ebf56a763`) converted many PER-CLICK sites to
canvas-scoped sync, but never touched the mount-time seam this task targets: `__init__`/
`restore_state` still left `_local_source_records` at empty defaults, so a warm revisit's
first `compose_content` painted the "Loading..." placeholder; `on_mount` then applied the
cached snapshot and issued its own explicit `refresh(recompose=True)` (a second compose,
required only because the first compose was stale); `_refresh_local_source_snapshot`'s
worker then landed and called `_apply_local_source_snapshot`, which recomposed a THIRD
time unconditionally for any row not in the Ingest/Prompts/Search rail-only carve-out --
even when the DB confirmed the cache verbatim. Measured: 2 composes per warm revisit
before this fix (the reconcile's would-be 3rd compose was already suppressed for the
common row classes by the existing rail-only branch; the general canvases were not).

**Fix (`tldw_chatbook/UI/Screens/library_screen.py`):**
- New `_seed_local_source_snapshot_from_cache()` helper, called from the tail of both
  `__init__` and `restore_state`. Re-checks the same `LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS`
  freshness window `on_mount` already used and, on a hit, calls
  `_apply_local_source_snapshot(*cached_snapshot)` -- safe pre-mount because that method
  only touches the DOM when `self.is_mounted` (never true yet at either call site), so a
  hit is a pure attribute seed. Seeded in BOTH places because `restore_state` runs after
  the real `_selected_conversation_id` is restored, which `_apply_local_source_snapshot`'s
  existing out-of-page-selection carry-forward (`_carry_selected_conversation_into_
  snapshot`) needs; `__init__`'s own empty default would carry nothing.
- `on_mount`'s cache-apply-and-recompose block is now gated on `not self._library_loaded`
  (the pre-mount seed's only setter before this point) -- it becomes a no-op fallback,
  reached only when the seed found no fresh-enough cache (miss, or TTL boundary race).
- `_apply_local_source_snapshot` now computes an `unchanged` flag (old attrs vs. the
  incoming snapshot, always false while `_library_loaded` is still False since that
  transition must clear the "Loading..." placeholder) BEFORE overwriting state, and skips
  its `refresh(recompose=True)` when true -- except while
  `_library_notes_pending_focus_waits_for_snapshot` is armed, since that flag's own release
  (`_release_library_notes_focus_after_snapshot`) is only ever scheduled from inside that
  same recompose branch (the notes-editor "Back" flow) and would otherwise strand set
  forever. The Ingest/Prompts/Search rail-only branch is untouched (already targeted, not
  a full recompose).

**Evidence (measured, this branch):**
- Compose count for a warm revisit before the reconcile fetch resolves: 2 -> 1. Pinned by
  `test_library_shell_repeat_visit_composes_exactly_once_when_data_is_unchanged`
  (`Tests/UI/test_library_shell.py`), which also asserts the reconcile landing with
  unchanged data adds no further compose. Both halves of the fix were mutation-tested
  (temporarily reverted, confirmed RED with the exact expected failure message, then
  restored) per `lessons-testing-evidence.md`.
- Synthetic in-process latency sample (5 runs each, same Pilot harness, wall clock from
  `push_screen` to shell-settled) via a scratch probe (not part of the suite): before
  (seed disabled) mean 441.5 ms (min 359.5, max 546.9), composes=2 every run; after mean
  299.5 ms (min 219.8, max 357.8), composes=1 every run -- ~32% faster in this synthetic
  harness. Caveat: this is in-memory fake-service timing dominated by Pilot/event-loop
  overhead, not a real terminal's widget-mount/CSS-apply cost (the audit's own Console/
  Watchlists numbers, ~0.9-1.35s per screen switch on real hardware, are the more relevant
  order of magnitude for what one fewer ~300-500-widget recompose actually saves).
- New tests added: `test_library_shell_repeat_visit_composes_exactly_once_when_data_is_
  unchanged`, `test_library_shell_init_seeds_local_source_snapshot_from_cache`,
  `test_library_shell_restore_state_seeds_local_source_snapshot_from_cache`.
- Regression coverage run (targeted, not the full suite -- see notes below): the existing
  "166" app-scoped cache suite, the task-252/task-15457 recompose-count spies
  (`test_library_selection_updates.py`, `test_library_canvas_sync_defects.py`), every
  notes-editor "Back"/focus-handoff test (the `unchanged`-skip's one carve-out), the
  out-of-page conversation carry-forward test, and the real app.py navigation round-trip
  tests in `test_screen_navigation.py` (save_state -> restore_state -> mount) all pass
  unmodified.
- One pre-existing failure, unrelated: `test_screen_navigation.py::
  test_action_library_notes_files_back_returns_to_database` fails at HEAD
  (`ebf56a763`, before any of this task's edits) with `AttributeError:
  'WorkspaceProbe' object has no attribute 'cancel_reload_confirmation'` --
  a test-double/production drift introduced by the 15457 merge itself (`git log -S` on
  that method name resolves to the merge commit), nowhere near this diff (`action_library_
  notes_files_back`, line ~12630, vs. this diff's `__init__`/`restore_state`/`on_mount`/
  `_apply_local_source_snapshot`). Not fixed here -- out of this task's scope.

**Files changed:** `tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_shell.py`.
