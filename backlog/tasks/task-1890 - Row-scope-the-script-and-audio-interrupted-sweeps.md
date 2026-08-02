---
id: TASK-1890
title: Row-scope the script and audio interrupted sweeps
status: In Progress
assignee: []
created_date: '2026-08-02'
updated_date: '2026-08-02 12:10'
labels:
  - watchlists
  - briefings
  - audio
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed during the whole-branch review fix wave for `chore/briefings-residuals-1810-1812`
(verdict: `.superpowers/sdd/briefings-residuals/whole-branch-verdict.md`, adjudication (b)),
which fixed `fail_interrupted_briefings`'s `exclude` to be row-scoped rather than
watchlist-scoped (task-1812, AC #3) and then closed the residual claim-registration
window that fix left open (this branch's own fix wave, Important 1). Both sibling sweeps
in the same family were deliberately left out of that scope and still exclude by the
coarser key:

- `fail_interrupted_scripts` (`tldw_chatbook/Subscriptions/briefing_cast.py`) excluded by
  `briefing_id` (the claim key), not by the claimed row's own id.
- `fail_interrupted_audio` (`tldw_chatbook/Subscriptions/briefing_audio.py:1342`, `AND
  script_id NOT IN (...)`) excludes by `script_id` too.

This is the same bug class task-1812 fixed for briefings: a `script_id` (or, for audio, a
script whose scripts share one id) can have more than one `generating` row over its
lifetime -- a crash-zombie row left by a prior process, coexisting with a freshly-claimed
live row for the SAME key. A coarse, key-scoped `exclude` cannot tell them apart, so it
shields both rather than only the live one.

Unlike the briefings case, this errs toward *over*-protection, never over-sweeping: a
row-scoped exclude only ever narrows which rows survive, so leaving the coarse exclude in
place cannot cause a live row to be falsely marked `interrupted`. There is no correctness
hole here -- a zombie merely survives longer than it needs to, until a sweep runs while
its key is entirely unclaimed.

But task-1811 (this same branch) gave the coarse audio exclude a user-visible surface it
did not have before: `WatchlistsCollectionsScreen`'s Synthesize refusal toast
(`tldw_chatbook/UI/Screens/watchlists_collections_screen.py:5784`, "is already being
synthesized for this script") can now name a row that is not actually live -- a
crash-zombie audio row shielded by an unrelated live claim on the same `script_id`,
surfaced as if it were the thing blocking Synthesize. The *decision* to refuse is still
correct (something IS claimed for this script); the *row named in the message* can be
dishonest.

Reference: task-1812 (the briefings-side fix this generalizes) and this branch's
whole-branch verdict file for the adjudication that filed this task rather than folding
the fix into 1811/1812's own scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `fail_interrupted_scripts`'s `exclude` is scoped to the claimed script's own row id, not merely its `script_id`, mirroring `fail_interrupted_briefings`'s row-scoped shape (task-1812)
- [x] #2 `fail_interrupted_audio`'s `exclude` is scoped to the claimed audio row's own id, not merely its `script_id`, the same way
- [x] #3 Both row-scoped sweeps additionally handle the unrecorded-claim window the same way this branch's briefings fix does (a claim taken before its row id is recorded must still spare that row from a concurrent sweep) -- reference `chore/briefings-residuals-1810-1812`'s `pending_briefing_claim_watchlist_ids()` shape
- [x] #4 A same-`script_id` crash-zombie script row and a live claim coexist in one sweep: the zombie is failed as interrupted, the live row is untouched (script sweep coexistence test)
- [x] #5 A same-`script_id` crash-zombie audio row and a live claim coexist in one sweep: the zombie is failed as interrupted, the live row is untouched (audio sweep coexistence test)
- [x] #6 A claim taken but not yet row-recorded survives a sweep run inside that exact window, for both the script and audio sweeps (window regression tests, mirroring this branch's briefings window test)
- [x] #7 The Synthesize blocking toast (`watchlists_collections_screen.py`) no longer names a crash-zombie audio row as "already being synthesized" once the live claim's row id is recorded -- the row-scoped sweep fails the zombie before the toast is composed, so only the genuinely live row's label appears (in the brief pre-recording window the pending-claim guard deliberately spares the whole script, zombie included -- the same tradeoff as the briefings sweep)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mirror the merged briefings shape (`briefing_service.py`: `_ACTIVE_BRIEFING_CLAIM_ROW_IDS`, `pending_briefing_claim_watchlist_ids()`, row-id `exclude` + `exclude_watchlists` pending-claim guard) into `briefing_cast.py` (`fail_interrupted_scripts`, claims keyed by script/briefing id) and `briefing_audio.py` (`fail_interrupted_audio`, claims keyed by script_id) — same registry-dict + frozen-snapshot + finally-cleared discipline, row id recorded as soon as the INSERT returns.
2. Update the two sweeps' SQL from key-scoped `NOT IN` to row-id `NOT IN` + pending-key `NOT IN`; update every call site (screen sweeps `_sweep_and_guard_cast`/`_sweep_and_guard_audio` and any handler/service callers) to pass the new exclude pair.
3. Tests per ACs 4-6 mirroring the briefings coexistence + window tests; AC 7 pinned at the screen: zombie audio row + live claim → toast names only the live row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mirrored briefing_service's task-1812/1810-1812 shape into both sibling modules, generalizing "row id, not key" scoping and the unrecorded-claim window guard.

briefing_cast.py: added `_ACTIVE_CAST_CLAIM_ROW_IDS` (briefing_id -> script_id), `active_cast_claim_row_ids()`, `pending_cast_claim_briefing_ids()`; `_claim_cast` gained an optional `script_id` param (mirrors `_claim_briefing`'s `briefing_id` param) and clears the row-id entry in its `finally`; `generate_script` records the row id immediately after `_start_script`'s `to_thread` hop returns (no `await` in between). `fail_interrupted_scripts` gained `exclude_briefings`; `exclude` is now `id NOT IN (...)` (was `briefing_id NOT IN (...)`).

briefing_audio.py: identical shape, keyed by script_id -> audio_id (`_ACTIVE_AUDIO_CLAIM_ROW_IDS`, `active_audio_claim_row_ids()`, `pending_audio_claim_script_ids()`); `_claim_audio` gained an optional `audio_id` param; `generate_script_audio` records the row id immediately after `db.create_briefing_audio`'s own `to_thread` call returns. `fail_interrupted_audio` gained `exclude_scripts`; `exclude` is now `id NOT IN (...)` (was `script_id NOT IN (...)`).

watchlists_collections_screen.py: both sweep call sites per artifact (`_fail_interrupted_scripts_if_safe`/`_sweep_and_guard_cast`/`_cast_script`, and `_fail_interrupted_audio_if_safe`/`_sweep_and_guard_audio`/`_synthesize_audio`) now snapshot the row-id accessor plus the pending accessor and pass both through. `active_cast_claims`/`active_audio_claims` imports dropped from the screen (no longer what the sweep wants); the coarse accessors themselves stay unchanged (test-only consumers now; no production callers remain).

Tests: mirrored test_briefing_service.py's three sections (spares-an-excluded-row-both-directions updated to be row-scoped with a padding-row fixture; new row-scoped coexistence test; new pending-claim-window tests) into test_briefing_cast.py and test_briefing_audio_pipeline.py. Widened 4 stale-signature `_recording_sweep` monkeypatch stubs in test_watchlists_artifacts_pane.py (2 cast, 2 audio) to accept the new exclude_briefings/exclude_scripts kwarg -- these would otherwise TypeError once the screen's own call sites pass it. Added one screen-level test for AC #7 (test_the_blocking_toast_never_names_a_crash_zombie_sharing_the_live_claims_script), pinning that a crash-zombie audio row sharing a script_id with a live claim (recorded row id, via _claim_audio's new audio_id param) never appears in the Synthesize blocking toast -- only the genuinely live row's label does, and the zombie is independently confirmed swept (failed/interrupted) rather than merely absent from the message by coincidence.

Mutation-tested both sweeps (Edit-revert -> RED -> restore, clean git status between): reverting exclude's SQL from id-scoped back to key-scoped REDs the respective coexistence test (swept count goes from 1 to 2, i.e. protection lost, not merely widened); dropping the exclude_briefings/exclude_scripts SQL clause REDs the respective pending-window test (swept goes from 0 to 1).

Verified: Tests/Subscriptions/ (535 passed), Tests/Watchlists/ (365 passed), Tests/Scheduling/ (264 passed) -- 1164 total, 0 failures.

No deviations from the reference shape: both cast and audio claims turned out to need the identical three-part treatment (row-id registry, frozen-snapshot accessor, pending-claim accessor) despite cast's INSERT landing as the LAST statement of its to_thread hop (briefings' is the FIRST) -- the pending window exists regardless of where in the hop the INSERT falls, since the claim is taken before the hop starts either way.
<!-- SECTION:NOTES:END -->
