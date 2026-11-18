---
id: TASK-41532
title: "Dev's push-to-dev Derived Artifacts run 43917600414 failed four nodes at dev 5e3c8b2bd4"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-18 04:56'
created_date: 2026-11-17 05:39
labels:
- follow-up
- tests
- ci
priority: high
references:
- Tests/UI/test_library_shell.py
- Tests/Backup_Recovery/test_operational_owners.py
dependencies:
- TASK-41505
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reported by the coordinator at the filing review (round 15). The run failed:
- `Tests/UI/test_library_shell.py::test_slash_focuses_the_prompts_filter_not_the_rail_search`
- `Tests/UI/test_library_shell.py::test_library_media_edit_save_stays_gated_through_detail_refresh`
- `Tests/UI/test_library_shell.py::test_library_media_complete_facets_commit_during_initial_recovery[loading-All]`
- `Tests/Backup_Recovery/test_operational_owners.py::test_run_log_real_bound_writer_is_excluded`
Whether the three library_shell nodes share TASK-41505's cause (a single pause racing a handler) is not known; read the run logs first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four nodes passes in the push-to-dev job, shown by a CI run on the fix
- [x] #2 The failure shape of each node from run 43917600414 is recorded, with any shared cause named
- [x] #3 If the library_shell nodes share TASK-41505's cause, the notes say so and fix them there
<!-- AC:END -->

## Implementation Plan

1. Read run 43917600414's failed logs and the earlier push-to-dev runs, and record each node's failure shape (AC#2).
2. Reproduce each library_shell node locally under load at base; trace what moves the state the test asserts.
3. Fix each test to wait for the condition it asserts (no fixed pause), and prove it red-before under a forced delay and green after, alone and at `-n 4` under load.
4. Backup_Recovery node: check whether TASK-41117's drain fix (on this base) already covers it; run it at base.
5. AC#1 needs a push-to-dev run on the merge; leave it to the coordinator.

## Implementation Notes

Test-only fixes for the three library_shell nodes (the rare one's cause was found in fix round 1), and nothing for the Backup_Recovery node, which TASK-41117 already fixed on this base (45b1949ca3). AC#1 is open: it needs a push-to-dev run on the merge. Run 44218463462 (dev 45b1949ca3, which has TASK-41117's fix) was cancelled at 02:54 UTC on 2026-11-18 by newer dev pushes before any job started, so it says nothing either way.

**Failure shapes (AC#2).** Read from run 43917600414 and every completed push-to-dev run since the Library shell job started (19 runs, 43382155356 to 44190585155, 2026-11-13..17; the Backup_Recovery job has 11 of them, from 43743285529). No cause is shared by all four.
- `test_slash_focuses_the_prompts_filter_not_the_rail_search` (CI shard 10): red in 19 of 19 runs. The check that fails is the one before `/` is pressed, `assert not prompts_filter.has_focus`: focus is already in `#library-prompts-filter`. Cause: when Prompts entry focus first runs and finds no `.library-prompt-row`, it takes the empty-list fallback, the filter (`_focus_library_list_entry`). Under load the rows arrive about 1 s after the rail press, after that first attempt, so focus stays in the filter. A `set_focus` trace (serial, load ~35, 3 of 4 runs red) showed the entry landing on the filter with 0 rows mounted, then the browse sync putting focus back on the filter once the row mounted. This is not TASK-41505's cause: no number of pauses moves focus off the filter. It is the product gap TASK-32301 already tracks (a cold Prompts visit lands in its filter): in `_focus_library_list_entry`, Prompts and Conversations have no wait for loading rows (Notes got one in TASK-41129, and Skills has one), so the test's precondition held only when the rows won the race. TASK-32301's description now names that cause.
- `test_library_media_edit_save_stays_gated_through_detail_refresh` (CI shard 9): red in 14 of 19 runs, `assert len(service.type_calls) == initial_facets + 1` (1 == 2). Cause: `_complete_library_media_mutation` clears `bulk_delete_in_flight` first and only then starts the page refresh and the facet refresh, each on its own worker. The test waited for the flag and then asserted both calls. Locally the page call also lost the race (`search_calls`, 1 == 2). This is TASK-41505's cause: the test asserted a worker's effect after waiting only for an earlier flag. That task's notes now name this node as fixed here.
- `test_library_media_complete_facets_commit_during_initial_recovery[loading-All]` (CI shard 4): red in 1 of 19 runs (this one). `#library-media-type-choices never mounted within 30.0s`. The visible text still shows the toolbar, so the type strip never opened. The "Couldn't load media" copy in that text is a side effect, not the cause: it is the test's own 30 s page gate expiring (`Library Media browse failed` logged at 17:32:02.945, as the wait ended). The first 40 local runs at `-n 4` (load 44-80) passed, and a probe over 24 of them showed the press reaching its handler before the opener was removed. Fix round 1 reproduced it once in 20 runs (load ~31), with the round-1 diagnostic: `disabled=False, display=False, pruning=True, parent=None`, focus on a replacement type filter. `display=False` comes from the pruning, not from CSS: `dom.py`'s `display` returns False whenever `_pruning` is set. So a canvas recompose that the press did not cause pruned the opener, and the press never reached the handler. A pruned node's already-queued messages are usually still delivered late, because `Widget.on_prune` closes the queue after them. So this press was lost in a way not yet pinned down. That defect is TASK-41703. What triggered that recompose is not identified either: 168 instrumented runs did not catch it. It is not TASK-41505's shape, because no fixed pause comes before the wait that failed.
- `Tests/Backup_Recovery/test_operational_owners.py::test_run_log_real_bound_writer_is_excluded`: red in 9 of 11 runs. The message's first line, `TTS shutdown cleanup failed (CancelledError)`, is teardown noise: the child prints `unhandled exception during asyncio.run() shutdown` for `TTSService._complete_shutdown` on passing runs too (checked locally on `[resume]`). The real failure is at the bottom of the message: `File "<string>", line 63`. In the script as composed at 5e3c8b2bd4 (the setup block inserted), line 63 is `assert runtime.pause.drain(time.monotonic()+1)`. That is the 1 s drain TASK-41117 replaced (7e98d6bc04). Its fix is on this base but not on 5e3c8b2bd4 or a03035f039, the two newest red runs. At base, the node and the four handoff cases passed (5 passed, `-n 4`, load 35-37).

**Fixes** (`Tests/UI/test_library_shell.py`):
- slash: after the row mounts, the test waits for the entry window to disarm, focuses the row, and waits until the row holds focus. Only then does it check the precondition and press `/`.
- edit save: the wait now covers the interlock and both refresh calls. The exact `== +1` assertions stay after it.
- `[loading-All]` (fix rounds 2-3): after the press, a bounded loop (30 s, matching the selector wait after it) waits for `type_choices_visible`. Once the pressed opener is detached (`parent is None`), the loop drains every message pump from the canvas up to the screen (`call_later` a marker on each, in order). It re-presses the live, unpruned replacement opener only if the strip is still closed after the drain. This works around TASK-41703.
  - Why this never toggles twice: by the time the opener is detached, every queue being removed above it has emptied. A press that survived is already queued at or above the canvas, ahead of the markers, so it is handled before the drain ends. This rests on message order, not timing.
  - Each marker wait is capped at the time left before the deadline. A `TimeoutError` or `NoMatches` inside the drain is caught, so a failure always ends in the loop's diagnostic. On a timeout, the message names the number of presses and the last opener's `disabled`, `display`, pruning and parent, plus the focused widget. `Button.press` does nothing on a disabled or undisplayed button, which is why those two are named.

**Red before, green after.**
- Base at `-n 4`, 12 repeats each (load 76-87): slash 2 of 12 red (`assert not True`, line 7300), edit save 2 of 12 red (`1 == (1 + 1)`). Slash serially: 3 of 4 red (load ~35).
- Forced race. A temporary edit, reverted afterwards: `await asyncio.sleep(1.0)` at the top of `LibraryPromptBrowseController._load` and `await asyncio.sleep(0.5)` at the top of `LibraryMediaBrowseController._load_facets`. The old tests failed 3 of 3 each (line 7300 `assert not True`; line 10243 `1 == (1 + 1)`). The new tests passed 3 of 3 each (load 58-67).
- Diagnostic check (round-0 message): with the press removed temporarily, the round-0 wait failed with "The type opener press never opened the strip: opener pruning=False, parent=Horizontal(id='library-media-toolbar-choosers' ...)". The current message reads "N type opener press(es) never opened the strip: last opener disabled=..., display=..., pruning=..., parent=..., focused=...".

**Runs after the fix.**
- Alone (serial): the three tests (5 cases), 3 runs, 15 of 15 passed (load 60-86).
- Under load (`-n 4`, 10 repeats of each case): 50 of 50 (load 61-81), then 50 of 50 (load 61-107).
- CI shards that hold the changed nodes, run serially with the lane's `--timeout=180`, all three at once, load 57-99: shard 4 passed 62 of 62; shard 9 passed 60 of 62; shard 10 passed 61 of 62. The changed nodes passed in all three. The three failures are other nodes: `test_library_shell_media_type_filter_narrows_list` and `test_browse_row_error_callout_keeps_the_right_tint_across_a_retry` in shard 9, and `test_library_note_pilot_conflict_can_originate_from_every_region[context]` in shard 10. All 15 of their cases passed alone (`-n 4`, 3 repeats, load 53-61).
- The same two shard commands on base (the original test file), run at the same time (load 34-71): shard 9 passed 62 of 62. Shard 10 failed 3 of 62: the slash node (red, as in CI) and two other nodes, `test_library_shell_history_manual_expand_survives_unrelated_refresh` and `test_browse_row_error_callout_remounts_when_the_failure_shape_changes`. The other nodes that fail change from run to run on both base and head, so they are load flakes outside this task.
- `Tests/Architecture/test_screen_size_ratchet.py`: 25 passed (serial). No product module changed.
- `./scripts/preflight.sh`: all checks passed.

**Files.** `Tests/UI/test_library_shell.py`; TASK-41505's notes (one line).

**Fix round 1 (review, 2026-11-18).** The slash test's comment and the slash bullet above cite TASK-32301 instead of a new follow-up, and TASK-32301's description gained one line naming the cause. The `[loading-All]` wait also reports the opener's `disabled` and `display`, and waits 30 s. The slash test's entry-settle wait reports the focused widget on failure.
- **The diagnostic caught the lost press** (see the `[loading-All]` bullets above), so the wait now presses the live replacement opener. Review r16dc-rev2 found this round's guard against a double toggle unsound; it was replaced in fix round 2.
- **Red/green.** A temporary edit, reverted, awaited `LibraryMediaCanvas.recompose()` right after `opener.press()`, to force the prune. The round-1 test (diagnostic only) failed 9 of 72 runs with the same message. The new test passed 72 of 72 (load 18-48).
- **Re-run.** The changed nodes (5 cases), 6 repeats at `-n 4`, three rounds: 30 of 30 each time, 90 of 90 (load 21-34). Preflight passed.

**Fix round 2 (review r16dc-rev2, 2026-11-18).**
- **The round-1 double-toggle guard was unsound.** A pruned opener does not mean its press was lost: a press already queued is still delivered late. The round-1 predicate could therefore re-press while the first press was in flight, and toggle the strip twice. The reviewer's plain-Textual probe showed 31 such re-presses in 200 trials. The loop now drains the pumps before deciding (see the `[loading-All]` fix bullet). The comment and these notes no longer say Textual drops a pruned node's messages.
- **Control (temporary edits, reverted).** Each trial counted presses against handler runs, using the handler's `then` callback names on `_sync_library_canvas`. 48 trials (16 repeats x 3 cases, `-n 4`) for each control and loop:

| Forced after the press | Loop | Trials, as presses / handler runs | Double toggles |
|---|---|---|---|
| `canvas.refresh(recompose=True)`, not awaited (load 20-31) | new | 47 at 1/1; 1 at 2/1 (first press lost) | 0 |
| same | round-1 | 48 at 1/1 | 0 |
| `await canvas.recompose()` (load 25-36) | new | 43 at 1/1; 5 at 2/1 (first press lost) | 0 |
| same | round-1 | 45 at 1/1; 3 at 2/1 | 0 |

  In the app, every re-press followed a truly lost first press. The double toggle the reviewer showed in plain Textual did not appear here, so the drain is shown safe in the app, not shown necessary.
- **TASK-41703** files the product defect: a background canvas recompose can drop a pending button press. The test comment cites it.
- **Wording.** TASK-32301 now says Prompts and Conversations have no wait for loading rows (Skills has one) and gained AC#4 for the cold Prompts visit. The `display=False` clause and the round-0 diagnostic label are above.
- **Re-run.** The changed nodes (5 cases), 8 repeats at `-n 4`: 40 of 40 (load 27-32). `[loading-All]` alone, 24 repeats at `-n 4`: 24 of 24 (load 28-32). Preflight passed.

**Fix round 3 (review r16dc-rev3, 2026-11-18).**
- **The drain no longer starts while the opener is only `_pruning`.** At that point a late press can still sit in the pruned toolbar's queue, below the canvas marker, and the round-2 trigger could then re-press ahead of it. The reviewer's plain-Textual probe (`pruning_trigger_probe.py`) forces that interleaving; re-run here: with the round-2 trigger, presses 3 and handler runs 3 (open, close, open) in 20 of 20 trials; with `parent is None` only, 1 and 1 in 20 of 20.
- **Drain failures.** Temporary edits, reverted: the opener was removed so the drain runs, and `TimeoutError()` or `NoMatches()` was raised at the top of the drain. Each run ended in the loop's assertion, "1 type opener press(es) never opened the strip: last opener disabled=False, display=False, pruning=True, parent=None, ...". A first control that stubbed `screen.call_later` was invalid, because `pilot.pause` itself needs `call_later` (WaitForScreenTimeout).
- **TASK-41703** now says a press *can* be lost when it is still below the canvas child the recompose removes at the moment that child stops taking messages, notes that most such presses are only late, and has a deterministic AC#1 (a gated press, red N of N before the fix).
- **Re-run.** The changed nodes (5 cases), 8 repeats at `-n 4`: 40 of 40 (load 36-48). `[loading-All]` alone, 24 repeats at `-n 4`: 24 of 24 (load 38-50). Preflight passed.
