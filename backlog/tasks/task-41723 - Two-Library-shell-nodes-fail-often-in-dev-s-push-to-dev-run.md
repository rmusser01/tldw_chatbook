---
id: TASK-41723
title: "Two Library shell nodes fail often in dev's push-to-dev run"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: '2026-11-23 23:30'
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41532's review (round 16), reading 19 push-to-dev runs: `test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata` failed 8 times (also red in run 44190585155) and `test_library_media_initial_error_is_unknown_and_retry_is_unique` failed 4 times. TASK-41532 named only four other nodes, so the job will likely stay red without these. TASK-39152 already re-pinned the ingest-canvas node to wait for the cleared form and it still fails in CI; the CI failure state shows `already_in_library=1`, a lead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each node's failure shape across those runs is recorded, with its cause
- [x] #2 Each passes 10 of 10 at `-n 4` with the load recorded, or the product defect is fixed with a red-before test
<!-- AC:END -->

## Implementation Plan

1. Read every failed push-to-dev Library shell job since 2026-11-16 (read-only `gh run view --log-failed`) and tally both nodes' failure shapes.
2. Ingest-canvas node: trace who writes the form's path and title after the submit clears them (throwaway probe), reproduce naturally and under a forced order, and fix the race where it lives.
3. Red-before test in a file a PR lane runs whole (`Tests/UI/test_library_ingest_readiness.py`).
4. Retry node: confirm round 20's fix holds in CI and locally (10 of 10 at `-n 4`).
5. Soak both nodes 20 times at `-n 4` under load; run the push shard serially with CI's flags.

## Implementation Notes

**2026-11-22 (round 20, branch fu/r20-libraces): the `test_library_media_initial_error_is_unknown_and_retry_is_unique` half is covered by 1ae18ce611**, the partial port of another contributor's PR #3535 (recorded in TASK-39152). Failure shape: `'Media' == 'Media (2)'` on the title assertion after the Retry wait, in dev push runs 14e13291aa, c1e9d12c2d and d46f753d82 (the three found in the runs read for TASK-39152, 2026-11-13..22). Cause: the Retry's loading sync already removes Retry and moves focus to the type filter, and `_apply` sets `applied_result` before its canvas sync repaints the title, so the old wait could pass on the stale "Media". The wait now also requires the "Media (2)" title, and its timeout message names the title (808c7ab7d2). With the page and facet syncs delayed 0.5 s (a temporary source edit, reverted), the old wait failed 4 of 4 with that message and the new one passed 3 of 3. The ingest-canvas node is not touched, so neither AC is fully met and nothing is ticked: AC#1 still needs that node's shape and cause, and AC#2 still needs 10 of 10 runs for both nodes.

**Round 24 (2026-11-23, branch fu/r24-library, base dev b31e3e9241).**

**CI record (AC#1).** Read-only `gh run view --log-failed` over the 150 newest push-to-dev Derived Artifacts runs (2026-11-16..23). 26 of them got their Library shell jobs to an end (the rest were cancelled first; latest attempts only), 45 shard jobs failed.
- `test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata` failed in 13 of the 26, every time with the same shape: "The submitted form never cleared" with `path='…/tides.txt'`, `title='Tides 101'`, `preflight_checking=False` and `already_in_library=1`. In the two logs with captured stderr (runs 45209243472, 45124056579), the file is ingested (media_id 1), then a second content-hash pre-check of the same file runs 1.7-1.8 s later.
- `test_library_media_initial_error_is_unknown_and_retry_is_unique` failed in none of the 26. Round 20's shape (`'Media' == 'Media (2)'`) stands. Locally, under load, a second shape showed up (see TASK-42473): focus on `library-media-row-0`.

**Cause of the ingest node.** A throwaway probe logged every write to the form's path and title, the path and title `Input.Changed` the screen handled, and each screen recompose. It caught one natural failure in 3 runs (load ~18):
1. Start runs the pre-check; its result adds the `generic` type group, a structural change, so `_refresh_library_ingest_canvas_preserving_context` rebuilds the screen with the path and title still in the form.
2. The readiness worker's result arrives as an asyncio callback during that rebuild's mount. It submits and clears path and title (`_enqueue_library_ingest_snapshot`).
3. The rebuilt `Input`s announce the values they were composed with (Textual posts a `Changed` for a non-empty `value=`). One arrived 0.6 s later, from an Input already being removed: `PATH Changed value='…/tides.txt' form='' … attached=False pruning=True`.
4. The screen's echo guard drops a `Changed` only when it equals the form's current value, now empty. So it took both echoes as edits: path and title went back into the form, the path's debounce re-ran the pre-check (now `already_in_library=1`), and the next rebuild composed the restored values. The form never cleared.

Forced order (probe, not committed): hold the screen's recompose after the structural rebuild, submit inside the hold, release. Base: 8 of 8 failed with the CI shape. Fixed: 8 of 8 passed.

**Fix (`Widgets/Library/library_ingest_canvas.py`).** The canvas composes path, title, author and keywords through `_form_field`, which records the value each one was composed with, and `_stop_form_field_echo` stops a field's first `Changed` when it equals that value: its mount echo, whenever it arrives. Only the first message is checked: Clear empties the path without a message, so typing the same path again must still count. A first cut that kept a "last value rendered or passed on" record, like the option fields' rule, failed exactly that case.

**Red/green.** `Tests/UI/test_library_ingest_readiness.py::test_a_rebuilds_echo_of_its_rendered_form_never_reaches_the_screen` (a whole-file target of "Run admission-sensitive suites 2"): it rebuilds a canvas holding all four values, then makes one edit per field (the path after a silent Clear, back to the same path) and checks that exactly the four edits reach the screen. Base: 3 of 3 red (`('title', 'Tides 101') != ('title', 'edited')`, the echoes leaking). Record-based cut: 3 of 3 red ("The edits never arrived", the re-typed path stopped). Fixed: 3 of 3 green. Lane time: call 2.99 s (`-p no:xdist --timeout=300`, load ~54).

**The retry node.** Fixed in TASK-42473's commit (its focus change exposed it): the test now waits for the Media entry window to close before its code-driven Retry press. 40 of 40 at `-n 4` (load 50-62).

**Runs on the final tree (AC#2).**
- Both nodes plus the new test, 20 runs at `-n 4`: 60 passed (load 40-42).
- The push shards that hold them, serially with the job's flags (`--timeout=180 --durations=0`): shard 12/14 (the ingest node) 62 passed, and shard 3/14 (the retry node) 62 passed, side by side at load 40-60.
- Earlier, before the retry fix, the ingest node passed 20 of 20 at `-n 4` beside the retry node (load ~40). Shard 12 passed serially at base too (62, load ~28): the race did not reproduce locally without the probe or a forced order.

**Lesson.** Added to `backlog/docs/lessons-textual.md`.

**Files.** `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`, `Tests/UI/test_library_ingest_readiness.py`, `backlog/docs/lessons-textual.md` (and `Tests/UI/test_library_shell.py` in TASK-42473's commit).

**Checks (final tree, shared by TASK-41723, 42470 and 42473).** `Tests/CI` at `-n 4`: 1823 passed, 2 skipped. `Tests/Architecture` serially: 1491 passed, 5 skipped, 1 failed. The failure was `test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_canonical_schema_and_arithmetic`, at its 300 s timeout while the run shared the host at load 33-67. Alone it passed in 194 s (load ~45-50). `./scripts/preflight.sh` passed; `ruff check tldw_chatbook --select F` passed; `Tests/lint/ruff_gate.py` passed (RATCHET OK).
