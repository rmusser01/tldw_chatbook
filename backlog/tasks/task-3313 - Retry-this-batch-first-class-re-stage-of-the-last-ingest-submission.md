---
id: TASK-3313
title: 'Retry this batch: first-class re-stage of the last ingest submission'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-08 20:30'
updated_date: '2026-08-09 15:49'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved by the owner via task-3310 (ruling 3). After Start, the ingest form auto-clears to invite the next source — but the likeliest next action after a failure (or after installing the dependency a warning just named) is the SAME source again, and today that means re-typing/re-browsing; per-row Retry is buried in the queue. Alex-persona flag from the 2026-08-07 critique, re-confirmed by the arc's live verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a submission reaches a terminal state, a single visible action re-stages that submission's source (path/URL) with its options and metadata restored to the form
- [x] #2 The action is keyboard-reachable and advertised in the ingest shortcut set
- [x] #3 Re-staging runs a fresh preflight (tooling installed since the last run is picked up; the old forecast is not reused)
- [x] #4 The affordance survives the in-place update discipline (object-identity test across queue ticks) and appears only when a last submission exists
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. State layer (`library_ingest_state.py`): add `show_retry_last: bool = False` to
   `LibraryIngestCanvasState` and a `last_submission_available: bool = False` kwarg to
   `build_library_ingest_state`. Visible only when a snapshot exists AND no job is active
   (QUEUED/PARSING/WRITING) — the affordance appears when the batch has settled, matching
   AC#1's "after a terminal state" framing while never showing without a last submission.
2. Screen (`library_screen.py`): capture a session-scoped snapshot
   (`self._library_ingest_last_submission`) inside `_do_submit_ingest`, after the submit
   call and BEFORE the form clears: resolved source, title/author/keywords (raw text),
   analyze/chunk/chunk_size, and a per-group copy of `type_options`. Session-scoped by
   design (recorded choice): the jobs DB persists sources but not the form's staged
   options/metadata, so a durable snapshot would need new storage — out of scope here.
3. Canvas (`library_ingest_canvas.py`): an always-mounted, display-managed compact button
   `#library-ingest-retry-last` after the queue panel (the queue outcome area) — NEVER
   conditionally composed and NOT inside the recomposing queue panel, so it keeps object
   identity across ticks. `_update_library_ingest_dynamic_regions` owns its visibility.
4. Press handler: restore all snapshot fields into the form, invalidate the old preflight
   (the stale forecast must never be reused), context-preserving recompose (the form
   widgets re-render from state), then `_trigger_library_ingest_preflight(source)` for a
   FRESH forecast, and focus the path field.
5. Keyboard: `Binding("r", "library_ingest_retry_last", show=False)` + `check_action`
   gate (Ingest canvas + snapshot exists; audited by the bindings-gate test) + an
   `("r", "retry last batch")` row in `LIBRARY_INGEST_SHORTCUTS` (footer + F1).
6. Tests: builder visibility matrix; restore fidelity (mutation: drop the options restore
   → RED); fresh-preflight assertion by toggling the patched analyzer between runs
   (mutation: drop the trigger → RED); object identity across dynamic-region ticks;
   keyboard reachability + advertisement.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Shipped**: a "Retry this batch" button below the ingest queue (with keyboard
accelerator `r`) that re-stages the last submitted batch — source, title, author,
keywords, analyze/chunk/chunk_size, per-type options — back into the form and runs a
FRESH pre-flight.

- **State carrier**: `LibraryIngestLastSubmission` (frozen dataclass in
  `library_ingest_state.py`), captured in `_do_submit_ingest` AFTER the submit call and
  BEFORE the form auto-clears, values copied not aliased. **Session-scoped by choice**
  (recorded per the task's option): the jobs DB persists sources but not the staged
  options/metadata, so a durable snapshot would need new storage; after a restart the
  affordance simply stays hidden. Deliberately NOT cleared by rail re-entry — it is
  submission history, not form state.
- **Visibility rule**: `show_retry_last` = snapshot exists AND no QUEUED/PARSING/WRITING
  job (the queue has settled) — satisfies AC#1's "after a terminal state" without ever
  appearing without a last submission (AC#4), and avoids inviting a duplicate batch
  mid-run.
- **Placement/discipline**: canvas-level, always-mounted, display-managed button
  (`#library-ingest-retry-last`) after the queue panel — deliberately OUTSIDE the
  recomposing `LibraryIngestQueuePanel` so it keeps object identity across job ticks;
  `_update_library_ingest_dynamic_regions` owns its visibility in place.
- **Re-stage flow** (`_restage_library_ingest_last_submission`): restore form →
  invalidate the old pre-flight (the stale forecast is never reused) →
  context-preserving recompose → `_trigger_library_ingest_preflight(source)` (the same
  seam typing/blur uses, so the gate updates ride the normal pipeline) → focus the path
  field.
- **Keyboard**: `Binding("r", …, show=False)` + `check_action` gate (Ingest canvas AND
  snapshot present — audited by `test_library_screen_bindings_are_all_gated_or_universal`)
  + `("r", "retry last batch")` in `LIBRARY_INGEST_SHORTCUTS` (footer + F1). A focused
  Input consumes the printable key first, so `r` still types in text fields (pinned).
- **Tests** (`Tests/UI/test_library_ingest_retry_last.py`, 10 tests): builder visibility
  matrix; snapshot capture + copy semantics; restore fidelity with the form CORRUPTED
  between submit and retry; fresh-forecast assertion by flipping the patched analyzer
  between runs; object identity across dynamic-region ticks incl. display flip with an
  active job; `r` re-stages from non-text focus / types inside the path field;
  check_action gate; shortcut-set advertisement.
- **Mutation evidence**: dropping the options restore → RED
  (`assert 'parakeet-onnx' == 'faster-whisper'`) — the FIRST version of that test was
  vacuous (options persist across submits by design), caught by the mutation check and
  fixed by corrupting the form first; dropping the fresh-preflight trigger → RED
  ("fresh pre-flight never ran after re-stage") — initially masked by the harness's own
  0.8s typing debounce re-running the analysis at test speed (found via call-stack spy),
  neutralized in the harness. Both incidents recorded in
  `backlog/docs/lessons-testing-evidence.md`.
- **Files**: `tldw_chatbook/Library/library_ingest_state.py` (snapshot dataclass,
  `show_retry_last` + `last_submission_available`),
  `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (button),
  `tldw_chatbook/UI/Screens/library_screen.py` (capture, restage, binding, gate,
  display management), `Docs/User_Guide/library/import-and-export.md` (control row,
  keyboard section, fix-a-warning task, stamp),
  `Tests/UI/test_library_ingest_retry_last.py` (new).
- **Verification**: consolidated ingest battery 644 passed (16 files incl. the new
  suites); nav bindings-audit subset 10 passed; `test_library_shell -k ingest` 28
  passed / 14 failed — all 14 verified as task-3315's pre-existing
  `_ingest_local_stt_jobs` harness drift (each failure log carries that AttributeError;
  the raise point is inside `submit(...)`, upstream of this task's code).
xhigh review + live-verify round (2026-08-09): two defects.
(a) Gate divergence. `check_action("library_ingest_retry_last")` stated its own copy of the
visibility rule with the settled-queue half MISSING, so mid-run -- exactly while the button is
deliberately hidden to prevent a duplicate batch -- the `r` key stayed live. Fixed by extracting the
ONE predicate, `library_ingest_retry_available(jobs, last_submission_available=...)` in
tldw_chatbook/Library/library_ingest_state.py; `build_library_ingest_state` derives `show_retry_last`
from it and `check_action` calls it directly. Duplicating the condition was the defect, so the test
asserts builder and gate agree on both sides of the same job snapshot. Mutation check (feed
check_action an empty job list) sends it RED.
(b) No consent on a destructive overwrite. A re-stage replaces path/title/author/keywords/options
wholesale with no undo, and `r` fires it from any non-text focus. It now uses the incumbent two-press
grammar (Clear-finished, Start consent) WHEN the re-stage would discard work -- i.e. when a non-empty
form field or option value differs from what the snapshot would put back. Right after a submit
nothing differs (path/title are cleared and the rest IS the snapshot), so the common case still
re-stages on one press. Both routes share `_restage_library_ingest_last_submission`, so the key can
never be the looser of the two, and the affordance's own LABEL carries the pending state
("Press again to replace form") because the `r` route has no gate line of its own -- rendered from
the new `LibraryIngestCanvasState.retry_confirm_armed` and synced in place by the dynamic-region
updater. `test_retry_last_restores_the_form_and_runs_a_fresh_preflight` deliberately dirties the form
before pressing, so it was updated to press twice; the two consent legs have their own tests.
<!-- SECTION:NOTES:END -->
