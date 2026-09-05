---
id: TASK-31220
title: Media storage - root-cause the session-progressive silent degradation
status: Done
assignee:
  - '@claude'
created_date: '2026-09-03 22:30'
updated_date: '2026-09-05 11:37'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique 2026-09-03 P0 (user: full root-cause hunt): in a ~25-min live session, a completed review set never durably persisted (UI 'All 6 reviewed' but DB done=0/active=1/completed_at=NULL), the Sets picker failed twice then went silently dead, Read later persisted nothing, and a plain row click stuck on 'Media item is unavailable.' with dead Retry - spanning BOTH DBs. The picker worker's except Exception is the one review-set wrapper without traceback logging, so no trace exists. Lead: stuck DB thread w/ open transaction exhausting the asyncio.to_thread pool; CancelledError (BaseException) explains the silent-dead phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The picker worker's exception path logs the traceback
- [x] #2 The root cause of the wedge is identified with reproduction evidence
- [x] #3 A durable fix lands with a regression pin, and displayed state can no longer silently diverge from durable state
- [x] #4 One storage-health surface tells the user when local storage is unhealthy
- [x] #5 The bulk-delete receipt is derived from the service result: a write that did not land renders as a failure with a reason, never as ✓
- [x] #6 The bulk-mutation interlock is released on every path, and Retry, row opens and select mode are never LEFT gated behind it once a mutation ends (they still wait while one is genuinely in flight)
- [x] #7 Undo is enabled whenever the receipt says ✓, and the receipt never says ✓ when Undo cannot be
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 1: instrumented reproduction on a real MediaDatabase through the real seam (fresh-reader visibility, long-lived-reader staleness, contended write) — facts before any fix; no production change unless the fresh reader missed the write (it did not).
2. Task 2: the wedge — one _claim_library_media_mutation seam guarding the interlock claim at all six sites (release on any raise before the worker starts), Retry paints `Couldn't retry · <reason>`, rows open under a mutation-only gate; retire the pins that encoded the wedged behaviour.
3. Task 3: receipt/Undo consistency — Undo exempt from the stale gate, `✗ undo failed · n of m · <reason>` with Retry undo, focus on Undo after a full-success delete via the completion seam's focus channel.
4. SDD per task (review + scoped re-review), final whole-branch review + one fix round, single-instance live verification, PR E.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause of the storage wedge (this task's "session-progressive silent degradation" and critique #5's P0 aftermath): six sites claimed `_library_media_bulk_delete_in_flight = True` and every worker released it in a `finally`, but none guarded the claim itself — any raise between the claim and `run_worker` (fence, scheduling) left the flag stuck True, gating Retry, row opens and select mode until restart. Fixed by routing all six claims through `_claim_library_media_mutation`, which releases on failure; a source-invariant test fails on any seventh hand-written claim. ONE flag, ONE worker group (ADR-055) kept. Retry never silently no-ops (`Couldn't retry · timed out|<os/sqlite message>|<class>`; renamed from `Retry failed` at the final review because that is the Analyze receipt's button label), rows open read-only under `Media changed; retry…`, the bulk-delete AND review-set receipts' Undo are exempt from the stale gate, and the bulk-delete Undo takes focus after a full-success delete (Enter undoes); a failed undo paints `✗ undo failed · n of m · <reason>` with Retry undo.
What Task 1 established on a real DB: the single-instance trash write is visible to a fresh SQLite reader at receipt time; a long-lived reader is NOT stale (legacy isolation level, bare SELECT opens no snapshot); a contended write raises immediately, paints no ✓, notifies, releases the interlock. The critique-#5 `✓` over an untouched row therefore stays unexplained for the single-instance case; it occurred only with two app instances on one profile (the app's guard is advisory), and the surviving hypotheses (wrong-row via a stale retained id; a different DB file) are recorded above under Reproduction evidence.
AC wording amended at close: "Retry, row opens and select mode are never LEFT gated after a mutation ends" — while a mutation is genuinely in flight they still wait (correct).
Trade-offs: cancel-before-first-step can leak the flag at teardown only (commented at the seam); the rail's "Media N" count is stale after an Undo (pre-existing, filed as a rider).
Verification: per-file suites vs base in separate processes; whole-file test_library_shell.py and test_library_media_side_by_side.py compared to the task-31249 census; preflight green; live single-instance delete/undo at 235x52 and 100x30; failed-undo and failed-retry receipts in app-tests with raising seams.
Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py, tldw_chatbook/Widgets/Library/library_media_canvas.py, tldw_chatbook/Library/library_media_state.py, Tests/UI/test_library_media_bulk_delete_real_db.py (new), Tests/UI/test_library_multiselect_media.py, Tests/UI/test_library_media_browse_controller.py, Tests/UI/test_library_media_side_by_side.py, Tests/UI/test_library_shell.py, Tests/UI/test_library_media_render_fixes.py, Docs/security/production-diagnostic-inventory.json.
<!-- SECTION:NOTES:END -->

## Renumbering

Renumbered from task-31202 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).

## Reproduction evidence (critique #5, 2026-09-04 evening, dev c09717a7cb)

Reproduced twice, independently, by two assessment agents running the app under tmux at 235x52 against the real profile — with a second app instance alive on the same profile (the app's own "Another copy of tldw is already using this profile" guard fired and was bypassed) and the host out of POSIX semaphores. Sequence: select mode → select one item → Delete → confirm → receipt `✓ deleted · 1 item · in Trash` with `○ Undo` and `Media changed; retry to load a current page.`; every row, Export, Select, sort and the pager disabled; Retry inert (two clicks, 3 s apart, byte-identical captures); Dismiss clears the receipt but not the gate; `s` inert; a Media → Notes → Media round-trip does not clear it; the DB row is untouched (`is_trash=0, deleted=0`); only killing the process recovers.

Mechanism traced in source — CORRECTED the same evening: `_delete_library_media_selection` treats "no exception" as success, which is sound for the local backend (`local_media_reading_service.delete_media_item` raises `ValueError` when `mark_as_trash` returns `False`, `KeyError` when the row is missing; `mark_as_trash` commits inside its transaction before returning `True`). A `✓` over an untouched row is therefore NOT explained by the code read. Open candidates: the assessment's own DB read on a stale WAL snapshot (long-lived `MediaDatabase` connection opened before the app's commit); the app writing through a different connection/path; a later revert. The profile-lock guard is advisory only (`_maybe_warn_second_instance`: "Everything keeps working"), so both instances shared one DB. First step of the fix PR: an instrumented reproduction (real DB, fresh read connection after the receipt) before any code change. `_complete_library_media_mutation` raises the mutation gate through `reconcile_committed_mutation` and refreshes only while the screen holds authority. `handle_library_media_retry`, `handle_library_media_row` and `_toggle_library_media_select_mode` all early-return while `_library_media_bulk_delete_in_flight` is set — exactly the three controls observed inert — so the flag not being released on that path explains the wedge. Undo is stale-gated by design, so a `✓` with a dead Undo is what the code does whenever the gate is up.

Fix shape: derive the receipt from the result (`False` = failure with a reason); release the interlock in a `finally` on every path; never gate Retry behind the interlock it exists to escape; enable Undo iff the receipt says `✓`. Snapshot: `.impeccable/critique/2026-09-05T06-05-33Z__tldw-chatbook-ui-screens-library-screen-py.md`.
