---
id: TASK-32534
title: >-
  Library Notes: lasting sync shows "✓ Up to date" beside "Manual check failed",
  writes a Chatbook edit to disk with no receipt, and never pulls a disk edit
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:45'
updated_date: '2026-09-14 16:28'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A, persona Alex / solo operator, Keep-a-folder-synced workflow on a $HOME vault. P1. B did not reach this leg (NOT-EXERCISED).

**What happened.** Root activated ("60 applied · durable receipt recorded", A 65). Edited the synced note "Sam" in Chatbook: after 5 s `People/Sam.md` on disk carried the Chatbook line (`git status` → `M`) with no receipt row anywhere (A 68 + cat). Appended a line to `Daily/2026-09-06.md` on disk → Manage sync folders → Check changes → still "✓ Up to date · Next: Check changes" (A 71) → Check changes again → "Manual check failed. Review root status, then try again." while the root row still reads "✓ Up to date" (A 73). sqlite: the note "2026-09-06" stayed at version 1 without the disk text. Captures: A 68–73.

**Cause.** The copy is PROVEN: `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py:1157-1165` — `sync_now` wraps `self._runtime.request_sync_now(root_id)` in a bare `except Exception:`, sets the generic status line, returns to the `roots` phase and leaves the row's last-known status in place; nothing is logged. This is a different method from the setup-time Check that task-32243 fixed (`check_setup` / `_check…` name their reason; `sync_now` does not). The note→file write without a receipt and the non-pickup of the disk edit are INFERRED (not traced). Not a wave-3 regression: the path was unreachable before 32243 made lasting sync enterable; tasks 32518/32519 fixed activation refresh and Resume, not this.

Adjacent open task: 32451 (the root row's placeholder name). Docs contradicted: notes.md says a root with nothing changed returns to "✓ Up to date" and edits surface as "◌ Changes available" / "⚠ Needs attention".
**Wave-4 probe (2026-09-13, fresh profile + git vault at 235x52, plus a headless runtime built as `Tests/UI/test_library_notes_files_sync_journey.py::_start_real_conflict_stack` does).** Live: after activation a Chatbook edit to "Sam" is NOT written to disk on its own (the polling watcher hints only on a disk-side change); the first disk append then triggers one automatic pass that writes `People/Sam.md` AND pulls `Daily/2026-09-06.md` into the note (version 2) with no receipt anywhere (`wave4-caps/sync-roots/roots-00-no-receipt.txt`) -- AC#3 PROVEN. A second disk append followed by an immediate **Check changes** leaves the note at version 2 forever while the row reads "✓ Up to date" (`roots-03-race-check.txt`, `roots-04-race-second-check.txt`) -- AC#4 PROVEN; headless (`probe-ac4-manual.txt`, `probe-ac4-race.txt`): `check_root` right after the append returns an UPDATE_NOTE plan, but `_ProductionRuntimeAdapter.observe_root` (`notes_sync_runtime.py:786`) overwrites `_root_signatures[root_id]` -- the watcher's change baseline -- so `changed_root_ids` never reports the root again and the automatic pass that would apply the edit never runs; with the manual check landing after the watcher's poll (>=0.6 s) the edit is applied normally. Neither observation-reuse validation nor the baseline advance is involved (the manual plan is correct). The second Check did not raise on this walk; `sync_now`'s bare except (AC#1/#2) stays PROVEN from code -- no reason, no log, no row re-projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed manual check flips the root row to a failed or needs-attention state that names the exception category and a next action; "✓ Up to date" is never rendered beside a failure status line
- [x] #2 sync_now records the failure in the log with the exception type and reason code (no path), the way the setup Check does since task-32243
- [x] #3 Every note→file write performed by lasting sync leaves a receipt row visible in Manage sync folders (what was written, when, from which note)
- [x] #4 A file edited on disk after activation surfaces on the next Check as changes available or needs attention and its text reaches the note after review, or the row states why it cannot
- [x] #5 Walked live at 235x52 on a $HOME vault: Chatbook edit → disk edit → Check → row state, receipt and note version captured; notes.md stamp updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a fresh scratch profile + $HOME git vault (Chatbook edit -> disk write with no receipt; disk append -> Check -> Up to date; second Check -> Manual check failed beside Up to date) and headlessly (check_root after a disk append; second check_root exception type/token).
2. RED: controller test (failed request_sync_now flips the row to Needs attention with a reason + next action, logs error_type/reason_code metadata only, never renders Up to date beside a failure); receipts test (an automatic UPDATE_FILE leaves a receipt row naming path/title/effect/time); store test (list_completed_operations newest-first bounded); runtime test (a disk edit after activation surfaces on the next check; a second check does not raise).
3. Fix: sync_now and the three sibling bare excepts route through check_failure_line (metadata-only diagnostic) and record a per-root failure overlay projected by refresh_roots; extend the reason-code copy map; check_root publishes a status for its pre-reconcile refusals; store.list_completed_operations + a write_receipts projection rendered as a Receipts section in the roots canvas; fix the disk-edit cause where the probe found it.
4. Inventory re-pin, full chunked test comparison vs a detached origin/dev baseline, live walk at 235x52 and 100x30 with captures, guide notes.md body + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:BEGIN -->
All five ACs live at 235x52 and 100x30 on a scratch profile + a 60-file git vault under $HOME.

**AC#1/#2 — a refused action names itself.** The four bare `except Exception` branches on `LibraryNotesSyncController` (sync_now, apply_reviewed, resolve_cleanup, _run_root_control) set a generic line, left the row at its last projection and logged nothing, so "Manual check failed" stood beside "✓ Up to date". Each now routes through `check_failure_row`, which logs the refusal as metadata (`reason=` / `error_type=` / `root_id=`, never the message) and records a per-root overlay that `refresh_roots` projects: "⚠ Needs attention · <verb> failed — <reason> · Next: <action>". The next action you run on that root clears it.

Two things only the live walk found. `_refusal_reason` filters on `_CHECK_REFUSAL_COPY`, so codes present only in `_CHECK_FAILURE_ROW` were dead -- and a Check on a paused root raises `root_admission_closed` (pause_root closes admission), not `sync_root_not_active`, so the row read "Check failed — RuntimeError · Next: Check changes": the category, pointing at the control that had just refused. And because the overlay rewrites the row's status, the canvas's status-keyed button set offered Pause and Recovery beside a row reading "Next: Resume" -- with no Resume. The named action now drives the control.

**AC#3 — receipts.** `NotesDeviceStateStore.list_completed_operations` reads the journal's completed rows; `NotesSyncRuntimeOwner.write_receipts` labels them through `binding_labels` (wave-4 Task 2's seam, copied here until it lands -- drop the copy at merge if theirs is on dev); the controller projects them and a "Receipts" section on Manage sync folders lists the newest 20 across every listed root as "when · effect · path · note".

**AC#4 — a disk edit reaches the note.** `_ProductionRuntimeAdapter.observe_root` overwrote `_root_signatures[root_id]` -- the baseline `changed_root_ids` compares against -- so a manual Check landing before the watcher's poll consumed the disk change: the automatic pass that applies it never ran and the note stayed stale behind an unchanged row. `changed_root_ids` owns that advance; `observe_root` now only seeds it. Neither observation-reuse validation nor plan correctness was involved (the manual plan was always right).

**AC#5 — live.** Captures under `wave4-caps/sync-roots/`: `roots-11-disk-edit-review` (both receipt rows), `roots-12-note-version` (sqlite: both notes at version 2 carrying the other side's text; `git -C vault status` shows both files modified), `roots-14-race-check-applied`, `roots-13-failed-row` (+ `-before-reasoncode-fix`), `roots-18-second-root-activated`, `roots-19-two-roots-receipts`, `roots-16/17` at 100x30.

Tests: `Tests/UI/Library_Modules/test_library_notes_sync_controller.py` (failed row + log category, receipts projection, paused-root reason, two-table key invariant, overlay clearing, activation receipt line), `Tests/Notes/test_notes_sync_runtime.py` (disk edit after activation, write_receipts labels), `Tests/Notes/test_notes_device_state_store.py` (completed operations newest-first bounded), `Tests/UI/test_library_notes_w4_sync_roots.py` (failed row offers the control it names). Sibling pins asserting the old copy were updated to the new strings. Guide `Docs/User_Guide/library/notes.md` gained the failed-check paragraph, the Receipts paragraph and a verification stamp.
<!-- SECTION:NOTES:END -->

## Fix round 1 (review 2026-09-14)

Four of the five review Importants land here; the fifth is 32545's.

- **Duplicate dict keys (finding 1).** Three entries added to `_CHECK_REFUSAL_COPY` were already defined above it, and Python keeps the last literal -- `root_lease_unavailable` lost "Close any other Chatbook window using it" and started pointing at a Reconnect that cannot help. Originals restored; the pin asserts the SOURCE via an `ast` scan that fails on any repeated constant key in the module, because the dict object cannot show a duplicate that is already gone.
- **The overlay overwrote `status` (finding 2).** The canvas reads `status` for `check_blocked` and Pause suppression, so an offline root whose Check was lease-refused came back with an enabled Check and a Pause. The row now carries `failed_action` beside a truthful `status`; only the labels are overlaid, and the canvas prefers `failed_action` for the button set. Pinned through the controller with the real `NotesSyncRootRefused("root_lease_unavailable", reason_code="root_offline")`.
- **A paused root blanked the Receipts section (finding 3).** `write_receipts` reached `_admit_task`, which `pause_root` closes. Fixed at the cause -- receipts are a read, so `_read_binding_labels` takes the `_register_task` path while still requiring cutover admission -- plus the per-root guard the review asked for. Task 2's `binding_labels` seam is untouched. Pinned on the real runtime.
- **The overlay outlived its route (finding 4).** `resolve_cleanup` and `apply_reviewed` success now clear it; the pin is parametrised over all four routes. The live re-walk found the inverse: an accepted Resume cleared the row but left the failure sentence as the status line, so a cleared refusal now restates the row's own labels.

Live (235x52, same profile + vault): `roots-30-paused-root-receipts-survive`, `roots-31-paused-check-failed-row`, `roots-32-resume-clears-overlay`. The offline case is pinned, not walked -- a missing folder does not make the runtime report `offline`; that needs a second process holding the lease.

Cross-group check (coordinator's file-notes finding): every user-visible string this round adds, removes or rewords was grepped against the whole `Tests/` tree, and the FAILED-name SET was compared over full files against a detached `origin/dev`. That found one branch-red/dev-green name -- `test_import_back_retains_canvas_and_shows_truthful_lasting_availability`, which asserted `"unavailable" in status_line.casefold()`, a proxy any rewording breaks silently. Tightened to the exact sentence.
<!-- SECTION:NOTES:END -->
