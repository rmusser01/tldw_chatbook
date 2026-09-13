---
id: TASK-32534
title: >-
  Library Notes: lasting sync shows "✓ Up to date" beside "Manual check failed",
  writes a Chatbook edit to disk with no receipt, and never pulls a disk edit
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-13 06:45'
updated_date: '2026-09-13 15:15'
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
- [ ] #1 A failed manual check flips the root row to a failed or needs-attention state that names the exception category and a next action; "✓ Up to date" is never rendered beside a failure status line
- [ ] #2 sync_now records the failure in the log with the exception type and reason code (no path), the way the setup Check does since task-32243
- [ ] #3 Every note→file write performed by lasting sync leaves a receipt row visible in Manage sync folders (what was written, when, from which note)
- [ ] #4 A file edited on disk after activation surfaces on the next Check as changes available or needs attention and its text reaches the note after review, or the row states why it cannot
- [ ] #5 Walked live at 235x52 on a $HOME vault: Chatbook edit → disk edit → Check → row state, receipt and note version captured; notes.md stamp updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a fresh scratch profile + $HOME git vault (Chatbook edit -> disk write with no receipt; disk append -> Check -> Up to date; second Check -> Manual check failed beside Up to date) and headlessly (check_root after a disk append; second check_root exception type/token).
2. RED: controller test (failed request_sync_now flips the row to Needs attention with a reason + next action, logs error_type/reason_code metadata only, never renders Up to date beside a failure); receipts test (an automatic UPDATE_FILE leaves a receipt row naming path/title/effect/time); store test (list_completed_operations newest-first bounded); runtime test (a disk edit after activation surfaces on the next check; a second check does not raise).
3. Fix: sync_now and the three sibling bare excepts route through check_failure_line (metadata-only diagnostic) and record a per-root failure overlay projected by refresh_roots; extend the reason-code copy map; check_root publishes a status for its pre-reconcile refusals; store.list_completed_operations + a write_receipts projection rendered as a Receipts section in the roots canvas; fix the disk-edit cause where the probe found it.
4. Inventory re-pin, full chunked test comparison vs a detached origin/dev baseline, live walk at 235x52 and 100x30 with captures, guide notes.md body + stamp.
<!-- SECTION:PLAN:END -->
