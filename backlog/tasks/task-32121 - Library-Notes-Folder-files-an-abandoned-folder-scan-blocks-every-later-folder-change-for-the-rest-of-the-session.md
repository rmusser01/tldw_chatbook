---
id: TASK-32121
title: >-
  Library Notes Folder files: an abandoned folder scan blocks every later folder
  change for the rest of the session
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:35'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - p0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN live x3 on a clean sequence: a small folder links in under 2 s; picking the home directory times out at 30 s; the next small folder then sits on 'Changing folder… · still working' and times out too, and so does every pick after it. Cause: `_change_root_with_deadline` (library_file_notes_workspace.py) cancels only the asyncio task, while `set_root` runs `service.scan` in `asyncio.to_thread` under `operation_lock=self._service_lock`; the abandoned scan thread keeps running and the next `set_root` waits behind it. The timeout copy `ROOT_CHANGE_TIMEOUT_COPY` never painted at 0.5 s sampling, so the failure is silent. Both assessors hit it through task-32122 (the picker hands over the browsed directory, which opens at the home folder). The busy row also paints two buttons labelled Cancel. Critique #8's task-32055 added the deadline; this is the half it did not cover. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancelling or timing out a folder change stops or isolates the scan so that a later change to a different folder completes in its normal time within the same session
- [x] #2 After a timed-out change, choosing a small folder links within 5 s with no restart
- [x] #3 The timeout copy ('Folder change timed out · previous folder kept…') is painted and stays visible until the next action
- [x] #4 A scan still running after about 3 s reports progress (entries seen so far) and offers 'Keep waiting' or 'Choose another folder' rather than a bare still-working line
- [x] #5 The busy row never shows two controls labelled Cancel
- [x] #6 Covered by a test where the first scan blocks past the deadline and a second root change on another folder still succeeds
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: an abandoned scan must not block the next folder change (fake service, monkeypatched 0.4s deadline).
2. Add ScanCancelled + should_cancel/on_progress to FileNotesService.scan (per-directory and per-file checks).
3. Per-change threading.Event set by _abandon_root_change_task; bounded service-lock acquire in the scan thread.
4. Failing test: the timeout copy paints and survives the final repaint; render the reason in the root row too.
5. Progress + Keep waiting / Choose another after the patience window; one Cancel control in the busy row.
6. Live-verify on the power profile; docs stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (proven live x3): `_change_root_with_deadline` cancelled only the asyncio task, while `set_root` ran `service.scan` in `asyncio.to_thread` under `operation_lock=self._service_lock`. `scan` is `@_serialized`, so the abandoned thread held that lock for the whole home directory and every later `set_root` queued behind it.

Approach -- fix at the shared seam, not the one path:
- `FileNotesService.scan(*, should_cancel, on_progress)` + new `ScanCancelled`. The cancel check runs between directories in `_walk_candidates` (the only place an `os.walk` can be stopped) and between files in the load loop; raising unwinds the `@_serialized` `with`, releasing the lock. `reconcile()` unchanged by default.
- One `threading.Event` per root change, set by `_abandon_root_change_task` -- the seam the deadline, Cancel, Escape, the back cue and the navigation flush already share.
- `_scan_for_root` acquires the service lock with a bounded wait, so a scan parked in one uninterruptible syscall (dead network mount) makes the NEXT change report the timeout copy instead of hanging silently.
- The timeout reason now owns the FOLDER ROW until the next attempt (`_report_root_change_reason` -> `_root_action_reason`). It used to be written only to the action-status line at the bottom of the editor pane while the row reverted to the kept folder -- why 0.5 s live sampling never caught it.
- Slow scans report `Changing folder… · 1,240 entries so far` (patience timer is now `set_interval`) and offer Keep waiting (one extra budget; `_change_root_with_deadline` loops on `asyncio.wait`) and Choose another (abandon + reopen the picker).
- AC5: the shared `StructuralWait` line's trailing `· Cancel` sat beside the actual Cancel button -- two controls with the same label in one busy row. This surface renders its own line instead; `StructuralWait` is untouched, so export/skill-import keep their copy.

Trade-off: the busy line does NOT repeat the two button labels the brief suggested -- measured, the row leaves the status 46 cells at 120 columns and that line elides to 'Changing folder… · sti...ting · Choose another'.

Tests: `Tests/UI/test_library_crit8_waits.py` gains a `_LockHoldingScan` fixture that holds the lock exactly like `@_serialized` (the older fake replaced the decorated method and never did, which is why this survived crit8) plus five tests; `Tests/Notes/test_file_notes_service.py` gains cancel/lock-release and progress tests. Seven existing scan fakes now take `**kwargs`.

Live (power profile, 235x52): home directory timed out with the copy painted and held, then the vault linked in ~3 s -- the sequence that used to wedge. Captures in SCRATCH/notes-crit/wave/file-notes/caps/04-08.

Known, deferred (review round 1): below 120 columns the folder row cannot fit the status line plus Cancel/Keep waiting/Choose another -- at 120 the status has 46 cells and the three buttons take 48. The buttons clip rather than wrap; no capture or test covers that width yet.

Files: tldw_chatbook/Notes/file_notes_service.py, tldw_chatbook/Widgets/Library/library_file_notes_workspace.py, Tests/UI/test_library_crit8_waits.py, Tests/Notes/test_file_notes_service.py, Docs/User_Guide/library/file-notes.md.
<!-- SECTION:NOTES:END -->
