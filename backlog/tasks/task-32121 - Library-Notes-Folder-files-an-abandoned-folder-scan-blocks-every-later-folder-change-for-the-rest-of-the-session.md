---
id: TASK-32121
title: >-
  Library Notes Folder files: an abandoned folder scan blocks every later folder
  change for the rest of the session
status: In Progress
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 05:49'
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
