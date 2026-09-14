---
id: TASK-32534
title: >-
  Library Notes: lasting sync shows "✓ Up to date" beside "Manual check failed",
  writes a Chatbook edit to disk with no receipt, and never pulls a disk edit
status: To Do
assignee: []
created_date: '2026-09-13 06:45'
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed manual check flips the root row to a failed or needs-attention state that names the exception category and a next action; "✓ Up to date" is never rendered beside a failure status line
- [ ] #2 sync_now records the failure in the log with the exception type and reason code (no path), the way the setup Check does since task-32243
- [ ] #3 Every note→file write performed by lasting sync leaves a receipt row visible in Manage sync folders (what was written, when, from which note)
- [ ] #4 A file edited on disk after activation surfaces on the next Check as changes available or needs attention and its text reaches the note after review, or the row states why it cannot
- [ ] #5 Walked live at 235x52 on a $HOME vault: Chatbook edit → disk edit → Check → row state, receipt and note version captured; notes.md stamp updated
<!-- AC:END -->
