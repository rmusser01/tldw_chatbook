---
id: TASK-32519
title: "Lasting sync: Resume after Pause always lands in ✕ Failed and leaves the root paused"
status: To Do
assignee: []
created_date: '2026-09-13 00:35'
updated_date: '2026-09-13 03:40'
labels:
  - library
  - notes
  - rider
dependencies:
  - TASK-32269
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Pause** on an active lasting-sync root works, but **Resume** never brings
the root back — with nothing edited on either side, on a root that was
"✓ Up to date" a moment earlier. The row goes to "✕ Failed · Next: Review
changes" with the status line "Action needs attention. Review settings, then
Check changes."; every **Check changes** after that reports "Manual check
failed. Review root status, then try again."; and **Review** opens a review
that says "That folder is paused. Resume it, then Check again." with
"0 safe · 0 need attention" and a stale flag — a loop with no exit. **A
restart does not recover it**: the row comes back as "Ⅱ Paused · Next:
Resume" (the ✕ Failed status is not persisted), **Check changes** while
paused fails ("Manual check failed"; `check_root` raises
`sync_root_not_active`), **Resume** fails identically
(`binding_review_required`), and a file edited on disk after the restart is
never synced. The stored state is root `paused` with all 60 bindings
`paused` (`tldw_chatbook_notes_sync_state.db`). Disconnect is disabled in
this release, so a paused root stays paused for good.

**Cause (proven, dev 7159fc0b99).** `NotesSyncRuntimeOwner.pause_root`
(`tldw_chatbook/Notes/notes_sync_runtime.py`) calls
`store.transition_root(root_id, PAUSED)`, and the store
(`notes_device_state_store.py`, `transition_root`) cascades that to every
binding: `UPDATE notes_sync_bindings SET state = 'paused' WHERE state =
'active'`. `_resume_root` then reviews the root *before* it transitions it
back — `_review_candidate(root)` → `_ProductionRuntimeAdapter.observe_root`
— and `observe_root` refuses any root whose bindings are not all
`ACTIVE`/`CANDIDATE` with `RuntimeError("binding_review_required")`
(line 638). `_resume_root` swallows that into
`NotesSyncControlResult(False, "failed", "review_changes")`, so the root row
reads ✕ Failed while the root stays `PAUSED`; the follow-up **Check changes**
raises `RuntimeError("sync_root_not_active")` from `check_root` (line 1904)
for the same reason, which the controller renders as "Manual check failed".
The `transition_root(ACTIVE)` branch that would flip the bindings back to
`active` is reached only after the review that cannot run. Traceback captured
with an import-time probe wrapping those three methods, on a fresh seeded
profile: `wave3-caps/docs-sweep/resume-trace.log` (copied from the
scratchpad `resume-trace.log`).

This is not new to the wave: the task-32269 sync walk ended its Resume step
on the same "✕ Failed · Next: Review changes" (`wave3-caps/sync/cap-18-resumed.txt`)
and recorded the step as done. The docs sweep first reproduced it after a
two-sided edit while paused, then reproduced it with no edits at all — the
edit was never the trigger.

Evidence: wave-3 docs sweep on dev 7159fc0b99, 2026-09-12/13 —
`wave3-caps/docs-sweep/b11-list-after-restart-235x52.txt` (root ✓ Up to date
after restart) → `b12-paused` (Ⅱ Paused · Next: Resume) → `b13-resume-no-edits`
(✕ Failed, "Action needs attention") → `b14-check-after-resume` ("Manual
check failed") → `b15-review-after-failed-resume` (review says paused, 0
items, stale). Earlier run with a two-sided edit: `56-paused` → `61-resume-attempt`
→ `62-check-after-resume` → `63-conflict-review` → `67-resolution-history`.
Restart after the failed Resume:
`b16-roots-after-restart-following-failed-resume` (Ⅱ Paused) →
`b17-check-after-restart` ("Manual check failed") → `b18-resume-after-restart`
(✕ Failed) → `b19-check-after-second-resume`; that run's probe log is
appended to `resume-trace.log`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Resume on a paused root with no pending changes returns it to "✓ Up to date · Next: Check changes" and a following Check changes finishes
- [ ] #2 Resume on a paused root with pending changes (either side, or both) returns it to service and surfaces those changes as attention items ("⚠ Needs attention · Next: Review changes"), never "✕ Failed"
- [ ] #3 A root that genuinely cannot resume says why on its row and its Review page does not describe it as still paused
- [ ] #4 A test on the real runtime pins pause → resume → check, and pause → edit both sides → resume → check
<!-- AC:END -->
