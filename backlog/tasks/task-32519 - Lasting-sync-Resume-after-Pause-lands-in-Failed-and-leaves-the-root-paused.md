---
id: TASK-32519
title: "Lasting sync: Resume after Pause lands in ✕ Failed and leaves the root paused"
status: To Do
assignee: []
created_date: '2026-09-13 00:35'
labels:
  - library
  - notes
  - rider
dependencies:
  - TASK-32269
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With an active lasting-sync root whose note had been edited on both sides
while the root was paused (edit in Chatbook, edit the file on disk),
**Resume** in Manage sync folders does not return the root to service. The
row goes to "✕ Failed · Next: Review changes" with the status line "Action
needs attention. Review settings, then Check changes."; every **Check
changes** after that reports "Manual check failed. Review root status, then
try again."; and **Review** opens a review that says "That folder is paused.
Resume it, then Check again." with "0 safe · 0 need attention" and a stale
flag — so the user is sent round a loop with no exit, and the two-sided
change is never surfaced as an attention item. The wave-3 sync walk
(task-32269) ended on the same "✕ Failed · Next: Review changes" after its
Resume (`wave3-caps/sync/cap-18-resumed.txt`) but recorded the step as
done.

Evidence: wave-3 docs sweep on dev 7159fc0b99, 2026-09-12 —
`wave3-caps/docs-sweep/56-paused-235x52.txt` → `61-resume-attempt`
(✕ Failed) → `62-check-after-resume` (manual check failed) →
`63-conflict-review` (review says paused, 0 items, stale) →
`67-resolution-history` (empty).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Resume on a paused root with pending two-sided changes returns the root to service and surfaces those changes as attention items ("⚠ Needs attention · Next: Review changes"), never "✕ Failed"
- [ ] #2 A root that does fail to resume says why on its row and the Review page does not describe it as still paused
- [ ] #3 A test pins pause → edit both sides → resume → check on the real runtime
<!-- AC:END -->
