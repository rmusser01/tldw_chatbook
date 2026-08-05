---
id: TASK-2391
title: >-
  Empty realtime transcript row is never persisted and its status is never shown
status: To Do
assignee: []
created_date: '2026-08-05'
labels:
  - realtime
  - console
dependencies: []
priority: medium
---

## Description (the why)

task-2364 added `MessageMetadata.transcript_status`, and the realtime wiring stamps
`"empty"` when a committed turn produces no transcript. Two gaps remain, found by that
task's review (findings F3):

1. The row it stamps has empty content, and the store defers persistence for content-less
   rows — so the status exists only in memory and is lost on restart.
2. Nothing reads `transcript_status` anywhere. The person looking at the screen still sees
   an unexplained blank row, which was the third consequence task-2364's description set
   out to close.

The data model can now express "why this row is empty"; the user still cannot see it.

## Acceptance Criteria (the what)

- [ ] A committed voice turn that produced no transcript is visible to the user as an
      explained state (not a silently blank row), in the transcript itself.
- [ ] That explanation survives a restart, or the row is not created at all — pick one and
      say which in the notes; a row that exists only until restart is not acceptable.
- [ ] `transcript_status` has at least one real consumer, or is removed as dead weight.
