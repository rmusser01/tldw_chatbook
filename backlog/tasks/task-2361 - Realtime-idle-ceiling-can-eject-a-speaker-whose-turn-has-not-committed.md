---
id: task-2361
title: 'Realtime: idle ceiling can eject a speaker whose turn has not committed'
status: To Do
assignee: []
created_date: '2026-08-04'
labels: [realtime, voice]
dependencies: []
priority: low
---

## Description (the why)

The idle ceiling counts activity as turn-commit or reply-end. A user who STARTS speaking
just before the deadline (speech_started, no commit yet) is cut off mid-utterance with
"idle for N minutes" (V4 final review M3). Matches the spec's letter, surprises an
attending user.

## Acceptance Criteria (the what)

- [ ] `on_speech_started` while live refreshes the idle anchor (in both barge-in modes'
      reachable paths) so an in-progress utterance is never cut by the cost guard.
- [ ] A genuinely silent session still exits at the ceiling.
- [ ] FSM tests pin both directions.
