---
id: TASK-15862
title: >-
  Wake-turn UI staleness: transcript, run-state chip, and tab glyph can freeze
  until the session is next viewed
status: To Do
assignee: []
created_date: '2026-08-13 21:43'
labels:
  - fleet
  - console
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-2 Task 7's live pass found the wake DELIVERY layer correct and durable in every scenario, but the UI around a wake turn can go stale indefinitely. Three observed shapes, one likely root: (1) a wake turn completing while the user views another Console session leaves the woken session's tab glyph stuck at RUNNING (●) for minutes instead of flipping to the unvisited-outcome glyph — it clears only when the session is viewed; (2) a mount-claim wake delivering into the VIEWED session froze mid-delivery: the assistant reply row stayed empty while the full reply sat in the DB, the status row read 'Run: Agent running.' and the composer read 'Send blocked — finish provider setup to continue' (misleading — provider setup was fine) for 4+ minutes, healing instantly on a session switch; (3) the same freeze recurred on the post-restart poked delivery. Likely the transcript poll / repaint pipeline is armed by user-driven send paths and never armed (or re-armed at the terminal edge) for a wake delivery task — the same self-stopping-poll family as task-15664.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A wake turn's streamed reply and its terminal state paint in the viewed session without requiring a session switch
- [ ] #2 A wake turn completing in a non-viewed session flips that session's tab glyph off RUNNING at the terminal edge
- [ ] #3 The composer's blocked-state copy during a wake turn names the actual reason (busy with a wake turn), not provider setup
<!-- AC:END -->
