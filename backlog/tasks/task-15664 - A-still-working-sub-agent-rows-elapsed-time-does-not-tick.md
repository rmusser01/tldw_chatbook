---
id: TASK-15664
title: 'A still-working sub-agent row''s elapsed time does not tick between publishes'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live during PR 3a-1 Task 7 verification. The Sub-agents panel keeps a cross-turn survivor's row after its reply finishes, but the row's elapsed segment is only rewritten when something else repaints the rail (the child's own next step, the next user message, drilling into the row and back). A child that had been working for roughly a minute still displayed `. 1s`; the same row read `. 18s` and `. 1m 11s` immediately after unrelated interactions repainted it. The status glyph and the "N working" summary stayed correct throughout, so this is a stale number rather than a wrong state - which arguably makes it more misleading, since the number looks authoritative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A live sub-agent's elapsed segment advances on its own while the row is visible, with no other interaction
- [ ] #2 The refresh does not repaint the whole rail on a timer when no sub-agent is live
- [ ] #3 A test drives a live row across a clock advance and fails when the elapsed value is frozen
- [ ] #4 The Known gaps entry added for this in Docs/User_Guide/console/agent-runs-and-tools.md is removed when it is fixed
<!-- AC:END -->
