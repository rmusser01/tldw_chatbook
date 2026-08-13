---
id: TASK-15664
title: A still-working sub-agent row's elapsed time does not tick between publishes
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 21:30'
updated_date: '2026-08-13 16:21'
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
- [x] #1 A live sub-agent's elapsed segment advances on its own while the row is visible, with no other interaction
- [x] #2 The refresh does not repaint the whole rail on a timer when no sub-agent is live
- [x] #3 A test drives a live row across a clock advance and fails when the elapsed value is frozen
- [x] #4 The Known gaps entry added for this in Docs/User_Guide/console/agent-runs-and-tools.md is removed when it is fixed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce red: only-survivor state (transcript poll self-stopped), clock advance, rendered elapsed frozen\n2. Add ConsoleChatController.fleet_has_unsettled_children (drain-paired counter sweep)\n3. 1s survivor tick armed at the transcript poll's self-stop edge + mount hedge; stops itself with one final settle paint\n4. Geometry-asserted tests incl. clock-advance red; remove the Known-gaps entry
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in fleet PR 3a-2 Task 4 (branch feat/fleet-autowake). Root cause confirmed as surveyed: the 0.2s transcript poll self-stops when no run occupies a slot, and a cross-turn survivor occupies no slot, so nothing repainted the rail. Fix: a 1s survivor tick (ChatScreen._console_fleet_survivor_tick) armed at the poll's own self-stop edge (+ a mount hedge), driven by the new ConsoleChatController.fleet_has_unsettled_children (the drain-paired unsettled counter from PR3a-2 Task 3 -- cheap dict reads, NOT the prune-on-read coordinator sweep task-15666 flags). The tick skips beats while the 0.2s poll runs, and when the last child settles it stops itself FIRST and paints once more -- that final pass flips the row to its terminal glyph and surfaces the new unseen-completion badge (AC#2: no recurring repaint of an idle rail). AC#3's red was reproduced against unmodified production on the test's own assertion (rendered '· 1s' frozen after a 60s clock advance): Tests/UI/test_console_fleet_survivor_tick.py. Known-gaps entry removed and the outlives-the-reply section updated (AC#4).
<!-- SECTION:NOTES:END -->
