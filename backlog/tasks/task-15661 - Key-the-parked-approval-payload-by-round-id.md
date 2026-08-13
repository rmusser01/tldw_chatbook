---
id: TASK-15661
title: 'Key the parked approval payload by round id (fleet F7)'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parked approval payload lives in a single slot, so two approval rounds that are parked at the same time overwrite each other. This is pre-existing for sibling sub-agents within one turn and is already documented in the file as an accepted limitation, but cross-turn survivors (PR 3a-1) widen the window in which two rounds can be parked together. Key the payload by round id so each parked round keeps its own.

**Exposure update, 2026-08-13 (fleet PR 3a-2, `feat/fleet-autowake`):** auto-wake adds a machine-initiated turn class (`AGENT_WAKE`) that by design runs in sessions the user is not viewing, so approval rounds raised by a woken turn park by default rather than exceptionally. The overwrite MECHANISM is unchanged (the per-session `_parked_approval_payloads` slot), and the wake never fires INTO a session with a card already pending (busy-session deferral gate, pinned) — but the population of parked rounds grows: a woken turn's tool can park a round while an earlier turn's survivor parks another in the same session, with no user action involved at any point. The fixer should treat wake turns as a first-class producer of parked rounds when testing AC #1/#4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two rounds parked at the same time each retain their own payload
- [ ] #2 Answering one parked round does not alter or clear the other's payload
- [ ] #3 The accepted-limitation comment in the source is removed rather than reworded
- [ ] #4 A test parks two rounds concurrently and fails when the slot is shared again
<!-- AC:END -->
