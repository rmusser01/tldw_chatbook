---
id: TASK-21122
title: >-
  Persona Buddy enabled-state runtime costs - ungated 10Hz repaint, 10Hz resolution retries, per-keypress config-write workers
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - persona-buddy
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21122).

Verified on the pin (`persona_buddy_widget.py`): (a) `set_interval(0.10, refresh_from_controller)`
(:227) repaints unconditionally at 10 Hz - no change gate; each tick takes the controller lock
twice, rebuilds an ~11-field visual identity, and issues three `Static.update` calls while a
separate frame timer already animates; (b) the resolution loop (:257-313) can retry a failed
visual at 10 Hz, each iteration four `to_thread` hops + DB reads, until the confirm fence
lands; (c) every h/j/k/l geometry keypress spawns a non-exclusive config-write worker
(:671-690, :779-799) - held-down key repeat queues an unbounded backlog of file writes.

## Acceptance Criteria

- [ ] The 10 Hz poll repaints only when (snapshot generation, preferences generation, visual identity) changes - or is replaced by controller-posted messages; the frame timer remains the sole animation clock
- [ ] The unconfirmed-unavailable resolution path uses capped exponential backoff (0.1 -> 2 s)
- [ ] Geometry persists are coalesced (exclusive worker or ~250 ms debounce)
- [ ] A Static.update counter with a static visual shows ~0 updates/s at steady state (was ~30/s)
