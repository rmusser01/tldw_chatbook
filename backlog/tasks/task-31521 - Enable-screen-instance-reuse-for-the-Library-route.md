---
id: TASK-31521
title: Enable screen-instance reuse for the Library route
status: To Do
assignee: []
created_date: '2026-09-04 21:30'
labels:
  - performance
  - library
dependencies:
  - task-24452
priority: high
---

## Description (the why)

TASK-24452 landed opt-in screen-instance reuse and proved it on Home.
Library's measured headroom is the largest relative win: a warm
installed-instance switch cost 33 ms CPU against 340-1,540 ms fresh
(-95%, interleaved arms, 2026-09-04) -- and reuse also retires the
per-visit costs already filed as task-24456 (43 config reads/visit) and
task-24457 (8 SQLite connections/visit), since `on_mount` runs once.
Enablement is gated on an audit of the remount-invalidation machinery:
`LibraryScreen.on_unmount` bumps request generations, revokes reader/apply
authority, and invalidates six controllers, all designed around
fresh-construction; per-visit data refresh must move to
`on_screen_resume`, and the ingest-registry listener keeps firing while
suspended (recompose-on-hidden cost to disposition).

## Acceptance Criteria (the what)

- [ ] Every `LibraryScreen.on_unmount` invalidation is dispositioned for reuse (suspend/resume pairing, app-lifetime move, or documented as unnecessary under a live instance)
- [ ] Revisits show fresh data (browse lists, sync status, ingest queue) without a full re-mint -- per-visit refresh runs from `on_screen_resume`
- [ ] Repeat visits resume the same instance (TASK-24452 guard pattern applied to the library route)
- [ ] Library switch CPU improves in an interleaved A/B against the fresh-instance baseline
- [ ] The flush-veto path (`flush_pending_work`) still protects unsaved note edits across suspend/resume cycles
