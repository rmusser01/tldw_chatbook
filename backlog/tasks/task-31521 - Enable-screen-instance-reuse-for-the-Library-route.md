---
id: TASK-31521
title: Enable screen-instance reuse for the Library route
status: In Progress
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

## Implementation Plan (the how)

Audit completed 2026-09-04 (28-row disposition table over `on_unmount`,
timer inventory, staleness map). Headline: Textual does NOT auto-cancel a
suspended installed screen's timers, and three Library timers are already
unguarded even today (media selection, media filter, ingest-path debounce
-- they rely on real-unmount auto-cancel, which reuse removes).

1. `on_screen_suspend` (new): stop all six one-shot timers; flip
   `reader_mounted_authority`/`_library_media_trash_mounted_authority`
   off; bump the request-generation fences. Do NOT wipe caches (notes
   tree, sync receipts/history, collections capture) -- wipe-on-hide was
   modeled on fresh construction; a live instance's caches are current.
2. `on_screen_resume` (extend): re-arm the two authority flags, then run
   `on_mount`'s per-visit re-kick block (conversations page, media browse
   + facets, media trash, source snapshot, notes tree, prompts, skills,
   collections capture) gated by `_library_selected_row_id` exactly as
   `on_mount` gates them -- the browse controllers' `invalidate()` strands
   a canvas in "Loading..." forever without a re-kick. One ingest-UI
   reconciliation pass.
3. Ingest-registry listener: counting + cross-tab toast stay
   unconditional; gate the DOM/DB branches on a suspended flag; reconcile
   once on resume.
4. Flip `reusable=True` on the library route.
5. Guards: same-instance resume test, no-timer-fires-while-suspended
   (measured via the timer-audit hooks), stale-data-refresh-on-resume
   test, flush-veto still honored. Mutation-test the route flag and the
   resume re-kick.
6. Interleaved A/B vs base through the real nav handler.

Deliberately unchanged this PR (in `on_unmount`, now true-teardown-only):
persistence drains/awaits, file-notes `workspace.shutdown()` (the DB
replica now survives visits -- part of the win), ingest listener removal,
`super()`. Product-flag rows surfaced in the PR body: note-import cancel
on leave, prompt multi-select cleared on leave, external-submit cancel --
under reuse these keep running / keep selection (recommended), noted as
behavior deltas.
