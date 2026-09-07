---
id: TASK-31521
title: Enable screen-instance reuse for the Library route
status: Done
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

- [x] Every `LibraryScreen.on_unmount` invalidation is dispositioned for reuse (28-row audit table, 2026-09-04; dispositions encoded in the suspend/resume hooks' docstrings)
- [x] Revisits show fresh data (browse lists, sync status, ingest queue) without a full re-mint -- per-visit refresh runs from `on_screen_resume` (mutation-tested)
- [x] Repeat visits resume the same instance (TASK-24452 guard pattern applied to the library route)
- [x] Library switch CPU improves in an interleaved A/B against the fresh-instance baseline (arrival CPU -14% overall / -21% steady-state warm on a quiet machine; the window is dominated by the outgoing fresh-instance Console -- the headline lands with task-31520)
- [x] The flush-veto path (`flush_pending_work`) still protects unsaved note edits across suspend/resume cycles (runs on the outgoing screen before suspend, unchanged; audit section (h) found no death-after-flush assumption)

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

## Implementation Notes

Flipped `reusable=True` on the library route after executing the audit's
dispositions. `on_screen_suspend` (new) stops all six armed debounce
timers -- three (`media selection`, `media filter`, `ingest path`) had NO
explicit stop anywhere and relied entirely on Textual's real-unmount
auto-cancel, which reuse removes. Deliberately NOT done at suspend: cache
wipes, generation bumps, authority flips -- a suspended instance is live,
and in-flight results landing on it mean fresher data at resume;
rejecting them would re-create the stranded-"Loading" class.
`on_screen_resume` is the single per-visit dispatch seam (mount no longer
dispatches -- ScreenResume fires on the initial push too, the Qodo #2402
finding-4 double-run): `_refresh_library_visit_surfaces` re-kicks the
active row's surfaces with `on_mount`'s original gating, entry-vs-repeat
semantics for media trash (repeat preserves the user's filter context).
Ingest-registry listener: counting + cross-tab toast stay live while
suspended; DOM/DB branches gate on the suspended flag with one
resume-time reconciliation pass.

Two pre-existing tests pinned the fresh-instance lifecycle and were
updated to the reuse contract with their load-bearing halves kept
(serialization via the app-lifetime lock; fetch-completing-inside-the-
dispatching-handler reconciles once -- the hazard moved from Mount to the
resume seam and the gate moved with it).

Verification: `Tests/UI/test_library_screen_reuse.py` (3 guards; suspend
timer-stops and resume dispatch each mutation-tested red); all reuse
suites 14 passed; paired arms vs pristine dev -- test_library_shell
224-vs-226 with only-mine ZERO after the contract updates, compose-once
in-order failure set IDENTICAL to base, notes_reader/honesty/resize/
footer sets identical. Interleaved A/B (quiet machine): Library arrival
CPU -14% overall / -21% steady-state warm; wall -7%. The end-to-end
window is dominated by the outgoing fresh-instance Console -- the
headline switch numbers land with task-31520. Structural wins pinned by
guards: zero widget re-mint per revisit, the file-notes DB replica and
mount-path config reads now once per app run (retires the task-24456/
24457 per-visit costs), zero Library timers firing while hidden.

Files: `UI/Screens/library_screen.py`, `UI/Navigation/screen_registry.py`,
`Tests/UI/test_library_screen_reuse.py` (new),
`Tests/UI/test_library_shell.py`, `Tests/UI/test_library_entry_compose_once.py`.
