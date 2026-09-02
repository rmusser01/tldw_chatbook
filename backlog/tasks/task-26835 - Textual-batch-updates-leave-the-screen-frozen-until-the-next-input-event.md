---
id: TASK-26835
title: Textual batch updates leave the screen frozen until the next input event
status: Done
assignee:
  - '@claude'
created_date: '2026-09-01 14:27'
updated_date: '2026-09-01 14:40'
labels:
  - console
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dominant cause of TASK-26834's felt button lag, verified link by link in Textual 8.x source with the in-terminal probe. Any refresh(recompose=True) runs inside 'async with self.batch()'. If the Screen's queue drains while the batch is open, Screen._on_idle skips its paint-timer resume (batch guard). When the batch closes, App._end_batch calls check_idle on the APP's pump, and App._on_idle handles only resize -- nothing ever pokes the Screen again. Dirty widgets sit unpainted with the update timer paused until the next real input event lands on the Screen's queue. Measured: a rail section-header click completed all scheduling by +12.7ms and painted at +234.7ms, unblocked by a mouse-move; tray recomposes freeze the UI 300-700ms when the mouse holds still.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The claimed mechanism is either reproduced with a failing test or refuted with live-state evidence (REFUTED -- see notes)
- [x] #2 The refutation and the actual mechanism are recorded where the fix work lives (TASK-26834)
- [x] #3 No fix was shipped against the invalid premise
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: bare Textual app, recompose under batch with no input, assert a paint arrives promptly (fails unpatched)\n2. Utils/textual_batch_repaint.py: install patch on App._end_batch that pokes the current screen when the last batch closes\n3. Wire install next to install_stylesheet_fastpath in app.py\n4. Sweep console UI files incl. snapshots; preflight; PR with probe-verified before/after
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This task's premise is REFUTED, twice over, and no fix was shipped against it.

1. The bare-app reproduction paints promptly unpatched (first correction,
   above): Textual's widget-idle relay posts Update/Layout to the screen and
   closes the loop my derivation had open.
2. Probe v6 then sampled the live invariants during real stalls. The frozen
   windows show `screen.Idle guarded(batch=2..3)` and
   `STATE batch=2 timer=PAUSED dirty=71 lay=1 rep=1` at +250ms: the paint is
   withheld because NESTED App-level batches stay open for 250-400ms while
   tray recomposes mount dozens of widgets and run their reconcile storm
   inside the batch. First paint is the batch closing -- no input event is
   involved. The windows that DID end on a mouse-move (`dirty=0` for 26
   straight ticks) owed nothing at all: clicks with no visible response,
   a probe framing artifact plus at most a missing click-ack.

The real mechanism and its fix directions live on TASK-26834. The bare-app
freeze test written here was deleted rather than shipped: it pins framework
behavior that was never broken.

Lesson credit: the no-fix-without-failing-repro rule is what stopped this
twice -- the mechanism read plausibly in source both times.
<!-- SECTION:NOTES:END -->

## Correction (2026-09-01, before any fix): the composed mechanism is UNPROVEN

The description above says "verified link by link" -- each link IS verified in
source, but the COMPOSITION does not reproduce. A bare-app test that recomposes
under batch and waits with raw `asyncio.sleep` (no input-shaped messages)
paints promptly, unpatched: `Tests/UI/test_batch_repaint_unfreeze.py`, 2
passed. Two relays my derivation missed both close the loop in the simple
case:

- `Widget.refresh()` pokes its own pump, whose Idle runs `_check_refresh`,
  which posts `messages.Update/Layout` TO THE SCREEN -- queue activity, Idle,
  resume, paint.
- The screen's Idle is edge-triggered but those posts provide the edge.

So the frozen windows (all work done by +12.7ms, paint at +234.7ms on a
mouse-move; no recompose events in that window at all) have a mechanism not
yet named. One noted hole, unproven: `_check_refresh` clears
`_repaint_required` but posts NOTHING when `self.display` is False -- a
refresh on a hidden widget is silently dropped rather than deferred.

**No fix until a failing reproduction exists.** Next instrument (probe v6)
samples the live invariants during stall windows -- `app._batch_count`,
update-timer active flag, `len(screen._dirty_widgets)`,
`_layout/_repaint_required`, `len(screen._callbacks)` -- plus trace events
for `Screen._on_idle` (and which branch it took) and update-timer resumes.
The frozen window's sequence will show directly which invariant sticks.
