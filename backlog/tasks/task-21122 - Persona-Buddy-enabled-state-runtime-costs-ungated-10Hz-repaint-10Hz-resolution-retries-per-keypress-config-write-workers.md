---
id: TASK-21122
title: >-
  Persona Buddy enabled-state runtime costs - ungated 10Hz repaint, 10Hz
  resolution retries, per-keypress config-write workers
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 17:31'
labels:
  - performance
  - persona-buddy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21122).

Verified on the pin (`persona_buddy_widget.py`): (a) `set_interval(0.10, refresh_from_controller)`
(:227) repaints unconditionally at 10 Hz - no change gate; each tick takes the controller lock
twice, rebuilds an ~11-field visual identity, and issues three `Static.update` calls while a
separate frame timer already animates; (b) the resolution loop (:257-313) can retry a failed
visual at 10 Hz, each iteration four `to_thread` hops + DB reads, until the confirm fence
lands; (c) every h/j/k/l geometry keypress spawns a non-exclusive config-write worker
(:671-690, :779-799) - held-down key repeat queues an unbounded backlog of file writes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 10 Hz poll repaints only when (snapshot generation, preferences generation, visual identity) changes - or is replaced by controller-posted messages; the frame timer remains the sole animation clock
- [x] #2 The unconfirmed-unavailable resolution path uses capped exponential backoff (0.1 -> 2 s)
- [x] #3 Geometry persists are coalesced (exclusive worker or ~250 ms debounce)
- [x] #4 A Static.update counter with a static visual shows ~0 updates/s at steady state (was ~30/s)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: run buddy/Persona_Buddy/base-app-screen tests, tee counts; record pre-existing reds.
2. Red-first probe: count Static.update/Widget.refresh for a STATIC visual over a fixed window; expect ~30/s on base.
3. Gate refresh_from_controller on an explicit paint key (snapshot generation, preferences generation, state, collapsed, working geometry, screen size, display, visual identity); keep _ensure_resolution outside the gate; keep the frame timer the sole animation clock. Avoid rebuilding the visual-identity tuple when snapshot.visual is the same frozen object.
4. Capped exponential backoff (0.1 -> 2.0 s, doubling, capped attempt count) on the unconfirmed-unavailable retry path, keyed per resolution authority, enforced both inside the loop and in _ensure_resolution so a finished worker cannot be respawned at 10 Hz.
5. Coalesce geometry persists behind a ~250 ms debounce with an unmount flush; keep the in-memory patch immediate so intent and merge semantics are unchanged; cancel a pending debounce when an immediate (non-geometry) persist supersedes it.
6. Re-run probes (static ~0, animated still ticks, change still repaints promptly) and the full test set; A/B every red.
7. Diagnostic inventory check; DoD.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gated the Persona Buddy overlay's 10 Hz poll on a paint-authority tuple, put capped exponential backoff on the unconfirmed-unavailable resolution retry, and coalesced geometry writes behind a 250 ms debounce with an unmount flush. All three defects lived in `Widgets/Persona_Widgets/persona_buddy_widget.py`; no controller change was needed.

## Paint gate (AC1, AC4)

`refresh_from_controller` now builds `_paint_authority(...)` and returns before any `Static.update` when it equals the last painted one. Membership is derived from what the paint path READS, not from generations alone: snapshot generation + preferences generation + state + enabled/open/collapsed, the working geometry, `screen.size`, `display`, whether a drag/resize interaction is active, the visual's `available`/`reason` pair (they select `_paint_frame`'s no-frames placeholder and are NOT in the identity tuple), and the visual identity tuple. `screen.size` is load-bearing: `on_resize` re-applies geometry but never recomputes the compact/minimal classes, so that transition was riding on the ungated poll.

Two supporting changes: the ~11-field visual-identity tuple is no longer rebuilt per tick when `snapshot.visual` is the same object (both it and `PersonaBuddyPreparedFrame` are frozen slots dataclasses, so object identity implies field identity); and `_ensure_resolution` stays OUTSIDE the gate so a semantic-authority change still restarts resolution promptly. Animation-reset semantics are byte-identical: the reset is still keyed on `visual_identity != self._visual_identity`, so a fresh-but-equal snapshot still preserves the frame deadline. The frame timer remains the sole animation clock.

Textual gating verified rather than assumed: `set_class` -> `add_class`/`remove_class` early-return on no change and `Button.label` is an equality-gated reactive, so the class/label writes in a tick were already free; `Static.update` re-visualizes and calls `refresh(layout=True)` unconditionally, which is why the gate has to sit above it.

## Resolution backoff (AC2)

Both retry paths in `_resolution_loop` now go through `_await_resolution_retry`: 0.1 s doubling to a 2.0 s cap, at most 8 attempts per resolution authority, then the loop abandons. Because `_ensure_resolution` would otherwise respawn a finished worker on the next 100 ms tick, the backoff state lives on the VIEW (`_retry_authority`/`_retry_attempts`/`_retry_ready_at`) and `_ensure_resolution` honours it. The second path (post-reconcile, still unconfirmed) only backs off when the authority has NOT moved, so the legitimate authority-changed restart is still immediate.

## Geometry coalescing (AC3)

`_geometry_action`, `action_reset_geometry` and `on_mouse_up` now call `_schedule_geometry_persist`, which applies the patch in memory immediately (intent, merge order and displayed geometry unchanged) and debounces the blocking write by 250 ms, flushed from `on_unmount`. A non-geometry persist discards any pending debounce because its merged snapshot already carries the geometry. Chose the debounce over an exclusive worker deliberately: `_drain_owned` shields the write task, so cancelling the outer worker does NOT prevent the write and `exclusive=True` would have serialized the backlog without shrinking it.

## Evidence

Counter probe on `Static.update` (and `Widget.refresh`) over the Buddy subtree, static visual, 2 s window -- base 60 updates / 60 refreshes (30.0/s each), after 0 / 0. Animated 3-frame visual: frames 0,1,2 all still seen, 46 updates/s -> 13/s (the pure frame clock). Change latency unchanged at one poll tick: state 92 ms, collapse 114 ms, visual 90 ms. 20 held 'l' presses: 20 persists -> 1, final geometry identical. Live PTY probe (`Tests/Live/persona_buddy_terminal_probe.py`) A/B'd base vs branch: 18/18 checks pass on both, every observed field identical including `resolution_calls` and the geometry restored across an app restart.

## Files

- `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py`
- `Tests/UI/test_persona_buddy_widget.py` (6 new tests; the 8-keypress test now asserts coalescing instead of >=8 writes)
- `backlog/docs/lessons-testing-evidence.md`

## Review fix round (2026-08-23)

Review verdict was FIX-FIRST on the geometry leg. Two majors plus four cheap
pins; the paint gate itself survived a 39-stimulus differential harness with a
working negative control (zero missed changes) and the `compare=False`
reliance was verified sound in production (44 identity recomputations, zero
mismatches).

### MAJOR-1 -- the debounced write was LOST at app exit (and raised on quit)

`TldwCli._shutdown` runs `_shutdown_app_owned_lifecycles()` -> `_shutdown_
persona_buddy()` -> `controller.shutdown()` BEFORE `super()._shutdown()`
closes screens. Admission is therefore already closed when Textual unmounts
the widget, so `on_unmount`'s flush hit `persist_preferences_revision`'s
`RuntimeError("persona_buddy_shutdown")` guard: the user's LAST nudge was
dropped and the quit produced a `WorkerFailed` traceback. At base every
keypress wrote immediately, so this window did not exist -- the 250 ms
debounce created it.

Fixed in three places:

- `app.py` gained `_flush_persona_buddy_geometry()`, called from
  `_shutdown_persona_buddy` immediately BEFORE the `controller.shutdown()`
  task is created -- i.e. while the controller still accepts writes. It walks
  every screen in `_screen_stacks` (all modes, de-duplicated) and awaits any
  `flush_persona_buddy_geometry` it finds; duck-typed, so nothing imports the
  widget module at shutdown.
- `BaseAppScreen.flush_persona_buddy_geometry()` drains its mounted view.
- The widget gained `flush_pending_geometry_persist()` (awaitable, swallows a
  refused write) and its timer-driven `_flush_geometry_persist` now wraps the
  persist in `try/except Exception`, so a teardown path that skips the app
  hook can never turn the flush into a traceback. Both share
  `_take_pending_geometry_persist()`.

Pinned by `test_debounced_geometry_survives_the_real_shutdown_order`
(parametrised 0.05 / 0.15 / 0.40 s) and
`test_unmount_after_controller_shutdown_raises_no_worker_error`, both using
the REAL `PersonaBuddyController` in a test app that reproduces `TldwCli`'s
shutdown order. A/B against the previous commit: 0.05 s and 0.15 s FAIL with
`WorkerFailed: RuntimeError('persona_buddy_shutdown')`, 0.40 s passes -- an
exact reproduction of the review probe.

### MAJOR-2 -- the load-bearing backoff guard had no test

Deleting `if self._resolution_retry_is_pending(authority): return` left 47/47
green. Now pinned by `test_exhausted_retries_are_not_respawned_by_the_poll`,
which shrinks the ladder via monkeypatch, waits for the cap to be reached AND
the worker to finish, then asserts the resolve count is frozen for 0.6 s. The
existing 1.2 s backoff test could not see it: during the in-loop sleeps the
worker is unfinished, so `_ensure_resolution` early-returns on the
`_requested_authority` check -- the guard only matters after the loop exits,
~9.1 s in at the shipped ladder.

### Also taken

- `test_resolution_backoff_ladder_doubles_to_a_two_second_cap_and_stops` pins
  the ladder exactly ([0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 2.0], sum < 10 s).
- `test_post_reconcile_retry_backs_off_when_authority_did_not_move` pins the
  second retry path (the one that had no delay at all before this task).
- `test_superseding_persist_discards_the_pending_geometry_debounce` pins the
  discard in `_apply_and_schedule_preferences`.
- `test_mouse_resize_into_compact_repaints_from_geometry_alone` pins
  `preferences.geometry` in the paint tuple via a MOUSE resize past the
  compact threshold (the interaction flag is allowed to bake in first, so the
  working geometry is the only member that moves). The pre-existing compact
  test drives that transition through `screen.size` instead.
- `self._painted_authority = authority` moved to AFTER the paint work: as
  written, a raise mid-paint recorded the authority as painted and never
  retried, where the ungated version simply repainted on the next tick.
  Pinned by `test_a_failed_paint_is_not_recorded_as_painted`.

Every one of the six was mutation-verified: reverting the guard / uncapping
`_MAX_RESOLUTION_RETRIES` to 1,000,000 / dropping either backoff / dropping
the discard / dropping `preferences.geometry` / restoring the old assignment
order each reds exactly its pin. Dropping the second-path backoff does not
merely red -- it HANGS the event loop until pytest-timeout kills it,
confirming that path was a genuine tight spin.

### Re-verification

Steady-state probe unchanged: static visual 0.0 `Static.update`/s and 0.0
subtree refreshes/s over 2 s; animated 3-frame visual still paints frames
0,1,2 at 13/s; 20 held keypresses still coalesce to 1 durable write with the
final geometry intact. Live PTY probe 18/18, every field identical to the dev
base except `resolution_calls` 15 vs a 16-17 base spread (natural variance,
and lower is the direction the backoff pushes). Buddy suites 155 passed / 1
pre-existing failure; `Tests/UI/test_persona_buddy_widget.py` +
`test_persona_buddy_app_mount.py` 57 passed, 3x random-order, no flake.
Diagnostic inventory drift was exactly the one `logger.debug(
"persona_buddy_geometry_flush_failed")` this round added -- a fixed category
string interpolating nothing -- reviewed via `--statements ... --since` and
then regenerated; full `./scripts/preflight.sh` green.

### Pre-existing reds and flakes observed (NOT from this task)

- `Tests/Packaging/test_persona_buddy_import_closure.py::
  test_app_import_does_not_execute_persona_visual_or_pil` fails on the dev
  base `ae817fefe` with zero changes: PIL + `tldw_chatbook.Persona_Visual`
  are resident after `import tldw_chatbook.app`, i.e. TASK-21103's lazy-import
  guarantee has regressed through some other import path. Worth its own task.
- All 14 `Tests/UI/test_destination_shells.py` failures reproduce with an
  identical failure set on `ae817fefe`.
- `test_compact_keyboard_move_preserves_preferred_size_on_restore[viewport1|
  viewport2]` is a pre-existing load-sensitive flake on BOTH arms. Its timing
  profile did change under the debounce: `_wait_until(len(persisted) == 1)`
  now takes ~250 ms instead of ~0. Reported, not fixed.
- `Tests/Live/persona_buddy_terminal_probe.py` has its own load-sensitive
  race, also pre-existing: `_write_probe_report` calls `query_one` on a Buddy
  view whose removal is in flight (`open=False`, children already detached)
  and raises `NoMatches`. Seen once in 19 runs on this branch -- during a
  concurrent full pytest run -- and then REPRODUCED on the dev base at 1/12
  under deliberate 6-way parallel load, versus 0/20 serial on the base and
  0/18 serial on the branch. Harness-side, not product-side; reported only.

## Rebase onto dev `b2b1e2e0d` -- RE-DERIVED, not merged (2026-08-23)

Dev landed "repair and distill Persona Buddy chrome" (TASK-20941 + TASK-21000)
while this branch was in review: 537 changed lines in `persona_buddy_widget.py`
and a 1501-line rewrite of `Tests/UI/test_persona_buddy_widget.py`. The paint
path this task's gate was derived from no longer exists, so the branch was
reset onto dev and re-derived rather than textually merged. The three
pre-rebase commits are preserved at tag `prerebase-t21122`.

### What dev changed in the paint path

* **The chrome is now pet-only.** The drag handle, the status line and the
  hint line are gone; `compose` yields ONE `Static` (`#persona-buddy-frame`)
  plus two absolutely-positioned icon buttons. `_paint_frame` therefore issues
  **one** `Static.update` per tick instead of three -- dev's distillation
  already cut the measured repaint cost from 30/s to 10/s on its own.
* **The painted visual no longer comes from `snapshot.visual`.** Dev added
  `_AcceptedRender`, pinned in the resolution loop, carrying `authority`,
  `visual_identity`, `collapsed` and a `content_width`/`content_height`
  content box. `_paint_frame` reads the accepted render (through
  `_accepted_render_for_paint`), the `_ACTIONABLE_ALERTS` text for
  `snapshot.state`, and the compact class.
* **New paint inputs**: `_sync_control_presentation` reads `self.app.focused`
  (labels expand to words only while focused) and writes tooltips/labels;
  `_display_geometry` fits the box from `_display_content_boxes`;
  `_compact_presentation_for_geometry` hides an accepted pet that no longer
  fits; `_sync_viewport_generation` publishes terminal-size authority.
* **Not changed**: the 10 Hz `set_interval` poll is still ungated, the
  unconfirmed-unavailable path still sleeps a flat `_POLL_SECONDS`, the
  post-reconcile path still `continue`s with no delay, and every geometry
  keypress still spawns a config-write worker. **All three defects survive
  dev's rewrite; none of this task is obsolete.**

### How the tuple changed

Dropped (lost their subject): the `snapshot.visual` identity tuple and the
`available`/`reason` pair -- `_paint_frame` no longer reads `snapshot.visual`
at all. Also dropped: my `_visual_identity_for` per-tick caching helper, which
dev superseded by pinning `visual_identity` into `_AcceptedRender` at resolve
time.

Added: `accepted.authority`, `accepted.visual_identity`, `accepted.collapsed`,
`accepted.content_width`, `accepted.content_height`,
`_display_content_boxes[collapsed]`, and the focused control's id (a sound
proxy for dev's `self.app.focused is collapse` because ids are unique within
the one mounted view). Kept: both controller generations, `state`, `enabled`,
`open`, `collapsed`, the working geometry, `screen.size`, `display`, and the
interaction flag.

### The regression the re-derivation surfaced

The rebased branch failed 4 of dev's 21 live PTY probe tests, deterministically
(3/3), while passing 232 in-process tests. A six-variant bisect isolated it to
the geometry debounce alone -- every variant with the debounce off passed 21/21.

Root cause: `_apply_geometry` assigns `self.absolute_offset`, which is a plain
attribute, not a style, so it schedules no arrange. Now that the box is
content-fitted, a pure move changes no style at all, so the new offset only
reached the compositor when some *later* repaint happened to run -- and the
repaint that used to run was the one the per-keypress config write triggered
immediately afterwards. Coalescing those writes removed it, and the probe,
which reads painted control regions from the first report whose *preferred*
geometry matches, then clicked at the buddy's old position. Fixed at the
source: `_apply_geometry` now refreshes with `layout=True` only when the offset
actually changes, so a move composites itself and costs nothing at rest.

### Re-verified after the rebase

| measurement | dev `b2b1e2e0d` | this branch |
|---|---|---|
| static visual, 2 s | 20 updates / 20 refreshes (**10.0/s**) | 0 / 0 (**0.0/s**) |
| animated 3-frame | 19.8/s, frames {0,1,2} | **10.0/s**, frames {0,1,2} |
| state -> alert | 99 ms | 99 ms |
| collapse | 100 ms | 101 ms |
| visual swap | 93 ms | 91 ms |
| 20 held `l` | **20 persists** | **1 persist**, same final x |

Tests: 232 passed / 0 failed across the buddy widget, app-mount, Persona_Buddy,
boundary, on-mount-guard, packaging-closure and live-PTY suites (the packaging
red reported pre-rebase is fixed on dev). 107 passed 3x in random order, no
flake. Ten source mutations re-run against the rebased code: eight RED on their
pins (dropping the second-path backoff still HANGS the event loop until
pytest-timeout fires); two GREEN -- see below.

### Differential harness and its negative control

A 45-stimulus differential harness (settle, fingerprint, force an ungated
repaint, fingerprint again) reported zero misses -- and so did its negative
control, i.e. it was blind and its clean run was worthless. Strengthened (a
mouse resize that crosses the compact threshold in both directions, plus a
pure-move drag) and re-run as a per-member drop: `screen.size` (3 misses) and
`display` (1 miss) are caught by the harness; `preferences.geometry` is caught
only by the dedicated non-forcing pin test (the harness's own forced repaint
resyncs the gate mid-sequence); the remaining members are individually
redundant with `snapshot.generation`, because the real controller bumps its
generation on every state, preference and lease change. `accepted.visual_identity`
and `snapshot.state` are therefore kept as defensive members, not load-bearing
ones -- recorded here so nobody mistakes their green mutation for a proof.

### Test disposition

All twelve pinned tests survived and were re-applied on top of dev's rewritten
file (dev's structure kept, my assertions re-added): the backoff ladder, the
`_ensure_resolution` guard, the second-path backoff, the supersede discard,
`preferences.geometry`, the paint-failure ordering, steady-state repaint,
animation clock, held-key coalescing, unmount flush, and the three shutdown
tests in `test_persona_buddy_app_mount.py` (untouched by dev). Adapted rather
than lost: `test_real_changes_still_repaint_within_one_poll` now drives dev's
alert text / tooltip / accepted-render instead of the retired status line, and
gained a `focus` leg; `_StatefulController` was deleted because dev's
`_FakeController` now carries `state`. One dev test was edited:
`test_keyboard_move_resize_reset_collapse_close_exact_bindings` asserted
`len(persisted) >= 8`, which directly contradicts AC 3, and now asserts the
coalescing property instead. Nothing was dropped for lack of a subject.

### Pre-existing dev defect found, reported, NOT fixed

After `controller.shutdown()`, `action_toggle_collapse` and `action_close`
still raise `WorkerFailed: RuntimeError('persona_buddy_shutdown')` from the
unguarded `await persist_preferences_revision(...)` in
`_apply_and_schedule_preferences`. This branch fixes only the debounced
geometry path (measured: `move` clean, `collapse` and `close` still raise).
Guarding the other path is a semantic decision -- what should happen to the
`reconcile=True` leg when the write is refused -- and belongs in its own task.

## Deliberately not done

Controller-posted messages (the AC's alternative): the controller is mutated from worker threads under an RLock via `Persona_Buddy/console_adapter.py`, so its unused `scheduler=call_after_refresh` hook is not safe to drive from those call sites without a threading change out of this task's scope. `_ensure_resolution`'s per-tick authority tuple is left ungated -- it is pure computation with no repaint, and gating it would risk the resolution-restart latency the existing tests pin. Relocating the reconcile hook off BaseAppScreen is TASK-21123.
<!-- SECTION:NOTES:END -->
