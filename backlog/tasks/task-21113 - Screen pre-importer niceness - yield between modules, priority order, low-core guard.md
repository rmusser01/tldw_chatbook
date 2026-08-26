---
id: TASK-21113
title: >-
  Screen pre-importer niceness - yield between modules, priority order, low-core guard
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
labels:
  - performance
  - startup
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21113).

`_preimport_screens` (app.py:11762-11886) is a tight no-yield loop importing ~21 screen modules
(~117k lines + closures) on a GIL-holding daemon thread starting 0.2 s after first paint. A
live A/B on multi-core hardware measured NO first-interaction penalty (keystroke median 82 vs
99 ms, 30 s CPU 2.51 vs 2.75 s - within noise), so this is design hardening for 1-2-core
machines, not a measured regression: on such boxes the entire post-boot window has a CPU
competitor for every keystroke. Keep the `TLDW_SCREEN_PREIMPORT` overrides.

## Acceptance Criteria

- [x] A sleep between route imports gives the event loop back time proportional to the
      import that just ran
      **AMENDED 2026-08-24 before implementing.** The original wording, "a short sleep
      ... bounds GIL contention to one module at a time", describes something a sleep
      cannot do: there is only ever one import thread, so contention is already one
      module at a time, and a sleep cannot subdivide a single `import_module`. Measured
      on the pin, the loop's cost is three indivisible chunks (see Implementation
      Notes), so a *flat* sleep would treat a 0.5 ms module like a 110 ms one. The gap
      is proportional instead.
- [x] ~~Import order starts with the configured default tab, then the heavy trio~~
      **DROPPED 2026-08-24, provably a no-op** (evidence in Implementation Notes): the
      whole-registry pass is armed at the tail of `_post_mount_setup`, and both boot
      paths complete `_push_initial_screen` first, so the configured default tab's
      module is already in `sys.modules` before this order is read. Reordering would
      have moved a dict hit. Recorded in the code next to
      `SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS` so it is not re-proposed.
- [x] The pre-import pass parks while a screen navigation is resolving (bounded), and is
      heavily throttled below 4 usable CPUs -- throttled rather than disabled, with the
      reason recorded
- [x] The existing A/B env override still works both ways
- [x] The change is not slower or more complex on normal hardware: measured, interleaved,
      three arms
- [x] The low-core gate is coherent across both pre-importers: one shared
      `_usable_cpu_count`, applied to the speculative whole-registry pass and
      deliberately not to task-21110's certain-work initial-screen warm-up, with the
      asymmetry justified in the code

## Implementation Plan

1. Re-measure the loop before tuning it: task-21731 changed the module population it
   walks, and the filing's cost estimate predates that.
2. Establish whether the default-tab ordering AC is reachable at all, from the boot
   ordering rather than from the wording.
3. Build the A/B harness first -- a main-thread 20 Hz consumer whose inter-tick gaps
   are the GIL contention this task is about -- with the control arm running the
   merge-base function's own source.
4. Implement only what the measurements support; amend or drop ACs that they contradict.
5. Mutation-check every test.
6. Walk quit mid-import on a real thread.

## Implementation Notes

Shipped: a proportional inter-route yield, a bounded park while a navigation is
resolving, and a low-core throttle -- reached through one shared `_usable_cpu_count()`.
Dropped: the default-tab ordering, which the boot sequence makes a no-op.

### What the re-measurement changed (the filing is a hypothesis)

Measured on the pin, isolated profile (scratch HOME/XDG/`TLDW_CONFIG_PATH`), one
process, after warming the initial route -- which is the real boot condition, see
below:

| | |
|---|---|
| routes in the pass | 21 |
| whole pass, initial screen NOT yet warm | 704 ms (`chat` alone 206 ms) |
| **whole pass, initial screen already warm (the real case)** | **361 ms** |
| of which `library` 110.6 + `settings` 95.9 + `personas` 82.3 | **288.8 ms = 80%** |
| the other 18 routes, combined | 72.8 ms |
| package modules imported by the pass | 636 -> 1330 |

**That distribution is the finding.** Three modules are 80% of the cost and each is an
indivisible `import_module`; the other eighteen are 4 ms apiece. A flat inter-route
sleep -- what the AC asked for -- would hand the same courtesy to a 0.46 ms module as
to a 110 ms one and would not touch the three bursts that matter. So the gap is
`min(previous_import_cost x ratio, cap)`: the thread gives the loop back what it just
took. On the numbers above that inserts ~0.35 s of quiet across the pass on this
machine, turning a continuous competitor into a ~50%-duty-cycle one.

**The ordering AC is dead on arrival.** `_schedule_screen_preimport` is armed by
`_schedule_deferred_startup_work()`, the last statement of `_post_mount_setup()`, and
both boot paths run `_push_initial_screen()` to completion first
(`_run_no_splash_post_mount_setup` awaits push then setup; the splash handler pushes,
then `call_after_refresh(self._post_mount_setup)`). task-21110 additionally warms that
same module during the splash. So the configured default tab is *always* already in
`sys.modules` when this list is consulted -- and if the initial push raised, the pass
never runs at all. Reordering would have moved a dict lookup. Recorded in the code.

A useful side effect of the proportional gap: because `chat` (route #1) is already warm
its "cost" is ~0.2 ms, so the gap before `library` is ~0.2 ms -- the heavy trio still
lands early, exactly as `SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS` intends.

### A/B, interleaved, three arms

Harness: the main thread runs a 20 Hz tick doing a fixed slice of Python work (the
shape of a Textual render) while the pass runs on its daemon thread; the inter-tick gap
distribution is the GIL contention in question. One arm per process (imports are
process-global). The **control arm execs the merge-base `_preimport_screens`'s own
source**, extracted from `ceb4196a4` with `git show` + `ast`, not a paraphrase.
n=6 per arm, interleaved base / normal / low-core, medians:

| arm | pass elapsed | worst gap during pass | degraded ticks (>idle+10 ms) | total excess latency inflicted |
|---|---|---|---|---|
| base `ceb4196a4` | 0.37 s | 86.2 ms | 1 | 42.6 ms |
| after, normal tier | 0.77 s | 78.0 ms | 1 | 36.4 ms |
| after, low-core tier forced | 1.89 s | 75.2 ms | 1 | 23.8 ms |

Idle reference, identical in all three arms: median gap 54-55 ms, worst 55.1 ms, zero
gaps over 100 ms. So the pass really does inflict a visible spike -- 55 ms -> ~80 ms
for one tick -- in every arm.

**Read honestly: on this hardware the change is a wash, and that is the expected
result.** Excess latency 42.6 -> 36.4 -> 23.8 ms is the right direction but the
per-run ranges overlap (base 6.5-49.9, normal 25.3-52.1), and the worst single gap is
unchanged within noise because the gap cannot subdivide a 110 ms import. Expected harm
to a *randomly timed* keystroke is unchanged by construction: the same ~0.37 s of CPU
is spent either way, only spread over 2x (or 5x) the wall clock. What the spreading
buys is that a *burst* of interaction -- a user typing during the post-boot window --
now falls in a quiet gap about half the time instead of never, and on a 1-2 core box
those gaps are what let the loop drain its queue. That is the mechanism this task
targets, and it is not reproducible on an M-series; the review's own live A/B found no
first-interaction penalty here either.

**Not slower on normal hardware.** Nothing waits on this pass: it is armed strictly
after `_ui_ready` (pinned by the pre-existing
`test_screen_preimport_scheduled_only_after_ui_ready_flips`), and its longer wall time
is invisible to time-to-interactive. The full pre-import + navigation suites are green
unchanged (189 passed), including the end-to-end test that every registry module is
still warmed.

### Why throttle and not disable below 4 cores

Disabling would push each screen's import back onto the event loop at first
navigation -- work the user has actually asked for -- on the machines least able to
absorb it, to avoid speculative work whose measured harm here is one 25 ms tick. The
low-core tier is the same mechanism with 3x the yield and a 1.5 s cap, so a 400 ms
import on a slow box is followed by ~1.2 s of quiet.

### Coherence across the two pre-importers

Both share `_preimport_screens` and one enable switch (`TLDW_SCREEN_PREIMPORT`), and
now one core-count helper. The core-count question is deliberately answered differently
for them, and the reason is in both docstrings: the whole-registry pass is speculative,
so a slow machine should be throttled; task-21110's initial-screen warm-up is the boot's
own certain import, and moving it off the loop is worth *more* on a slow machine (that
task measured splash-close-to-usable -46% on a cold first boot). Throttling it would put
a certain cost back on the loop to dodge a speculative one. No branch was needed: the
gap and the park are strictly *between* routes, so a one-route list is untouched --
pinned by `test_single_route_preimport_is_untouched`, which asserts on the pause CALL,
not merely on the sleep (a zero gap sleeps nothing but still probes the nav lock).

### Quit / unmount walk

- `_preimport_screens` now re-checks `_shutting_down` before each route, so a quit
  mid-pass stops the thread instead of finishing the remaining registry. Both quit paths
  set that flag (`on_shutdown_request`, and the confirmed-quit path). It is read with
  `getattr(..., False)` so a lightweight test double cannot break the loop.
- The navigation park exits immediately on `_shutting_down` and is bounded at
  `SCREEN_PREIMPORT_MAX_NAVIGATION_POLLS` (5 s), so a navigation blocked on a confirm
  dialog throttles the pass rather than stranding the thread.
- The park counts polls rather than accumulating floats: summing 0.05 a hundred times
  lands either side of 5.0 depending on rounding, which made the bound off by one at
  random -- caught by the bounded-park test on its first run.
- The thread stays a daemon and holds no locks while sleeping; nothing new is awaited on
  the loop, so no `WorkerFailed` or unretrieved-task surface is added.
- Proved end to end by `test_preimport_thread_terminates_promptly_on_shutdown`: a real
  thread mid-pass over 200 routes, `_shutting_down` set, thread joins within 5 s.

### Tests

`Tests/UI/test_screen_preimport_pacing.py`, **18 tests**. Mutation matrix run
automatically, **13 deliberate defects, 13 caught** (one, "pause before every route
including the first", initially SURVIVED because a zero-cost first pause sleeps
nothing; the single-route test was strengthened to assert on the pause call itself and
it is now caught):

| mutant | caught by |
|---|---|
| no gap between routes | gap-between-routes, gap-tracks-cost |
| pause before every route incl. the first | gap-between-routes, single-route-untouched |
| flat gap, ignores the previous cost | gap-tracks-cost |
| no cap on the gap | gap-tracks-cost |
| low-core tier falls through to the normal tier | pacing-switches-tier |
| no navigation park | parks-while-locked, park-bounded, shutdown-breaks-park |
| park bound raised past the limit | park-bounded |
| navigation probe always reports idle | probe-reads-a-real-lock |
| no shutdown check in the route loop | shutdown-stops-the-pass |
| shutdown ignored while parked | shutdown-breaks-park |
| `_usable_cpu_count` trusts `os.cpu_count()` unguarded | cpu-count-never-zero |
| low-core tier no heavier than the normal tier | low-core-yields-more |
| low core disables the feature | low-core-does-not-disable, env-override x2 |

Suites green: `test_screen_preimport_pacing` (18) + `test_screen_preimport` (17) +
`test_splash_initial_screen_preimport` (24) + `test_screen_navigation` = **189 passed**.

**Modified/added files:** `tldw_chatbook/app.py`,
`Tests/UI/test_screen_preimport_pacing.py` (new).
