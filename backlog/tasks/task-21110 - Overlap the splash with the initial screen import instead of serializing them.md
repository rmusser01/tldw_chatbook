---
id: TASK-21110
title: >-
  Overlap the splash with the initial screen import instead of serializing them
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21110).

With the splash enabled (default, 1.5 s), boot is strictly serial: `__init__` -> splash runs
1.5 s doing nothing else -> splash closes -> THEN chat_screen (~20k lines + closure) imports
and composes on the loop (app.py:8343-8386 -> 12611-12642 -> 11166-11222). The screen
pre-importer only starts after `_post_mount_setup`, so it cannot help the first screen. The
1.5 s window is pure wasted overlap on exactly the machines that hurt.

## Acceptance Criteria

- [x] The resolved initial route's screen module import is kicked off (on a thread) when the splash mounts, so splash time overlaps the import
- [x] Time-to-interactive with splash on, measured on the isolated-profile probe, improves by roughly the warm import cost of the initial screen; numbers recorded in the task
- [x] Boot with splash disabled is unchanged; no import races introduced (the existing per-module import lock semantics are relied on, not fought)
- [x] The warm-up resolves the initial route through the same alias / shell-destination lookup the real push uses, so it can never warm a different module than the one that gets imported
- [x] The splash-animation cost of the overlap is measured, not assumed, and the start delay is chosen from those numbers

## Implementation Plan

1. Build the measurement harness first: a one-boot-per-process, isolated-profile probe that
   records splash mount/close, the initial `import_module` (with the thread that paid for it),
   the screen push, and the first frame after it -- plus every splash animation frame, so the
   stutter cost of a GIL-holding import thread is measurable rather than assumed.
2. Take a baseline on the unmodified branch; only then implement.
3. Add a `resolve_screen_route()` to the screen registry: the existing lazy lookup, minus the
   `load_screen_class()` call, so a caller can learn which module the initial screen needs
   without importing it on the loop.
4. Schedule the existing `_preimport_screens` worker for that single route on a daemon thread,
   from the splash branch of `on_mount`, behind the pre-importer's existing on/off gate.
5. Sweep the start delay (0.0 / 0.2 / 0.5 s) against both time-to-interactive and animation
   frame timing, interleaved so machine drift hits every arm equally; pick from the numbers.
6. Walk the failure paths live: quit during the splash, an import that raises on the thread,
   splash disabled, splash dismissed mid-import, first boot after upgrade (cold bytecode).
7. Tests, each mutation-checked against a deliberately broken implementation.

## Implementation Notes

`TldwCli` now warms the initial screen's module on a daemon thread 0.2 s after the splash
mounts, instead of importing it on the event loop 1.5 s later when the splash closes. This
moves a start time; the machinery is the pre-importer task-15472 already shipped
(`_preimport_screens`, with its per-module-import-lock race semantics) invoked for one route.

**Changed**

- `UI/Navigation/screen_registry.py`: new `resolve_screen_route(target)` -- the existing
  `_lookup_route` (aliases -> direct routes -> shell destination model) exposed publicly,
  returning the `ScreenRoute` without calling `load_screen_class()`. Same resolution the real
  push goes through, so the warmed module cannot drift from the imported one.
- `app.py`: `SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS = 0.2`, a new
  `_initial_screen_preimport_thread` handle, `_initial_screen_preimport_route()` and
  `_schedule_initial_screen_preimport()`, and one `else:` branch in `on_mount` -- the exact
  site where, with the splash up, the pre-existing code scheduled nothing at all.

**Measured — isolated profile, scratch HOME/XDG/`TLDW_CONFIG_PATH`, splash pinned to one card,
one boot per process, arms interleaved (the machine was under heavy concurrent load from other
sessions; interleaving is what makes the medians comparable). n=10 per arm, medians:**

| phase | splash overlap OFF | ON (0.2 s) |
|---|---|---|
| initial import paid ON the event loop | **0.307 s** | **0.000 s** |
| initial import paid on the thread | 0 | 0.338 s |
| splash close -> screen usable (first frame) | **1.410 s** | **1.083 s** (-0.327 s, -23%) |
| whole-process time to interactive | 4.576 s | 4.282 s (-0.294 s) |

First boot after an upgrade (every `.pyc` in the package discarded before each boot), n=4:

| phase | OFF | ON (0.2 s) |
|---|---|---|
| initial import on the loop | **0.944 s** | 0.000 s |
| splash close -> screen usable | **2.103 s** | **1.128 s** (-0.975 s, -46%) |
| whole-process time to interactive | 6.857 s | 6.007 s (-0.850 s) |

The work moved rather than relocating: of the two phases after the push, `mount+post` rose
0.867 -> 0.972 s and `->first_frame` fell 0.209 -> 0.119 s, a net +0.015 s -- noise across the
boundary between them, not the 0.307 s reappearing. The end-to-end span that contains both
(splash close -> first frame) is what fell by 0.327 s.

**The tradeoff, measured.** The splash animates on the event loop at 20 Hz and the import
thread holds the GIL, so the overlap costs animation smoothness. Same interleaved arms, frames
rendered during a 1.5 s splash (ideal 30):

| arm | frames | worst gap | p95 gap | gaps >100 ms / 10 boots | close -> usable |
|---|---|---|---|---|---|
| no warm-up | 30 | 51.0 ms | 50.9 ms | 0 | 1.410 s |
| 0.0 s delay | 28 | 111.5 ms | 69.8 ms | 6 | 1.106 s |
| **0.2 s delay** | **30** | **86.8 ms** | **52.9 ms** | **2** | **1.083 s** |
| 0.5 s delay | 30 | 83.6 ms | 51.8 ms | 2 | 1.087 s |

0.2 s was chosen: it recovers the dropped frames and nearly all of the p95 that a 0 s start
costs, at no boot-time cost. 0.5 s is not measurably smoother and eats headroom the case with
the most to gain cannot spare -- a cold first-boot import takes 0.94 s, which fits inside a
1.5 s splash from 0.2 s but not from 0.5 s. This is a real, if small, regression in splash
smoothness (worst inter-frame gap 51 -> 87 ms, ~2 late frames per boot on a fast machine, more
on the low-core hardware this targets) and it is deliberate.

**Failure paths, walked live (each A/B'd against `TLDW_SCREEN_PREIMPORT=0` on the same code):**

1. *Quit during the splash* -- Ctrl+Q with the thread mid-import: clean exit, no unhandled
   exception, quit not delayed. Re-run with the import artificially stretched to 4 s so the
   daemon thread was provably still inside `import_module` at interpreter teardown: exit code
   0, no `Fatal Python error`, no `Exception ignored`.
2. *Import raises on the background thread* -- a `RuntimeError` (which the registry does not
   catch) is swallowed by `_preimport_screens`, nothing is cached in `sys.modules`, and the
   real push then raises the identical error to the identical caller. Verified byte-for-byte
   against the control arm. It does not vanish and does not leave a broken first screen.
3. *Splash disabled* -- the new branch is not reached, no thread is created, boot is
   byte-identical. Covered by a test as well as the live probe.
4. *Splash dismissed early, mid-import* -- dismiss -> screen pushed took 1117 ms with the
   warm-up and 1144 ms without: the main thread blocks on CPython's own per-module import
   lock and is no worse off than the synchronous import it replaced. When the delay outlasts
   the splash entirely, the `splash_screen_active` guard means no thread is ever started.
5. *First boot after upgrade* -- the largest win (table above); the 0.94 s cold import still
   completes inside the splash window from a 0.2 s start.

**Two things the work proved wrong, both recorded because they cost real time:**

- **Textual 8's `set_timer(0.0)` never fires.** `Timer._run` computes `(now - start) /
  interval`, so a zero interval raises `ZeroDivisionError` inside the timer's own task and the
  callback is silently dropped (nobody retrieves that task's exception). The first delay sweep
  showed the "0.0 s" arm as a stutter-free win *because no pre-import had happened at all* --
  its `import_on_loop` was still 430 ms. `on_mount` now falls back to `call_after_refresh` for
  a zero delay, and a test covers it.
- **Monkeypatching an `@on`-decorated handler on the class is invisible to the decorated
  dispatch.** Textual's metaclass captures decorated handlers as raw function objects at class
  creation, and the naming-convention fallback skips only functions carrying `_textual_on` --
  so patching `TldwCli.on_splash_screen_closed` left the original running *and* got the patched
  copy dispatched a second time. The first probe recorded "splash closed at 6.09 s" on a boot
  where it closed at 3.53 s. The probe instruments `SplashScreen.close` instead.

**Tests** -- `Tests/UI/test_splash_initial_screen_preimport.py`, 24 tests: route-resolution
parity with the real push (7 configured defaults including alias routes and the first-run Home
redirect), off-thread execution, idempotence, non-interference with the whole-registry
pre-importer in both orders, all four gates, failing/missing-module behaviour unchanged, the
zero-delay fallback, the delay-constant bounds, and two live `run_test()` boots asserting *which
thread* paid for the initial import and that a splash-disabled boot never creates the thread.
Every one was mutation-checked: 13 deliberate defects, all 13 caught.

**Modified/added files:** `tldw_chatbook/app.py`, `tldw_chatbook/UI/Navigation/screen_registry.py`,
`Tests/UI/test_splash_initial_screen_preimport.py`.
