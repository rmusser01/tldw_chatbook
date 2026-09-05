---
id: TASK-24452
title: Console re-mints 559 widgets on every visit with no warm benefit
status: In Progress
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - ui
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Switching to the Console constructs 559 widgets (212 Static, 108 Button, 44 Horizontal,
24 Vertical), performs 1,668 `stylesheet.apply()` calls, 402 `update_nodes` passes over 971
nodes, and 907 `set_class` calls -- every single visit. The measurement is identical cold and
warm (1.89-1.98 s), so nothing is cached or reused between visits.

The dominant chain is `Button.watch_flat` -> `set_class` -> `app.update_styles` ->
`stylesheet.update_nodes` -> 974 applies, accounting for roughly 1.5 s of the switch. 108
Buttons per visit each fire that watcher on construction.

Two independent levers: avoid re-minting the widget tree per visit, and collapse the 402
separate `update_nodes` passes into a batched update during construction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
Re-scoped 2026-09-04 (owner call: "tackle 24452"): the root cause is the
fresh-screen-instance-per-navigation architecture, and flipping Console to
reuse safely requires a per-subsystem lifecycle audit of its ~dozen
`on_unmount` teardowns. THIS task delivers the reuse mechanism, proven
end-to-end on the lowest-risk route; the Console and Library enablements
(where the measured wins are -80% and -95% switch CPU) are task-31520 and
task-31521, gated on their audits.

- [x] #1 An opt-in per-route screen-instance reuse mechanism exists (`ScreenRoute.reusable`): the instance is installed, suspended instead of unmounted on switch-away, and resumed on return
- [x] #2 Repeat visits to a reusable route resume the SAME instance with no widget re-mint, pinned by a guard test
- [x] #3 Reuse is scoped to the runtime identity that built the instance; an identity flip invalidates the cache (guard-tested)
- [x] #4 Per-visit refresh for the enabled route runs from `on_screen_resume`, guard-tested by mutation (deleting the hook fails a test)
- [x] #5 Non-reusable routes keep today's fresh-instance lifecycle, pinned by a guard test
- [x] #6 The enabled route's behaviour is unchanged across a switch-away-and-back cycle (screen-reuse suite + home/master-shell/navigation-recovery suites green; nav-suite reds shown identical to pristine dev)
- [ ] #7 Console re-mint elimination itself: moved to task-31520 (measured headroom recorded there); Library: task-31521
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
NOT IMPLEMENTED in the 2026-08-29 review pass. The root cause was identified and is larger than
this task as filed.

The app hands `switch_screen` a FRESHLY CONSTRUCTED screen every time and restores state from
`screen_state_store` (`app.py` ~11709). Verified live: three visits to Library produced three
distinct instance ids. That single decision is what makes the Console re-mint 559 widgets per
visit -- and it is also the cause of task-24456 (43 config reads per Library visit) and
task-24457 (8 SQLite connections per Library visit).

So the real fix is screen instance reuse, which is an architecture change: the snapshot/restore
design deliberately assumes fresh construction, and screens may rely on it. That is an owner
call, not a drive-by refactor.

The cheaper half of this task -- wrapping screen construction in `app.batch_update()` to collapse
the 402 separate `update_nodes` passes -- remains available and independent, and was not attempted.

Note that task-24450 already took ~175 ms off the Console switch (1.96 -> 1.78 s mean of 6) by
making each of those style applications cheaper rather than fewer.
<!-- SECTION:NOTES:END -->


## Implementation Plan (the how)

1. Measure whether Textual installed-screen reuse avoids the 2026-07-11
   freeze class and what it buys: bypass-probe Chat/Library instance
   switching + typing-cost check with a suspended screen mounted.
2. Add `ScreenRoute.reusable` (opt-in) + an app-side identity-scoped
   instance cache; integrate at `_complete_screen_navigation` (skip
   construction and snapshot-restore on a cache hit; install on first
   build).
3. Enable for Home (no unmount teardown, no timers); move its per-visit
   refresh to `on_screen_resume`.
4. Guard tests + mutation-test the flag and the resume hook.
5. Interleaved A/B through the real navigation handler; file the Console/
   Library enablement follow-ups with the measured headroom.

## Implementation Notes

Mechanism: `ScreenRoute.reusable` (default False) + TASK-24452 helpers in
`app.py` (`_reusable_navigation_screen` / `_retain_reusable_navigation_
screen`). A reusable route's screen is constructed once, INSTALLED
(`App.install_screen`), and re-switched to on later visits -- Textual
suspends installed screens instead of unmounting them, so the 2026-07-11
re-mount-races-teardown freeze cannot start (no teardown ever runs
mid-session). The cache is scoped to `RuntimeIdentity`; a local<->server
flip drops and disposes the instance (bounded documented leak if the flip
happens while the screen is current). Snapshot-restore is skipped on a
cache hit (the live instance IS the state); the outgoing save_state path
is unchanged (projection consumers).

Enabled for Home; its three refresh workers re-trigger from
`on_screen_resume` (exclusive groups coalesce the first-visit
mount+resume double-fire). Guard suite `Tests/UI/test_screen_reuse.py`
(5 tests); the route flag and the resume hook were each mutation-tested
red. Honest measurement note: Home arrivals show NO wall/CPU win through
the real nav handler -- the arrival window is dominated by the OUTGOING
screen's teardown, and Home was always cheap to build. The wins live in
Chat (-80% switch CPU) and Library (-95%), measured via installed-instance
bypass probes on 2026-09-04, and land with task-31520/31521 after their
lifecycle audits. Verified: screen-reuse + home + master-shell +
navigation-recovery suites green (142 passed); full test_screen_navigation
red set shown IDENTICAL to pristine dev (31 pre-existing, membership
flake-stable); perf-guard suite 26 passed.

Files: `UI/Navigation/screen_registry.py`, `app.py`,
`UI/Screens/home_screen.py`, `Tests/UI/test_screen_reuse.py`,
`backlog/tasks/task-31520`, `backlog/tasks/task-31521`.
