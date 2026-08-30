---
id: TASK-24452
title: Console re-mints 559 widgets on every visit with no warm benefit
status: To Do
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
- [ ] #1 Repeat visits to the Console construct materially fewer widgets than the first visit
- [ ] #2 The number of `update_nodes` passes during a Console screen switch is reduced relative to the pre-change baseline
- [ ] #3 Console screen-switch wall time improves measurably in an interleaved A/B on the same machine
- [ ] #4 A guard pins the per-visit widget construction count so the warm path cannot silently regress
- [ ] #5 Console behaviour and focus placement are unchanged across a switch-away-and-back cycle
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
