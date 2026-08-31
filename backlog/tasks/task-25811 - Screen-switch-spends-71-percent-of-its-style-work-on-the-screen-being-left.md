---
id: task-25811
title: Screen switch spends 71% of its style work on the screen being left
status: To Do
assignee: []
created_date: '2026-08-30'
labels:
  - performance
  - console
  - navigation
priority: high
---

## Description (the why)

Navigating away from a screen restyles the screen you are leaving, and on
a large screen that dominates the switch.

Instrumenting `Stylesheet.apply` across an ordinary Console → Library →
Console → Library navigation, and attributing each apply to the screen its
node belongs to:

```
library#2: total_applies=1577
    1124 applies under ChatScreen#4992     <- the screen being LEFT
     247 applies under LibraryScreen#8608  <- the screen being BUILT
```

**71% of the switch's style work is spent on the outgoing screen**, at ~5.4
applies per node across a 207-node ChatScreen. The user is waiting for the
screen they asked for; most of the work is on the one they are done with.

## Evidence

dev `0ef6f3fd4e`. Reproducible to the call across three runs — the apply
counts were identical (332 / 389 / 1,577 / 384) every time:

| navigation | screen built | nodes | applies | apply ms | wall ms |
|---|---|---:|---:|---:|---:|
| → Library (1st) | LibraryScreen | 96 | 332 | 105.0 | 301.6 |
| → Console (1st) | ChatScreen | 207 | 389 | 79.9 | 160.1 |
| **→ Library (2nd)** | LibraryScreen | 96 | **1,577** | **540.0** | **1,003.2** |
| → Console (2nd) | ChatScreen | 207 | 384 | 76.4 | 103.4 |

CSS apply is **50–72% of switch wall time**. The 2nd Library visit costs
4.7× the 1st; the 1st is cheap only because the screen it left was the
small splash, not because anything got faster.

Live instance counts also climb over the same navigation
(`ChatScreen: 1 → 2`, `LibraryScreen: 1 → 2`), consistent with the
fresh-screen-per-switch behaviour already filed as TASK-24452. The new
part here is that a retained, no-longer-current instance keeps absorbing
style applies.

Full method: `Docs/Design/2026-08-30-holistic-perf-review.md` §3.

## Acceptance Criteria (the what)

- [ ] Establish WHY the outgoing screen is restyled — display/visibility
      change, teardown, or the app's own call — and record the trigger
      before changing anything
- [ ] Style work on a screen that is being left is eliminated or bounded,
      without breaking the case where that screen is later re-shown
- [ ] Applies for the Console → Library (2nd) navigation drop materially
      from the 1,577 baseline, measured with the same instrumentation
- [ ] The first and second visit to a screen cost comparable applies —
      the 4.7× asymmetry is the symptom to remove
- [ ] Confirm whether the retained non-current instances are a leak in
      their own right, or only reachable because TASK-24452's
      fresh-screen-per-switch keeps them alive; if a leak, file separately
      with its own evidence

## Notes

Interacts with TASK-25810: the outgoing-screen restyle pays that finding's
93% candidate overhead too, so landing 25810 first will shrink this
number without addressing its cause. Measure them independently.
