---
id: task-25811
title: Screen switch spends 71% of its style work on the screen being left
status: Won't Do
assignee: []
created_date: '2026-08-30'
labels:
  - performance
  - console
  - navigation
priority: high
---

## RETRACTED 2026-08-31 — the premise was a measurement artifact

**Do not implement this task.** The finding it was filed on is wrong.

`switch_screen` posts `ScreenResume` to the NEW screen, and
`post_message` is asynchronous. The probe's navigation helper returned
as soon as `app.screen` changed -- before that message drained -- so the
next measurement window opened with the PREVIOUS navigation's resume
still queued, and attributed it to the switch under test. The screen it
named as "being left" was the screen that had just become current.

Re-measured with a full message drain before each window:

| navigation | nodes | applies | apply ms | wall ms | on outgoing |
|---|---:|---:|---:|---:|---:|
| -> Library #1 | 97 | 286 | 50.4 | 252.1 | **0** |
| -> Console #1 | 265 | 485 | 78.6 | 232.8 | **0** |
| -> Library #2 | 97 | 286 | 45.5 | 116.9 | **0** |
| -> Console #2 | 265 | 485 | 75.5 | 248.8 | **0** |

**Zero** style work lands on the outgoing screen, and there is no 4.7x
revisit asymmetry -- applies are proportional to node count.

The reproducibility that made the original convincing (three runs,
identical counts) proved only that the artifact was deterministic.
Changing the window, not repeating it, was what exposed it.

**What survives, and needs no new task:** CSS apply is 20-39% of
screen-switch wall time (45-79 ms). That is real, smaller than claimed,
and already addressed by TASK-25810's ancestor filter. Screen instances
accumulating across visits is TASK-24452, filed separately and
unaffected.

Full correction: `Docs/Design/2026-08-30-holistic-perf-review.md` section 6.

---

## Original description (WRONG -- kept so the error stays legible)

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

## Root cause (AC #1 — established 2026-08-30)

Tracing `Screen._on_screen_resume` / `_on_screen_suspend` with instance
identity across one Console → Library navigation:

```
RESUME  ChatScreen#7872      <- the screen being LEFT, resumed FIRST
SUSPEND LibraryScreen#8560   <- an older RETAINED Library instance
RESUME  LibraryScreen#8624   <- the incoming screen
SUSPEND ChatScreen#7872      <- the outgoing screen, suspended right after
```

**The outgoing screen is resumed at the start of a navigation that is about
to replace it, then suspended moments later.** The resume runs
`_on_screen_resume -> dom.update_node_styles -> app.update_styles` across
its whole 207-node subtree, and all of that work is discarded.

Stack attribution of the outgoing screen's 1,107 applies:

| applies | trigger |
|---:|---|
| 499 (45%) | `_on_screen_resume` → `update_node_styles` → `app.update_styles` |
| 306 (28%) | `widget.mount` → `_compose` (widgets mounted INTO the screen being left) |
| 116 (10%) | `widget.update_styles` → `update_node_styles` |
| 57 | `stylesheet.update_nodes` |
| 31 | `widget.mount` → `app._register` |

Two leads for the resume itself, neither yet confirmed — establish which
before fixing: `_handle_screen_navigation_locked` calls
`_dismiss_navigation_overlays()` **before** `switch_screen`, and popping an
overlay resumes the screen beneath it; alternatively Textual's own
`switch_screen` stack handling may resume the top before replacing it. The
stack was `['Screen', 'ChatScreen']` (depth 2) at measurement time with no
overlay open, which argues against the first lead but does not settle it.

The 306 mount/compose applies are a separate question worth its own answer:
why are widgets being composed into a screen that is being replaced?

## Probe hazards for whoever picks this up

* A navigation helper that waits "until the screen has > N nodes" returns
  **immediately** when the current screen already qualifies. Wait for the
  screen **identity** to change and assert it did — one probe here reported
  `INCOMING == OUTGOING` because of exactly this.
* Bucket applies by explicit OUTGOING / INCOMING / RETAINED role. Lumping
  "not the outgoing screen" together produced 334/1,377 and appeared to
  refute the 1,107 figure; the corrected bucketing reproduced it exactly.
