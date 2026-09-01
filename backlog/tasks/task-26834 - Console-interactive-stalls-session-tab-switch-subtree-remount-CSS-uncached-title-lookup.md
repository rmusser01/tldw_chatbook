---
id: TASK-26834
title: >-
  Console interactive stalls: session-tab switch, subtree remount CSS, uncached
  title lookup
status: To Do
assignee: []
created_date: '2026-09-01 06:23'
labels:
  - console
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In-terminal sampling of the real app (2026-08-31, iTerm2, 55 clicks) shows Console clicks are fast at the median (15.7ms to first paint) but carry a heavy tail: p95 735ms, max 1255ms. The tail, not the median, is the reported button sluggishness. Each stall over 100ms was stack-sampled at 10ms resolution and attributes to three causes below; screen NAVIGATION cost (nav-console 1255ms) is excluded here as already owned by TASK-1320/TASK-24452/TASK-24455.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking a session tab paints its answer in under 150ms on the reference terminal, or the remaining wait is attributed to a named worker operation with its own task
- [ ] #2 Interactions that restyle or remount a Console subtree no longer spend >100ms in stylesheet apply/selector matching on the main thread
- [ ] #3 _active_console_provider_model_display_uncached no longer performs a registry get_workspace per invocation on the click path
- [ ] #4 Improvements are measured with the same in-terminal sampler that produced the baseline, before and after
<!-- AC:END -->

## Evidence (real app, real terminal — not a headless harness)

Probe: `Screen._forward_event` stamps each Click; `Screen._compositor_refresh`
closes the window; a daemon thread samples the main thread's stack every 10ms
while a window is open. iTerm2 3.5.14, local, 55 clicks. Headless harnesses
measured this same app at ~10-30ms per click for a week and missed all of it —
none of the causes below reproduce without real session state.

```
all interactions (n=55): median 15.7ms  p95 734.8ms  max 1255.2ms
slowest: nav-console 1255ms(owned elsewhere) | workspace-tree 848ms |
         session-tab 735ms | RoleplayMarkdownParagraph 518ms |
         session-tab 402ms | model-search-picker-input 378ms
```

### Cause 1 — waiting, not computing (~720ms sampled in `selectors.py:select`)

The largest bucket has the main thread IDLE in the event loop while a click's
answering paint is pending. Session-tab clicks (735ms, 402ms, full=0) fit
this: the click dispatches work to a worker and the UI paints only on its
callback. The main-thread sampler cannot see into the worker; probe v3
(all-thread sampling) exists to name it. Do not start on this cause without
that attribution — "waiting" has looked like three different culprits already
this week and been wrong twice.

### Cause 2 — CSS selector matching over remounted subtrees (~550ms sampled)

Five distinct stacks, all `stylesheet.apply -> _check_rule -> match`, arriving
via `app.py:_register` (fresh mounts) and `update_nodes/update_styles`
(restyles). They pass through `textual_css_fastpath.py:127` — the NARROWED
branch, so the TASK-15450-era fastpath is installed and working; the residual
cost is per-NODE volume, not per-rule: a session-tab switch remounts a
transcript's worth of widgets and each pays narrowed matching. Same disease as
TASK-24452's "re-mints 559 widgets", but for in-Console interactions rather
than screen entry. Likely shape of a fix: reuse/recycle transcript widgets on
session switch instead of remounting.

### Cause 3 — uncached registry lookup on the click path (~60ms sampled)

`registry_service.py:499:get_workspace` inside
`workspace.py:3855:_console_workspace_session_title` inside
`chat_screen.py:6778:_active_console_provider_model_display_uncached`. The
function's own name says uncached; the samples say it runs during click
handling.

### Excluded, already owned

`click:nav-console` 1255ms is screen-switch cost: TASK-1320 (mount I/O off the
message pump, In Progress), TASK-24452 (559 re-minted widgets), TASK-24455
(598 query_one lookups). Evidence here corroborates them; no new task filed.

## Second run (v3, all-thread sampling): the waits are TIMERS, not work

The v3 probe sampled every thread while the main thread sat idle in a
click->paint window. Result: **no thread in the process was computing.**
The two large "other threads" buckets decode as noise, recorded so nobody
chases them:

- `thread.py:90:_worker`, 528 samples — idle ThreadPoolExecutor threads
  blocked in C-level `SimpleQueue.get` (the C frame is invisible, so the
  Python caller shows as top-of-stack). 528 ≈ 87 sampling ticks x ~6 pool
  threads.
- `ui_responsiveness.py:79:_drain_stalls`, 87 samples — the stall monitor's
  own drain thread, blocked the same way.

So the residual stall class (this run: `model-search-picker-input` 352ms and
192ms with 720/126-byte answering paints, `SelectOverlay` 300ms,
`ConsoleModelPopover` 142ms) is **scheduling latency**: the answering paint
arrives when a timer or deferred-callback chain fires, not when work
completes. The Console leans heavily on `call_after_refresh` chains, and each
hop waits for a refresh tick. The picker's `_BLUR_RESTORE_DELAY_SECONDS` is
only 0.05s, so no single constant explains 300ms — multi-hop chains are the
suspect, unproven.

Cause-2 CSS matching measured ~150ms total this session vs ~550ms in run 1 —
the cost tracks how much remounting the session's clicks did, consistent with
the per-node-volume reading.

**Instrument preserved** as `Helper_Scripts/console_latency_probe.py` (this
task's AC requires re-measuring with the same sampler; the scratchpad copy
dies with the session). Known limitation documented in its docstring: the
window closes at the FIRST paint, so interactions that ack early
under-report — median figures are optimistic, the tail is trustworthy.

**Next instrument, if the timer-chain suspicion needs proof:** log
`Timer.__init__`/`call_later`/`call_after_refresh` call sites with timestamps
while a window is open, so a 300ms wait decomposes into named hops.
