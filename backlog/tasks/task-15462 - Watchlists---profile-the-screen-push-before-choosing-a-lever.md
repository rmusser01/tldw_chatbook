---
id: TASK-15462
title: 'Watchlists: profile the screen push before choosing a lever'
status: Done
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: Watchlists is the heaviest screen never profiled — 0.89 s on fast hardware, no deferral shipped, no widget survey, and `compose` runs `resolve_latest_follow_item()` inline (`watchlists_collections_screen.py:2554`). Per the owner's stability preference this is investigation-first: run the task-2725 method (widget survey + cProfile of one push) BEFORE choosing a lever — the series' headline lesson is that hidden-widget weight predicts nothing when a screen is sync/DB-bound (Schedules' 1.11 s evaporated; Console's deferral measured 4-8% and was reverted). If widget-bound, apply the established defer-past-first-paint recipe (traps banked in tasks 2725/2900/2901); if service/DB-bound, the levers are tasks 15463/15464.

Depends on: 15460/15461 landing first will change the profile — run the profile at whatever order is current and say so. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A recorded cProfile + widget survey of one Watchlists push names the top costs, committed to the task
- [x] #2 The chosen lever (deferral, service, or none) is justified against the profile; a wrong-lever conclusion is recorded like task-2902 if applicable
- [x] #3 If a lever ships: first-paint latency before/after plus the recipe's mechanism and integrity tests — N/A, marked complete honestly: **no lever ships**. All three levers this task named were measured and refuted (below); the one lever the profile actually found is handed over as a follow-up rather than riddened onto an investigation task, exactly as task-2725 declined to rider its own recommended fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fresh baseline first — the audit's 0.89 s predates 15460/15461/15463/15464.
   Measure live (real app, real terminal) AND headless, both on one seeded
   corpus, both against the other screens so a ratio exists.
2. task-2725 method: widget survey (total / hidden / top hidden roots) +
   cProfile of one warm push.
3. Instrument the application layer directly (per-method timers + a
   thread-exact sqlite trace on the loop connection) so "framework vs app"
   is measured, not inferred from a framework-dominated profile.
4. Decide against the profile: deferral / service-data / batching the swap
   mounts / none. Refute or adopt each with a number.
5. Ship only a lever the evidence supports; otherwise record and hand over.
<!-- SECTION:PLAN:END -->

## Investigation Notes (2026-08-13 — profiled, all three named levers refuted, NO LEVER SHIPPED)

<!-- SECTION:NOTES:BEGIN -->
Profiled at dev `6ee09eebc` — i.e. **after** all four of this programme's
Watchlists merges (15460 in-place filtering, 15461 scoped rebuilds, 15463
backend singleton/off-loop, 15464 items query). Corpus seeded through the
real DB APIs: 3 watchlists, 12 sources, 300 items (~3 KB bodies), 40 runs,
6 briefings, 5 alert rules. Isolated HOME/XDG/`TLDW_CONFIG_PATH` throughout;
no repo or live-config writes.

### 1. Fresh baseline — the audit's 0.89 s is now 0.50 s

**Live** (real `TldwCli` under tmux 235x52, real config/services/stylesheet;
only the measurement injected — `handle_screen_navigation` wrapped to log its
own duration, which is the audit's click→nav-highlight metric; navigation
driven through the command palette; 3 reps after warming every route's
module import):

| route | median | runs |
|---|---|---|
| logs | 110 ms | 124/110/78 |
| workflows | 113 ms | 117/106/113 |
| artifacts | 171 ms | 650/170/171 |
| library | 175 ms | 310/175/171 |
| mcp | 226 ms | 291/226/216 |
| settings | 237 ms | 251/224/237 |
| schedules | 263 ms | 481/263/240 |
| **watchlists_collections** | **498 ms** | 620/445/498 |
| personas (Roleplay) | 580 ms | 732/570/580 |

Median of the other eight: **201 ms**. Watchlists is **2.48×** it — outside
task-2725's ≤2×-median rule, and stated that way rather than rounded down.
But in absolute terms the four merges moved it **0.89 s → 0.50 s (-44%)**,
and it is no longer the app's heaviest screen (Roleplay is, again).

**Headless** (same harness, `run_test(size=(235,52))`, 3 reps): watchlists
328 ms paint / 719 ms settled; median of the other eight 197 ms → 1.66×.
Headless is used below for everything that needs repeatability.

### 2. Widget survey — nothing to defer

318 widgets at settle, **10 hidden (3%)**. Every hidden root is a single
widget: `Static#items-new-items-pill`, `Static#items-empty-state`,
`Static#footer-token-count`, `Static#internal-db-size-indicator`,
`Static#nav-overflow-hint-left`, `Button#nav-overflow-hint`, two
`SelectOverlay`, two `NonSelectableStatic`. Top classes: `Static` 129,
`_ArticleRow` 100, `Button` 26, `NavigationButton` 13, `_DayHeader` 12.

For contrast, the screen the recipe was built for (task-2725, Personas) had
494 widgets with **358 hidden in one stack**. There is no deferrable mass
here at all.

### 3. cProfile + application-layer instrumentation — it is not the app

cProfile of one warm push (2.886 s under the profiler; ~8.9 M calls). Top
cumulative entries are Textual, top to bottom: `stylesheet.apply` 723 calls
/ 1.184 s, `update_node_styles`→`App.update_styles` **221 subtree sweeps** /
0.651 s, `Widget._compose` 503 / 0.765 s, `widget.mount` 153 / 0.275 s,
compositor `render_update` 0.277 s, `css/model.py:__hash__` **2,167,650
calls**. The 2725 shape exactly: widget count is the lever, per-widget CSS
cost the multiplier.

Application code, sliced out of the same profile (`tldw_chatbook/` only),
totals **~5 ms of tottime**. Largest app-code entries are thin wrappers
around framework work: `watchlist_tree.compose` 0.159 s cumulative,
`article_list.watch_items`/`_rebuild_rows` 0.145 s,
`watchlists_tab_strip.compose` 0.084 s.

Direct instrumentation (per-method timers + `sqlite3.Connection.
set_trace_callback` on the loop thread's connection, which is thread-exact
rather than timing-based) confirms it independently:

| | |
|---|---|
| loop-thread sqlite statements, whole push | **13** (all sub-ms; 10 distinct) |
| `SubscriptionsDB.__init__` during a push | **0** (15463 holds) |
| every screen/controller/service method, summed | **~10 ms**, nesting double-counted |
| `resolve_latest_follow_item` (the audit's compose-inline concern) | 3 calls, **0.1–0.3 ms total** |

**`compose` running `resolve_latest_follow_item()` inline is a non-issue.**
It is memoized per `WatchlistsConsoleHandoff` instance
(`_latest_console_follow_loaded`), so it resolves once per screen and costs
tenths of a millisecond. The audit flagged it from the code, correctly; the
measurement retires it.

### 4. The structural fact: a median screen plus a 224-widget feed

Item-count sweep (sources/runs/briefings held constant, feed page varied;
4 doses, headless, 4–5 reps each):

| items in feed | widgets at paint | paint median | settled median |
|---|---|---|---|
| 0 | **86** | **200 ms** | 434 ms |
| 24 | 170 | 218 ms | 580 ms |
| 60 | 260 | 244 ms | 665 ms |
| 100 (page cap) | 344 | 342 ms | 720 ms |

Monotone dose-response, slope **~0.55 ms/widget** at paint (~1.1 ms/widget
at settle). Read it as: **the Watchlists screen's own chrome pushes in
200 ms — dead on the median of every other screen. Everything above the
median is the article feed**, and the feed is 224 widgets because
`_load_items` pages at a hard-coded `limit=100` and every `_ArticleRow` /
`_DayHeader` is a `ListItem` wrapping one `Static` (112 rows × 2).

The rows are built exactly **once** per push (`_ArticleRow.__init__` = 100
calls; `_build_rows` runs 3× but twice over an empty list). No duplicated
work to remove.

### 5. The three named levers, each refuted with a number

**(a) Defer-past-first-paint — REFUTED.** 3% of the tree is hidden, all of
it single widgets. There is nothing to defer. The one thing that *could* be
deferred past paint is the feed itself (the 112 rows mount inside the switch
window: 310 of the 318 widgets are present when `handle_screen_navigation`
returns) — and deferring it is deferring the screen's payload. On Personas
the deferred mass was non-active-mode surfaces nobody was waiting for; here
the rows *are* what the user came to read, so moving them past first paint
moves the pixels, not the wait. Declined on those grounds, not on cost.

**(b) A service/data fix — REFUTED, and already done.** 13 sqlite statements
on the loop thread for the entire push, one `SubscriptionsDB`, ~10 ms of
service time. 15463 and 15464 took this lever; there is nothing left on it.

**(c) Batching the swap mounts — REFUTED for the push path, and it found a
real (but immaterial) defect on the way.** Exactly **one** region swap
happens per push, and tracing it showed why: the screen's reactive default
is `region_layout = RegionLayout()` (nothing collapsed), while the shipped
first-run default is `_FIRST_RUN_DEFAULT = RegionLayout(collapsed={RIGHT_
RAIL})`. So every single visit **composes the expanded Inspector rail, and
`on_mount`'s `_apply_layout(load_region_layout())` immediately tears it down
and mounts the one-line collapsed header instead**. `on_mount`'s own comment
documents the correctness half of this ("the WatchlistsWorkbench child was
built with whatever `region_layout` held at THAT moment"); nobody had
measured the waste half.

Measured, noise-free, by instrumenting the swap: **13 widgets discarded, 1
mounted in their place.** At the sweep's 0.55 ms/widget that is ~5–10 ms of
a ~450 ms push — **1–2%**. A prototype of the fix (seed `region_layout` from
`load_region_layout()` before compose) was built and verified to remove the
swap entirely (`_apply_layout` sees `equal=True`, zero `_swap_region_widget`
calls, identical 310/318 final widget counts) — and a paired A/B could not
detect it: **preseed faster in 6/12 pairs, median delta −1 ms.** Not shipped:
1–2%, against moving a config-writing loader (`load_region_layout` performs
a one-time synchronous migration write) into screen construction and ahead
of the `_last_persisted_collapsed` priming that `_schedule_layout_persist`'s
no-op guard depends on. Recorded here so the next reader does not re-derive
it. (Batching remains relevant to *section switches*, which 15461 measured
separately; it is simply not in the push path.)

### 6. The one lever the profile does point at — handed over, not riddened

Collapse `_ArticleRow` / `_DayHeader` from a `ListItem`-wrapping-a-`Static`
into a single self-rendering `ListItem`. That removes **112 widgets**, 36%
of the screen's tree. Two independent estimates agree: the sweep slope
(112 × 0.55 ms ≈ 62 ms ≈ **15–18%** of the push) and a prototype in which
the rows render the same `Text` with no child widget (interleaved runs:
paint 414/479 ms → 335/372 ms). It is not shipped here because it is a
structural rewrite of `article_list.py` — a file hardened by 15460 three
tasks ago — requiring an audit of `.article-row`/`.article-day-header` and
any `ListItem > Static` selectors across the 21,479-line bundle, plus
rerouting `_repaint_row`'s `query_one(Static).update()`, plus re-validating
in-place filtering, `j`/`k` cursor skipping and highlight styling. That is
its own reviewable change, not a rider on a profiling task — the identical
call task-2725 made about its own recommended fix. **A follow-up task is
being filed by the controlling session** (not from this worktree: five-digit
ids need a repo-wide sweep). A cheaper variant worth considering in the same
task: `_load_items`' hard-coded `limit=100` is not viewport-proportionate.

### 7. Method note worth keeping

On this machine, cross-process A/B of a screen push is **not resolvable**
below ~30%: repeated identical configurations ranged 360–925 ms within a
single run. A naive dev-first-then-preseed comparison across processes
reported the layout pre-seed as "35% faster"; the same change under paired
ABBA ordering came out at −1 ms, and the noise-free widget count said 1–2%
all along. **Widget count was the reliable predictor here and wall-clock A/B
was not** — the exact mirror of the defer series' lesson, which is that
widget count *over*-predicts when a screen is sync/DB-bound (Schedules,
Console). Watchlists is neither: it is genuinely widget-bound, so counting
widgets beats timing them. Any future claim on this screen should be
anchored on a widget count or a dose-response sweep and only then confirmed
by wall clock.

### Verdict

Watchlists is **widget-bound, not DB-bound and not deferrable**. Its own
chrome is a median-cost screen; its excess over the median is 224 widgets of
article feed that the user opened the screen to read. The four merged tasks
took it from 0.89 s to 0.50 s. The remaining prize is one structural change
worth ~15–18%, filed separately. No lever ships from this task.

Evidence produced: `Docs/Design/2026-08-11-input-latency-audit.md` (Watchlists
row updated to point here). Probe recipe: seed via the real DB APIs against
a `_build_test_app` instance before `run_test`; drive
`app.handle_screen_navigation(NavigateToScreen(route))` directly (the
production handler, minus only the worker-dispatch hop); survey with
`screen.query("*")` walking each node's ancestor chain for `display`;
cProfile enabled around the whole push so loop callbacks are captured;
`set_trace_callback` on `app.subscriptions_db.conn` with a
`threading.get_ident()` guard for loop-thread-exact sqlite counting.
<!-- SECTION:NOTES:END -->
