---
id: TASK-22214
title: Budget and genuinely pace the screen pre-importer's grown payload
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 04:51'
labels:
  - performance
  - startup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22214).

The whole-registry pre-importer payload grew +99 modules / +74,524 LOC since the pin
(568k -> 552k LOC compiled on a daemon thread; library_screen route 92,758 -> 135,933 LOC,
settings 43,762 -> 72,963). Pacing (`app.py:12725-12731`) is
`min(previous_cost * SCREEN_PREIMPORT_YIELD_RATIO, SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS)`
with the cap at 0.10 s (`app.py:795-796`) — for a 1.2 s route compile that is ~92% GIL
duty exactly while the user first touches the UI. `_usable_cpu_count` falls back to
`os.cpu_count()` on macOS (no sched_getaffinity), so laptops take the unthrottled tier.
Honest history: TASK-21113 shipped as a wash because a sleep cannot subdivide one
`import_module` — the lever here is payload size, route order, and the gap CAP, not finer
sleeping.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pre-importer GIL duty cycle over the first 5 s after mount is measured (tip) and reduced, or the payload is trimmed per-route with the top growers listed and justified
- [x] #2 The gap cap / yield ratio is retuned with measurements at both a high-core and a low-core tier, honestly reporting overlap if results wash (the 21113 precedent)
- [x] #3 First-navigation latency to Library and Settings measured before/after (the pre-import exists to protect it — do not trade it away silently)
- [x] #4 A payload budget (module count or LOC per route) is pinned so the next +30k LOC lands in review, not in users' laps
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Census the pre-importer payload per-route (marginal tldw modules + LOC in pass order, app already imported -- the pass's real condition); list top growers.
2. Build the duty-cycle probe first and run it against the CURRENT 0.10s cap (the red baseline): headless Pilot boot, thread-side per-route import-vs-gap bookkeeping + loop-side heartbeat sampler over the first 5s after mount; A/A control per the 22213 positional-bias lesson; both tiers (low-core via monkeypatched _usable_cpu_count).
3. Baseline first-navigation latency to Library and Settings (after-pass and mid-pass conditions).
4. Implement: raise the normal-tier and low-core gap caps so the proportional yield actually bites on the grown route costs; slice the gap sleep (22200 _interruptible_sleep precedent) so quit mid-gap exits within one slice; keep the heavy-first route order (reordering trades away exactly the first-navigation protection AC3 guards -- justify in code).
5. Pin a payload budget guard modeled on test_app_import_weight.py: total pass-added modules + LOC and a per-route marginal LOC cap, born at the measured census, with raise procedure + honest blind spots.
6. Re-measure duty + first-nav (both tiers); state the longer total pre-import window as the accepted cost.
7. Targeted suites + --collect-only sweep, tee everything; preflight; mutation tests (restore 0.10 cap -> cap-floor test reds; break budget -> guard reds); teardown walk on a real thread with a long gap in flight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped: the gap caps raised so the proportional yield actually binds (0.10 -> 2.0 s
normal, 1.5 -> 6.0 s low-core), the gap sleep sliced so a quit never waits one out, and a
new payload budget guard born at the measured census. Route order deliberately NOT
changed. The low-core half of the retune is an honest wash -- see below.

### The finding, confirmed as stated

The census (app already imported -- the pass's real condition) measures the pass at **715
modules / 564,326 LOC** beyond the app import, of which **478 modules / 365,692 LOC** is
beyond app+chat (chat is always a `sys.modules` dict hit at pass time). Top growers:
library 161 mods / 133,517 LOC, ccp/personas 66 / 53,429, settings 50 / 56,421,
watchlists_collections 38 / 30,279, stts 26 / 19,570. Eighteen routes cost under 20k LOC
each; four routes are 74% of the payload.

The cap really had turned the proportional yield back into the flat sleep it replaced.
Proof is the requested-gap series on a cold pass -- BEFORE `[0.0, 0.1, 0.1, 0.002, 0.003,
0.1]` (clipped flat on exactly the expensive routes), AFTER `[0.0, 0.529, 0.245, 0.003,
0.113, 0.303]` (tracking cost). library alone costs 156-183 ms warm and 525-615 ms
bytecode-compiling.

### Measurements (AC #1, #2)

Method: headless Pilot boot instrumented on both sides -- thread-side spans of every
`load_screen_class()` and every requested gap, loop-side 20 ms asyncio heartbeat. In-pass
GIL duty = import time / pass wall time. Isolated scratch profile. **A/A control run
first** (the TASK-22213 positional-bias lesson): 58.5% vs 59.8% pass duty, stall 492 vs
491 ms -- noise floor ~1.5 points. A/B pairs interleaved AND run in both orders.

| arm | duty before | duty after | worst 1 s busy |
|---|---|---|---|
| normal tier, warm (n=4/arm) | 49.7-58.0% | 47.4-47.8% | wash (~465 ms both) |
| normal tier, cold (n=2/arm) | 66.2-66.6% | 47.8-48.5% | 783 -> 681 ms |
| low-core tier, warm (n=2/arm) | 23.4-23.5% | 23.6-24.2% | **WASH** |
| low-core tier, cold (n=2/arm) | 24.1-25.0% | 23.7-24.1% | **WASH** |

**Read honestly.** The win is entirely on the normal tier, and it is largest on the
cold/bytecode-compiling arm (the stand-in for slow hardware): duty 66.4% -> 48.3% with
non-overlapping ranges, worst-1 s busy 783 -> 681 ms, also non-overlapping. On the warm
normal arm the duty ranges do not overlap either (before min 49.7 > after max 47.8) but
the loop-side metrics -- worst-1 s busy and total excess stall -- are a **wash**; on an
M-series the loop is not starved either way, exactly as TASK-21113 found.

**The low-core tier is a wash in both cache states and the raise there is hardening, not
a measured gain.** At ratio 3.0 the old 1.5 s cap was already nearly non-binding
(3 x 525 ms = 1.58 s), so it clipped exactly one route's gap by ~0.3 s across a ~6.6 s
pass. Raising it to 6.0 s removes that clip for hardware slower than anything measurable
here. Reported as overlap, per the 21113 precedent.

### First navigation (AC #3) -- not traded away

The accepted cost is a longer pass: warm 0.90-0.99 -> 1.14-1.24 s, cold 2.43-2.48 ->
3.43-3.51 s. Nothing waits on it. What that costs in warm-at times, measured from
`_ui_ready`:

| | before | after |
|---|---|---|
| library warm at (warm / cold) | 351 / 700 ms | 371 / 693 ms |
| settings warm at (warm / cold) | 499 / 1152 ms | 616 / 1510 ms |
| LAST route warm at (warm / cold) | 1119 / 2607 ms | 1373 / 3672 ms |

Library -- route #2, ahead of every large gap -- is unchanged. Settings slips 117 ms warm
/ 358 ms cold, and the last route ~254 ms warm / ~1.07 s cold. Against that, a click
landing MID-pass is measurably *faster*, because the thread is now usually in a gap
rather than mid-import: first-nav to Library at 0.35 s after ready, 3 interleaved pairs,
**63.5 -> 17.8 ms median** (ranges do not overlap); Settings 144 -> 109 ms median, with
its module not yet warm in *either* arm at that instant. After the pass completes, both
screens' modules are warm in both arms and the residual nav cost is compose/mount, which
this task does not touch (it is variable, 12-744 ms, in both arms alike).

Route order was left heavy-first and the reasoning recorded next to
`SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS`: moving the big routes last would shrink the early
window's payload but widen the window in which a first click to Library/Settings pays a
synchronous import on the loop -- trading away the exact thing the pre-importer exists to
protect.

### Payload budget (AC #4)

`Tests/Performance/test_screen_preimport_payload_budget.py`, modelled on the
import-weight guard: a subprocess census walking the pass's own route order, pinning
total pass-added modules (478, budget 500), total LOC (365,692, budget 380,000) and a
single-route LOC cap (library 133,517, budget 145,000). Headroom is deliberately smaller
than the +30k the AC names, so the next such growth reds. Documented blind spots: LOC is
a proxy for cost, not cost; marginal attribution is walk-order-dependent (trust totals
over rows); third-party payload is invisible; chat's own closure is excluded by design
(TASK-22213's census owns it). Raise procedure requires naming the routes that grew, in
the same commit, and the test prints the full table on pass so it can be diffed.

### Teardown

The gap sleep is now sliced into 0.05 s steps with a `_shutting_down` check between them
(the TASK-22200 `_interruptible_sleep` precedent). Walked on a real daemon thread with a
real 2.0 s gap in flight: **quit -> thread exit 0.008 s**. Same walk against the
un-sliced version: **1.801 s**. Without slicing, the wider cap would have made every quit
during the post-boot window wait out a gap.

### Tests / mutation

Suites: pre-import pacing 21 + preimport 17 + splash-initial 24 + guards (import weight,
`_ui_ready` census, payload budget) = **68 passed, 3 skipped**. Full four-suite run
including `test_screen_navigation` was 193 passed / 1 failed --
`test_action_library_note_editor_back_honors_dirty_guard`, which fails identically with
this branch's `app.py` replaced by base `f0e896122`'s: **a pre-existing dev red, not
this change** (bound-method identity in the Library notes editor; nothing to do with
pre-import). `--collect-only` sweep: 59,600 collected, 28 errors, all missing optional
deps (numpy/playwright) in Audio/TTS/RAG/Transcription/Web_Scraping.

Mutation matrix, 6 mutants, 5 caught + 1 informative survivor:

| mutant | result |
|---|---|
| restore the 0.10 s cap | CAUGHT (cap-floor test) |
| un-slice the gap sleep | CAUGHT (slicing + shutdown tests) + walk 1.801 s |
| slice but ignore `_shutting_down` | CAUGHT (shutdown-mid-gap test) |
| budget one LOC under the measured census | CAUGHT (census reproduces to the exact LOC) |
| census walks routes but imports nothing | CAUGHT (anti-vacuity) |
| import an ALREADY-resident module on a route | survived -- **correctly**: it adds no payload. The real-growth version (+40k LOC module on the workflows route) is CAUGHT and names the route |

### Modified/added files

`tldw_chatbook/app.py` (caps + sliced gap + the measurement record and the route-order
decision in comments), `Tests/UI/test_screen_preimport_pacing.py` (+3 tests; the gap
arithmetic test now asserts on the pause call, since slicing changed the raw sleep
shape), `Tests/Performance/test_screen_preimport_payload_budget.py` (new).
<!-- SECTION:NOTES:END -->
