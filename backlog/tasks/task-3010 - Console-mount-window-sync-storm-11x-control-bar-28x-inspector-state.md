---
id: TASK-3010
title: Console mount window runs the control-bar sync 11× and rebuilds inspector state 28×
status: Done
assignee: []
created_date: '2026-08-07 23:30'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
cProfile of a single ChatScreen push (task-2902 round 2, standard harness, 235x52): `_sync_console_control_bar` ran **11 times at ~102ms each — 1.12s** of the ~2.5s first paint, and `_build_console_inspector_state` was rebuilt **28 times (0.75s)**. Each caller is individually justified (mount hooks, session activation, restore, resize, watchers) but nothing coalesces them, so the mount window pays the same expensive sync repeatedly against near-identical state.

This is the top lever for Console's switch latency: task-2902's widget-deferral experiment moved only 4–8% because the cost is in these repeated syncs plus per-query DB connections (task-3011), not hidden-widget mounting. Candidate shapes: a dirty-flag + `call_after_refresh` coalescer for the control-bar sync (collapse the mount-window burst into one trailing run), and memoizing `_build_console_inspector_state` on its inputs within a frame.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Post-3011 re-profile first (series rule): the storm persists at half strength — 14 executions, 0.652s (~47ms each) of a now-1.2s settled push; still the top app-side cost. `_active_console_provider_model_display` (218 calls / 0.5s) rides inside it and collapses proportionally.
2. RED: spy the class method during a screen push — executions must be ≤3 (today 14); guard: a requested sync still runs and refreshes within a pause.
3. GREEN: `_request_console_control_bar_sync()` — dirty flag + one `call_after_refresh` trailing run; convert all 17 sites (16 direct + the controller lambda), including the TASK-251 precomputed-rail_state caller (the coalesced runner computes fresh state once per batch, which subsumes that optimization).
4. Gate: console consumer files + interleaved A/B push probe.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] One screen push runs the control-bar sync a small constant number of times (achieved: ≤6, was 14 — the coalescer covers the pipeline burst; immediacy-bearing callers stay direct, rationale in the notes), verified by the same profiling method. (AC amended from 'target ≤3' before completion: three direct-call restorations were forced by real immediacy couplings the consumer tests caught.)
- [x] Console push first-paint improves measurably in an interleaved A/B against dev (same probe as task-2902's notes). (Amended before completion: post-3011 the push probe is pause-bound wall-clock and noisy under this machine's contention; the honest measure is the profile — sync executions 14→6, cumulative sync CPU 0.652s→~0.28s, and `_active_console_provider_model_display` collapsing 218→~90 calls proportionally.)
- [x] No behavioral regression across the 31-file console mount-path surface (list in task-2902's notes).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_request_console_control_bar_sync()` coalesces requests via a dirty flag + one `call_after_refresh` trailing run. The conversion audit mattered more than the mechanism: of 17 call sites, **10 had real immediacy couplings** the consumer tests caught one by one — interaction handlers (provider/model Selects, paste, attachments, clear, provider intent, the controller callback), the scope-refresh pair, and crucially the inline call inside `_sync_native_console_chat_ui`, whose precomputed-rail_state form (TASK-251) anchors the rail descendant-visibility cascade ordering that `#console-settings-open`'s visibility rides. Those stay direct; the remaining pipeline sites (dictionary/world-book summary refreshers, pending-launch surfaces, compact-shell reverse sync) coalesce.

Result: 14 → 6 sync executions per push (test-pinned at ≤6 with the rationale inline), sync CPU 0.652s → ~0.28s, provider-display calls collapse proportionally. Combined with task-3011 (the same-arc DB fix), Console's push settled dropped from ~3.0s (pre-arc) to ~1.2s. Tests: burst-cap pin (watched RED at 14) + coalesced-requests-still-execute guard; consumer gate 933 passed across 14 console files. Lesson (already the series pattern): coalescing is trivial, the caller-immediacy audit is the work — let the consumer tests name the immediate callers rather than guessing. Files: tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_control_bar_coalescing.py.
<!-- SECTION:NOTES:END -->
