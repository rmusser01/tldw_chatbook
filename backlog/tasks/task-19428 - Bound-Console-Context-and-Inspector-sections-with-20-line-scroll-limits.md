---
id: TASK-19428
title: Bound Console Context and Inspector sections with 20-line scroll limits
status: In Progress
assignee: []
created_date: '2026-08-20 07:10'
updated_date: '2026-08-21 15:35'
labels:
  - console
  - ux
dependencies:
  - TASK-19638
  - TASK-19639
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep dense Context and Inspector sections readable in an expanded terminal by
showing up to 20 content lines per section, scrolling additional content inside
that section, and exposing product-standard fold hints wherever content remains
below a visible fold.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every direct named Context and Inspector section body grows naturally up to a 20-rendered-content-line ceiling; additional content scrolls inside its allocated viewport with a separate local `▼ more — scroll` hint row.
- [ ] #2 Section hints are absent when all 0-20 content lines fit the allocated viewport and at the scroll end, and reappear after scrolling upward while content remains below; exactly 21 rows at a full allocation yields 20 content rows plus the separate hint.
- [ ] #3 In normal header-fit mode, Context allocates shorter sections only what they need, prioritizes the most recently activated section, redistributes unused height up to 20 lines, and marks an unfunded open body `· no room` with a transient `[>]` reprioritization action.
- [ ] #4 When Context header chrome cannot physically fit, its outer body scrolls with a distinct pinned `▼ more sections — scroll` cue; every open non-empty body receives an honest base row, every header/body remains reachable, and persisted open preferences are unchanged.
- [ ] #5 Inspector retains the distinct pinned `▼ more sections — scroll` cue whenever content exceeds the counterfactual no-hint viewport; 10→11→10 content transitions and terminal resize add/remove the slot without sticky overflow, while local and outer active owners have dimensionally stable non-color focus treatment.
- [ ] #6 Pointer scrolling hands off at nested-scroll boundaries; Tab/Shift+Tab preserve header-control, viewport, and body-control DOM order; focused descendants auto-reveal; removed focus targets recover next-then-previous; and rail badges/collapse/reopen/responsive focus/stored preferences do not regress.
- [ ] #7 Inspector-local `n/p` commands implement the specified boundary and non-boundary anchors without wrapping or stealing editable input; footer hints refresh on Inspector focus transitions and F1 help evaluates active focus at invocation.
- [ ] #8 Every existing Inspector row/action has one semantic owner and Review Changes renders under Changes; injected STRICT policy rejects unknown ownership, while RESILIENT production keeps known sections, omits Other/unknown children, deduplicates safe stable-ID diagnostics, and clears incomplete status in place on the next valid state.
- [ ] #9 `ConsoleLeftRail` atomically reconciles all Context allocations from one post-refresh measurement snapshot after every named invalidation, every Context/Inspector mutation and outer-body resize requests coalesced reconciliation, Inspector changes invalidate the outer hint, and same-tick multi-section updates cannot mix old/new geometry.
- [ ] #10 Production-CSS Textual compositor tests cover 235x52 and 160x45 all-open expanded states plus default and explicit-open safeguards at 120x30 and 80x24, including 20/21 boundaries, normal constrained reprioritization, short-height outer fallback, content shrink, live mutation, resize, recompose, focus recovery, and scroll clamping.
- [ ] #11 The implementation follows ADR-077 and preserves TASK-15110's simultaneous-header outcome whenever headers physically fit, while keeping every header/body reachable through the short-height outer fallback.
- [ ] #12 Specialized child constraints no longer override the shared ceiling: Sources' legacy 6/10-row caps and Session Settings' CSS 9-row minimum plus inline 9-row maximum are retired, and a one-row settings body occupies one content row.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add pure Context allocation and fold-hint policies with unit tests.
2. Add the shared 20-content-line bounded-section widget with local hint, scroll, resize, and focus tests.
3. Integrate atomic Context allocation, active-section behavior, focus recovery, and named invalidations.
4. Enforce exhaustive Inspector ownership with STRICT development and RESILIENT production policies.
5. Wrap every Inspector group, retire Sources/Session Settings legacy caps, and reconcile live mutations.
6. Add counterfactual outer-fold hints plus Inspector-local n/p, footer, and live F1 behavior.
7. Prove expanded and constrained production-CSS geometry at 235x52, 160x45, 120x30, and 80x24.
8. Update the Console guide and TASK-19428 implementation notes.
9. Run only the focused changed-functionality tests and scoped static checks.

ADR required: yes
ADR path: backlog/decisions/077-console-bounded-rail-section-scrolling.md
Reason: ADR-077 defines the approved nested-scroll, ownership, focus, and constrained-height interaction model.
<!-- SECTION:PLAN:END -->
