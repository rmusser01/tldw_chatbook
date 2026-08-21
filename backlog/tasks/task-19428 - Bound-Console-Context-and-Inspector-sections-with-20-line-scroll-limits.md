---
id: TASK-19428
title: Bound Console Context and Inspector sections with 20-line scroll limits
status: To Do
assignee: []
created_date: '2026-08-20 07:10'
labels:
  - console
  - ux
dependencies:
  - TASK-19638
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
- [ ] #5 Inspector retains the distinct pinned `▼ more sections — scroll` cue whenever outer content remains below; local and outer active owners have dimensionally stable non-color focus treatment.
- [ ] #6 Pointer scrolling hands off at nested-scroll boundaries; Tab/Shift+Tab traverse viewport then enabled descendants, focused descendants auto-reveal, removed focus targets recover deterministically, and rail badges/collapse/reopen/responsive focus/stored preferences do not regress.
- [ ] #7 Inspector-local `n/p` commands navigate direct boundaries without wrapping or stealing printable input; context-sensitive footer/F1 help advertises them only while Inspector is active.
- [ ] #8 Every existing Inspector row/action has one semantic owner, Review Changes renders under Changes, and unknown ownership raises in test/development rather than producing an Other section; production shows a compact incomplete-data status and stable diagnostic.
- [ ] #9 Every named Context/Inspector mutation path explicitly requests coalesced post-refresh reconciliation, and Inspector section changes also invalidate the outer hint.
- [ ] #10 Production-CSS Textual compositor tests cover 235x52 and 160x45 all-open expanded states plus default and explicit-open safeguards at 120x30 and 80x24, including 20/21 boundaries, normal constrained reprioritization, short-height outer fallback, content shrink, live mutation, resize, recompose, focus recovery, and scroll clamping.
- [ ] #11 The implementation follows ADR-077 and preserves TASK-15110's simultaneous-header outcome whenever headers physically fit, while keeping every header/body reachable through the short-height outer fallback.
- [ ] #12 Specialized child constraints no longer override the shared ceiling: Sources' legacy 6/10-row caps and Session Settings' CSS 9-row minimum plus inline 9-row maximum are retired, and a one-row settings body occupies one content row.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-21-console-bounded-rail-section-scroll-design.md`.

ADR required: yes

ADR path: `backlog/decisions/077-console-bounded-rail-section-scrolling.md`

Reason: this establishes a long-lived cross-rail layout, scroll-ownership, and
keyboard/pointer interaction contract.
<!-- SECTION:DESIGN:END -->
