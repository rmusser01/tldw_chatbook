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
- [ ] #1 Every direct named Context and Inspector section body grows naturally up to a 20-rendered-content-line ceiling; additional content scrolls inside its allocated viewport with a separate `▼ more — scroll` hint row, except a constrained Context body may receive zero rows after headers take priority.
- [ ] #2 Section hints are absent when all 0-20 content lines fit the allocated viewport and at the scroll end, and reappear after scrolling upward while content remains below; exactly 21 rows at a full allocation yields 20 content rows plus the separate hint.
- [ ] #3 Context allocates shorter sections only what they need, redistributes unused height up to the 20-line ceiling, keeps every Context header visible, and deterministically assigns zero-height bodies in DOM order when a constrained explicit-open viewport cannot fund every body plus its honest hint.
- [ ] #4 Inspector retains a separate outer fold hint whenever any outer content remains below the rail viewport; every existing row/action has one named section owner, and no hint covers, reorders, duplicates, or changes the semantics of Sources, Scope, run state, Changed Files, Tools, Approvals, Artifacts, live-work sources, or Session Settings.
- [ ] #5 Pointer scrolling hands off at nested-scroll boundaries; keyboard scrolling, focus order, rail badges, collapse/reopen behavior, responsive focus transfer, and stored rail preferences do not regress.
- [ ] #6 Production-CSS Textual compositor tests cover 235x52 and 160x45 all-open expanded states plus default and explicit-open safeguards at 120x30 and 80x24, including 20/21-line boundaries, deterministic zero-body allocation, content shrink, resize, recompose, and scroll-position clamping.
- [ ] #7 The implementation follows ADR-077 and preserves TASK-15110's all-Context-headers-visible outcome while replacing its fixed-percentage body cap.
- [ ] #8 Specialized child constraints no longer override the shared ceiling: Sources' legacy 6/10-row caps and Session Settings' CSS 9-row minimum plus inline 9-row maximum are retired, and a one-row settings body occupies one content row.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-21-console-bounded-rail-section-scroll-design.md`.

ADR required: yes

ADR path: `backlog/decisions/077-console-bounded-rail-section-scrolling.md`

Reason: this establishes a long-lived cross-rail layout, scroll-ownership, and
keyboard/pointer interaction contract.
<!-- SECTION:DESIGN:END -->
