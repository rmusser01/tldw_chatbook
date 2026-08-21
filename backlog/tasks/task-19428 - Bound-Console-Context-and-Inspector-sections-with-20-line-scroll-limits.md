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
- [ ] #1 Every direct named Context and Inspector section body grows naturally through 20 rendered content lines; additional content scrolls inside a 20-line viewport with a separate `▼ more — scroll` hint row.
- [ ] #2 Section hints are absent for 0-20 content lines and at the scroll end, and reappear after scrolling upward while content remains below.
- [ ] #3 Context allocates shorter sections only what they need, redistributes unused height up to the 20-line ceiling, and scrolls longer sections sooner when required to keep every Context header visible.
- [ ] #4 Inspector retains a separate outer fold hint for complete sections below the rail viewport; no hint covers, reorders, duplicates, or changes the semantics of Sources, Scope, run state, Tools, Approvals, Artifacts, live-work sources, or Session Settings.
- [ ] #5 Pointer scrolling hands off at nested-scroll boundaries; keyboard scrolling, focus order, rail badges, collapse/reopen behavior, responsive focus transfer, and stored rail preferences do not regress.
- [ ] #6 Production-CSS Textual compositor tests cover 235x52 and 160x45 expanded states plus 120x30 and 80x24 responsive safeguards, including 20/21-line boundaries, content shrink, resize, recompose, and scroll-position clamping.
- [ ] #7 The implementation follows ADR-077 and preserves TASK-15110's all-Context-headers-visible outcome while replacing its fixed-percentage body cap.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-21-console-bounded-rail-section-scroll-design.md`.

ADR required: yes

ADR path: `backlog/decisions/077-console-bounded-rail-section-scrolling.md`

Reason: this establishes a long-lived cross-rail layout, scroll-ownership, and
keyboard/pointer interaction contract.
<!-- SECTION:DESIGN:END -->
