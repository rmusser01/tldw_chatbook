---
id: TASK-22505
title: >-
  Defer or reduce TieAwareStylesheet's 8 full bundle reparses inside the first-paint window
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - performance
  - startup
  - css
priority: medium
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Measured by TASK-22222's permanent counter: `TieAwareStylesheet`'s tie-breaker lowering arms
14 times and causes **8 actual full reparses** of the ~834 KB boot bundle, ALL inside the
first mount (none in the following second). A/B against a neutered arm: 20 vs 12
`Stylesheet.parse()` calls, identical fresh and warm. These are warm/parse-cached so far
below the module docstring's 125-380 ms cold price, but 8 full passes over the bundle during
first paint had never been quantified before and sit squarely in the TTI window this review
was commissioned over.

Related: boot-parsed CSS is now budgeted at 860,000 B (measured 833,841, already +20 KB
since the review) — the TASK-21115 ratchet counts SOURCES and pushes bytes into these eager
sheets by design; the two budgets price the same trade in different currencies.

## Acceptance Criteria

- [ ] The reparse count during first mount is reduced (batch the base-class offers, defer arming past first paint, or reparse once at the end) with the counter proving it
- [ ] The tie-breaker correctness the mechanism exists for is preserved: the TieAware staleness tests stay green and a computed-style spot-check matches before/after
- [ ] Boot-to-`_ui_ready` measured before/after with the interleaved A/A-controlled method (an honest wash is an acceptable outcome)
