---
id: TASK-23029
title: >-
  Boot budgets are all within 2-4% of breach two days after being pinned
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - process
  - performance
  - startup
priority: high
---

## Description

Four boot budgets were pinned on 2026-08-25 "just above reality". Two days later every one is within
2-4% of breach, and at observed merge rates each breaches within a day or two of normal traffic.

| guard | budget | now | headroom |
|---|---|---|---|
| boot import weight | 660 modules | 657 | **3 (0.5%)** |
| `_ui_ready` census | 970 | ~950 | ~20 - family assertion already RED |
| boot CSS bytes | 860,000 | 842,236 | **17,764 (2.1%)** |
| pre-import payload | 500 / 380k LOC | 481 / 368,814 | **19 (3.8%) / 11,186 (2.9%)** |

This is the finding that outranks the individual costs. Three reviews in six days have each brought
these numbers down and each time they were consumed within days. The guards work - they caught every
regression in this review - but a budget with 0.5% headroom converts the next ordinary feature into a
red build, which trains people to raise the budget.

## Acceptance Criteria

- [ ] A decision is recorded on whether these are budgets (raise deliberately, with review) or ratchets (never raise, fix the cause)
- [ ] Whichever it is, the guard says so in its failure message, so the next person hitting it knows which move is legitimate
- [ ] A breach names the specific edge or module that consumed the headroom, not just the total - tracing it currently takes an import tracer and an hour
- [ ] Consider whether headroom itself should be reported per-PR, so consumption is visible before the breach

## Evidence

See the four rows above, all measured on `c6218918d1`. Separately: the guards forbidding
`Chat.trajectory_export` on the first-paint path were written 2026-08-25 and breached within ~24
hours by the current tip (TASK-23020), which touched neither guard file and routed around an explicit
in-code comment forbidding exactly that.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
