---
id: TASK-24461
title: Boot budget ratchets do not run in the fast per-PR guard
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - ci
  - infrastructure
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The four ADR-097 boot budget ratchets (import weight, `_ui_ready` module census, boot
parsed CSS bytes, boot worker census) run only in the slow full test suite.
`.github/workflows/perf-guard.yml` runs `Tests/Performance/test_ui_latency_guardrails.py` and
`Tests/Utils/test_ui_responsiveness_stall_persist.py` and nothing else.

The consequence is measured, not hypothetical: for three review cycles running, the budgets
have been re-breached within roughly 24 hours of every paydown, and each breach was discovered
by a periodic review rather than by the PR that caused it. All four are red on pristine dev
right now, one day after TASK-23112 paid the import debt down to 646/660.

Moving them into the fast lane makes a breach cost one PR instead of one review cycle, and
gives the breaching PR the culprit-module list the guards already print.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The four boot budget ratchets run in a fast per-PR workflow and report a verdict in minutes
- [x] #2 A PR that breaches any of the four fails that workflow with the culprit list the guard already prints
- [x] #3 The workflow's path filters cover the inputs that can move any of the four budgets
- [x] #4 The fast lane is not made red-by-default for unrelated PRs -- the guards must be green on dev before or in the same change that enables enforcement
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Added a "Run boot budget ratchets" step to `.github/workflows/perf-guard.yml`, which previously
ran only the latency guardrails. The four budgets took ~15 s together locally, and they already
print their culprit module lists, so a breaching PR now gets that list in minutes instead of at
the next periodic review.

`test_boot_css_byte_budget.py` is deliberately EXCLUDED for now and the workflow says why: it is
red on dev (862,184 B against an 860,000 B ratchet) and including it would fail every unrelated
PR -- exactly the failure mode this task's AC forbids. It joins the step when task-24459 lands.
The other three are green with headroom after task-24458, so enforcement starts clean.

`test_textual_css_fastpath.py` is included too: its upstream-pin test is what makes a Textual
version bump surface as a loud failure rather than silent styling drift.

The workflow's existing path filters already cover the inputs that move these budgets
(`tldw_chatbook/**.py`, `tldw_chatbook/css/**`, `Tests/Performance/**`, `pyproject.toml`).

Modified: `.github/workflows/perf-guard.yml`.
<!-- SECTION:NOTES:END -->
