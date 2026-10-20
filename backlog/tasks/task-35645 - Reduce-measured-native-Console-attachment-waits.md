---
id: TASK-35645
title: Reduce measured native Console attachment waits
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20 00:24'
updated_date: '2026-10-20 00:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identify the native preparation and replay work behind variable Console attachment latency and reduce the next measured bottleneck while retaining fresh storage admission and exact retirement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native preparation timings separate queueing, callback work and attachment replay with causal limits recorded.
- [ ] #2 A focused failing regression covers the measured fix and passes without weakening storage safety or freshness.
- [ ] #3 Serial native before/after comparisons retain all outcomes, exact process exits and safety guard counts.
- [ ] #4 Targeted verification and independent review are complete, PR 3166 is updated and auto-merge stays off.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: bounded Home display-demand correction under existing ADR-126/230; no storage/schema, callback ownership, cache TTL or service contract change. Native diagnostics measured 15 of 21 Home content seam starts after Home suspension, alongside fresh display coordinator refusals and repeated FULL. 1. Add a held-read regression: hide Home while its first content read runs; require that read to finish, skip subsequent content queries and retain the prior complete snapshot. Cover visible and revisited Home plus hidden-before-start. 2. Stop only not-yet-issued Home content work when the mounted view is hidden, leaving started reads awaited through their existing retirement and retaining all existing storage/Console checks. 3. Run targeted Home and config/workspace lifetime checks, then serial d24919ab40 Home-only baseline/current ABBA profiles. 4. Independently review, retain adverse as well as improved outcomes, update PR3166 and leave auto-merge off.
<!-- SECTION:PLAN:END -->
