---
id: TASK-35645
title: Reduce measured native Console attachment waits
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20 00:24'
updated_date: '2026-10-20 00:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identify the native preparation and replay work behind variable Console attachment latency and reduce the next measured bottleneck while retaining fresh storage admission and exact retirement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native preparation timings separate queueing, callback work and attachment replay with causal limits recorded.
- [x] #2 A focused failing regression covers the measured fix and passes without weakening storage safety or freshness.
- [x] #3 Serial native before/after comparisons retain all outcomes, exact process exits and safety guard counts.
- [x] #4 Targeted verification and independent review are complete, PR 3166 is updated and auto-merge stays off.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: bounded Home display-demand correction under existing ADR-126/230; no storage/schema, callback ownership, cache TTL or service contract change. Native diagnostics measured 15 of 21 Home content seam starts after Home suspension, alongside fresh display coordinator refusals and repeated FULL. 1. Add a held-read regression: hide Home while its first content read runs; require that read to finish, skip subsequent content queries and retain the prior complete snapshot. Cover visible and revisited Home plus hidden-before-start. 2. Stop only not-yet-issued Home content work when the mounted view is hidden, leaving started reads awaited through their existing retirement and retaining all existing storage/Console checks. 3. Run targeted Home and config/workspace lifetime checks, then serial d24919ab40 Home-only baseline/current ABBA profiles. 4. Independently review, retain adverse as well as improved outcomes, update PR3166 and leave auto-merge off.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stopped unissued mounted-hidden Home content queries while awaiting each already-issued read through its existing cleanup. Hidden partial builds preserve the preceding complete snapshot; ordinary revisit refresh and direct unmounted callers remain. ADR required: no; existing ADR126/230 boundaries retained. Original regression6fail/1pass, focusedGREEN7pass, final50targeted checks pass158.23s. Independent final review has no additional findings; Home architecture and monitoring-API lesson updated. Twelve safety/ownership modules are unchanged. Serial Home-only d249 baseline/current ABBA retains all24drafts/attachments, four exits0 and allguards0. Pooled attachment2152.17->1639.77ms(-23.8%) and input262.72->238.43ms, but reverse attachment pair+6.2%worse and worst input566.69->731.16msworse: no consistent latency or threshold qualification. Separate corrected-observer controls verify22->8actualcontentworkers; both six-return journeys succeed and guards0. Failed local PY_UNWIND setup exit1 retained and corrected with a filtered global event; earlier fixture marker mistakes recorded without bypass. Remaining direct control-bar native fallback is documented. Fourteen exact disposable roots removed after all-target validation; unrecorded earlier error root not inferred/deleted. Lint/format pass; no full suite. Published code and all evidence to PR3166, with the leading description recording adverse as well as improved results. Final independent review passes with no additional findings. Auto-merge is verified off. Keep In Progress because consistent latency and tail-performance qualification remain unmet; the measured direct control-bar native fallback is documented as remaining work.
<!-- SECTION:NOTES:END -->
