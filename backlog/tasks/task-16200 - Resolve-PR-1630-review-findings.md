---
id: TASK-16200
title: Resolve PR 1630 review findings
status: Done
assignee: []
created_date: '2026-08-14 06:18'
updated_date: '2026-08-14 07:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close every actionable PR 1630 review finding without weakening the bounded TTS artifact validator or provider endpoint safety policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optional PyAV loading uses the centralized dependency seam.
- [x] #2 Playable audio paths pass shared validation while exact-file race checks remain enforced.
- [x] #3 Public audio validation API has complete Args and Returns documentation.
- [x] #4 Schemeless exact IPv6 loopback endpoints are accepted while remote schemeless endpoints remain blocked.
- [x] #5 Dotted provider display names canonicalize correctly for model probes.
- [x] #6 Focused regression and acceptance tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests for all five findings.
2. Implement bounded fixes using existing dependency, path, and endpoint contracts.
3. Run focused and first-run acceptance verification.
4. Document results and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Routed optional PyAV loading through the shared dependency seam and gated artifact paths through shared validation without weakening no-follow, bounded-read, or file-identity checks.
- Accepted exact schemeless IPv6 loopback endpoints, reused canonical provider aliases for discovery, and normalized continuation endpoint comparisons.
- Adapted rebased first-run, settings, todo, and Studio contracts to current `dev`; explicitly auto-sized the unframed Speech inspector so its controls remain reachable.
- Verified 1,344 acceptance tests with four expected PyAV skips, four Pocket TTS roleplay integrations, 140 auto-speak/first-run tests, and 865 review regressions. Compilation, diff checks, and scoped formatting passed; isolated mypy findings were outside changed lines.
<!-- SECTION:NOTES:END -->

## ADR Check

ADR required: no

ADR path: N/A

Reason: These are bounded correctness, compatibility, and UI layout fixes within existing dependency, endpoint, validation, and settings contracts.
