---
id: TASK-31640
title: Restore bounded Architecture contracts on rebased dev
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:11'
updated_date: '2026-09-05 16:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair verified inventory drift and runtime ownership regressions exposed by the comprehensive dev test review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The eight non-size Architecture failures pass without broadening debt exemptions.
- [x] #2 TLS sessions retain configured verification and timeout defaults, and refresh workers keep their exclusive ownership.
- [x] #3 Timer repaint changes preserve dynamic layout and avoid layout work for fixed-size values.
- [x] #4 Focused behavior suites and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Architecture failures after rebasing onto latest dev.
2. Repair the TLS factory and moved/generated-source inventories.
3. Reconcile timer classifications and fixed-size repaint semantics.
4. Repair briefing refresh ownership and validate grouped dispatcher exemptions.
5. Run affected behavior tests, Architecture checks, Ruff, and diff validation.

Design: `Docs/superpowers/specs/2026-09-05-dev-architecture-repair-design.md`.

ADR required: no
ADR path: backlog/decisions/079-network-tls-trust-policy.md (existing)
Reason: restore existing timeout, TLS, timer-layout, and worker ownership contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused the timeout-safe TLS session factory; corrected moved diagnostic and maintained-source inventories; classified new and renamed timer sites and disabled layout only for fixed-size values; grouped daily-report work and coordinated briefing refreshes. The worker census now verifies nested dispatch wrappers against the expected exclusive group. Evidence: 106 Architecture/TLS checks passed; the affected UI run passed 258 of 263 initially, then all five failed cases passed targeted reruns (three required loopback socket access outside the sandbox; two needed scheduled-worker-aware test assertions). Scheduling/egress follow-up passed 107 tests. Ruff, changed-range formatting, and git diff --check passed. Existing ADR-079 applies; no new ADR required. The remaining Architecture failures are size ratchets, including one additional media-controller breach found after rebasing to 93388ba69b.

Post-implementation review found that the grouped-loader sole-reference waiver ignored nested closure references. Added a synthetic nested mutate/refresh regression, observed the failure, then counted descendant-scope references as well. All 22 worker inventory tests now pass, including the new red/green regression; Ruff formatting/check and diff checks pass.
<!-- SECTION:NOTES:END -->
