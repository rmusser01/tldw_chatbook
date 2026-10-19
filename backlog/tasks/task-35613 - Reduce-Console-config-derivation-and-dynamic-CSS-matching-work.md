---
id: TASK-35613
title: Reduce Console config derivation and dynamic CSS matching work
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19 19:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce measured redundant work during Windows Console interactions without changing settings freshness, storage checks, or rendered styles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Independent readiness and provider-label builds read current settings once, observe later changes, and clear the memo on exceptions.
- [ ] #2 Impossible final pseudo-state selectors avoid ancestor matching while full style and invalidation behavior remains equivalent to upstream.
- [ ] #3 Serial before/after Windows profiles and focused benchmarks record results and limitations against a pinned baseline.
- [ ] #4 Targeted regression, compatibility, static checks and review complete with storage safety checks intact.
- [ ] #5 PR 3166 updated with code and reproducible evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: routine optimization of existing synchronous derivation and stylesheet delegation; no storage, authority, cache lifetime, or UI contract changes. Existing ADR-126 admission and ADR-230 finite display boundaries remain unchanged.
1. Attribute remaining config and CSS samples; measure repeated calls and preserve necessary path posture reads.
2. Add failing regressions, scope independent derivations using the existing synchronous memo, and reject impossible final pseudo states at the matcher seam.
3. Verify dynamic style fidelity and fresh config behavior; run serial native Windows benchmark and ABBA return profiles against aeaac674e3b19e2af5989be2a145a0316856248a.
4. Review, document honest results, update PR, and clean only owned diagnostic fixtures.
<!-- SECTION:PLAN:END -->
