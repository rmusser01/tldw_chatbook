---
id: TASK-35613
title: Reduce Console config derivation and dynamic CSS matching work
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19 19:37'
updated_date: '2026-10-19 20:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce measured redundant work during Windows Console interactions without changing settings freshness, storage checks, or rendered styles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Independent readiness and provider-label builds read current settings once, observe later changes, and clear the memo on exceptions.
- [x] #2 Impossible final pseudo-state selectors avoid ancestor matching while full style and invalidation behavior remains equivalent to upstream.
- [x] #3 Serial before/after Windows profiles and focused benchmarks record results and limitations against a pinned baseline.
- [x] #4 Targeted regression, compatibility, static checks and review complete with storage safety checks intact.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reduced independent readiness and provider-label settings loads from 3 to 1 using the existing synchronous memo, including freshness and exception cleanup. CSS rejects false final pseudo states using Textual checks while retaining candidate invalidation flags and full positive matching. No storage/config safety implementation changed; no new ADR required.
23 distinct targeted checks passed, including real screen fidelity and the concurrently integrated measured-spend fix. New/modified helper and test lint/format checks pass; ChatScreen retains exactly its baseline 22 lint findings and formatting debt. Independent review found no blockers. One 45-second cold fixture timeout was retained and resolved with a bounded 90-second allowance; no full sweep ran.
Warm full-screen CSS improved 165.34 to 129.41 ms (21.7%), 526 widgets and zero style mismatches. Serial ABBA medians were mixed, and the final integration pair retained a worse updated first-return outlier: no overall latency gain is claimed. All 36 return journeys retained drafts and attached, with zero guard failures and successful exits.
A concurrent shared-checkout rebase required an isolated managed worktree. Controls pin f152dfc4; final compatibility/profile evidence includes 1bebf20b. See Docs/Development/evidence/windows-performance-2026-10-18/SUBJECT_RESULTS.md and updated Console architecture/verification lessons. PR update pending.
<!-- SECTION:NOTES:END -->
