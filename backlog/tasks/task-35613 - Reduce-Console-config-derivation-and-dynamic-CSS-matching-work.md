---
id: TASK-35613
title: Reduce Console config derivation and dynamic CSS matching work
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 19:37'
updated_date: '2026-10-19 21:14'
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
- [x] #5 PR 3166 updated with code and reproducible evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: routine optimization of existing synchronous derivation and stylesheet delegation; no storage, authority, cache lifetime, or UI contract changes. Existing ADR-126 admission and ADR-230 finite display boundaries remain unchanged.
1. Attribute remaining config and CSS samples; measure repeated calls and preserve necessary path posture reads.
2. Add failing regressions, scope independent derivations using the existing synchronous memo, and reject impossible final pseudo states at the matcher seam.
3. Verify dynamic style fidelity and fresh config behavior; run serial native Windows benchmark and ABBA return profiles against aeaac674e3b19e2af5989be2a145a0316856248a.
4. Review, document honest results, update PR, and clean only owned diagnostic fixtures.

Verification follow-up after syncing the completed remote work: rerun config/CSS/return compatibility, then repeat a same-screen CSS benchmark and serial ABBA returns on current head. Restore only the two original derivation method bodies and original CSS helper in the diagnostic control, preserving later Console return and measured-spend fixes in both arms. Record all results and update PR3166. No new ADR: diagnostic-only control; product/security contracts unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reduced independent readiness and provider-label settings loads from 3 to 1 using the existing synchronous memo, including freshness and exception cleanup. CSS rejects false final pseudo states using Textual checks while retaining candidate invalidation flags and full positive matching. No storage/config safety implementation changed; no new ADR required.
23 distinct targeted checks passed, including real screen fidelity and the concurrently integrated measured-spend fix. New/modified helper and test lint/format checks pass; ChatScreen retains exactly its baseline 22 lint findings and formatting debt. Independent review found no blockers. One 45-second cold fixture timeout was retained and resolved with a bounded 90-second allowance; no full sweep ran.
Warm full-screen CSS improved 165.34 to 129.41 ms (21.7%), 526 widgets and zero style mismatches. Serial ABBA medians were mixed, and the final integration pair retained a worse updated first-return outlier: no overall latency gain is claimed. All 36 return journeys retained drafts and attached, with zero guard failures and successful exits.
A concurrent shared-checkout rebase required an isolated managed worktree. Controls pin f152dfc4; final compatibility/profile evidence includes 1bebf20b. See Docs/Development/evidence/windows-performance-2026-10-18/SUBJECT_RESULTS.md and updated Console architecture/verification lessons. PR update pending.

Published fixes and evidence to PR #3166, updated its description with mixed end-to-end results and retained outliers, and removed all 12 exact owned private fixture directories. Auto-merge remains disabled for the new head pending fresh review. Final combined-code profile/control both completed all six returns with no guard violations.

Current-code confirmation on 3742efae: 19 targeted native config/CSS/return/spend checks passed. Four alternating same-screen CSS rounds improve warm application from 144.37 to 117.70 ms (18.5%), with zero style differences across 526 widgets. The scoped control restores only the two original derivation methods and CSS helper, preserving later return/spend fixes; all other ChatScreen AST is checked identical.

Serial ABBA input medians were 165.69 / 168.12 / 233.70 / 195.87 ms: no overall latency improvement is verified, and updated tails worsen. All 24 drafts were retained, all attaches completed, all five exits were zero, and network/profile guard counts were zero. Seven safety modules remain identical to pre-optimization sources.

Independent review found no product blocker; its minor benchmark-receipt validation finding was fixed and negative-tested. Helper/test lint and formatting pass; legacy debt is unchanged. See SUBJECT_CONFIRMATION.md, subject-confirmation.json and exact exit receipts. No new ADR or product/security change in this confirmation. Six exact completed fixture directories were cleaned.

<!-- SECTION:NOTES:END -->
