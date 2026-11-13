---
id: TASK-40727
title: Bound dev CI with required smoke, nightly regression and release qualification
  tiers
status: In Progress
created_date: 2026-11-13 21:54
assignee:
- '@codex'
updated_date: 2026-11-13 22:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shorten normal development feedback without losing nightly regression coverage or allowing unqualified release candidates. Keep the self-hosted migration baseline independent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing dev required context always reports and fails closed on preflight, smoke and selected affected checks; high-risk or unclassified changes retain broad regression coverage.
- [ ] #2 One immutable changed dev SHA receives nightly broad regression coverage; unchanged successful SHAs do not repeat, failures retry, and scheduled execution is installed on the default branch before lighter gating is enabled.
- [ ] #3 Promotion to main and publication require full-suite and packaging evidence for the exact candidate; failures, cancellations and skipped expected checks cannot qualify.
- [x] #4 Focused policy tests cover event routing, changed-path selection, unknown/deleted files, failed or missing evidence and rollout disabled behavior; current runner isolation and queue semantics are preserved.
- [x] #5 Document the tier policy, coverage tradeoffs, rollout and rollback; keep opt-in disabled until live qualification and default-branch schedule activation succeed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/360-tiered-development-and-release-ci.md
Reason: Changes required-check selection, nightly scheduling and release qualification boundaries; amends ADR-103 while preserving ADR-359 isolation.
1. Add a conservative changed-path selector and explicit reviewed source-to-test map; retain broad coverage for risk-sensitive, deleted or unknown inputs and keep smoke/artifact checks unconditional.
2. Wire selection into existing required aggregation behind CI_TIERED_ENABLED, preserving all existing broad test commands, queue events and runner routes.
3. Add changed-SHA nightly orchestration and exact-candidate promotion/release qualification using reusable existing suites; make publication depend on the full gate.
4. Add focused behavioral and workflow contracts, run targeted checks, review the diff and prepare a separate draft PR without competing with #3529's baseline.
5. Document rollout requiring an active default-branch schedule and live evidence before enabling the lighter policy; record remaining live qualification explicitly.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the opt-in three-tier workflow policy under ADR-360 (amending ADR-103, preserving ADR-359). The stable required context aggregates preflight, representative subsystem smoke or the existing broad lanes, and exact-candidate release qualification. Preflight now requires the same pinned Ruff F gate and debt ratchet. Broad test commands, matrices, deadlines and package build steps were compared against the parent and preserved exactly. Package build is reusable, and both publish paths require comprehensive-test success. The nightly dispatcher selects changed dev SHAs, retries failures, and qualifies every smoke selection before opt-in; Library shell coverage moves from every dev push to nightly only when enabled.

Validation: exact combined light-job shell invocation with CI contracts and all selected smoke targets: 1974 passed, 1 intentional child-test skip in 156.48s locally. Earlier isolated smoke union: 291 passed in 68.98s. CI plus package metadata contracts: 1706 passed, 1 intentional skip. Final shallow-checkout/selector and derived-workflow contracts: 118 passed in 24.33s. Actionlint, Ruff for new Python, source F gate/debt ratchet, whitespace, task/ADR validators pass. Self-review checked fail-closed missing/skipped results, fork routing, nightly identity/retries, reusable-workflow concurrency and exact event-SHA checkouts. No application full sweep or publication run.

Selection is representative subsystem coverage, not inferred dependency closure; high-risk names, shared inputs, new/deleted sources and unknown subsystems retain broad coverage. Exploratory whole legacy chat/navigation modules exposed existing profile-admission fixture failures and missing mock fixtures; product tests were not altered or removed. The selected maintained seams passed, while comprehensive qualification still owns the whole tree. This is a deliberate coverage/cadence tradeoff documented in scripts/ci/README.md.

Pending live acceptance: #3529 full baseline/merge; normal full gate on this policy PR; install and qualify regression-nightly.yml on default main; promotion candidate/package qualification; strict required CI on main (currently only deletion/force-push rules); enable CI_TIERED_ENABLED only afterwards. Opt-in is confirmed unset. No branch protection, server configuration or automation was changed. Task remains In Progress; Qodo exception for #3529 is not inherited.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
