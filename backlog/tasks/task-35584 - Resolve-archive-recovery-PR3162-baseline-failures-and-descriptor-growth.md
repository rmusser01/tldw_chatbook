---
id: TASK-35584
title: Resolve archive recovery PR3162 baseline failures and descriptor growth
status: In Progress
created_date: 2026-10-18 20:43
dependencies:
- TASK-35583
assignee:
- '@codex'
documentation:
- Docs/superpowers/qa/console/2026-10-18-archive-recovery-pr3162-rebase.md
- backlog/decisions/166-agent-assisted-archive-recovery.md
updated_date: 2026-10-18 21:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the causes of the ten remaining targeted-test failures and descriptor-growth warnings, and repair issues that undermine archive recovery or its qualification without weakening ownership, privacy, migration or UI-governance checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All ten recorded failures are reproduced, traced to their source and resolved where they affect PR3162, with any unrelated remainder explicitly evidenced.
- [x] #2 Descriptor growth in storage, captured follow-up and mounted recovery cohorts is attributed to actual retained resources; affected cleanup retires native handles without replaying effects, closing borrowed live handles or relying on warning suppression.
- [ ] #3 Focused regressions and impacted targeted cohorts pass; static/generated checks, independent review and updated PR qualification record exact changes, results and remaining limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Existing ADR166 plus SQLite kernel/native-custody and design-token ADRs apply; targeted compatibility/fixture fixes introduce no new boundary.
1. Read all ten exact failure traces and prior resource warnings, reproduce the independent trace/governance failures, and investigate descriptor owners with isolated per-test native-handle probes.
2. Follow Docs/superpowers/plans/2026-10-18-archive-recovery-baseline-repair.md. Helpers own disjoint trace/governance repairs; root owns descriptor diagnosis, shared cleanup, integration and Git. Preserve all security/ownership and verification guards.
3. Add focused actual RED/GREEN for proven defects, correct stale test setup at its shared source and verify repeated resource rounds plateau with final native retirement under ordinary GC settings.
4. Run affected targeted cohorts/static/CSS checks, independently review actual source and evidence, update task/new QA, and publish only the existing PR branch with exact-head lease. No full suite, extra model requests, merge or original-checkout changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resolved all ten recorded baseline failures without loosening sanitizer, migration, ownership or UI-governance checks. Archive search now retires only operation-owned native SQLite handles and bounds checkpoint waits at its existing deadline; failed Workspace initialization closes its own handle and preserves the original exception. Test-owned registry/media, replay worker and mounted runtime cleanup now retire their real owners. The mounted harness drives callbacks through its actual host and waits for real receipt acknowledgement.

Targeted evidence: trace8, governance/migration/CSS50, integrated storage282, ownership18, captured follow-up50, mounted9 plus delayed-ack3 all pass. Final integrated resource cohorts plateau at17 descriptors with zero SQLite files after complete teardown and session GC. Independent review has no actionable findings; all16 modified Python files pass AST/critical Ruff checks with no introduced per-file lint findings. Existing voice2 failures and inherited static debt are documented, not represented as green.

ADR required:no; existing ADR166/228/150/161 apply. Plan: Docs/superpowers/plans/2026-10-18-archive-recovery-baseline-repair.md. Final QA, source hashes, exact commands and review: Docs/superpowers/qa/console/2026-10-18-archive-recovery-baseline-repair.md and its companion directory. Updated existing testing-evidence lessons with the measured incidents. No full suite or new live/model requests; budget remains30/30 consumed. Source and review are complete; exact-lease publication of the existing draft PR remains pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
