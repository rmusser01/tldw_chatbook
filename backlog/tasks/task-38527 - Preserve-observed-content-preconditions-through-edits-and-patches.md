---
id: TASK-38527
title: Preserve observed-content preconditions through edits and patches
status: In Progress
created_date: 2026-11-01 17:30
dependencies:
- TASK-38524
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-4
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
assignee:
- '@codex'
updated_date: 2026-11-01 19:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 4 of the accepted native shared-worktree workflow design so preserve observed-content preconditions through edits and patches. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 File edits and patches carry the digest of the actual read bytes through the existing atomic writer, refusing observed concurrent replacement.
- [ ] #2 Workflow writes require present-content or absent-file preconditions and cannot degrade to blind overwrite on missing/evicted stamps.
- [ ] #3 Existing local/remote worker, exclusion, permission and ordinary caller behavior retains its documented contract; partial multi-file effects remain honest.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: direct implementation of accepted ADR-285 atomic file preconditions; no new boundary.
1. Add held-read race tests for edit/patch, disappearance/create, and partial effects; retain red output.
2. Hash actual newline-preserving read bytes at existing atomic write sites.
3. Narrow workflow fs_write using checked private automatic-work context and read-ledger observations; preserve ordinary writes.
4. Exercise native local/remote workers and provider regressions, static checks, self-review and exact owned-path commit; leave acceptance pending independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Edit and modify-patch now hash actual newline-preserving UTF-8 reads and pass the digest through the existing atomic writer; create patches retain absence checks. Workflow fs_write uses checked private AutomaticWorkContext and run-keyed ReadLedger, refuses missing/evicted observations and conflicting model CAS, and permits explicit atomic absence. Ordinary blind/explicit-CAS behavior and existing worker receipts remain unchanged. Partial patch errors identify prior retained effects; no whole-patch transaction or retry is implied. Rebuilt the existing remote worker bundle after verifying clean non-owned inputs; updated Docs/Architecture/tool-catalog.md. Implements existing ADR-285 (no new ADR). Targeted verification: final tool/bundle/import-closure modules 170 passed; provider subset with exact-node supported-profile diagnostic 27 passed. Original legacy provider result retained separately: 12 raw_source_selection_changed failures, 7 passed; new cases explicitly enroll in bootstrap_profile. Static comparison: 72 existing/72 current Ruff diagnostics, zero introduced; new module clean, changed ranges formatted, compile and diff checks passed. Independent review pending: retain In Progress and unchecked AC. Full evidence: .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-4-report.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
