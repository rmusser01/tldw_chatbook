---
id: TASK-37630
title: Repair raw shell output and search outcome handling
status: Done
created_date: 2026-10-29 22:34
assignee:
- '@codex'
updated_date: 2026-10-30 04:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep raw shell error details available to models, let empty searches complete without exhausting retries, and preserve Windows command text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shell results retain stderr and all executor-provided output beyond 4000 characters.
- [x] #2 Unambiguous no-match grep, rg and findstr calls do not trigger the ordinary failure streak; genuine and ambiguous command failures still do.
- [x] #3 Windows cmd commands retain quoting and preserve intended output and exit codes.
- [x] #4 Focused regression tests, lint, documentation and ADR review are complete.
- [x] #5 The new CLI regressions run automatically in the existing Linux Python 3.12 and 3.14 PR workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md. Reason: refine search outcomes and optional listing worker arguments while retaining ADR-078/094/101 authority. Follow Docs/superpowers/plans/2026-10-29-cli-tool-recovery.md with focused red/green tests, implementation, documentation, targeted checks and fresh review.
PR routine follow-up: add a fail-first CI routing contract, select the new regression modules in bounded separate invocations of the existing Tool Recovery job, retain the two-version matrix and timeout, and verify the contract before publication. ADR required: no new ADR; existing ADR-254 and CI execution structure apply.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed the redundant 4000-character provider slice and retain executor stdout/stderr. Simple grep/rg/findstr exit-1 results with complete empty stderr are successful empty searches; genuine errors, compounds, cmd expansions and truncated previews retain failures. Exact cmd text is launched without CRT re-escaping. Final review reproduced hidden stderr under the executor cap; added RED/GREEN accumulator-to-provider coverage and excluded truncated results from normalization. Fixed the integration fixture to select its schema/profile explicitly, and made the fresh-import refusal fixture use its owned directory instead of POSIX /tmp. Evidence: agents-final 80 passed; review-green 25 passed; real-process/native/import run 142 passed with 7 explicit POSIX skips, plus repaired import fixture 1 passed. New files lint clean; no new E9/F diagnostics over base. ADR-254 and user/architecture docs updated. No full suite, push or PR requested.
PR routine: the new regressions were absent from PR selections. Added separate raw-agent and real-worker selections to the existing bounded Linux3.12/3.14 Tool Recovery matrix, retained the job budget and artifact receipts, and added a CI routing contract. Routing RED:1 failure; GREEN:5 contract/workflow-safety tests. No dependency or platform-matrix expansion.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
