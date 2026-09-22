---
id: TASK-32904
title: "Ruling needed: the unreachable Evals legacy run stack"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-dead
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
About 9,000 lines -- `eval_runner`, `specialized_runners`, `dataset_validator`, `dataset_loader`,
`base_runner`, `ui_integration`, `_run_admitted_evaluation` -- with **no reachable entry point**:
`handle_start_evaluation` does not exist and `ABTestOrchestrator` is constructed nowhere. Three test
files keep it green.

This is a ruling, not a cleanup, for one reason: **it carries a `subprocess` code-execution sandbox**
governed by ADR-031. Unreachable code that can execute code is a different risk from unreachable code
that cannot -- it is attack surface that nobody is reviewing precisely because nobody thinks it runs.
Deleting it is probably right; confirming it is genuinely unreachable from every entry point, including
plugins and any server-driven path, is the work.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reachability is confirmed or refuted from every entry point, plugins included
- [ ] #2 If unreachable, it is deleted with its three test files and ADR-031 is updated to say so
- [ ] #3 If reachable, the path is documented and the sandbox is reviewed as live code
<!-- AC:END -->
