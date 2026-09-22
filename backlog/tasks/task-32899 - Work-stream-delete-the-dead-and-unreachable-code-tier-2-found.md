---
id: TASK-32899
title: "Work stream: delete the dead and unreachable code tier 2 found"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-dead
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
About 24 deletion candidates in packages none of TASK-32807's six sub-tasks names, each with production
importer counts resolved by AST walk (relative levels resolved, function-body imports included) and
reachability resolved by **running** `resolve_screen_route()` rather than reading the registry.

Two entries are deliberately **not** deletions and must not be swept into one:
- the `Evals/` legacy run stack (~9,000 lines) is unreachable but carries a `subprocess` code-execution
  sandbox -- it needs a ruling (TASK-32904), not a cleanup PR;
- `Widgets/Tamagotchi/`'s widget half (2,181 lines) is a product decision (TASK-32905), because the
  storage half is wired into backup/recovery and the private-SQLite allowlist.

Watch for test-only lifelines: several candidates are kept green by tests that exist only to import them,
and `Local_Inference/mlx_lm_inference_local.py` has **30 tests asserting dead code's behaviour** -- a
future fixer will "repair" it by mistake. Delete the tests with the code.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each deletion states its production importer count and its route reachability
- [ ] #2 Test-only lifelines are deleted with the code they pin
- [ ] #3 `Prompt_Management/Prompt_Engineering.py`'s metaprompt is extracted for task-474 before deletion
- [ ] #4 Deletions land one PR per package family, not one PR for all of them
<!-- AC:END -->
