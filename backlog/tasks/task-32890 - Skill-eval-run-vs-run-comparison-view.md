---
id: TASK-32890
title: 'Skill eval: run-vs-run comparison view'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:32'
updated_date: '2026-09-22 03:21'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review B1 (MEDIUM). Nothing compares two run groups: iterating on a skill - the core loop this feature exists for - means eyeballing two monospace reports side by side. Power-user gap surfaced in both UAT and HCI review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A 'Compare with previous' action exists on a completed run group
- [x] #2 Comparison shows per-dimension and composite deltas (direction + magnitude) between the two runs
- [x] #3 Comparison identifies which run is newer and preserves methodology_version/depth differences visibly
<!-- AC:END -->


## Implementation Notes

- MVP: `SkillEvalDetail` resolves the most recent OTHER run group on the same bench (`list_runs(task_id=bench)`, newest-first, first report-bearing sibling) and renders a "Compare with previous" toggle (`#skill-eval-compare`). The section shows newer-vs-older identity (each side's composite, depth, methodology version), the signed composite delta, per-dimension signed deltas, and "new this run"/"dropped this run" markers for dimensions present on only one side.
- Tests: `test_compare_with_previous_run_group` (two seeded groups; +2.8 composite and +0.08 dimension deltas asserted, newer/older line present).
- Files: `tldw_chatbook/UI/Evals/skill_eval_detail.py`, `Tests/UI/test_evals_skill_eval_screen.py`.
