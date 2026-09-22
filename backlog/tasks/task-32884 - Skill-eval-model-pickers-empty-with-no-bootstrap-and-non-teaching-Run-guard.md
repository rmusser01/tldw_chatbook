---
id: TASK-32884
title: 'Skill eval: model pickers empty with no bootstrap and non-teaching Run guard'
status: Done
assignee: []
created_date: '2026-09-21 22:30'
updated_date: '2026-09-21 23:22'
labels:
  - ux
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review A2 (HIGH). Generator/judge Selects render blank when no eval models are configured, and the Run guard toast says 'Pick generator and judge models first.' without saying WHERE models come from - while providers configured in the Lab's own Models mode do not feed these pickers (separate eval_models registry). Invisible seam between two screens 20px apart. Evidence: live walkthrough 2026-09-21; also UAT finding F1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty generator/judge Selects show a message naming where to add eval models instead of a blank overlay
- [x] #2 Run guard toast names the fix destination, not just the problem
- [x] #3 A user who only configured a provider in Lab Models mode gets actionable guidance toward an eval model
<!-- AC:END -->

## Implementation Notes

- `set_targets([])` swaps both pickers to a "no eval models configured" prompt plus one guidance row ("+ New target" in a bench editor, or "Create sample bench") carrying a sentinel the change handler rejects (reset to NULL + informational notify) — same pattern the empty store picker got in TASK-32883.
- Run guard chooses its toast by picker state: zero real options → teaching toast naming both creation paths; models available but unpicked → the original "Pick generator and judge models first."
- Tests: `test_empty_model_pickers_show_bootstrap_guidance`, `test_run_guard_toast_teaches_the_fix_when_no_models_exist`, `test_run_guard_toast_stays_plain_when_models_exist`. Panel 21/21; screen 19/21 (same single pre-existing dev red as before, unrelated).
- Files: `tldw_chatbook/UI/Evals/skill_eval_panel.py`, `Tests/UI/test_evals_skill_eval_panel.py`.
