---
id: TASK-32885
title: 'Skill eval: Run not guarded on subject'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:30'
updated_date: '2026-09-21 23:15'
labels:
  - ux
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review A3 (HIGH). Run guards generator+judge NULL but not subject: with models picked and 'No subject selected.' the user can launch a real run that fails in the worker (SubjectError) and leaves a red failed row in the rail. Inconsistent with the models guard; pollutes run history on a first-timer's very first interaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run is disabled (with visible reason, matching the screen's Blocked-badge+callout convention) until a subject is set
- [x] #2 Launching without a subject is impossible from the panel; no failed run row is created for a missing subject
- [x] #3 Covering test: Run press with NULL subject neither dispatches the worker nor creates a run row
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add panel test: Run with models set but no subject posts nothing; set_subject with a real ref restores runnability
2. Track effective subject ref in SkillEvalPanel (set_subject filters the "(no subject set)" sentinel; picker/input choices update it)
3. Guard on_button_pressed Run branch with teaching notify
4. Run targeted tests: Tests/UI/test_evals_skill_eval_panel.py
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `SkillEvalPanel` now tracks the effective subject (`_subject_ref`): set by a real store pick, a non-empty directory path, or a screen-restored ref via `set_subject` (the `"(no subject set)"` remount sentinel never counts). A NULL picker pick never clears it — matching what a run would actually use since the screen persists every `SubjectChanged` immediately.
- Run guard order: subject first (teaching toast: "Pick a subject skill or type a directory path first."), then generator/judge as before. A subjectless Run now posts nothing — no worker, no failed run row.
- Tests: `test_run_without_subject_posts_nothing` (models set, no subject → nothing; sentinel doesn't count; restored ref arms Run) and `test_directory_path_subject_enables_run` (typed path is a valid subject). 15/15 panel tests green; screen suite 20/21 (pre-existing dev red `test_skill_eval_bench_can_be_deleted` fails identically without this change — screen unmounts mid-test, same class as HCI finding B7, needs its own investigation).
- Also marked both skill-eval UI suites `bootstrap_profile` (TASK-32628 admission signature; sanctioned TASK-32873 opt-in) — separate commit.
- Files: `tldw_chatbook/UI/Evals/skill_eval_panel.py`, `Tests/UI/test_evals_skill_eval_panel.py`, `Tests/UI/test_evals_skill_eval_screen.py`.
