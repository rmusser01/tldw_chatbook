---
id: TASK-32883
title: 'Skill eval: empty subject picker dead end'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:29'
updated_date: '2026-09-21 23:18'
labels:
  - ux
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review A1 (HIGH). A first-time user opening the skill-eval launch panel gets a subject Select that renders a BLANK overlay when the skills store is empty - no message, no pointer to where skills live - and the directory-path Input documents nothing (SKILL.md convention, validation). Users cannot tell 'no subjects' from 'broken control'. Evidence: live walkthrough 2026-09-21, /tmp/uat-skill-eval/HCI-REVIEW.md persona 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Store Select shows an empty-state row (e.g. 'No skills in your store - install one or type a directory path below') instead of a blank overlay when the store is empty
- [x] #2 Directory-path Input carries an inline hint naming the expected layout (folder containing SKILL.md)
- [x] #3 Typed path is validated on blur with an inline not-found message; no silent acceptance of invalid paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Panel tests: empty store shows guidance prompt + non-blank overlay row that posts nothing; path input carries SKILL.md hint and inline not-found validation
2. Implement: sentinel guidance option in set_subjects([]), prompt swap, _SkillDirectoryValidator + hint Static, handler ignores sentinel pick
3. Targeted: Tests/UI/test_evals_skill_eval_panel.py
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Empty store: picker prompt becomes "No skills installed — type a path below" and the overlay holds one guidance row carrying a sentinel value; picking it posts nothing (handler rejects the sentinel, resets to NULL, informational notify) — no more blank dead-end overlay. Populated stores keep the neutral prompt.
- Directory input: inline hint Static ("A folder containing SKILL.md") plus an advisory `_SkillDirectoryValidator` (mirrors `subject_from_directory`: absolute dir containing SKILL.md) — invalid paths get "No SKILL.md found at that path" inline and invalid input styling; run-time resolution stays authoritative (feedback is not a block).
- Tests: `test_empty_store_picker_shows_guidance_not_a_blank_overlay`, `test_populated_store_picker_keeps_the_standard_prompt`, `test_directory_input_hint_and_inline_validation`. Panel 18/18 green; screen 19/21 (1 pre-existing dev red unrelated: `test_skill_eval_bench_can_be_deleted`).
- Files: `tldw_chatbook/UI/Evals/skill_eval_panel.py`, `Tests/UI/test_evals_skill_eval_panel.py`.
