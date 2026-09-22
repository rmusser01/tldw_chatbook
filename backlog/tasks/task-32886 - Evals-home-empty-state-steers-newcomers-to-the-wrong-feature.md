---
id: TASK-32886
title: 'Evals home: empty state steers newcomers to the wrong feature'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:30'
updated_date: '2026-09-21 23:32'
labels:
  - ux
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review A4 (HIGH). The Evals home's recommended first step is 'Create a sample bench' (word bench). The skill-eval CTA is the third of four near-identical '+ New' rail buttons, explained only by a hover-only tooltip (keyboard-inaccessible, contrary to the repo's TASK-1076 visible-callout convention). Users arriving for skill evals are actively pointed elsewhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the skills store has skills but no skill-eval benches exist, the empty state recommends '+ New skill eval' instead of the word-bench sample
- [x] #2 Each '+ New' rail button carries a visible (non-hover) one-line caption or the disabled-reason hints are visually attached to their own button
- [x] #3 The 'Create or import a dataset first.' hint cannot be read as belonging to '+ New skill eval'
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Empty-states tests: skills-present rail steers to skill-eval; +New skill eval visible caption; hints interleaved under their own buttons
2. Rail: skills_available param, steering hint branch, interleaved hints + caption
3. Screen: _store_has_skills flag + on-mount probe worker, pass through compose_lab_rail
4. Targeted: Tests/UI/test_evals_empty_states.py + skill-eval suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Rail gains `skills_available` (ctor param, default False); the fully-empty Benches branch steers skills-holding users to "+ New skill eval" (`#evals-rail-skill-eval-hint`, same first-run-hint styling) and suppresses the word-bench "Start here" copy. The sample-bench button itself stays reachable.
- Screen probes the store once after mount (`_probe_store_skills_presence` worker, `store_skill_names`, silent-degrading like `_feed_skill_eval_subjects`) and refreshes the mounted rail in place (`refresh(recompose=True)`, the rail toggle's own pattern) when the answer changes; `compose_lab_rail` passes the flag through.
- `_new_bench_actions` interleaves each disabled-reason hint directly under its own button (inside the row container the rail-width CSS stacks vertically), so "Create or import a dataset first." can no longer be read as "+ New skill eval"'s precondition; "+ New skill eval" gains a permanent muted caption ("grade one skill: static + judge + simulation") — ids/classes preserved where tests scope on them.
- Tests: 3 new in `test_evals_empty_states.py` (steering, caption, hint ordering); both TASK-1076 guards still green. Empty-states 107/107; panel 21/21; screen 19/21 (same single pre-existing dev red).
- Files: `tldw_chatbook/UI/Evals/library_rail.py`, `tldw_chatbook/UI/Screens/evals_screen.py`, `Tests/UI/test_evals_empty_states.py`.
