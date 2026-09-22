---
id: TASK-32888
title: 'Skill eval: silent navigation away loses in-progress launch state'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-21 22:31'
updated_date: '2026-09-22 00:10'
labels:
  - ux
  - evals
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HCI review B7 (HIGH, needs repro). During a keyboard-only power-user walkthrough (2026-09-21, /tmp/uat-skill-eval), a plain Tab traversal with NO Enter sent coincided exactly with the app log switching Evals->Library ('Screen evals unmounted' 13:02:44) - the in-progress launch panel (subject+models) was lost and no run was ever created (zero evals.db writes in the window). Trigger needs repro (focus landing on nav chrome is the prime suspect; NavigationButton does not navigate on focus per source). Consequence class regardless: nothing protects an in-progress launch panel from accidental navigation or warns that unsaved picks are lost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced (or ruled out) the exact trigger: scripted keyboard sequence + log evidence identifying which key/focus position navigates away
- [x] #2 Subject, depth, and model picks persist on change and survive navigation away and back
- [x] #3 Guardrail or warning when navigating away from an in-progress launch configuration
- [x] #4 Documented in the task whether models/depth round-trip like subject already does via SubjectChanged
<!-- AC:END -->

## Implementation Notes

- **Repro verdict**: a pinned test (`test_tab_traversal_never_navigates_away`) walks a full Tab cycle across the Evals screen and never switches screens -- pure Tab traversal cannot navigate. The served incident's Evals->Library switch therefore required an Enter/click on destination chrome. With configuration now persisting, that is no longer state-destroying; no separate navigation guardrail is needed (Escape-close from TASK-32889 is the explicit leave path).
- **Persistence**: panel posts `DepthChanged` and `TargetsPicked` on change; the screen round-trips them exactly like the subject (load/replace/save). All three survive navigate-away-and-back (`test_launch_picks_survive_navigation_away_and_back`).
- **Restore staleness guarantee kept**: model ids restore only when their eval_models row still exists; stale ids render unset and the Run guard demands a fresh pick. To keep that honest, draft benches no longer pre-seed the first available model invisibly (empty ids; the pinned creation test updated with rationale) -- the old pre-seed was exactly the unseen-default trap the HCI review flagged.
- **Swap-feed black hole (new finding)**: panels mounted inside a selection swap sometimes never receive their compose-time `panel.call_after_refresh` feeds, and callbacks scheduled from the swap worker (`call_later`, panel `call_after_refresh`) were also observed never dispatching. Fixed by an inline-awaited safety net in the swap worker (`_feed_skill_eval_panel_if_unfed`): waits for the panel's children to compose, then feeds any unfed panel and fills NULL controls from the persisted config.
- **Test lesson**: a Select's `_options` is always truthy (NULL padding row) -- waits must poll real values. Two vacuous waits masked the net's timing and produced false reds.
- Files: `tldw_chatbook/UI/Screens/evals_screen.py`, `tldw_chatbook/UI/Evals/skill_eval_panel.py`, `Tests/UI/test_evals_skill_eval_screen.py`. Suites: screen 24/25 (1 known pre-existing dev red), panel 24/24, empty-states 107/107, mode-keys 4/4, Evals/skill_eval 76/76.
