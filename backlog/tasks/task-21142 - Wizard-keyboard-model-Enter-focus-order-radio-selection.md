---
id: TASK-21142
title: 'Wizard keyboard model: Enter, focus order, radio selection'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:14'
updated_date: '2026-08-25 07:26'
labels:
  - ux
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings N-1, N-2, N-8, N-9 (findings.md section N): Enter never advances; the abandon action is the first Tab stop after step content; radio highlight is not selection so Down+Next silently keeps the Quick track; Back does not restore focus to the step's primary control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enter advances the wizard when the focused widget does not consume it; Input submit advances; track radio Enter advances
- [x] #2 Track-choice selection follows the highlight (Down then Next yields the Full track)
- [x] #3 Tab from step content reaches Next before any abandon action
- [x] #4 After Back, focus lands on the step's primary control with a visible indicator
- [x] #5 Existing wizard tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify N-9 live post-F-1 heal\n2. SetupRadioSet: selection follows highlight; enter advances (message to container)\n3. Container: enter binding + Input.Submitted -> advance\n4. SetupWizardNavigation subclass: DOM order Next/Back/Cancel + dock CSS so visuals stay Cancel-left\n5. Pilot tests per behavior; live tmux verify; suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
--------------------------------------------------
N-9 re-verified first: focus-after-Back was the SAME orphaned-focus family as F-1 and is already cured by the TASK-21139 heal (Pilot probe: focus restored to the radio, arrows move highlight; live: Down+Space selects). The remaining fixes:

N-8: SetupRadioSet(RadioSet) — selection follows highlight (WAI-ARIA semantics) via action_next_button/action_previous_button overrides selecting the highlighted button; all 10 wizard RadioSets swapped. This also moots the near-invisible highlight glyph.

N-1: Enter advances — SetupRadioSet binds Enter to an AdvanceRequested message (its toggle is redundant once selection follows highlight); SetupWizardContainer binds enter->action_next for other non-consuming foci; Input.Submitted advances, EXCEPT #setup-provider-api-key where Enter keeps launching the credential probe (TASK-1506's deliberate design). Key-hints line now reads 'Enter / Ctrl+N next ...'.

N-2: discovered Textual's focus order is VISUAL (y,x) via _focus_sort_key, not DOM order — the initial DOM-reorder + dock approach measurably changed nothing. Fix: SetupWizardNavigation adopts the Windows-wizard footer (progress docked left; right-aligned Back/Next/Exit cluster), making Next the first enabled Tab stop after step content. BaseWizard untouched (house rule); .setup-navigation CSS scoped so Chatbook wizards keep the stock footer.

Tests: 4 new app-level contract tests (arrow-selects, Enter-from-radio, Tab-reaches-Next, Enter-in-model-input). Live tmux: Enter on Welcome advances; Down selects Full; full footer on one baseline. User guide gains a Keyboard section. 

Post-suite addendum: the first full run caught a REAL regression in the initial cut — Textual's RadioSet._on_mount itself calls action_next_button() to seat the initial highlight, and following it auto-selected the first option on every mount (would have clobbered AppearanceStep's deliberately-unselected fresh-run theme radio). Fixed by gating _select_highlighted on self.has_focus (selection follows the highlight only during user navigation). Full combined suite: 864 passed (one order-dependent flake in test_rerun_over_settings_review_settings_returns_to_settings passed in isolation and on repeat run).
<!-- SECTION:NOTES:END -->
