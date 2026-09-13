---
id: TASK-21140
title: 'Wizard step errors: always-visible surface'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:14'
updated_date: '2026-08-25 06:41'
labels:
  - ux
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings W-1, G-3, plus the hidden show_step_error surface (findings.md sections W/G and the F-1 investigation): step errors render into a .setup-step-error Static at the bottom of an overflowing scroll region, invisible at common terminal sizes; the empty strip paints an error background on Welcome; the Notes step uses the error slot for neutral info (red-on-maroon reassurance); the error copy references a nonexistent Skip-this-step control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A step commit failure is visible without scrolling at 140x40 and 80x24 (error surface pinned near the footer)
- [x] #2 Empty error surfaces paint no background anywhere in the wizard
- [x] #3 Notes-step informational text renders in neutral styling and survives a real error being shown
- [x] #4 No error copy references controls that do not exist
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add pinned error Static (#setup-step-error-pinned) to container chrome next to _ProviderSaveStatus\n2. Retarget SetupStep.show_step_error to it; clear on show_step\n3. Remove empty per-step tail .setup-step-error Statics; re-class Notes info to setup-step-note (+ CSS); Voice inline error via show_step_error\n4. Replace '(Retry, or Skip this step.)' with real affordances\n5. Update tests querying step-local slots; add pinned-strip contract test\n6. Regenerate modular tcss; live tmux check at 140x40 + 80x24
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Step errors now render on a pinned Static (#setup-step-error-pinned) in the SetupWizardContainer chrome between the step body and the nav bar — visible at any terminal size; cleared on every show_step. SetupStep.show_step_error retargeted; the nine per-step empty tail .setup-step-error Statics removed (kills UAT W-1's empty maroon stripe, verified by ANSI comparison against the original UAT capture); NotesSyncStep reassurance re-classed to new .setup-step-note (neutral, verified live); VoiceSetupStep inline speed error routed through show_step_error; '(Retry, or Skip this step.)' replaced with 'Retry with Next, or go Back.' (only real affordances). CSS: .setup-step-note added to _wizards.tcss; modular tcss regenerated. Tests: 4 queries retargeted to the pinned strip; new contract test (failure renders on strip + honest copy + clears on step change; had to wait for can_proceed — advance_programmatically silently drops presses before validation settles, a pre-existing behavior worth knowing). Suites: Tests/Wizards/test_first_run_setup_wizard.py + live-contract + visual-audit = 462 passed. Docs/User_Guide/First_Run_Setup.md updated + restamped (also corrected 'every step has a Skip' phrasing — no such control exists).

Files: FirstRunSetupWizard.py, css/features/_wizards.tcss, css/tldw_cli_modular.tcss (regenerated), Tests/Wizards/test_first_run_setup_wizard.py, Tests/UI/test_first_run_wizard_live_contract.py, Docs/User_Guide/First_Run_Setup.md.
<!-- SECTION:NOTES:END -->
