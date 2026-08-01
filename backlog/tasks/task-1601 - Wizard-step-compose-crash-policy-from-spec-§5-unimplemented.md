---
id: TASK-1601
title: Wizard step compose() crash policy from spec §5 unimplemented
status: Done
assignee: []
created_date: '2026-07-29 22:10'
updated_date: '2026-07-29 22:11'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md section 5 (Error handling) specifies two distinct failure classes with two distinct policies: a step's compose() crash must auto-skip that step, show a one-line notice, and render a reasoned (X-with-reason) row for it in the Summary matrix; separately, a commit/validation failure keeps the user on the step with inline Retry/Skip. Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md carries this same requirement into the implementation plan. Only the second policy (commit/validation failure -> SetupStep.show_step_error + inline Retry/Skip, see tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py's SetupStep.commit()/show_step_error() and each step's commit() implementations) was implemented across the 12 build tasks that shipped task-1264. The first policy -- what happens when a step's own compose() raises -- has no auto-skip, no one-line notice, and no reasoned Summary row anywhere in SetupWizardContainer, WizardContainer.show_step(), or SummaryStep/build_summary_rows() in first_run_setup_state.py. A crashing compose() today would surface as an unhandled exception (or, depending on where Textual catches it, a dead/blank step the user cannot get past), not the graceful auto-skip the spec calls for.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SetupWizardContainer catches a step's compose()-time crash, advances past that step automatically instead of leaving the wizard stuck, and shows a one-line notice that a step was skipped due to an error
- [x] #2 SummaryStep's read-back matrix (build_summary_rows in first_run_setup_state.py, or an equivalent reasoned row) reflects the crashed step with a reason rather than silently omitting it
- [x] #3 A test drives a step whose compose() deliberately raises and asserts the wizard survives, auto-skips it, and the Summary reflects it
- [x] #4 Full Tests/Wizards/ and Tests/UI/test_first_run_wizard_live_contract.py suites stay green
- [x] #5 BaseWizard.py is not modified (per this project's established constraint that the wizard framework is subclassed, never edited)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed during the final-fix-wave review (branch feature/first-run-setup-wizard, HEAD 77153eaaf at review time). Spec: Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md section 5 (Error handling), bullet 'Two failure classes, two policies'. Plan: Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md. Implemented policy (commit/validation failure) lives in tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py's SetupStep.commit()/show_step_error(); the compose()-crash auto-skip policy has no implementation anywhere in that file or in BaseWizard.py's WizardContainer.show_step().
<!-- SECTION:NOTES:END -->
