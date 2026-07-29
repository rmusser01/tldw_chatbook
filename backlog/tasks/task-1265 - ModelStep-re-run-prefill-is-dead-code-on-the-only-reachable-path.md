---
id: TASK-1265
title: ModelStep re-run prefill is dead code on the only reachable path
status: To Do
assignee: []
created_date: '2026-07-29 22:09'
updated_date: '2026-07-29 22:10'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During the final whole-branch review of the first-run setup wizard (backlog/tasks/task-1264*.md, Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md, Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md), auditing ModelStep's re-run prefill branch in tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py (ModelStep.on_show, the has_provider_entry check around the prefill_model_id assignment) found it can never actually fire on the only navigation path the wizard exposes. SetupWizardContainer._advance() (same file) unconditionally does self.wizard_data[step_id] = step.get_step_data() for every step it advances past, regardless of whether that step's commit() persisted anything (e.g. skip-safe steps that return True, "" with nothing selected still get a wizard_data entry). Since Model is only ever reached by advancing past Provider (active_step_ids in first_run_setup_state.py never allows an out-of-order jump -- _next_active_index/_previous_active_index only move one active step at a time), wizard_data[STEP_PROVIDER] is always already set by the time Model's on_show() runs, so has_provider_entry is always True and the not-has_provider_entry prefill branch (and its pinning test, Tests/Wizards/test_first_run_setup_wizard.py::test_model_step_rerun_prefills_from_config_when_no_provider_entry_yet) exercise a state the real UI can never produce.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either ModelStep's re-run prefill fires on a genuinely reachable path (e.g. by not writing a wizard_data entry for skipped/no-op steps, or by deriving has_provider_entry from persisted config instead of session wizard_data), or the dead branch and its pinning test are removed with the resulting behavior documented as intentional
- [ ] #2 Tests/Wizards/test_first_run_setup_wizard.py reflects whichever outcome is chosen (no test pins an unreachable state as if it were live behavior)
- [ ] #3 Full Tests/Wizards/ and Tests/UI/test_first_run_wizard_live_contract.py suites stay green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed during the final-fix-wave review (branch feature/first-run-setup-wizard, HEAD 77153eaaf at review time). See ModelStep.on_show()'s has_provider_entry check and SetupWizardContainer._advance() in tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py.
<!-- SECTION:NOTES:END -->
