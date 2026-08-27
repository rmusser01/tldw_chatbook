---
id: TASK-21146
title: Fold model-lists consent into wizard summary
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:15'
updated_date: '2026-08-25 21:58'
labels:
  - ux
  - wizard
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT finding H-1 (findings.md): finishing the wizard lands in Console which immediately opens the 'Check model lists online?' consent modal - a fourth decision at the moment of first chat, and setup itself already contacted the provider. Surface the consent as an unchecked-by-default option on the wizard Summary; Console must respect the recorded answer and not re-ask. Privacy default must not weaken: no consent means no online checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wizard Summary offers the model-lists consent, default off
- [x] #2 After completing the wizard, Console does not show the consent modal (either answer)
- [x] #3 Users who skip the wizard still get the existing Console consent flow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find consent modal + persisted flag\n2. Summary checkbox (default off) writing the same flag on completion\n3. Console skips modal when an answer exists; skip-wizard path unchanged\n4. Tests + live run
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary composes a SetupCheckbox (default OFF, deny-by-default — same privacy posture as the modal) shown only while [model_catalog].refresh_consent_recorded is false (reruns never re-ask; fail-closed on a bad read). SummaryStep.commit persists the exact contract _handle_model_catalog_consent writes: refresh_consent_recorded=true, plus auto_refresh_enabled=false on deny — so the Console modal can never fire after a completed wizard, while skip-the-wizard paths never write consent and keep the existing modal flow. 'model_catalog' added to WIZARD_OWNED_SECTIONS (the live run caught the allowlist rejection — and, satisfyingly, the TASK-21140 pinned error strip surfaced 'Saving the model-list preference failed. Retry with Next, or go Back.' exactly as designed).

Two follow-on fixes the work surfaced: (1) stock Checkbox conveys checked state by color only — live UAT read the UNCHECKED box as checked (constant ▐X▌ glyph); SetupCheckbox mirrors TASK-1497's structural-glyph fix (✓/blank) and now backs both wizard checkboxes; (2) Summary focus raced the async row-render reveal — preferred_focus() now lands on the primary exit button (Enter finishes setup), and the focus-walk test helper mirrors production's preferred-first logic.

Live: fresh run shows '▐ ▌ Keep model lists fresh' (unchecked reads unchecked), completion writes the deny pair, no consent modal after. Suites: 877 passed. User guide updated.

Files: FirstRunSetupWizard.py, first_run_setup_state.py, Tests/Wizards/test_first_run_setup_wizard.py, Docs/User_Guide/First_Run_Setup.md.
<!-- SECTION:NOTES:END -->
