---
id: TASK-25818
title: First-run stepper marks steps complete that were never configured
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 06:45'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The setup wizard stamps a checkmark on each step the user passes through, regardless of whether its required values were set. Its own summary screen contradicts the stepper, reporting the provider and model as not configured while every step chip above shows complete. Users read the stepper, not the summary, and believe setup succeeded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A step chip shows complete only when that step's required values are set
- [ ] #2 A skipped or incomplete step is visually distinct from a completed one
- [ ] #3 The stepper and the summary screen never disagree about the same step
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The three-state tracker already existed (SetupWizardProgress renders complete/attention/upcoming, and build_setup_progress already downgrades completed steps via attention_ids, documented as 'a visited step whose probe demonstrably failed must not wear the ✓'). The defect was that attention_ids was fed ONLY by provider_probe_failure(), so a step walked through without configuring anything still showed ✓ -- while the summary screen reported it as unconfigured. Fix: new pure helper setup_attention_ids(wizard_data, probe_failed=) feeding the same mechanism, flagging a visited-but-unconfigured Provider or Model. Deliberately scoped to those two: skipping voice or key encryption is legitimate under the wizard's skip-safe design (see TASK-25820), so flagging them would cry wolf. 5 unit tests + full wizard suite 392 passed / 0 failed.
<!-- SECTION:NOTES:END -->
