---
id: TASK-25820
title: First-run model step leaves the only recommended model unselected
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 06:35'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The model step queries the configured provider, finds the available models, and labels one as recommended, but leaves every option unselected. Pressing Next proceeds with no model chosen and stamps the step complete. When exactly one model is offered and it is already marked recommended, requiring a separate click adds a step whose only outcome is a misconfigured install.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A sole available model marked recommended is selected by default
- [ ] #2 Advancing without a model selected is either prevented or clearly reported
- [ ] #3 The summary reflects the model actually chosen
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
INVALID -- the current behaviour is deliberate. Do not implement this.

Implemented TDD-first (pre-press index 0 when nothing is selected, mirroring
the speech step's "pre-press ONLY the recommended option" rule). It broke 34
tests in Tests/Wizards/test_first_run_setup_wizard.py, led by:

  test_model_step_empty_selection_commits_nothing
  """Skip-safe: leaving the model step untouched must not touch config."""

The model step is INTENTIONALLY skip-safe: an untouched step must commit
nothing. Related pinning tests reinforce it --
test_model_step_with_provider_entry_present_does_not_prefill_stale_model,
test_model_step_provider_switch_does_not_resurrect_stale_pressed_radio,
test_typing_manual_model_clears_keyboard_selected_radio. Pre-selecting a model
the user never chose is exactly what these forbid, and it would write a model
id into config on a step the user only passed through.

The underlying UX complaint is still real but must be solved WITHOUT
auto-committing: e.g. make Next surface "No model selected -- pick one or
continue without a default", or mark the step incomplete in the stepper
(TASK-25818) so the summary's honesty is reflected earlier. Reverted.
<!-- SECTION:NOTES:END -->
