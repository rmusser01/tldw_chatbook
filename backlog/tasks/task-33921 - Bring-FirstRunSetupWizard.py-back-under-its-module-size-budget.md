---
id: TASK-33921
title: Bring FirstRunSetupWizard.py back under its module-size budget
status: Done
assignee:
  - '@claude'
created_date: '2026-10-03 01:38'
labels:
  - wizard
  - size-ratchet
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
FirstRunSetupWizard.py is 10,854 lines against its 10,404 budget (+450), red on dev in Tests/Architecture/test_module_size_ratchet.py. Most of the growth is the OmniVoice voice-service work from #2844 (+398 net on 2026-09-26). The ratchet asks for new code in a controller, widget or helper module, not a raised budget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FirstRunSetupWizard.py is at or under its size budget and the ratchet test passes
- [x] #2 The Voice step behaves the same, with its existing tests green
- [x] #3 Message handlers moved out of the wizard still fire (Textual registers on() handlers per message-pump class, so a plain mixin silently drops them)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move the whole VoiceSetupStep class (the OmniVoice growth) to its own module, so its @on handlers stay on the class.
2. Import it lazily in the wizard and re-export it, since the new module needs the wizard's step base classes.
3. Retarget the test patches of names that move; update path-keyed inventories; lower the ratchet row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
VoiceSetupStep (1,007 lines, mostly the OmniVoice work from #2844) moved whole to UI/Wizards/first_run_voice_step.py; the wizard is 9,866 lines and its ratchet row is lowered to that (PR #2973). Moving the class whole, rather than splitting it into a mixin, keeps its @on handlers and workers registered on it (AC3). The new module imports the wizard's step base classes, so the wizard imports the step lazily at its two use sites (step factory, resume isinstance) and re-exports it via a module __getattr__; the four test files that import it from the wizard are unchanged.
The five imports only the step used moved with it (math, tempfile, omnivoice_setup_state, run_omnivoice_preflight, run_omnivoice_provision), so the 13 test patches of those names were retargeted on purpose to the new module. Negative control: a patch left on the wizard module now raises AttributeError instead of silently missing and running the real provisioner.
Derived artifacts: the diagnostic inventory moved the step's six constant-message logger calls (identical digests, reviewed with --statements, no interpolation); the textual_await_dom_census row for _run_voice_sample (4) changed path only, hand-moved because --write also dropped five rows other work had resolved on dev.
Evidence: the 27 test files that reference the wizard, run on the branch and on origin/dev: 240 vs 239 failed, all baseline; the one branch-only failure (TestThemePickerShortlist::test_shortlist_then_show_all_expands) passes 5/5 on both trees. Voice-step tests: 26 passed. Tests/Architecture: 11 vs 12 failures on dev (the wizard's row now passes; the rest are other modules over budget on dev). Preflight green. AC3, checked in a pytest process: all 8 of the module's @on handlers are registered on the moved class (Textual's _decorated_handlers), and the wizard's re-export is the same class object.
<!-- SECTION:NOTES:END -->
