---
id: TASK-33921
title: Bring FirstRunSetupWizard.py back under its module-size budget
status: To Do
assignee: []
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
- [ ] #1 FirstRunSetupWizard.py is at or under its size budget and the ratchet test passes
- [ ] #2 The Voice step behaves the same, with its existing tests green
- [ ] #3 Message handlers moved out of the wizard still fire (Textual registers on() handlers per message-pump class, so a plain mixin silently drops them)
<!-- AC:END -->
