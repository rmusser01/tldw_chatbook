---
id: TASK-27017
title: Console send after workspace switch persists the wrong provider
status: To Do
assignee: []
created_date: '2026-09-01 19:04'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deterministic red on dev, found while baselining the tray recompose work and re-verified at 3f30fb686: test_console_send_after_workspace_switch_persists_to_selected_workspace expects the selected workspace's session to carry provider llama_cpp and gets openai -- 0 passed / 5 failed of 5 alone with -p no:randomly, identical on pristine dev and feature branches, so it is a real behavioural drift in session-settings provider derivation after a workspace switch, not a flake and not test infrastructure. Recorded on TASK-25715's ledger as finding 5; this task gives it an owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes, or the intended provider-derivation behaviour after a workspace switch is re-decided and the test updated to pin it
<!-- AC:END -->
