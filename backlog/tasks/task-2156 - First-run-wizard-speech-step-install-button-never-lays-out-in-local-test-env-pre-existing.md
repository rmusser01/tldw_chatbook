---
id: TASK-2156
title: >-
  First-run wizard: speech-step install button never lays out in local test env
  (pre-existing)
status: To Do
assignee: []
created_date: '2026-08-06 21:15'
labels:
  - tests
  - first-run
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_first_run_wizard_live_contract.py::test_speech_step_install_button_visible_at_120x40_without_scrolling fails deterministically (~13s): the test reaches the speech step, but #setup-speech-install never gets a non-zero region within the 10s wait — the real ModelArtifactService background worker over the isolated test data dir never settles, leaving the 'Checking installed models…' placeholder. Reproduced at 1db4e1362 (pre-dates the Console UX remediation batches 4-5); unrelated to TASK-2154 work. Same triage pattern as TASK-2155. Likely environmental (machine lacks STT capture backends / model artifacts), possibly a genuinely broken install-state worker — needs a real investigation pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Test passes in the local environment,Root cause documented in Implementation Notes
<!-- AC:END -->
