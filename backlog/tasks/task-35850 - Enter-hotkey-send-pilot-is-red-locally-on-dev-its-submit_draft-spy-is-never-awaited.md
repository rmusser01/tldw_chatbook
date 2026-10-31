---
id: TASK-35850
title: >-
  Enter-hotkey send pilot is red locally on dev: its submit_draft spy is never
  awaited
status: To Do
assignee: []
created_date: '2026-10-20 10:38'
labels:
  - console
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_send_disabled_state.py::test_enter_hotkey_still_sends_when_send_is_enabled fails locally on dev 0fef489ece. In its private-profile child it fails with 'Expected mock to have been awaited once. Awaited 0 times.' at the submit_draft spy (line 291), and the persistent diagnostics log a console_send_stage at phase=ui_submit with exception_type=RuntimeError. The test does not call the shared provider-selection helper; it builds its host with _ready_host. TASK-32819's notes listed it, with test_enter_hotkey_queues_draft_behind_accepted_run, as a baseline red on main in September, and no open task tracks it. The file is not in scripts/ui_pr_gate_census.txt, so whether CI runs it green is not known.

Found under TASK-35660 on 2026-10-20 (macOS, plain command, host load 48-72): it failed in one whole-file run on the TASK-35660 branch and in one on dev, the same way both times.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the RuntimeError at ui_submit is named, and the test passes on dev or is fixed so it does
- [ ] #2 If the cause is the local environment rather than the product, the test or its harness says so in its failure message
<!-- AC:END -->
