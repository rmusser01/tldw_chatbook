---
id: TASK-37812
title: 'Tests: the Console controller-exchanges file runs in no PR lane and fails under a plain local run'
status: To Do
assignee: []
created_date: '2026-10-29 17:30'
labels:
  - tests
  - console
dependencies: []
references:
  - backlog/docs/lessons-testing-evidence.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Chat/test_console_chat_controller_exchanges.py` holds the controller-level tests for Temporary-chat capture and Save & Send. No PR-gate lane runs it. Its Save & Send test was red on dev without anyone noticing until TASK-33621.17 (see the lessons entry).

A plain local run also fails some of its sends with `RecoveryRequired: raw_source_selection_changed` or 'Hooks unavailable', because the per-test sandbox profile is not the selected one. TASK-33621.17/.18 marked the five tests they touched `bootstrap_profile`.

Two tests still fail under the plain command, on dev 38c9d2edcf and on that branch alike:
- `test_temporary_capture_on_pauses_real_submit_before_gateway`
- `test_shipped_config_default_resolves_exchange_capture_true`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in the file passes under the plain local command
- [ ] #2 A PR-gate lane runs the file, or its send-path tests, so a red one blocks a merge
<!-- AC:END -->
