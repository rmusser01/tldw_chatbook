---
id: TASK-38111
title: >-
  Tests: several UI and Chat test files error at setup with RecoveryRequired
  raw_source_selection_changed when run on their own
status: To Do
assignee: []
created_date: '2026-10-30 05:56'
labels:
  - tests
  - test-infra
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-33621.8/.9 review on origin/dev ff815f8559. Run one file per command, as agents and developers do locally, and every test in these files errors at setup: Tests/UI/test_trace_export_ui.py, Tests/UI/test_console_inspector_workflow.py, Tests/UI/test_console_inspector_focus_carriers.py, Tests/UI/test_non_obscuring_focus_contract.py and Tests/Chat/test_trajectory_capture.py. They pass when given a bootstrap_profile marker, so their tests can only be checked locally through a marked copy, and a regression in them is invisible to a per-file run.

For test_trace_export_ui.py the traceback is: Tests/UI/conftest.py _disable_model_catalog_refresh monkeypatches tldw_chatbook.app.TldwCli, which imports tldw_chatbook.app for the first time inside the per-test profile; the app module runs load_settings() at import, and config admission refuses with RecoveryRequired('raw_source_selection_changed'). This is the same class as the eager imports Tests/conftest.py already makes (Chunking, RAG config profiles, model capabilities) and the per-file list in that fixture. The other files were reported with the same error and need their own traceback read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed file passes, or fails only on its own assertions, under the plain one-file command on a clean checkout
- [ ] #2 The fix does not bind config to the developer's real profile (the conftest's documented trap)
- [ ] #3 The cause of each listed file is confirmed from its own traceback, not assumed from test_trace_export_ui.py
<!-- AC:END -->
