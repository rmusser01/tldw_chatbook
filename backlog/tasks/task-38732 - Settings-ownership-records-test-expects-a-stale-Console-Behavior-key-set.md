---
id: TASK-38732
title: "Settings ownership-records test expects a stale Console Behavior key set"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- tests
- settings
priority: low
references:
- Tests/UI/test_settings_configuration_hub.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), `Tests/UI/test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries` fails. The expected CONSOLE_BEHAVIOR tuple in the test predates the `console.os_notifications.*` keys that the screen gained in a746d1af7a.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's expected key set matches the screen's ownership records, and the test passes on dev
- [ ] #2 No other test in the file regresses
<!-- AC:END -->
