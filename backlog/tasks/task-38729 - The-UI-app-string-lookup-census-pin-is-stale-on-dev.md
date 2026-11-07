---
id: TASK-38729
title: "The UI app-string-lookup census pin is stale on dev"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture
- tests
priority: medium
references:
- Tests/Architecture/test_app_string_lookup_census.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[UI]` fails with 'UI: 1276 lookups, pinned 1278. Lower the pin to 1276'. It was already red on dev c2cdb6b82e before the architecture-review Phase 2 merge, and stayed red at every branch of that program's merge chain.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The UI row in the app-lookup census is lowered to the measured count, and the test passes on dev
- [ ] #2 The task notes say which dev change dropped the count without lowering the pin
<!-- AC:END -->
