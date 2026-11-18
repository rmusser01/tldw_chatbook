---
id: TASK-41654
title: "The Widgets app-lookup census reads 35 on dev against a pin of 33"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- architecture
- tests
priority: medium
references:
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_lookup_census.tsv
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[Widgets]` fails on dev 45b1949ca3 and later: 35 literal `getattr`/`hasattr` lookups on the app object against a pin of 33. It passed on 5e3c8b2bd4, so the two new lookups arrived with the round-15 merges #3594..#3609. The census says not to raise the pin.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two new lookups reach their service through a typed attribute or explicit accessor
- [ ] #2 The Widgets census passes at its existing pin
<!-- AC:END -->
