---
id: TASK-42652
title: "The first admission-sensitive Console P0 profile runs up to 29.7 of 35 minutes while the second peaks at 20.2"
status: To Do
assignee: []
created_date: 2026-11-28 17:10
updated_date: 2026-11-28 17:10
labels:
- follow-up
- ci
priority: low
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_response_rules_critical_gate.py
- Tests/CI/test_console_p0_regression_gate.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-42533 split the admission-sensitive Console P0 job in two (#3782, merged 2026-11-26). Its first 10 PR runs met the target, ending at least 8 minutes under the 35-minute cap. Over all 80 completed PR runs to 2026-11-28 16:25Z, the two profiles are unbalanced:

| Profile | Holds | Median | p90 | Max |
|---|---|---|---|---|
| `admission-sensitive` | head | 23.2 min | 26.3 min | 29.7 min |
| `admission-sensitive 2` | tail | 17.2 min | 19.4 min | 20.2 min |

Three head runs went above 27 minutes: 27.6, 29.7 and 27.1. The projection in TASK-42533 had the two the other way round. Moving about 3 minutes of files from the head to the tail would leave both with room under the cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Files move from `admission-sensitive` to `admission-sensitive 2` (no file in both, each keeping its isolation profile and flags), chosen from measured per-file times
- [ ] #2 The lane pins (`Tests/CI/test_response_rules_critical_gate.py`, `Tests/CI/test_console_p0_regression_gate.py` and any other that lists the moved files) match the new lists
- [ ] #3 Over the next 20 PR runs after the change, each profile's slowest run ends at least 8 minutes under its cap
<!-- AC:END -->
