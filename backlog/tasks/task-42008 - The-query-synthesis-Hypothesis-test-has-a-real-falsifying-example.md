---
id: TASK-42008
title: "The query-synthesis Hypothesis test has a real falsifying example"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- dreams
priority: low
references:
- Tests/Dreams/test_query_synthesis.py
- tldw_chatbook/Dreams
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41709 (round 17), red at base. `test_no_wrapping_lets_a_mark_through_or_loses_its_goal` fails for `'Kyoto: **"[g1]"Wednesday 13" tour dates 2026"**'`: the query becomes `'Kyoto Wednesday 13" tour dates 2026"'` where `'Kyoto "Wednesday 13" tour dates 2026'` is expected. CI hits it whenever Hypothesis generates this shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Query synthesis keeps the quoted goal for this input, and the example is pinned with `@example`
<!-- AC:END -->
