---
id: TASK-42008
title: "The query-synthesis Hypothesis test has a real falsifying example"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
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
Found by TASK-41709 (round 17), red at base. `test_no_wrapping_lets_a_mark_through_or_loses_its_goal` fails for `'Kyoto: **"[g1]"Wednesday 13" tour dates 2026"**'`: the query becomes `'Kyoto Wednesday 13" tour dates 2026"'` where `'Kyoto "Wednesday 13" tour dates 2026'` is expected. The file is on no PR or dev-push lane; the release `test.yml` sweep runs it. The input comes from the generator's `front-glued` path with the `"{}"` wrapper, which keeps `exact=True`, yet the result is textually the `front-glued-wrapped` `"[g1]"` shape the same generator excludes as unreadable (~873-890), so the defect may be the generator's exactness rule rather than `_query_row`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named in the notes (generator exactness or product)
- [ ] #2 Either `_query_row` returns `'Kyoto "Wednesday 13" tour dates 2026'` for this input, or the generator marks this shape inexact
- [ ] #3 The input is pinned with `@example`, red before the fix
<!-- AC:END -->
