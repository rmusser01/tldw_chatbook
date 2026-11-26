---
id: TASK-42552
title: "Seventeen Console tests in four files no PR lane runs are red on dev"
status: To Do
assignee: []
created_date: 2026-11-26 04:50
labels:
- follow-up
- console
- dev-red
priority: medium
references:
- Tests/Chat/test_console_continuation_group_merge.py
- Tests/Chat/test_console_named_agent_model.py
- Tests/Chat/test_console_redirect_stream_cut.py
- Tests/Chat/test_console_request_provenance_routes.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 26 (TASK-42525's wide runs, 2026-11-25). 17 tests in `Tests/Chat/test_console_continuation_group_merge.py`, `Tests/Chat/test_console_named_agent_model.py`, `Tests/Chat/test_console_redirect_stream_cut.py` and `Tests/Chat/test_console_request_provenance_routes.py` fail identically on the round-26 stack and with the stack's tests on the base's source. No PR lane runs these four files, so nothing reports them, and a real regression in continuation grouping, named-agent model choice, redirect stream cuts or request provenance routes would read as old noise. It is not yet known whether the product or the tests are wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each red test, the notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [ ] #2 A product defect is fixed and its test shown red before and green after; a drifted test is repaired with every behaviour assertion kept
- [ ] #3 All four files pass on dev at `-n 4`, twice in a row
- [ ] #4 The four files run on a PR lane that blocks merges to dev, timed with that lane's flags
<!-- AC:END -->
