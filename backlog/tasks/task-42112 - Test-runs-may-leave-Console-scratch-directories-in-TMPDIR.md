---
id: TASK-42112
title: "Test runs may leave Console scratch directories in TMPDIR"
status: To Do
assignee: []
created_date: 2026-11-20 10:36
updated_date: 2026-11-20 10:36
labels:
- follow-up
- tests
- console
priority: low
references:
- tldw_chatbook/Chat/console_scratch_space.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41708 (round 17). After its full guarded sweep of the test suite (2026-11-19/20, every test file through the scratch wrapper, whose pytest basetemp is deleted after each run), the implementer reported about 45,000 directories, about 16 GB, left in the system `$TMPDIR`. About 28,000 of them were named `tldw-console-*`, the prefix `ConsoleScratchSpaceManager` passes to `tempfile.mkdtemp` (`Chat/console_scratch_space.py`). Its `dispose()` and `close()` remove them; a test that builds a Console runtime and never disposes it would leave one behind.

By 2026-11-20 10:36 UTC no `tldw-console-*` directory remained in `$TMPDIR`, and what removed them is not known. So the leak is reported, not yet reproduced. Free disk on the machine fell from about 223 GB to 30 GB during the sweep, while the wrapper's own basetemp stayed under 100 MB.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A local run of the Console UI and Chat test files is measured for `tldw-console-*` directories left in `$TMPDIR` afterwards, with the count recorded
- [ ] #2 If any are left, the tests or fixtures that leave them are named and fixed so a run leaves none, or each remaining source has its own task
<!-- AC:END -->
