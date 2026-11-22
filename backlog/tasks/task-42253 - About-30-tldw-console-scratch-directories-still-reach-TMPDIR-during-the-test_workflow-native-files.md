---
id: TASK-42253
title: "About 30 tldw-console scratch directories still reach TMPDIR during the test_workflow native files"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/conftest.py
- tldw_chatbook/Chat/console_scratch_space.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (TASK-42156, round 19). TASK-42112's session fixture in `Tests/conftest.py` gives `ConsoleScratchSpaceManager` a basetemp parent only when the caller passes none. During the `test_workflow_*` native files about 30 `tldw-console-*` directories still land in `$TMPDIR`, so a path that passes its own parent, or builds the manager before the fixture applies, is not covered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The path that creates the directories is named
- [ ] #2 A run of the `test_workflow_*` native files leaves no `tldw-console-*` directory in `$TMPDIR`, counted with a private per-worker `$TMPDIR` and recorded
- [ ] #3 A test pins the covered path
<!-- AC:END -->
