---
id: TASK-41039
title: "The learning-repository transaction tests cannot join the hypothesis-free PR lane"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- ci
- study
priority: medium
references:
- Tests/ChaChaNotesDB/test_study_functionality.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36949.2 added `Tests/ChaChaNotesDB/test_study_functionality.py::TestLearningRepositoryTransactions` (6 tests, passing on dev ecad404bd4 (2026-11-15)). They pin that learning writes stay inside one transaction and roll back together, the guard that matters once learning data lives in its own repository.

The Phase 3 lane batch meant to run them in the `pr-fast-lane-arch` job. That job installs only `-e . pytest pytest-asyncio pytest-timeout packaging aiohttp-jinja2`. The module imports `hypothesis` at top level (`:19-20`, for its other property-based tests), so in that job the whole file fails at collection; the coordinator reproduced this in a venv with only the job's install line (2026-11-15). The transaction tests therefore run on no PR lane, and a rollback regression would reach dev unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TestLearningRepositoryTransactions collects and passes in a venv with only the pr-fast-lane-arch install set
- [ ] #2 The six tests run on a PR lane; the lane entry is listed for the coordinator if a workflow edit is needed
<!-- AC:END -->
