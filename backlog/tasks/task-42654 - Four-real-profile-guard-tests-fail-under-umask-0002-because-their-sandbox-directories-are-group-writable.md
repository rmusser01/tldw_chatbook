---
id: TASK-42654
title: "Four real-profile guard tests fail under umask 0002 because their sandbox directories are group-writable"
status: To Do
assignee: []
created_date: 2026-11-28 19:37
updated_date: 2026-11-28 19:37
labels:
- follow-up
- tests
priority: low
references:
- Tests/test_real_profile_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 29 (2026-11-28) while checking TASK-42651 on Linux. Four tests in `Tests/test_real_profile_guard.py` fail in an `ubuntu:24.04` container run as a regular (non-root) user, both on dev 2df861ec7d and with TASK-42651's change:
- `test_an_xdist_workers_session_end_refusal_fails_the_run[both-conftests]`
- `test_an_xdist_workers_session_end_refusal_fails_the_run[root-conftest-only]`
- `test_an_xdist_workers_session_end_refusal_fails_the_run[ui-rooted]`
- `test_the_incidents_own_dir_fd_writer_cannot_replace_the_profile_file`

The 61 others pass.

The child sessions these tests start refuse to run with "Chatbook cannot start: its private storage location is not secure on this machine". The blocking path is the test's own `sandbox` directory, with mode `0775` (`shared_writable_parent`). The container user has Ubuntu's default umask `0002`, so the directories the tests create are group-writable. The app's refusal is correct; the tests depend on the developer's umask.

GitHub's hosted runners use umask `022`, and no CI workflow names this file. So nothing shows these reds, but any developer on Ubuntu's default umask gets them locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four tests pass under umask `0002` (their sandbox directories are created private, or the tests set and restore a safe umask), with every assertion kept
- [ ] #2 A test pins that the sandbox directories these tests create are not group- or world-writable
- [ ] #3 The file runs in a CI lane
<!-- AC:END -->
