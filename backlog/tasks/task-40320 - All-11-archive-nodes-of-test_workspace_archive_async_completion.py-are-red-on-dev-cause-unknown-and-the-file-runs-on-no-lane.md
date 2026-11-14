---
id: TASK-40320
title: "All 11 archive nodes of test_workspace_archive_async_completion.py are red on dev, cause unknown, and the file runs on no lane"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- workspaces
- dev-red
priority: low
references:
- Tests/UI/test_workspace_archive_async_completion.py
dependencies:
- TASK-39169
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream review r13p-rev1: 11 failed at base ae9f9e8d21 and at the cprod head). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Every archive node of `Tests/UI/test_workspace_archive_async_completion.py` fails, on both the Console and Settings surfaces; the unarchive node passes. The tests drive the real handlers through a `SimpleNamespace` `_host` (`:20-56`), so drift in that stand-in is the first suspect, but the cause has not been read. The file runs on no lane, so archive's async completion has no guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 11 nodes is classified (test drift with the commit, or a product defect fixed with a test that was red) and passes at `-n 4`
- [ ] #2 The file runs on a PR lane
<!-- AC:END -->
