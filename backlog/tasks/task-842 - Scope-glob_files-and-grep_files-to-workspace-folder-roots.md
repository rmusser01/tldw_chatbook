---
id: TASK-842
title: Scope glob_files and grep_files to workspace folder roots
status: To Do
assignee: []
created_date: '2026-07-27 02:36'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
read_file, write_file and list_directory honour dev's allowed_file_roots workspace folders, but glob_files and grep_files were scoped to the tool sandbox root only when they were added. The result is strictly narrower than their siblings, so it is safe but inconsistent -- an agent can read a file it cannot find by search. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 glob_files and grep_files honour the same root set as read_file,Sandbox-only configurations behave exactly as before,A test covers a workspace-bound folder for both tools
<!-- AC:END -->
