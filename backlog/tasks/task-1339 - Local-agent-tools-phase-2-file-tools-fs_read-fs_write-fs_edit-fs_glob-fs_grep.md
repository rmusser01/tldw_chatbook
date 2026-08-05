---
id: TASK-1339
title: >-
  Local agent tools phase 2: file tools
  (fs_read/fs_write/fs_edit/fs_glob/fs_grep)
status: In Progress
assignee: []
created_date: '2026-08-05 05:09'
updated_date: '2026-08-05 05:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md (phase 2). Plan: Docs/superpowers/plans/2026-08-04-local-agent-tools-phase2.md. ADR: backlog/decisions/032. Builds on task-1338 (phase 1, fs_list pilot). NOTE: fs_list already landed in phase 1 and is out of scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 fs_read pages line-numbered output with offset/limit and refuses binary files
- [ ] #2 fs_write creates/overwrites files confined to workspace root with mutates risk tag
- [ ] #3 fs_edit performs unique-match replacement with ambiguity errors and replace_all
- [ ] #4 fs_glob and fs_grep search the workspace with result caps
- [ ] #5 Legacy ReadFileTool/WriteFileTool delegate to the shared cores with unchanged legacy behavior
- [ ] #6 Local deny/timeout outcomes are audit-recorded
- [ ] #7 Tools remain reachable via find_tools/load_tools past the direct-disclosure threshold
- [ ] #8 All new tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-04-local-agent-tools-phase2.md
<!-- SECTION:PLAN:END -->
