---
id: TASK-32600
title: >-
  Restore the compact Notes import journey assertion to the shipped authority
  contract
status: To Do
assignee: []
created_date: '2026-09-15 02:48'
labels:
  - library
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 60x20 variant of test_database_notes_import_once_journey_is_painted_focused_and_retained stops at an obsolete requirement that library-notes-authority repeat Library notes. Below 64 columns, _authority_prefix intentionally omits that prefix because the immediately preceding source strip already names Library notes and Folder files (task-32360). The 120x36 variant passes. This early stale assertion leaves the compact import journey unverified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 60x20 journey verifies the visible source strip and meaningful authority status under the existing compact contract, then completes its import, focus and retained-receipt assertions.
- [ ] #2 The 120x36 journey continues to verify its complete authority copy and import behavior.
- [ ] #3 The repair retains assertions that distinguish Library-owned notes from folder files and does not restore redundant product copy simply to satisfy an obsolete test.
<!-- AC:END -->
