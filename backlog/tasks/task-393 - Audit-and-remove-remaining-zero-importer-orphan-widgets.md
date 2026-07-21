---
id: TASK-393
title: Audit and remove remaining zero-importer orphan widgets
status: To Do
assignee: []
created_date: '2026-07-19 04:23'
labels:
  - cleanup
  - ui
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-253 SearchWindow deletion (PR 669) verified importers for everything it removed and explicitly left a set of pre-existing zero-importer orphans for a separate audit because they were not reachable through the SearchWindow stack: at minimum Widgets/toast.py-style notification widgets, a detailed-progress widget, and an embedding-template-selector (exact filenames in PR 669's report and worth re-verifying). Sweep Widgets/ (and UI/ leaf modules) for zero-importer files, verify each with git grep across production code and tests, and delete confirmed orphans following the repo precedent (commit 628b1b8b, tasks 252/253). Note the codebase moves fast: re-verify every candidate on the current tree at implementation time rather than trusting this list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Widgets/ and UI/ module with zero production importers is either deleted or recorded with a documented reason to keep
- [ ] #2 Importer evidence (git grep) is captured per removed file in the task or PR description
- [ ] #3 Exclusive tests and CSS for removed widgets are cleaned up and the CSS bundle rebuilt if inputs changed
- [ ] #4 Full test suite shows zero regressions vs the pre-change baseline
<!-- AC:END -->
