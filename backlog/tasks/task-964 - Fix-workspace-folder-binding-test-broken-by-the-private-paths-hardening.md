---
id: TASK-964
title: Fix workspace folder-binding test broken by the private-paths hardening
status: To Do
assignee: []
created_date: '2026-07-27 18:06'
labels:
  - workspaces
  - tests
  - dev-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Workspaces/test_workspace_folder_bindings.py::test_add_folder_binding_rejects_duplicates_and_nesting fails on pristine origin/dev with PrivatePathError: unsafe_parent: missing_directory. Dev's recent private-paths hardening now resolves binding roots strictly, and this test binds a directory it never creates. Whether the fix belongs in the test (create the directory, as real callers do) or in the binding path (tolerate a not-yet-created root) is the decision to make -- production callers may legitimately bind a folder before it exists. Confirmed pre-existing and unrelated to TASK-857 despite touching the same code path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on a clean checkout,It is decided and documented whether binding a not-yet-existing root is legitimate,If it is legitimate the binding path handles it rather than raising
<!-- AC:END -->
