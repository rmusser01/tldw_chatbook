---
id: TASK-964
title: Fix workspace folder-binding test broken by the private-paths hardening
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:06'
updated_date: '2026-07-27 18:42'
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
- [x] #1 The test passes on a clean checkout,It is decided and documented whether binding a not-yet-existing root is legitimate,If it is legitimate the binding path handles it rather than raising
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce test_add_folder_binding_rejects_duplicates_and_nesting on the worktree and get the full traceback (not just the exception type) to find exactly which call raises PrivatePathError.
2. Determine whether the raise originates in add_folder_binding's own root validation or elsewhere.
3. Decide whether binding a not-yet-existing root is legitimate production behavior, using add_folder_binding's existing is_dir() gate as evidence.
4. Fix at the correct layer (test vs production) based on that decision.
5. Re-run Tests/Workspaces/test_workspace_folder_bindings.py and the broader Tests/Workspaces/ suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The full traceback (not just the exception type quoted in the task) shows the PrivatePathError does NOT come from add_folder_binding's own root-validation logic at all. add_folder_binding already requires the candidate folder to exist (resolved.is_dir() check, raising a domain-specific WorkspaceRegistryServiceError otherwise) and always has -- so binding a not-yet-existing root is already, deliberately, NOT legitimate in production, and that gate needed no change.

The actual raise happens in the test's own build_registry() helper: 'other = build_registry(tmp_path / "second-db")' constructs a brand-new WorkspaceDB whose OWN sqlite file lives under a directory (second-db) the test never created. BaseDB.__init__ used to auto-mkdir its parent (see inventory row P06, now 'migrated' -- disposition secure_default) but the private-paths hardening removed that, so opening a database whose containing directory doesn't yet exist now raises PrivatePathError instead of silently creating it. Real production callers never hit this because WorkspaceDB is always constructed under get_user_data_dir(), which is created as a side effect before its path is ever used.

Fix landed in the test only: build_registry() now creates tmp_path (mkdir(parents=True, exist_ok=True)) before constructing WorkspaceDB, mirroring what a real caller's directory is guaranteed to have. No production change -- the hardening is correct and add_folder_binding's existing non-existent-folder rejection is the right, unchanged behavior.

Before: 1 failed / 10 passed (test_workspace_folder_bindings.py). After: 11 passed. Tests/Workspaces/ overall: 138 passed, no regressions.
<!-- SECTION:NOTES:END -->
