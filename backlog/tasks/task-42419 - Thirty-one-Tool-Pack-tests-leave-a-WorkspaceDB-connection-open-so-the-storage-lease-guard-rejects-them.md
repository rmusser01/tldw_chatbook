---
id: TASK-42419
title: "Thirty-one Tool Pack tests leave a WorkspaceDB connection open, so the storage-lease guard rejects them"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
labels:
- follow-up
- tests
- tool-packs
- ci
priority: low
references:
- Tests/Tool_Packs/test_binding_guard.py
- Tests/Tool_Packs/test_activation.py
- Tests/conftest.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports, including the one that lanes `Tests/Tool_Packs`); paths start at the repository root.

**Why.** A storage lease left open by one test makes a later backup-pause test in the same process time out, far from the leak (`_StorageLeaseGuard`, `Tests/conftest.py:1236-1310`). Because of this leak, round 21 laned `Tests/Tool_Packs` in the dreams-tracking job, which has no `--storage-lease-guard`, instead of the guarded dreams-boundary job (the note at `derived-artifacts.yml:1459-1460`).

**What leaks.** `Tests/Tool_Packs` with `--storage-lease-guard` at `-n 4`: 439 passed, 31 errors at teardown, each "left 1 new live storage lease(s) open: SQLite .../workspaces.sqlite on thread MainThread" (`test_binding_holds_lifecycle_serialization_through_sqlite_commit` left 2, the second on a worker thread).
- 30 are in `Tests/Tool_Packs/test_binding_guard.py`. Its `_build` helper (line 177) constructs `WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="c1")` inside a `LocalWorkspaceRegistryService` (line 186) and never closes it.
- 1 is `Tests/Tool_Packs/test_activation.py::test_import_install_is_unbound_and_preserves_existing_workspace_policy`, which constructs a `WorkspaceDB` at line 988 the same way.

Other Workspaces tests close theirs with `registry.db.close()` (for example `Tests/Workspaces/test_agent_provisioning.py:70`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every WorkspaceDB the Tool Pack tests open is closed when its test ends, including when the test fails
- [ ] #2 `Tests/Tool_Packs` passes with `--storage-lease-guard` (serial and at `-n 4`) with no teardown errors
- [ ] #3 The lane note in `derived-artifacts.yml` that says Tool_Packs cannot run under the lease guard is updated to match
<!-- AC:END -->
