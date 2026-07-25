---
id: TASK-522
title: Isolate application database paths across pytest collection and app tests
status: Done
assignee: []
created_date: '2026-07-24 18:48'
updated_date: '2026-07-24 19:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure tests that import the application during collection and construct multiple TldwCli instances never resolve local database paths under the user's production data directory or reuse a lazy database from a prior test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Application database fallback paths resolve inside the per-test data directory even when config was imported during test collection
- [x] #2 Lazy config database instances do not carry over between tests
- [x] #3 The app initialization smoke test passes after prior config imports
- [x] #4 The code-repo integration test module reaches and completes its UI workflows without production-path database access
- [x] #5 Tests run with Tests/UI as the pytest root use the same application data isolation
- [x] #6 The shared full-app pilot closes app-owned database handles during cleanup
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regressions for collection-time config paths, cached DB singleton cleanup, and nested Tests/UI pytest-root isolation.
2. Update the shared test isolation fixture to patch the import-time fallback root, close config DB singletons, and shut down the prompt interop singleton before and after each test.
3. Re-export the canonical isolation fixture from Tests/UI/conftest.py and close app-owned DB handles in the shared full-app pilot.
4. Run the isolation regressions, app smoke, full code-repo integration module, Ruff, and diff checks.
5. Resolve independent review findings, document the no-ADR decision and implementation notes, then mark verified criteria complete.

ADR required: no
ADR path: N/A
Reason: This repairs test-only isolation without changing production storage, runtime contracts, or schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented collection-safe test isolation by redirecting config.BASE_DATA_DIR_CLI into each pytest sandbox, clearing lazy config databases, and shutting down initialized prompt interop state before and after every test. Re-exported the canonical fixture for the nested Tests/UI pytest root and made the shared full-app pilot close all app-owned database handles, aggregate cleanup failures, and preserve later close attempts. Added root and nested-root regressions, including a failing-handle cleanup assertion.

Verification: 10 passed for collection isolation + app initialization + the full code-repo integration module; 1 passed under Tests/UI as the direct pytest root; 11 passed for isolation + dictionary send + code-repo integration; Ruff and git diff checks passed; five modified/new harness files pass Ruff formatting. Tests/RAG/test_config_profiles.py received a comment-only correction and retains unrelated pre-existing whole-file formatting debt. Independent review approved with no remaining findings.

ADR required: no. This changes only pytest isolation and cleanup behavior; production storage, schemas, ownership, and runtime contracts are unchanged.

Modified: Tests/conftest.py, Tests/UI/conftest.py, Tests/textual_test_utils.py, Tests/test_environment_isolation.py, Tests/UI/test_environment_isolation.py, and a comment in Tests/RAG/test_config_profiles.py.
<!-- SECTION:NOTES:END -->
