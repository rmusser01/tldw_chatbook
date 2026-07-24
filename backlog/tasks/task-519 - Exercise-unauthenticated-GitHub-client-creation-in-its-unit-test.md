---
id: TASK-519
title: Exercise unauthenticated GitHub client creation in its unit test
status: Done
assignee: []
created_date: '2026-07-24 18:38'
updated_date: '2026-07-24 18:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the GitHub API client's no-token unit contract by actually accessing the lazy HTTP client under isolated environment/config inputs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The no-token test triggers lazy AsyncClient construction
- [x] #2 The test isolates environment and config token sources
- [x] #3 The constructed headers omit Authorization while retaining existing public-request headers
- [x] #4 No production GitHub client behavior changes
- [x] #5 The focused and full GitHub API client tests pass
- [x] #6 Task documentation records the merge-base failure, ADR decision, verification, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact lazy-construction test failure on feature branch and merge base.
2. Isolate token configuration/environment inputs and access the client property before inspecting the mocked constructor.
3. Assert the unauthenticated header contract and run the full GitHub API client test file.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This corrects a unit test that never exercised an existing lazy property; it changes no HTTP client, credential, dependency, or runtime architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Corrected the unauthenticated-client unit test so it exercises the existing lazy `client` property under deterministic no-token inputs before inspecting the mocked `httpx.AsyncClient` constructor.

Approach:
- Patched the module's `get_cli_setting` dependency to return each provided default and `os.getenv` to return `None`, preventing ambient config or environment credentials from affecting the test.
- Constructed `GitHubAPIClient` inside the patches, accessed `.client`, and asserted that `httpx.AsyncClient` was called once.
- Verified the constructor retained the public GitHub `Accept` and application `User-Agent` headers while omitting `Authorization`.
- Changed no production client or credential-resolution code.

RED evidence:
- Feature branch before the fix: the exact test failed because `mock_client_class.call_args` was `None`; constructing `GitHubAPIClient` alone does not create its lazy HTTP client.
- Merge base `ba6b45cdf4dd548796e072f5933cdcf44c8c0344`: the exact test failed with the same `NoneType` `call_args.kwargs` error.

Verification:
- Exact no-token regression: 1 passed.
- Full `Tests/Utils/test_github_api_client.py`: 32 passed.
- Ruff format check: file already formatted.
- Ruff check: all checks passed.
- `git diff --check` passed for the owned files.
- Scope review confirmed only the owned test and TASK-519 were changed.

ADR required: no
ADR path: N/A
Reason: This test-only correction exercises existing lazy construction and changes no HTTP client, credential, dependency, or runtime architecture.

Files modified:
- `Tests/Utils/test_github_api_client.py`
- `backlog/tasks/task-519 - Exercise-unauthenticated-GitHub-client-creation-in-its-unit-test.md`
<!-- SECTION:NOTES:END -->
