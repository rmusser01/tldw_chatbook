---
id: TASK-56
title: Address PR 302 Sync v2 review comments
status: Done
assignee: []
created_date: '2026-05-11 00:18'
updated_date: '2026-05-11 00:31'
labels:
  - sync
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/302'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #302 for the Chatbook Sync v2 client substrate without expanding the feature scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Crypto helpers narrow broad exception handling while preserving expected invalid-payload failures.
- [x] #2 Recovery bundle unwrapping validates bounded scrypt metadata from untrusted records.
- [x] #3 Local-first sync respects server max_batch_size by pushing outgoing envelopes in bounded batches.
- [x] #4 Sync v2 repository schema changes have an explicit version/migration path and dynamic SQL identifiers are validated.
- [x] #5 Server Sync v2 push validates outgoing domains against an explicit allow-list.
- [x] #6 New public Sync v2 entrypoints have Google-style docstrings where required.
- [x] #7 Relevant focused tests, full Tests/Sync_Interop, Bandit on touched production scope, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the reviewed code paths and existing tests for crypto, local-first sync batching, repository schema migration, server push validation, and public API docstrings. 2. Add red tests for the behavior-changing review items. 3. Implement narrow fixes with scoped docstring updates. 4. Run focused tests, full Sync_Interop, Bandit on touched production files, diff checks, push, and update PR review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #302 review threads: narrowed crypto exception handling, rejected unbounded recovery scrypt metadata before KDF work, chunked local-first pushes by max_batch_size, added SyncState schema version 2 migration and validated dynamic ALTER TABLE column identifiers, threaded push domain allow-lists, and added Google-style docstrings for public Sync v2 entrypoints.

Verification: targeted red tests failed before fixes and passed after; Tests/Sync_Interop plus Tests/tldw_api/test_sync_client.py passed 128 tests; Bandit on tldw_chatbook/Sync_Interop and tldw_chatbook/tldw_api/client.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved all actionable PR #302 review feedback across crypto hardening, Sync v2 batching/scoping, repository migration safety, and public API documentation. Preserved existing local-first error semantics while adding regression coverage for the reviewed issues.
<!-- SECTION:FINAL_SUMMARY:END -->
