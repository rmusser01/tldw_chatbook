---
id: TASK-16303
title: Repair current-dev ingest-order and dictation-fixture tests
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 05:41'
updated_date: '2026-08-14 05:44'
labels:
  - test-health
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic test evidence for two current-dev failures: folder ingestion must prove every discovered member is submitted exactly once without assuming filesystem enumeration order, and dictation capture teardown tests must construct the current service contract instead of an obsolete partial object.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Folder ingestion coverage proves every member is submitted exactly once while remaining order-agnostic.
- [x] #2 Dictation capture release coverage uses a fully initialized lazy service and still proves no recorder is constructed during teardown.
- [x] #3 The two focused files and adjacent suites pass with lint, diff, mutation, and no-new-format-drift evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: test-only corrections preserve existing ingestion and dictation runtime contracts.

1. Reproduce both current-dev failures in isolation and identify their current production contracts.
2. Make the smallest test-only corrections: compare folder members without ordering assumptions and build the lazy dictation service through its constructor.
3. Prove each old test shape fails, run the focused and adjacent suites, then run scoped static and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired two current-dev test contracts without changing production behavior. Folder-ingest evidence now compares the submitted member multiset independently of filesystem iteration order, and the dictation teardown fixture constructs the lazy service through its real initializer so the current processing queue and defaults exist while recorder creation remains lazy. Also removed one pre-existing unused local in the touched audio test. RED evidence: restoring the order-sensitive assertion failed 1/1; deleting the initialized processing queue failed both teardown nodes. GREEN: 131 focused/adjacent tests passed; scoped Ruff check and diff-check passed. Both touched files were already Ruff-format-red on HEAD, and the small edited hunks add no formatting drift. ADR required: no; test-only contract repair.
<!-- SECTION:NOTES:END -->
