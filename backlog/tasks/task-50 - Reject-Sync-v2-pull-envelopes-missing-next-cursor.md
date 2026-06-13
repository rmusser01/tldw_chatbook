---
id: TASK-50
title: Reject Sync v2 pull envelopes missing next cursor
status: Done
assignee: []
created_date: '2026-05-10 21:20'
updated_date: '2026-05-10 21:22'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first and restore pull handling so any response that includes envelopes must include a next_cursor before local apply or cursor persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync raises when a pull response returns envelopes without next_cursor before local apply.
- [x] #2 Restore raises when a pull response returns envelopes without next_cursor before local apply.
- [x] #3 The local-first failure records an apply error without advancing the remote pull cursor.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red local-first and restore tests for non-empty pull pages without next_cursor. 2. Extend shared pull pagination validation to require next_cursor for non-empty pages. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed local-first and restore did not raise for non-empty pull responses missing next_cursor before validation. Green/final verification: focused non-empty pull cursor tests 2 passed; affected local_first+restore files 40 passed; full Tests/Sync_Interop 105 passed; Bandit on local_first_sync_service.py, restore_service.py, and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended shared Sync v2 pull pagination validation so non-empty pull pages require next_cursor even when has_more is false. Local-first and restore now reject data pages that cannot be checkpointed before applying envelopes.
<!-- SECTION:FINAL_SUMMARY:END -->
