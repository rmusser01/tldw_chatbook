---
id: TASK-49
title: Reject Sync v2 pull pages missing next cursor
status: Done
assignee: []
created_date: '2026-05-10 21:16'
updated_date: '2026-05-10 21:18'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first and restore pull handling so a response with has_more=true must include a next_cursor before any local apply or cursor persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync raises when a pull response sets has_more=true without next_cursor before local apply.
- [x] #2 Restore raises when a pull response sets has_more=true without next_cursor before local apply.
- [x] #3 The local-first failure records an apply error without advancing the remote pull cursor.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red local-first and restore tests for has_more=true without next_cursor. 2. Add shared pull pagination validation before local apply. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed local-first and restore did not raise for has_more=true without next_cursor before validation. Green/final verification: focused pagination tests 2 passed; affected local_first+restore files 38 passed; full Tests/Sync_Interop 103 passed; Bandit on local_first_sync_service.py, restore_service.py, and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added shared Sync v2 pull pagination validation so has_more=true requires next_cursor before envelopes are applied. Local-first and restore now reject unpageable pull responses without applying local state; local-first also preserves the existing pull cursor.
<!-- SECTION:FINAL_SUMMARY:END -->
