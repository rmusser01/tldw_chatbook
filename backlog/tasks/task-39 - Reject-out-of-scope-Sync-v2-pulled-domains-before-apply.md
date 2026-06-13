---
id: TASK-39
title: Reject out-of-scope Sync v2 pulled domains before apply
status: Done
assignee: []
created_date: '2026-05-10 19:53'
updated_date: '2026-05-10 19:57'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first and restore apply paths so pulled server envelopes cannot mutate domains outside the caller's requested domain set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync rejects pulled envelopes whose domain is not included in the sync_once domains argument before mutating local state.
- [x] #2 Local-first sync records the failure without advancing the remote pull cursor.
- [x] #3 Restore rejects pulled envelopes whose domain is not included in the restore_selection domains argument before mutating local state.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for local-first and restore rejecting pulled envelopes outside requested domains. 2. Extend shared pull-boundary validation to check domain scope. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended pulled response scope validation so local-first sync and restore reject envelopes outside the requested domain set before applying local mutations. Added regressions for chat envelopes returned during notes-only pulls, verified cursor preservation for local-first, and reran focused tests, full Sync_Interop, Bandit, and diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
