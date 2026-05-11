---
id: TASK-38
title: Reject mismatched Sync v2 pulled datasets before apply
status: Done
assignee: []
created_date: '2026-05-10 19:49'
updated_date: '2026-05-10 19:51'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first and restore apply paths so pulled server batches cannot apply envelopes whose dataset identity differs from the requested dataset.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync rejects pulled envelopes whose dataset_id differs from the profile dataset_id before mutating local state.
- [x] #2 Local-first sync records the failure without advancing the remote pull cursor.
- [x] #3 Restore rejects pulled batches or envelopes whose dataset_id differs from the requested dataset_id before mutating local state.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for local-first and restore dataset mismatch rejection before apply. 2. Add pull dataset identity validation in the apply paths. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added pulled-dataset identity validation shared by local-first sync and restore apply paths. Local-first now records an apply failure and keeps the cursor unchanged on wrong-dataset pulled envelopes; restore now rejects wrong-dataset pull responses before local mutation. Verified focused tests, full Sync_Interop, Bandit, and git diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
