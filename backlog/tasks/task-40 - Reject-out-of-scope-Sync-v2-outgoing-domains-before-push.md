---
id: TASK-40
title: Reject out-of-scope Sync v2 outgoing domains before push
status: Done
assignee: []
created_date: '2026-05-10 20:07'
updated_date: '2026-05-10 20:09'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook local-first Sync v2 so ad hoc outgoing envelopes passed to sync_once cannot be pushed when their domain is outside the caller's requested domain set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once rejects outgoing_envelopes whose domain is not included in the domains argument before contacting the server.
- [x] #2 The failure is recorded on the Sync v2 profile without advancing the remote pull cursor.
- [x] #3 Existing valid local-first push/pull behavior remains covered by tests.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test for outgoing envelopes outside requested domains. 2. Add local pre-push scope validation for ad hoc outgoing envelopes. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added local-first pre-push validation for ad hoc outgoing envelopes so sync_once rejects envelopes outside the requested domain set before contacting the server. The failure is recorded as a push-stage profile error and the cursor remains unchanged. Verified the focused regression, local-first suite, full Sync_Interop, Bandit, and git diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
