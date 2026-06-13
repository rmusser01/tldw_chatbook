---
id: TASK-44
title: Reject duplicate Sync v2 push response envelope ids
status: Done
assignee: []
created_date: '2026-05-10 20:26'
updated_date: '2026-05-10 20:28'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook local-first Sync v2 so a push response cannot report the same client_envelope_id in multiple result sections or duplicate it within a section.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once raises when a push response repeats the same client_envelope_id across accepted, rejected, or conflicts.
- [x] #2 The duplicate-id response is rejected before pull, outbox dispatch, outbox attempt updates, or cursor advancement.
- [x] #3 The profile records a push failure that identifies the duplicate response id.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test for a push response that repeats one submitted client_envelope_id across accepted and rejected. 2. Extend push-response validation to reject duplicate client_envelope_id values before outbox marking or pull. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added push response duplicate-id validation so a client_envelope_id can appear only once across accepted, rejected, and conflicts. Duplicate response ids now fail before pull, outbox dispatch, attempt updates, or cursor advancement, with a profile-level push failure. Verified focused/local-first/full Sync_Interop plus Bandit and diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
