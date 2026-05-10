---
id: TASK-45
title: Reject own-device Sync v2 incremental pull envelopes
status: Done
assignee: []
created_date: '2026-05-10 20:31'
updated_date: '2026-05-10 20:33'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook local-first Sync v2 so incremental pulls that request include_own_changes=false cannot apply envelopes authored by the active profile device.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once raises when pulled envelopes have device_id equal to the active profile device_id.
- [x] #2 The own-device pull response is rejected before local apply or cursor advancement.
- [x] #3 The profile records an apply failure identifying the malformed pull response.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test for pulled envelopes from the active device despite include_own_changes=false. 2. Extend pulled response validation with an optional disallowed device id and wire local-first to pass the profile device. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added local-first pulled response validation to reject envelopes from the active profile device when sync_once pulls with include_own_changes=false. Same-device incremental pull responses now fail before local apply or cursor advancement and record a profile-level apply failure. Verified focused/local-first/full Sync_Interop plus Bandit and diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
