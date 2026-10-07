---
id: TASK-34422
title: Buddy modal fingerprint-gated polling
status: In Progress
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 09:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 / F9: the buddy conversation modal polls at 5 Hz rebuilding an O(session) snapshot and a 64KB transcript before checking for change even when idle
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Idle modal performs zero messages_for_session calls,Streaming still renders within one poll interval,Duplicate show_decisions call removed,Decisions coordinator invoked once per tick
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 10 (T10)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
