---
id: TASK-38606
title: Close DB connections on vacuum and integrity check errors
status: In Progress
created_date: 2026-11-02 14:29
labels:
- bug
- db
priority: low
updated_date: 2026-11-02 14:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
vacuum and check_integrity open short-lived connections and only close them on the success path so failed attempts on locked or corrupt files leak handles. See bug-sweep-2026-11-02-dev.md DB module findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 vacuum and check_integrity close their connection on error paths
- [ ] #2 A test or review confirms no handle leak on failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
