---
id: TASK-38608
title: Enforce artifact-share auth for non-loopback binds in the controller
status: In Progress
created_date: 2026-11-02 14:29
labels:
- security
- web
priority: low
updated_date: 2026-11-02 14:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The controller serves user artifacts to the LAN without auth when started with a non-loopback bind and no credentials; the gate currently lives only in the UI dialog. See bug-sweep-2026-11-02-dev.md services long tail findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 start_share raises for non-loopback bind without credentials at the controller layer
- [ ] #2 Existing dialog behavior is unchanged
- [ ] #3 Test covers the controller gate
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
