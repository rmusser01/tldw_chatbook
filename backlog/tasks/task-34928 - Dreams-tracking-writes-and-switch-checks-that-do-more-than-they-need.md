---
id: TASK-34928
title: Dreams tracking writes and switch checks that do more than they need
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- performance
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small items. Tracking a page or a question does not check the Dreams master switch: the Library hides the action when Dreams is off, but a call that reaches the service anyway re-enables a source and arms a reminder while Dreams is off. And the verifying backfill rewrites the provenance mark of every source Dreams ever created on every run, one write transaction each, even when nothing changed. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tracking while Dreams is off creates or changes nothing and says why
- [ ] #2 A verifying backfill that finds nothing to mark makes no writes
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
