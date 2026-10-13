---
id: TASK-34924
title: Restoring a backup can undo a Dreams switch-off
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- backup-recovery
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Switching Dreams off stops the sources, alert rules and reminders Dreams created. Those live in the subscriptions and scheduling stores, which a backup restores, while what Dreams recorded about them does not change. After restoring a backup taken while Dreams was on into a profile where Dreams is now off, the restored sources run again: the next start reads none of them, the dispatch gate allows them, and their checks and alerts run with Dreams off. Since the URL-provenance rework they are fetched under the public-only rule, so this is no longer a private-address exposure, but it is work the user switched off. The user guide does not mention it. TASK-34816's provenance criteria now look met by that rework; this is the part it does not cover. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a restore with Dreams off, no source, rule or reminder Dreams created runs until Dreams is switched on
- [ ] #2 After a restore with Dreams on, every live watch's source runs
- [ ] #3 A test restores each case with real stores and fails before the fix
- [ ] #4 The user guide says what a restore does to Dreams' watches
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
