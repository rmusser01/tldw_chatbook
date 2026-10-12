---
id: TASK-34802
title: A story captured through Dreams Ingest is re-fetched as a user-entered address
  on Retry
status: To Do
created_date: 2026-10-12 01:45
labels:
- dreams
- security
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams refuses to ingest a story whose address is not a public destination. A capture that was accepted and then failed can be retried later from the Library, and that re-fetch treats the stored address as one the user typed, so its own host counts as trusted. A story address that was public when ingested and resolves to a private address by the time of the retry is then fetched. Closing it means recording where a capture's address came from on the capture item, which is a schema step on the Library collections store, a store covered by backup and recovery. Found in the review of PR 2921 and left out of that change on purpose.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A capture whose address came from a Dreams story is fetched with no trusted origin on every attempt, including a retry and every redirect hop
- [ ] #2 A capture whose address the user typed keeps today's behaviour, including private addresses the user chose
- [ ] #3 An existing collections store upgrades in place and its recovery catalog entries are updated in the same change
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
