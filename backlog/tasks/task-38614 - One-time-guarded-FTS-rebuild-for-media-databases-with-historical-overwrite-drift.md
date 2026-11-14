---
id: TASK-38614
title: One-time guarded FTS rebuild for media databases with historical overwrite
  drift
status: To Do
created_date: 2026-11-03 02:53
labels:
- bug
- db
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-11-02 audit and TASK-38588 confirmed INSERT OR REPLACE on the external-content FTS5 table leaves stale tokens and fails integrity-check from the first historical overwrite. Existing user DBs may carry this drift. A one-time guarded rebuild migration should restore index integrity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A guarded one-time FTS rebuild runs for media DBs with historical overwrites
- [ ] #2 Integrity-check passes after the rebuild
- [ ] #3 Rebuild is idempotent and skipped for healthy DBs
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
