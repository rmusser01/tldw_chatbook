---
id: TASK-38594
title: Implement or remove chatbook REPLACE and MERGE conflict strategies
status: Done
created_date: 2026-11-02 14:28
labels:
- bug
- chatbooks
priority: medium
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The import wizard offers Replace and Merge radio buttons but the importer only implements SKIP and RENAME, so Replace creates duplicate-titled notes and fails on characters while merge_notes is dead code. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Replace updates in place instead of creating duplicates or failing
- [x] #2 Merge is implemented or removed from the wizard UI
- [x] #3 Dead merge_notes code removed or wired
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
REPLACE is update-in-place via versioned DB APIs (conversations retire-then-recreate, one ACTIVE row); MERGE removed from the wizard and rejected before any write; dead merge_notes deleted. Commit a76d67e6be.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
