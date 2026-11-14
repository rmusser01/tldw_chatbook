---
id: TASK-38615
title: Pass old values to FTS purge in meeting speaker rename
status: To Do
created_date: 2026-11-03 02:53
labels:
- bug
- library
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library/meeting_speaker_rename.py:390 still calls the media FTS update helper without old values so speaker renames leave stale FTS tokens - the same drift class TASK-38588 fixed at the DB layer. The old row content is available at the call site; a two-argument addition completes the fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rename path passes true old title or content to the FTS purge
- [ ] #2 Words only in the old speaker name stop matching after rename
- [ ] #3 Regression test covers the rename path
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
