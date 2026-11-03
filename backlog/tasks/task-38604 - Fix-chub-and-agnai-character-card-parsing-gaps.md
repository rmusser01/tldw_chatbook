---
id: TASK-38604
title: Fix chub and agnai character card parsing gaps
status: In Progress
created_date: 2026-11-02 14:29
labels:
- bug
- characters
priority: low
updated_date: 2026-11-02 14:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chub format detector is unreachable because the v2 pattern matches first so chub fields are dropped, and agnai persona objects are coerced to empty strings. Metadata-only impact. See bug-sweep-2026-11-02-dev.md module summary for Character_Chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chub cards are detected before the v2 pattern or chub fields are folded into v2 parsing
- [ ] #2 Agnai persona objects are stringified instead of dropped
- [ ] #3 Tests cover both formats
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
