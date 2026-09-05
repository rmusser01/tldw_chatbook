---
id: TASK-31696
title: Restore semantic Notes browse focus when returning from Folder Files
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 18:37'
updated_date: '2026-09-05 18:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the Files-to-Database return path that reads a removed receipt focus field, retaining semantic placement focus, scroll restoration and independent editor authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Files return paths resolve the current semantic Notes receipt without missing-field exceptions
- [ ] #2 Semantic focus role, selected note and scroll restore consistently across retained and rebuilt Notes paths
- [ ] #3 Existing return and editor authority assertions plus targeted state and static checks pass without screen budget increases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce existing Files return and editor-authority failures; trace obsolete receipt.focus against current semantic receipt fields.
2. Characterize exact focus identity projection for note/folder/filter roles and scroll offsets in pure state tests.
3. Move the two existing identical screen conversions into a receipt.focus_identity property and use it at the broken Files return, retaining callback guards and lifecycle behavior.
4. Run pure tree state and original return/focus matrix plus relevant Notes focus tests, scoped static checks, screen ratchet and parent review.
ADR required: no
ADR path: N/A
Reason: Existing pure semantic conversion ownership is deduplicated to repair a removed-field read; no lifecycle, focus policy, persisted receipt schema or runtime boundary changes.
<!-- SECTION:PLAN:END -->
