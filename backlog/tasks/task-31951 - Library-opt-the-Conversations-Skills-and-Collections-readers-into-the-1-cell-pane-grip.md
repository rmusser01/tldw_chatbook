---
id: TASK-31951
title: >-
  Library - opt the Conversations, Skills and Collections readers into the
  1-cell pane grip
status: Done
assignee:
  - '@claude'
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 17:41'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H Task 2 ruling: PR H made LibraryAdaptiveReaderPaneGrip's width per reader profile so Media could drop from five columns to one, which is the defect task-31633 fixed for Media. Conversations, Skills and Collections still carry the five-column grip - ten dead columns per surface - because H changed only the profile Media uses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The three sibling readers render a 1-cell grip with the same glyphs as Media
- [x] #2 Their resolver width pins are updated to the new reservation
- [x] #3 No reader surface reserves grip columns it does not paint
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-pin the three sibling profiles' resolver tuples and painted grips to the one-cell grip first (old values in the comments), then set grip_width=1 on the profiles. 2. Carry grip_width on the effective layout so the shell paints from the resolved value (delete the shell parameter). 3. Pin that every profile reserves exactly the grip columns it paints; pin the moved rail-open thresholds and the Skills items-priority floor at both band edges. 4. Live capture per sibling; guide update.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Profiles: Conversations/Skills/Collections grip_width=1 (screen_constants.py). The width rides AdaptiveReaderEffectiveLayout.grip_width; LibraryAdaptiveReaderShell reads it (its own grip_width parameter deleted, zero call-site edits); LibraryAdaptiveReaderPaneGrip.sync_width() is applied at construction and in sync_layout. Consequences (pinned both edges, "was" in comments): rail-open 118→110 (Conversations), 122→114 (Skills, Collections); Skills items-priority floor 90→82. Live: one-cell ‹/› on all three at 235x52. Riders: sync_layout re-apply now closes the stale-grip window; File Notes' placeholder layout hard-codes the default until its first resolve (circular import).
<!-- SECTION:NOTES:END -->
