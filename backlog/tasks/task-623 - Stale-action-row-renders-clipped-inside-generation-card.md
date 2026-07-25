---
id: TASK-623
title: >-
  Stale action row renders clipped inside the generation card and survives Escape
status: In Progress
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 10:45'
labels:
  - image-generation
  - console
  - ui
  - uat
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25: after a no-prompt generation completed while its message was (or became) selected, the selected-message action row rendered INSIDE the generation card's box — sandwiched between the image area and the Style/Source/Seed detail rows — clipped at the card's inner width (labels truncated at "Vie…", `keep`/`Save Image` unreachable). Escape (clear selection) did NOT remove it; the orphaned row persisted until the next full transcript recompose (tab switch away/back cleared it). A fresh keyboard selection simultaneously rendered a second, correctly-placed full action row below the message, so the in-card row is a stale duplicate — likely the row (or its mount anchor) from a render pass that raced the generation message's completion re-render. Clicks on the stale row did nothing (or hit the image widget), which is how it was noticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The selected-message action row always mounts below the message content (never inside the generation card), including when the generation completes or re-renders while selected.
- [ ] Clearing the selection (Escape) removes the action row in all cases; no orphaned row survives.
- [ ] A regression test covers select-during-generation-completion (or the closest reproducible ordering) asserting a single, correctly-parented action row.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
