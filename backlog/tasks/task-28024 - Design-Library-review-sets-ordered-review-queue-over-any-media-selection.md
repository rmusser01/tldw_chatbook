---
id: TASK-28024
title: 'Design - Library review sets: ordered review queue over any media selection'
status: To Do
assignee: []
created_date: '2026-09-02 04:23'
labels:
  - library
  - media-ux
  - design
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-approved direction (2026-09-01 media UX critique follow-up): reviewing a set of media items - a conference, a tag/keyword-filtered browse result, or hand-picked items - should be a first-class object, not a memory exercise. A review set is an ordered list of media item ids pinned at creation time, plus a cursor and per-item done marks, with an explicit completion state. Entry points: Review-these on the current browse result, on a search result, and Review-selected as a third Select-mode bulk action. While active, the viewer walks the set with Next/Prev, shows progress (12 of 40), resumes at the cursor across app restarts, and ends with an explicit all-reviewed state. DB/Library_Collections_DB.py is the likely persistence home (a set is nearly a collection plus cursor plus done marks). This is the DESIGN task: produce the spec (data model, lifecycle, UI surfaces, edge cases: item deleted mid-set, re-ingest dedup, multiple concurrent sets) and get explicit user approval before any implementation tasks are filed. Foundations that ship independently: task-28005 (viewer Prev/Next over current browse result) and task-28009 (read markers).

Foundations confirmed on dev tip 2026-09-02, which the review-set design should compose rather than rebuild: Reader mode persists across item selections (begin_selection preserves mode); moving the list selection auto-loads the Reader; per-item reading position is persisted (library_media_reading_progress). A set = pinned ordered ids + cursor + done marks on top of these.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written design covers data model, set lifecycle, entry points, viewer behavior, and the edge cases above
- [ ] #2 The design is explicitly approved by the user before implementation tasks are filed
- [ ] #3 The design states how it builds on or supersedes task-28005 and task-28009
<!-- AC:END -->
