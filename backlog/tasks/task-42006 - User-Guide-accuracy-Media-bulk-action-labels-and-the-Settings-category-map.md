---
id: TASK-42006
title: "The Media page names its bulk actions by labels the canvas does not show"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/library/media-and-conversations.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41526 (round 17). `media-and-conversations.md` calls Media's bulk actions "Export selected", "Delete selected" and "Review selected" in several places, but the canvas labels are "Export", "Review", "Analyze" and "Delete" (`library_media_canvas.py` ~898-939). (The `settings.md` category map's missing rows are TASK-35016's AC#12.) Conversations' own bulk action really is labelled "Export selected" (`library_conversations_canvas.py` ~246), so only Media's wording is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every mention of Media's bulk actions uses the canvas labels (Export, Review, Analyze, Delete), checked against the code; Conversations' "Export selected" is left as it is
<!-- AC:END -->
