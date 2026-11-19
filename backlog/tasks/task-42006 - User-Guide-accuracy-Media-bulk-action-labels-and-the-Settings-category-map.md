---
id: TASK-42006
title: "User Guide accuracy: Media bulk action labels and the Settings category map"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/library/media-and-conversations.md
- Docs/User_Guide/settings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41526 (round 17). `media-and-conversations.md` calls Media's bulk actions "Export selected", "Delete selected" and "Review selected" in several places, but the canvas labels are "Export", "Review", "Analyze" and "Delete" (`library_media_canvas.py` ~898-939). The `settings.md` category map has no rows for Tools & MCP, Network or Hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Media page uses the canvas labels throughout
- [ ] #2 The category map lists every Settings category, checked against the code
<!-- AC:END -->
