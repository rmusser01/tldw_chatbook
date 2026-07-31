---
id: TASK-1535
title: 'Cover-fit avatar thumbnails (fill the box, no distortion)'
status: Done
assignee: []
created_date: '2026-07-30 17:20'
labels: [enhancement, roleplay, console, rendering]
dependencies: [TASK-1533]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`mosaic_from_image` gains `fit="cover"` (object-fit: cover): scale to FILL
the cell box and center-crop the overflow, aspect always preserved. The
avatar thumbnails (Roleplay Inspector/editor, Console rail) use cover per
user choice; the full-size viewer and any future inline uses keep the
default contain (whole image visible).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fit="cover" paints every cell of the box for any source aspect, center-cropped, never stretched.
- [x] #2 Default stays contain; avatar thumbs pass cover explicitly.
- [x] #3 Geometry pinned by deterministic tests (asserted on the renderable's plain text -- console text export miscounts styled flat cells).
<!-- AC:END -->
