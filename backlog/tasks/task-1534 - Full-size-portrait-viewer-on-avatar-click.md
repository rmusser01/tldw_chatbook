---
id: TASK-1534
title: 'Full-size portrait viewer on avatar click'
status: Done
assignee: []
created_date: '2026-07-30 17:20'
labels: [enhancement, roleplay, console, ux]
dependencies: [TASK-1533]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking the character avatar (Console rail "Character" box or the Roleplay
Inspector portrait) opens `ConsoleImageViewerModal`: the image as large as
the viewport allows -- true raster via textual_image on graphics terminals,
a viewport-sized quadrant mosaic elsewhere (~10x the thumbnail's pixel
budget). Esc or any click closes. `ClickableAvatarBox` posts
`AvatarViewRequested` with no payload; each screen resolves the CURRENT
portrait at click time (selection can change between mount and click).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking either avatar surface opens the viewer with the portrait rendered viewport-sized (contain fit -- the whole image visible).
- [x] #2 Escape and click both dismiss.
- [x] #3 Covered by widget tests (click posts request; modal renders and dismisses); wiring live-verified with screenshots.
<!-- AC:END -->
