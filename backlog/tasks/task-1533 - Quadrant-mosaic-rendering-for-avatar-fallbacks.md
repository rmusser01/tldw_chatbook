---
id: TASK-1533
title: 'Quadrant-mosaic rendering for avatar fallbacks (2x horizontal detail)'
status: Done
assignee: []
created_date: '2026-07-30 16:40'
labels: [enhancement, roleplay, console, rendering]
dependencies: [TASK-1532]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The non-graphics avatar fallback rendered via rich_pixels half-blocks (1x2
subpixels per cell), which looks coarse in every environment that cannot do
TGP/Sixel (tmux, textual-serve/browser). Replace it with a quadrant-glyph
mosaic: 2x2 subpixels per cell from the universal Block Elements range —
double the horizontal detail with identical font coverage (sextants/octants
sample finer but their glyph coverage is spotty in browsers and stock macOS
terminal fonts, so they were deliberately not used).

Surfaces wired: Roleplay Inspector portrait, character editor thumbnail
(both via `PersonasScreen._build_avatar_pixels`), and the Console rail
"Character" section box (`_character_avatar_fallback_renderable`). The
graphics (TGP/Sixel) path is untouched.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Pure module `Utils/mosaic_render.py`: per-cell luminance-split 2-color
   quantization, quadrant glyph table, aspect-true sampling (2x horizontal
   density over the half-block grid; terminal cells are ~1:2).
2. TDD: glyph vocabulary, single-cell left/right-split resolution (the case
   half-blocks cannot express), box fit, RGBA compositing, square-image
   aspect (regression for the vertical-stretch bug caught live).
3. Swap the personas builder and the Console rail fallback to the mosaic;
   update tests pinned to rich_pixels mechanics to the new seam.
4. Live browser-served verification with screenshots on both surfaces.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A hard vertical edge resolves within a single cell (left/right-half glyphs) on the personas and Console rail fallback paths.
- [x] #2 A square image renders with correct aspect (no vertical stretch) and fills its cell box.
- [x] #3 Only universal Block Elements glyphs are emitted (browser/tmux/stock-font safe).
- [x] #4 Live visual verification on the Roleplay Inspector AND the Console rail Character box.
<!-- AC:END -->

## Implementation Notes

`mosaic_from_image(image, box_cols, box_lines)` resamples to a
`2*W*scale x H*scale` grid (scale = min(cols/W, 2*lines/H)) so quadrant
subpixels (half a cell wide, half tall) sample at double horizontal density
while the painted size stays aspect-true — the first cut reused the
half-block grid shape and rendered a square globe as a tall ellipse (caught
by live screenshot, pinned by `test_square_image_fills_square_cell_box...`).
Flat cells (< 8 luminance spread) paint as background-colored spaces;
otherwise a luminance-threshold split picks fg/bg groups and the glyph mask.

Verified live via textual-serve + Playwright screenshots against the local
llama.cpp instance: Roleplay Inspector globe is round and visibly sharper
than the half-block version; the Console rail Character section paints the
same mosaic (previously it also silently depended on the tmux/browser
graphics-mode blank fixed in TASK-1532). Suites: mosaic module 6, avatar
render 3, console avatar 14, editor avatar suite, plus the chat
controller/native-flow/preview regression set — all green.
