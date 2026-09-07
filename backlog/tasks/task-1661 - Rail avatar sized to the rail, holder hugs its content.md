---
id: TASK-1661
title: 'Rail avatar sized to the rail; holder hugs its content'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - console
  - images
  - ux
dependencies:
  - task-1660
priority: medium
---

## Description (the why)

User report after task-1660 restored native graphics: the portrait renders
crisply but sits small in the corner of a tall, mostly-empty box, with the
character name stranded at the bottom.

Two independent causes, neither of them cropping:

1. `ClickableAvatarBox` is a bare Textual `Container`, whose defaults are
   `width: 1fr; height: 1fr`. Nothing set a size on it — there is no CSS
   rule for `#console-character-avatar` and the compose site set no inline
   styles — so the holder claimed the whole rail section. The image sat at
   its default top-left origin and the name was pushed to the bottom.
2. The avatar box was hard-coded `CHARACTER_AVATAR_COLS/LINES` (16x8)
   regardless of rail width, so a ~50-column rail displayed a 16-column
   portrait.

Ruled out: nothing in the pipeline crops. `ConsoleImageRenderCache.prepare`
only downscales above the decode ceiling and PIL `thumbnail()` preserves
aspect; `get_pil` returns a copy; `fit_image_cell_size` is a contain fit.
The bust framing the user sees is the character card's own artwork.

## Acceptance Criteria (the what)

- [x] The avatar box derives from the rail's live width, clamped so a tall
      portrait cannot claim the whole rail
- [x] The holder hugs its content instead of expanding to the section
- [x] Both render paths agree on framing (contain, per user choice)
- [x] User confirms the portrait fills the rail in real Kitty/iTerm2

## Implementation Notes

`character_avatar_box(available_cols)` clamps between the historical 16x8
minimum and 44x22, deriving lines as half the columns (terminal cells are
~2x taller than wide, so this reads near-square). Width comes from
`_character_avatar_available_cols()`, which measures the rail SECTION
BODY and returns 0 before layout settles, falling back to the old
constants.

**Second-pass correction (user re-test):** the first attempt measured the
HOLDER, which had just been made `width: auto` -- circular, since the
holder hugs the very child being sized. Harness probe: holder reported 13
cols (the previous child) vs 27 for the section body, so the box clamped
to the 16-col minimum and nothing visibly changed. A regression test now
pins that the measurement comes from the section body and differs from
the holder's width. Verified after the fix: 27 cols -> box (27, 14),
mounted image 22x14 cells; a 45-col rail -> (44, 22). Both the graphics widget and the mosaic fallback use the
same box, and the mosaic switched `fit="cover"` -> `"contain"` so the two
paths frame identically (user chose whole-image over crop-to-fill).

The holder gets explicit `width/height: auto`. Three tests cover the box
math (fallback, scaling, clamp), the contain agreement, and the holder
sizing. Suites: 37 avatar/mosaic/warm-up green; live boot clean.
