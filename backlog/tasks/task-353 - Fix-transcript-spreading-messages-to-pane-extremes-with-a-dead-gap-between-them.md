---
id: TASK-353
title: >-
  Fix transcript spreading messages to pane extremes with a dead gap between
  them
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-22 14:30'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the first send, the user message renders at the very top of the transcript pane and the 'Assistant [failed]' + System rows at the very bottom, with ~40 empty rows between them. The pattern persists after more messages, across reloads of the persisted conversation, and in the selected-message state - it is layout, not scroll position or streaming anchoring.

**Repro:** Send one message in a fresh Console conversation and let it complete/fail. Observe the user message pinned top and the response pinned bottom with a large void between.

**Verifier note:** Real, but narrower than reported: j3-63/67 show the void sits immediately after the inline-image row and spans ~36-40 rows — matching the image row widget's max_height 40 clamp (console_transcript.py:932); non-image messages stack contiguously (j3-67 bottom run of 5 rows). So this is an inline-image row-height defect from task-215/PR #626 (post-dates every June-verified transcript capture), not a general transcript layout regression, and no ledger item covers image-row geometry — reviewer's suspected_regression is wrong but the defect is genuine and unrecorded. P2 appropriate: any conversation containing an image becomes unscannable.

**Source:** Console UX expert review 2026-07-20 (finding j3-transcript-giant-gap-layout; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-63-response-final.png`, `j3-67-shift-enter.png`, `j3-79-view-modal.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Messages should stack contiguously (top-anchored or bottom-anchored), so the conversation reads as a chronological flow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
DIAGNOSIS (pilot-measured, not yet fixed): the giant gap is the inline-image
row's height, not scroll/anchoring. `_image_row_widget`
(console_transcript.py ~970) sets `max_width=80, max_height=40` on the row
widget. PIXELS mode (rich_pixels Static, styles.height=auto) is correctly
CONTENT-SIZED — an 80x8 image → 4 rows, 16x16 → 8 rows, no void. GRAPHICS mode
(textual_image `Image`, styles.height=None) FILLS to max_height=40 for EVERY
aspect ratio (wide/square/tall all measured region.height=40); it renders the
image at correct aspect INSIDE the 40-row box and letterboxes the rest → the
~36-40 row void. `height="auto"` does NOT help (textual_image still fills 40).
Default render mode is "auto" (config.py:2743), which resolves to graphics when
the terminal claims support (xterm.js does), so the void is the default
experience with images.

FIX OPTIONS (need served-app + real-image verification): (a) compute the
aspect-correct display height for the graphics widget (H/W × display_cols ×
cell-aspect, capped 40) and set styles.height explicitly so the box matches the
image (risk: cell-aspect is terminal-dependent — textual_image is meant to own
this, so probe its API for a natural-size/sizing option first); (b) constrain
graphics like pixels; (c) if graphics letterboxing can't be tamed cleanly,
consider defaulting the served/limited path to pixels (content-sized, no void) —
a product decision. Pixels mode already correct, so the target is graphics-mode
height. Repro needs the textual-serve harness with a real image (graphics
protocol) — pilots fall back to synthetic sizing.
<!-- SECTION:PLAN:END -->


## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RESOLVED ON DEV by commit cf8d69c63 ("fix(console): explicit-fit graphics image
size in transcript") — landed concurrently while this task was being worked. It
added `fit_image_cell_size(pixel_w, pixel_h, box_cols, box_lines)` in
console_image_view and wired it into `_image_row_widget`, setting BOTH
`widget.styles.width` and `.height` to the aspect-fitted cell box for the
graphics widget (primarily to dodge textual_image's transient 0-size
`ValueError`, but it fixes THIS void too: the widget is now sized to the image's
aspect instead of filling max_height=40 and letterboxing). Verified on dev via a
pilot: wide 1024x256 → region.height=10 (was 40), very-wide 1600x200 → 5, square
→ 40 — the ~36-40 row void is gone, so messages stack contiguously.

(A parallel fix in PR #776 — `console_image_row_rows` — was a duplicate of the
same solution and was closed in favour of dev's more complete
`fit_image_cell_size`, which also handles the width dimension and the 0-size
race.)
<!-- SECTION:NOTES:END -->
