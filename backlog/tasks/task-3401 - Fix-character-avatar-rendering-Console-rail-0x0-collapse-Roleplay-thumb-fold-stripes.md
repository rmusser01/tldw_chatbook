---
id: TASK-3401
title: 'Fix character avatar rendering: Console rail 0x0 collapse + Roleplay thumb
  fold stripes'
status: In Progress
created_date: 2026-08-08 03:54
labels:
- ui
- rendering
- bug
updated_date: 2026-08-08 05:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Character card portraits are invisible on the Console rail and corrupted on Roleplay thumbnails, while the image data pipeline (DB bytes, PIL decode, mosaic content) is healthy. Console: the pixels-branch avatar Static (default width 100%) mounts inside the auto/auto ClickableAvatarBox (task-1661) and resolves to 0x0 under Textual 8.2.8 — even the no-character placeholder collapses, leaving only the name label. Roleplay: _build_avatar_pixels bakes a 24-cell-wide cover mosaic while all three thumb containers (#personas-inspector-avatar-thumb, #personas-char-editor-avatar-thumb, .personas-char-editor-expr-thumb) reserve max-width 24 PLUS padding 0 1, so every mosaic line folds at 22 content columns; continuation rows paint black on dark themes (stripes) and the folded ~17-row stack exceeds max-height 10 (bottom clipped). Reproduced headless against the owner's real DB image: console region 0x0, inspector region 22x17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console rail avatar paints a non-zero region with the active character portrait in pixels mode
- [ ] #2 Console avatar placeholder text remains visible (no 0x0 collapse) when no portrait is available
- [ ] #3 Roleplay inspector portrait and editor/expression thumbnails render with no fold (painted height equals mosaic rows, no black continuation stripes)
- [ ] #4 Regression tests pin the painted regions for the console builder and the roleplay thumb mount path
- [ ] #5 Existing console and personas avatar test suites still pass
- [ ] #6 Pre-existing mosaic test failures root-caused and fixed (Rich 15 file= harness emits no ANSI; noise-cell-box test statistically flaky) so the suite is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: routine bug fix — restores intended rendering behavior behind existing surfaces; no schema, boundary, contract, or UX-structure change.

1. Console (chat_screen.py::_build_character_avatar_widget): give the pixels-branch Static explicit styles.width/styles.height derived from the built renderable grid (mosaic Text lines), mirroring ConsoleImageViewerModal._build_full_size_widget; keep max-* as clamps. Give the no-portrait placeholder Static width auto so it cannot collapse in the auto/auto holder. Verified pattern in repro_avatar_fix.py (0x0 -> 11x8).
2. Roleplay: remove padding 0 1 from the three thumb containers (#personas-inspector-avatar-thumb, #personas-char-editor-avatar-thumb, .personas-char-editor-expr-thumb) so content width equals the mosaic build width (AVATAR_THUMB_COLS=24) — fixes fold at the source; and in the two mount helpers (PersonasInspectorPane.set_avatar_thumbnail, PersonasCharacterEditorWidget.set_avatar_thumbnail/set_expression_thumbnail) set explicit width/height on the wrapping Static from the Text grid so any future width mismatch degrades to a crop, never stripes. Verified pattern (22x17 folded -> 22x10).
3. Regression tests: console — mount the builder output in the real holder shape and assert a non-zero painted region + placeholder visibility; roleplay — assert mounted thumb Static region height equals mosaic rows (no fold) and line width <= container content width.
4. Run targeted avatar suites (console + personas), then the wider Tests/UI avatar/image sets.
5. Live verify headless with the owner DB image (repro scripts) before close-out.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
