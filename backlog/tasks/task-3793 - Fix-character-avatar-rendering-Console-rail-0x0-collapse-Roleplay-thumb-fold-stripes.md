---
id: TASK-3793
title: 'Fix character avatar rendering: Console rail 0x0 collapse + Roleplay thumb
  fold stripes'
status: Done
created_date: 2026-08-08 03:54
labels:
- ui
- rendering
- bug
updated_date: 2026-08-08 15:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Character card portraits are invisible on the Console rail and corrupted on Roleplay thumbnails, while the image data pipeline (DB bytes, PIL decode, mosaic content) is healthy. Console: the pixels-branch avatar Static (default width 100%) mounts inside the auto/auto ClickableAvatarBox (task-1661) and resolves to 0x0 under Textual 8.2.8 — even the no-character placeholder collapses, leaving only the name label. Roleplay: _build_avatar_pixels bakes a 24-cell-wide cover mosaic while all three thumb containers (#personas-inspector-avatar-thumb, #personas-char-editor-avatar-thumb, .personas-char-editor-expr-thumb) reserve max-width 24 PLUS padding 0 1, so every mosaic line folds at 22 content columns; continuation rows paint black on dark themes (stripes) and the folded ~17-row stack exceeds max-height 10 (bottom clipped). Reproduced headless against the owner's real DB image: console region 0x0, inspector region 22x17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console rail avatar paints a non-zero region with the active character portrait in pixels mode
- [x] #2 Console avatar placeholder text remains visible (no 0x0 collapse) when no portrait is available
- [x] #3 Roleplay inspector portrait and editor/expression thumbnails render with no fold (painted height equals mosaic rows, no black continuation stripes)
- [x] #4 Regression tests pin the painted regions for the console builder and the roleplay thumb mount path
- [x] #5 Existing console and personas avatar test suites still pass
- [x] #6 Pre-existing mosaic test failures root-caused and fixed (Rich 15 file= harness emits no ANSI; noise-cell-box test statistically flaky) so the suite is green
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
Two independent root causes, both in layout — the image data pipeline (DB bytes, PIL decode, mosaic content) was healthy end-to-end.

1. Console 0x0 collapse: since task-1661 (commit 622992419) the avatar holder is auto/auto; a default-width Static inside an auto container resolves to 0x0 under Textual 8.2.8, so both the portrait and the no-character placeholder mounted but painted nothing. Fix: `_build_character_avatar_widget` now sizes the pixels Static explicitly from the built mosaic grid via the new `mosaic_render.explicit_cell_size()` helper (max-* kept as clamps), and the placeholder gets `width auto`. Mirrors the proven `ConsoleImageViewerModal._build_full_size_widget` pattern.
2. Roleplay fold stripes: the three thumb containers reserved max-width 24 PLUS `padding: 0 1`, so every 24-cell mosaic line folded at 22 content columns; continuation rows painted as black stripes and the folded ~17-row stack exceeded max-height (bottom clipped). Fix: removed the padding (content width now equals the build width) and the mount helpers size the wrapping Static explicitly from the Text grid, so any future width mismatch degrades to a crop, never stripes.
3. Pre-existing harness failures (owner-challenged): Rich 15 no longer emits ANSI into `Console(file=StringIO, force_terminal=True)` — harness switched to `record=True` + `export_text(styles=True)`; the noise cell-box test was statistically flaky on Pillow 11.2.1 (LANCZOS-downscaled lines can bake to all-space) — contract repinned to `1 <= lines <= 5`.

Modified: chat_screen.py, mosaic_render.py, personas_inspector_pane.py, personas_character_editor_widget.py + 4 test files (2 new regression tests console, 3 roleplay, 2 mosaic unit). Branch fix/avatar-rendering-3401, commit 9148f8a74, PR against dev.

Test evidence: mosaic 12/12, console avatar 28/28, editor 14/14, inspector 31+1. Final combined run: 85 passed / 1 failed. The failure is `test_state_pushed_before_children_mount_defers_then_replays` (task-2727 deferral pin): fails 6/6 on this box with these changes applied, passed 1/1 on pristine; each changed file independently triggers it, so it is believed timing-sensitive rather than a logic regression. AC #5 is checked on the basis that every avatar/image suite passes and the single failure is an unrelated deferral-replay race — tracked in TASK-3794 per owner direction, not blocking this PR.

Rebase integration (2026-08-08): PR #1434 was rebased onto `origin/dev` at
`3023578c0`. The documentation conflict retained both dev's newer testing
lessons and this task's painted-region lesson. The task was renumbered from
3401 to 3793, and its follow-up from 3402 to 3794, because current dev already
owns the Console rail-label preference (renumbered from TASK-3401 to
TASK-14650 in PR #1465) and claimed TASK-3771
during the final pre-merge refresh. ADR check remains no/N/A:
this integration preserves the existing rendering fix and architecture.

Post-rebase verification was owner-scoped to files touched by this PR. The
four focused files completed with 86 passed / 2 deselected; both deselected
tests were run separately on the rebased branch and pristine `origin/dev` and
failed identically there (`test_rgba_input_is_composited_over_black`, due the
terminal's 16-color ANSI downgrade, and the TASK-3794 deferral-replay race).
Ruff reports the same 40 pre-existing findings on both trees, with no
branch-only finding, and `git diff origin/dev --check` is clean. A broad run
was stopped on owner instruction after its initial collection blocker was
also proven baseline-only (the optional `playwright` package is absent).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
