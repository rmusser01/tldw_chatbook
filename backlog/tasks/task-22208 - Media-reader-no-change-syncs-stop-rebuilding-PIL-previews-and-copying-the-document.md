---
id: TASK-22208
title: >-
  Media reader no-change syncs: stop rebuilding PIL previews and copying the
  document
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 22:25'
labels:
  - performance
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22208).

New with PR #2064. `_sync_library_media_viewer_state` computes the image-preview
projection BEFORE its `unchanged` test (`library_screen.py:33079-33085` vs `:33110`), so
every interaction — traversal step, mode switch, More toggle, Escape — synchronously runs
`build_media_image_widget` (`Widgets/Library/library_media_image_preview.py:127-175`): a
PIL LANCZOS resize plus a per-cell Python loop over the mosaic grid
(`Utils/mosaic_render.py:216-240`), on the event loop, then discards the widget when
nothing changed. Separately, every interaction rebuilds the full viewer display state:
`build_library_media_viewer_state` copies the whole content string (`str(...).strip()` at
`Library/library_media_viewer_state.py:305` — a trailing newline forces a full copy) and
the `unchanged` compare memcmps it (`:33087`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A no-change sync builds no preview widget and performs no O(document) string copies (probe)
- [x] #2 Preview construction is memoized by source (and/or moved off the loop) with the memo's invalidation stated
- [x] #3 The change test compares cheap identity (ids/revisions), falling back to content compare only when identity is inconclusive
- [x] #4 Per-interaction cost measured before/after on an image-typed item
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split the image-preview projection: cheap identity (status/hidden/available/source, no widget) feeds the unchanged compare; the widget is built only when the compare concludes changed (or when composing a fresh viewer).\n2. Memoize the mosaic renderable inside build_media_image_widget keyed by (image object identity, box_cols, box_lines); a fresh Static is constructed per compose (widgets cannot be remounted). Invalidation: new decoded image object, box dim change.\n3. Memoize build_library_media_viewer_state per detail-object arrival (keyed by detail identity + arrival_note/backend/canonical/external), so the O(document) str().strip() copy happens once per detail arrival, not per sync; route the sync, handoff-representation, search and mode-default call sites through it.\n4. Change test: identity-first (viewer.viewer IS the memoized state) with structural == fallback when identity is inconclusive (new detail dict with identical values must still compare equal - pinned by t22207 alternating-focus test); re-anchor viewer.viewer on the unchanged path so identity recovers after an identical re-fetch.\n5. Red-first probes: count build_media_image_widget/factory calls and content-copy work per no-change sync (nonzero today, 0 after); measure per-interaction wall time before/after on an image-typed item with a large document.\n6. Targeted suites + collect-only sweep, preflight, mutation tests (drop a memo key input -> stale-preview probe reds; drop identity fast-path -> copy probe reds).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Split the sync's preview projection, memoized the mosaic renderable, and memoized the viewer display state per detail arrival, so a no-change sync does zero preview building and zero O(document) copies.

Approach:
- `_library_media_image_preview_projection(build_widget=False)` returns only the identity facts (status/hidden/available/source image) the `unchanged` compare reads; the widget factory never runs on the no-change path. The changed path calls the projection again with the default to build a FRESH widget (a removed Textual widget cannot be remounted) and assigns the POST-build values so a build failure still flips the failure status consistently.
- `build_media_image_widget` memoizes the mosaic renderable in a single slot keyed by (decoded image OBJECT identity — strong ref held so id() reuse cannot alias, box_cols, box_lines). `fit`/`monochrome` never vary on this call shape and the mosaic samples colours from the image, so mode/theme are deliberately not key inputs (mode only selects the branch). Widgets are never reused — only the Text renderable.
- `_library_media_viewer_state_cached` memoizes `build_library_media_viewer_state` keyed by detail-object identity (the detail is only replaced wholesale or cleared, never mutated in place) plus arrival_note/backend/canonical_id/force_raw. The `str(content).strip()` copy now happens once per detail ARRIVAL. Routed through it: display-state build, console-representation/handoff payload (which ran INSIDE the unchanged compare per sync), content-search submit, match-advance, and the LIB-13 mode default.
- Change test: identity first (`viewer.viewer is viewer_state` — true on no-change syncs because the state is memoized), structural `==` only when identity is inconclusive (new arrival; byte-identical re-fetch must still compare equal — pinned by t22207's alternating-focus probe). The unchanged path re-anchors `viewer.viewer` to the memoized object so identity recovers after an identical re-fetch. NO field was removed from the 22207-proved compare.
- Accepted deviation (documented in the memo docstring): the "Updated: <age>" relative label freezes per detail arrival; recomputing it per sync was the cost, and a ticked-over age would otherwise force a full document recompose to repaint one metadata line.

Measured (Tests/UI/test_library_media_reader_no_change_sync_t22208.py, image-typed item):
- 5 pass-through keystrokes: preview factory calls 5 -> 0; display-state builds 10 -> 0; copied bytes ~270,010 -> 0 (27 KB doc).
- Direct no-change sync on a ~1 MB document with a 900x600 preview: median 3.892 ms (all memos mutated off = pre-task behavior) -> 0.012 ms.
- Keystroke wall time is pilot-frame dominated (~107 ms median both sides); the structural counters are the guarantee (15457 probe rule).

Mutation-tested: mosaic memo ignoring box_cols reds the memo-key probe; state memo ignoring detail identity reds the probe file; restoring per-sync builds (build_widget=True pre-compare + memo off) reds the factory/copy probes (5/10 again).

Verification: probe file 5/5; t22207 traversal probes green; preview/viewer-state/mosaic suites 107 passed; flow/shell/search/handoff 81 passed; side-by-side/trash/browse green — the 13 reds in test_library_entry_compose_once.py fail identically at base dev 76f130138 (pre-existing, non-media). Full collect-only: 59,449 collected; 28 errors are missing-optional-dep families (numpy/audio) untouched by this diff. preflight.sh all green.

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/Widgets/Library/library_media_image_preview.py, Tests/UI/test_library_media_reader_no_change_sync_t22208.py.
<!-- SECTION:NOTES:END -->
