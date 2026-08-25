---
id: TASK-22207
title: >-
  Library media reader: do not rebuild the document body per traversal keystroke
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25'
labels:
  - performance
  - library
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22207).

New with PR #2064. `UI/Screens/library_screen.py:7407-7419`: `on_descendant_focus` now
opens the reader on focus, so arrow-keying through the Items list drives it per keystroke.
`_select_library_media_reader_row` runs `_sync_library_media_viewer_or_recompose()`
synchronously (`:12424`) — the 120 ms settle timer debounces only the DB fetch, not the
recompose — and the `unchanged` test at `:33096-33100` compares `viewer.loading` against a
freshly-flipped pending flag, so it always falls through to `viewer.refresh(recompose=True)`
(`:33141`). Compose constructs a brand-new full-document body
(`Widgets/Library/library_media_viewer.py:296-311`): in rendered mode a fresh
`Markdown(content)` whose mount parses the whole document on the loop. The settle path
then recomposes a second time. Net: 2 full-document rebuilds per settled selection, 1 per
keystroke during fast traversal — and the wasted one re-parses the document being LEFT,
purely to paint a "Loading..." placeholder. There is no windowing of the reader body.
While Find is open, each rebuild adds 3 more O(document) match scans
(`library_media_viewer.py:296`, `library_media_content.py:34`, `:155`).

## Acceptance Criteria

- [x] Traversing N rows performs zero document-body rebuilds for pass-through rows; only the settled row renders, once (probe counts `LibraryMediaContentBody` constructions per 10-row traversal)
- [x] Showing the loading placeholder does not require recomposing the document body
- [x] A 1 MB document can be traversed past with no per-keystroke parse (measured)
- [x] Reader behavior (settle, modes, search) unchanged; existing reader tests green

## Implementation Plan

1. Baseline: teed runs of the reader suites (`test_library_media_reader_flow.py`,
   `test_library_media_reader_shell.py`, `test_library_per_click_recompose_t21116.py`,
   `test_library_recompose_ratchet.py`, `test_library_media_side_by_side.py`, the
   media-loading shell test) at branch base `983aa5878`.
2. Red-first probe file `Tests/UI/test_library_media_reader_traversal_t22207.py`:
   count `LibraryMediaContentBody` constructions (and `Markdown` builds) across a
   scripted 10-row focus traversal on the mounted harness with a gated detail
   service; assert 0 for pass-through rows and exactly 1 for the settled row.
   Confirm RED on the unmodified tree.
3. Fix, viewer side (`Widgets/Library/library_media_viewer.py`): make the loading
   placeholder a persistent, display-gated widget (compose always yields
   `#library-media-viewer-loading` when a media item is rendered, hidden unless
   loading; the empty-reader `#library-media-reader-empty` keeps its dual copy) and
   add `sync_loading_state(loading=..., message=...)` that patches copy + display in
   place. Display-gating (not mount/unmount) is deliberate: an async mount seam here
   is the TASK-21116 M3 DuplicateIds race class.
4. Fix, screen side (`UI/Screens/library_screen.py::_sync_library_media_viewer_state`):
   remove `loading`/`loading_message` from the recompose-deciding `unchanged`
   comparison (it then compares only real compose identity: display state built from
   the loaded detail, sub-state flags, highlights, search, mode, preview identity);
   on the unchanged path, patch the loading placeholder in place via
   `sync_loading_state`. Loading transitions alone no longer recompose the body;
   detail/settle transitions still recompose exactly once.
5. Evidence: per-keystroke traversal cost before/after with a ~1 MB markdown
   document fixture (rendered mode), plus the probe counts.
6. Mutation test: Edit-restore the loading-inclusive comparison, confirm the probes
   go red, restore the fix (Edit-based, never `git checkout`).
7. Failure/teardown paths: settle firing after the route changed; a fetch failing
   after the selection moved on; fast alternating focus between two rows (existing
   generation fences at `_library_media_preview_request_is_current` /
   `_matches_pending` are the pattern; assert no stale body wins and no
   DuplicateIds).
8. Targeted suites + `--collect-only` sweep + ratchet census (must stay ≤ 97) +
   `./scripts/preflight.sh`; tee everything and read counts from the tee.

## Implementation Notes

Two small changes at the mechanism, no new async seams, no windowing:

1. **`Widgets/Library/library_media_viewer.py`** — the pending banner
   (`#library-media-viewer-loading`) is now a PERSISTENT, display-gated child
   (composed once whenever a media item renders, hidden unless loading) and a
   new `sync_loading_state(loading=..., message=...)` patches its copy and
   visibility (or the empty-reader placeholder's copy) in place. Display-gating
   rather than mount/unmount is deliberate: an async mount seam on this surface
   is the TASK-21116 M3 `DuplicateIds` race class.
2. **`UI/Screens/library_screen.py::_sync_library_media_viewer_state`** —
   `loading`/`loading_message` are removed from the recompose-deciding
   `unchanged` comparison (every traversal keystroke flips the pending flag
   BEFORE this runs, so the old comparison always fell through to
   `viewer.refresh(recompose=True)`, re-parsing the document being LEFT to
   paint "Loading…"). The comparison now covers only real compose identity
   (viewer display state from the loaded detail, sub-state flags, highlights,
   search, mode, error, preview identity); the unchanged path calls
   `viewer.sync_loading_state`. Settle/detail transitions still recompose,
   exactly once.

### Evidence (all counts read from teed logs in `test-logs/`, gitignored)

- **Red-first** (`probe-red-first-2.txt`, unmodified tree): 10-row arrow-key
  traversal = **10 body builds** for pass-through rows; painting the banner =
  1 build; A↔B alternation = 5 builds; a deferred loading recompose landed in
  the stale-failure window (1 build).
- **After** (`probe-green-final.txt` / `final-green-verification.txt`): the
  same traversal = **0 pass-through builds, exactly 1 settled-row build**; the
  banner paints and clears with the SAME `LibraryMediaContentBody` widget
  identity and 0 builds; alternation re-settles identical content with 0
  builds (the re-fetched detail is a new dict — the display-state comparison
  is structural, so no stale body ever wins).
- **1 MB markdown fixture** (rendered mode, real `pilot.press("down")`
  keystrokes): pre-fix **median 16,889.688 ms per keystroke** (samples
  16,049–18,183 ms — a fresh 1 MB `Markdown` parse per keystroke; the settle
  then could not drain the parse backlog within 180 s; `probe-red-1mb-3.txt`).
  Post-fix **median 400.793 ms**, **0 body builds / 0 Markdown parses** during
  traversal, settle clean (`probe-green-1mb.txt`). Attribution probe
  (`test-logs/attribution-probe-no-banner-update.py`): with the banner patch
  fully suppressed the median is statistically identical (477 ms), so the
  in-place patch adds no measurable cost; the residual per-keystroke cost is
  the already-filed 22208/22210 work (preview projection, per-step progress
  write), not this seam. No wall-clock threshold is asserted (15457 rule).
- **Mutation test** (`mutation-red.txt`): Edit-restored the loading-inclusive
  comparison and removed the in-place patch → the three probes went red on the
  exact symptoms (10 / 1 / 6 builds); Edit-restored the fix (never
  `git checkout` over uncommitted work).
- **Failure/teardown paths** (probe file): a settle firing after the route
  changed paints nothing anywhere (0 builds, conversations canvas intact); a
  stale fetch failing after the selection moved on paints no error and no body
  (existing `_matches_pending`/generation fences — no new fence needed); fast
  alternating focus ends with one `#library-media-viewer-content` node, banner
  hidden, 0 builds.
- **Ratchet**: census at tip = **97** (pin 97); no whole-screen recompose
  statements added or removed. The viewer-scoped `viewer.refresh
  (recompose=True)` on the changed path is not in the census (by design).
- **Suites** (this branch, base `983aa5878`): probe file 6/6; reader flow +
  reader shell + ratchet + side-by-side **74 passed / 0 failed**;
  media-adjacent (content-search debounce, prompt search, scroll keys, image
  preview, trash, multiselect) **120 passed / 0 failed**; per-click t21116 +
  screen navigation **132 passed / 7 failed** — the 7 are byte-identical to
  the base run (`baseline-perclick-ratchet-sbs.txt`); canvas-sync trio
  **23 passed / 8 failed**, byte-identical to base (`base-canvas-sync.txt`);
  `test_library_shell.py -k media` **80 passed / 35 failed**, and every one
  of the 35 FAILED names reproduces at base (chunked node-id re-run,
  `base-media-failed-chunk{1,2}.txt`; sorted-set diff empty). Zero new reds
  anywhere.
- **Sweep**: `--collect-only` = **58,542 tests collected, 28 errors** — all 28
  are optional-dependency modules (numpy/audio/TTS/Confluence), none touching
  Library/UI or any changed file. `./scripts/preflight.sh` fully green
  (CSS bundle, path census, diagnostic inventory, task ids, table allowlist).

### Behavior notes

- One deliberate test retarget: `test_pending_banner_names_selected_b_and_
  loaded_a` waited on the banner's DOM PRESENCE, which is vacuous now that the
  banner is persistent — it now waits for the banner to be DISPLAYED.
- `Docs/User_Guide/library.md` got a "Verified against" stamp (no workflow
  changes; precedent: the TASK-21116 stamp).
- Lesson recorded in `backlog/docs/lessons-testing-evidence.md`: a
  programmatic `.focus()` is not the user's keystroke — `on_resize` arms
  `_library_notes_resize_settling` from the mount resize onward and only real
  input events clear it, so a Pilot probe must drive `pilot.press("down")`,
  not `row.focus()` (cost a debug cycle here).

### Pre-existing findings, NOT fixed here (verified, not this task's scope)

1. **Traversal-failure error paint gap**: with an older document rendered,
   traversing to a row whose CURRENT fetch fails settles `session.error` but
   never mounts `#library-media-viewer-error`/Retry —
   `_recompose_library_media_detail_if_unrendered` early-returns because the
   old detail is still "rendered identity". Reproduced IDENTICALLY at base
   `983aa5878` (scratch probe; error paints only when no detail was rendered,
   the case the existing retry test covers).
2. **Dev-base harness staleness**: 7 reds in
   `test_library_per_click_recompose_t21116.py`, 8 in the canvas-sync trio,
   and the 35 media-scoped shell reds all reproduce at base — the dominant
   signature is the harness boot recipe expecting `#library-rail-explore-all`,
   which no longer mounts on this dev base (post-#2064 reader-shell boot).
3. `test_library_shell_media_viewer_shows_loading_before_detail_loads` is red
   at base: a fresh press with `detail=None` renders
   `#library-media-reader-empty` (the empty branch returns before the banner),
   yet the test asserts `#library-media-viewer-loading` exists.
