---
id: TASK-22500
title: >-
  Virtualize the Library media reader body - it repaints every line of the
  document
status: Done
assignee: []
created_date: '2026-08-26'
updated_date: '2026-08-26 23:33'
labels:
  - performance
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Measured by TASK-22209's implementer while verifying its own (much smaller) win: the reader
body is a raw auto-height `Static` inside a `VerticalScroll`, so a 2.5 MB / 24,000-line
document gives the widget `height=45000` and `Widget._render_content` re-renders ALL 45,000
lines on EVERY repaint. `_render_content` alone measured **1.08-1.83 s**, and the click a
user actually feels is **~1.4-1.5 s both before and after** 22209's fix (whose 45 ms saving
is ~2% of the real cost).

This is distinct from TASK-22228 items 6-7 (recompose COUNT, now fixed) — this is paint
VOLUME per repaint. After the whole reader burn-down (22207 traversal, 22208 no-change
syncs, 22209 match nav, 22210 progress writes, 22228 press scope), this is the single
largest remaining Library reader cost and it dominates everything already fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A large document (>=2 MB) repaints in time proportional to the VIEWPORT, not the document: measure `_render_content` and end-to-end click latency before/after at 100 KB / 1 MB / 2.5 MB
- [x] #2 Scrolling, match navigation, highlight styling and search-scroll-into-view keep working across the window boundary (the existing reader gates stay green)
- [x] #3 The approach is stated: Textual line-api/virtualized widget, chunked mounting, or an explicit window with the trade-offs recorded
- [x] #4 A guard pins the property (render cost independent of document size) so it cannot regress silently
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: `VirtualizedRawContent` (`Widgets/Library/library_media_raw_view.py`, task 1-2)
replaces the raw `Static` in `LibraryMediaContentBody`'s Raw view with a `ScrollView`
subclass implementing Textual's line API directly. `Utils/text_wrap_index.py`'s
`WrapIndex` precomputes exact wrap breakpoints (via Rich's own `divide_line`, not a
`cell_len // width` approximation -- ragged real-world text drifted ~2.9% under the
approximation) once per (document, width) pair; `render_line(y)` maps a viewport row
through the index and renders only that one row's segment, so cost is O(viewport) per
repaint instead of O(document). Search highlighting, drag-select (`get_selection`),
match-scroll (`scroll_to_source_line`), and a debounced resize-triggered reindex
(task 8, 0.12s quiet period, mirroring TASK-22211's precedent) were all rebuilt against
this row-mapped model across tasks 3-8. Trade-off recorded and measured (task 9): the
index build is a genuine O(document) cost, but it now happens once per resize instead of
once per repaint, which is the right trade given repaints (every search keystroke, every
match-nav click) vastly outnumber resizes.

Task 9 measurements (before = merge-base 732105c2d's Static-based reader, checked out to
a scratch copy and instrumented identically; after = this branch; both measured via
direct time.perf_counter() brackets / instance-method monkeypatches around the actual
render calls, never harness wall time -- pilot.pause() costs ~30ms/call and would swamp
anything sub-millisecond):

| size    | before 1st paint (`_render_content` self-time, 2 calls) | after 1st paint (`render_line` self-time, 58 calls) | before search repaint (`Static.update`) | after `sync_search` repaint | after index build (direct) |
|---------|---:|---:|---:|---:|---:|
| 100 KB  | ~74 ms   | **0.51-0.54 ms** | ~52 ms    | **0.51-0.55 ms** | 6.3-7.1 ms   |
| 1 MB    | ~766 ms  | **0.65-1.30 ms** | ~604 ms   | **0.52-0.57 ms** | 58.4-60.9 ms |
| 2.5 MB  | ~2,283 ms| **0.81-0.90 ms** | ~1,743 ms | **0.59-0.66 ms** | 141.3-144.9 ms |

Before scales linearly with document size in both first paint and every search repaint
(matches the widget's own docstring note that the old Static "rendered twice" at first
paint -- confirmed directly: `_render_content` fires exactly twice before, zero times
after). After stays flat (sub-1.3ms) at every size for both first paint and repaint; call
count is also flat (58/18, viewport-bound not document-bound). The index-build cost is
real and scales with size but is paid once per resize, not per repaint -- an honest,
accepted trade, not a wash.

Markdown view (`mode="rendered"`) scope decision: measured directly (same three sizes,
realistic paragraph shape) rather than fixed. `Markdown.update()` (Textual's own widget,
which mounts one `MarkdownBlock`/`Static` per block) costs 129ms/2.3s/10.0s at
100KB/1MB/2.5MB -- a >10s hang at 2.5MB, and a structurally different problem (per-block
mount overhead in a different widget architecture) than the one this task solved. Filed
task-22660 with the full numbers rather than extending this task's scope, since task 9 is
this plan's close-out task and Markdown virtualization needs its own design pass.

Coverage gap closed: the pre-existing `test_renders_only_visible_rows_regardless_of_document_size`
guard only counts `render_line` calls, which Textual's compositor bounds by viewport
regardless of what the method body does -- mutation-tested a `render_line` that rebuilds
every row on every call (the exact "0.26ms cached vs 364ms rebuilt" regression named in
the task brief) and confirmed that guard stays GREEN under it. Added
`test_render_line_self_time_does_not_scale_with_document_size` (measures render_line
directly via perf_counter, no pilot.pause), confirmed it REDS under the same mutation,
then Edit-restored the mutation and reran the full file green (25/25).

Also mutation-tested (all confirmed red, then Edit-restored, `git diff --stat` clean
after each): WrapIndex.build's exact-vs-approximate wrapping, get_selection returning
None, and scroll_to_source_line skipping the wrap-index row mapping.

Full affected surface: 75 passed (Tests/Library/test_library_media_content.py,
test_library_media_raw_view.py, Tests/Utils/test_text_wrap_index.py, the three reader
gates t22207/t22208/t22209, test_library_media_reader_scroller_resolution.py,
test_library_recompose_ratchet.py -- ratchet still pinned at 74). Spot-checked
test_library_shell.py's 10 media/reader-relevant tests (that file's known pre-existing
baseline is 664-665 passed/63-64 failed in an unrelated conversation-reader area): all 10
pass. ./scripts/preflight.sh green (CSS bundle, profile-owned-path census, diagnostic
inventory, 2636 backlog task IDs with no duplicates).

Files: Tests/Library/test_library_media_raw_view.py (new self-time test);
backlog/tasks/task-22660 (new follow-up); Docs/User_Guide/library.md (Verified-against
stamp). No production code changed by this task -- it is measurement-only, closing out
tasks 1-8's implementation. Full report:
.superpowers/sdd/2026-08-26-library-reader-virtualization/task-9-report.md
<!-- SECTION:NOTES:END -->
