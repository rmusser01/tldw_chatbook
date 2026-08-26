# Library media reader: virtualize the Raw body (TASK-22500)

Status: design approved in chat 2026-08-26, pending spec review.
Base: dev `732105c2d`. Follows the 2026-08-24 holistic perf review's burn-down
(`Docs/Design/2026-08-24-holistic-perf-review.md`).

## Problem

The Raw view renders the whole document into one `Static` inside a `VerticalScroll`.
`Widget._render_content` builds every line into a cached strip list, so any repaint that
invalidates that cache re-renders the entire document.

Measured on this branch (2.2 MB / 24,000 source lines, 100x40 viewport):

| event | cost today |
|---|---|
| first paint | **1051 ms** (the document renders twice) |
| scrolling | **free** — cache persists, ~20 `render_line` per step |
| every `Static.update()` — each search keystroke, each match-nav click | **684 ms** |
| bare `refresh()` | 499 ms |

The finding that opened this task said "every repaint"; that is too broad and is corrected
here. Scrolling is already fine. The cost is **first paint plus every `update()`**, which is
exactly the ~1.4 s click TASK-22209 measured and could not explain from its own 45 ms saving.

## Approach

A `ScrollView` subclass renders only the rows in view (`render_line`), backed by an exact
wrap index built once per (content, width).

Rejected alternatives:

* **Chunked `Static`s** (~500-line blocks). A highlight change would repaint one block, but
  every block's auto-height must be measured, so **first paint stays ~1 s** — the larger half
  of the problem survives.
* **Explicit windowing** (mount only nearby lines), the chat transcript's TASK-15455/15777
  approach. It fixes both costs but re-implements scroll anchoring that `ScrollView` already
  owns, and that machinery produced the estimated-vs-measured skew and prune-releases-anchor
  bugs in this repo. Unnecessary here: only *rendering* needs to be lazy, not *mounting*.

## Prototype evidence

A throwaway prototype (6.47 MB / 24,000 ragged lines, virtual height 77,568 rows) measured:

| | prototype | today |
|---|---|---|
| index build | 400 ms | — |
| first paint (total) | **493 ms** | 1051 ms *at a third the size*; ~3 s extrapolated here |
| scrolling, `render_line` self-time | **0.027 ms/row** (10.6 ms per 400 calls) | free (cached) |
| search repaint, `render_line` self-time | **0.5 ms** (40 rows) | **684 ms** |
| jump to line 12,000 | exact via the index | approximate (see below) |

Wall-clock in the harness is dominated by `pilot.pause()` (300 ms for 10 scroll steps in
**both** arms). Measure `render_line` self-time; wall time in a Pilot harness will report a
scrolling regression that does not exist.

## Design

### 1. Structure

`LibraryMediaContentBody` (id `library-media-viewer-content`) becomes a plain container
hosting exactly one mode-specific scroller:

* Raw -> `VirtualizedRawContent(ScrollView)` (new, this design)
* Rendered -> today's `VerticalScroll` + `Markdown`, unchanged

Its public surface is unchanged: `content`, `is_markdown`, `sync_mode`, `sync_search`. The
viewer and screen keep calling exactly what they call today.

**Hazard.** Three sites in `library_screen.py` do
`query_one("#library-media-viewer-content", VerticalScroll)` inside `try/except`
(scroll-position capture, restore, and match scrolling). If the type stops matching they
**silently no-op** — the reader would quietly lose scroll restoration with every test still
green. These resolve the *active* scroller instead, and a test asserts each one actually
finds it.

### 2. The wrap index

One pass per (content, width) records each source line's wrapped row count and its running
virtual-row offset; `virtual_size` follows. `render_line(y)` maps a viewport row to
(source line, wrap segment) with `bisect` and renders that segment only.

**Exact wrapping is required.** A character-division approximation (`cell_len // width`) was
measured wrong on **12.4% of ragged prose lines**, drifting virtual height by 2.9% (2,942
rows on a 100k-row document) — a visibly wrong scrollbar and a broken match jump. Rich's
`divide_line` produces output **identical** to the public `Text.wrap` (verified over 24,000
ragged lines) at roughly half the cost: 426 ms vs 785 ms for 8.5 MB. Scaled to 2.5 MB that is **~125-155 ms** (the two
measured documents give 50 and 62 ms/MB; the spread is content-shape, not noise).

`rich._wrap.divide_line` is a **private API**. It is used behind a one-function adapter with a
`Text.wrap` fallback, and a test pins that the two agree — so a Rich upgrade that moves or
changes it fails loudly here rather than silently changing how documents wrap.

**Per-line segment cache.** Justified by one case only: a single pathological long line
(500k characters) costs 9.4 ms per `divide_line` call, which `render_line` would otherwise pay
per rendered row. For ordinary documents the cache measured no difference, so it stays small
and bounded rather than clever.

**Width changes re-index** (~125-155 ms at 2.5 MB). Resize coalescing follows TASK-22211's
hysteresis precedent so a drag-resize does not pay it per event.

### 3. Highlighting and match navigation

`RawContentHighlightPlan` keeps ownership of *which* source lines match. The widget applies
match / active-match styling per line as it renders, so moving the active match repaints only
the visible rows.

**Simplification this unlocks:** the plan's whole-document `Text` build (an O(document) pass
that exists only to hand `Static.update` a pre-styled blob) becomes dead. Only the match-line
list survives, which is what navigation and the "N of M" status actually consume.

**Correctness fix.** `_scroll_library_media_content_to_line` currently calls
`scroll_to(y=line_index)` — a *source line index* used as a *screen row*. Its own docstring
concedes it "is not pixel-perfect"; it drifts progressively once any line wraps, so on a
wrapped document the match jump lands increasingly far from the match. The index makes the
mapping exact.

### 4. Selection

Drag-to-select works today (verified with real mouse events: a drag produced
`'pha 1 beta gamma delta\nalpha 2 beta gamma delta\n...'`). `Static` provides it free;
a custom widget must implement `get_selection` or it disappears silently. The widget
implements it against the source text, and a test drags **across a wrap boundary** and
asserts the copied text.

Fidelity details that must match `Static`: `markup=False` (literal brackets — this repo has a
documented bracket-markup trap), tab expansion, and the empty-document message.

### 5. Testing

* Red-first probes: first paint, per-click repaint, and render cost independent of document
  size (the AC's guard) — all asserted on `render_line` **self-time**, never harness wall time.
* Equivalence against today's `Static` output for wrapped, ragged, unicode/wide-glyph, CRLF,
  tab-bearing, empty, and no-match documents.
* `divide_line` vs `Text.wrap` agreement (pins the private-API risk).
* Selection across a wrap boundary.
* The three scroller-resolution sites each find their target.
* Existing TASK-22207 / 22208 / 22209 reader gates stay green; the Library recompose ratchet
  (pinned at 74) does not move.
* Tests currently asserting on `raw.renderable` are converted to assert rendered lines, not
  deleted.

### 6. Markdown view

Per the scope decision: after the Raw work lands, measure the Rendered/Markdown path on the
same documents. Extend this task if it is also slow; otherwise file it with the numbers. It is
a widget tree, not text, so it is a different fix.

## Risks

| risk | mitigation |
|---|---|
| Reimplementing wrapping/selection that `Static` gave free | equivalence + selection tests above |
| Private `divide_line` API | adapter + fallback + agreement test |
| Scroll capture/restore silently breaking | active-scroller resolution + a test per site |
| Re-index storms on resize | coalescing (TASK-22211 precedent) |
| Harness wall time hiding the truth | assert `render_line` self-time |

## Expected outcome

First paint ~1051 ms -> ~125-155 ms at 2.5 MB (and ~3 s -> ~490 ms at 6.5 MB); every search
keystroke and match-nav click 684 ms -> under 1 ms of render work; scrolling unchanged; the
match jump becomes exact instead of drifting.
