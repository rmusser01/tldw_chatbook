# Library Media Viewer In-Place Search Design

**Task:** TASK-15458  
**Date:** 2026-08-12  
**Status:** Approved design

## Purpose

Remove full-screen remounts and unnecessary full-document Markdown parses from
Library media content search and mode switching. Also debounce the legacy media
panel's per-keystroke content search so one burst of typing produces at most one
render.

This work preserves the Library viewer's current Enter-to-search interaction,
match counting, wraparound navigation, Raw-mode highlighting, and source-line
scrolling behavior.

## Scope

### In scope

- Library media viewer content-search submission, clearing, next/previous
  navigation, and Rendered/Raw mode switching.
- Stable identity for the Library screen, media viewer, and any already-mounted
  Markdown document across those interactions.
- Lazy first mounting and subsequent reuse of the Rendered Markdown document.
- In-place Raw content highlighting and search-status updates.
- Debounced content search in `MediaViewerPanel` with exactly one Markdown
  update when a search is applied.
- Stale-debounce protection across query changes, media changes, clear, and
  unmount.
- Focused behavior, identity, render-count, scrolling, and latency evidence.

### Out of scope

- Changing Library content search from Enter-to-search to live search.
- Moving content matching or Markdown rendering to a worker thread.
- Replacing Textual's Markdown renderer or mutating its internal block widget
  implementation.
- Redesigning how source-line indices map onto rendered Markdown block geometry.
- Changing match cardinality: Library search continues to count one match per
  matching source line and mark only the first occurrence on that line.

## Existing Problems

The Library viewer is composed as part of the 26k-line `LibraryScreen`. Search
submission, match navigation, and content-mode toggles currently call a
screen-level `refresh(recompose=True)`. That discards and recreates the viewer,
including the full Markdown document, even when the only changed values are a
counter, one Raw-mode highlight style, or visibility.

The legacy `MediaViewerPanel` runs content search directly from every
`Input.Changed` event. Its display refresh first calls `Markdown.update("")`
and then `Markdown.update(content)`, causing two parses per keystroke. Its
next/previous behavior remains outside the Library-screen remount problem; this
task changes its typing path only.

## Design Decisions

### 1. Screen state stays canonical

`LibraryScreen` continues to own:

- `_library_media_content_query`
- `_library_media_content_match_index`
- `_library_media_content_mode`

Handlers update those values first, calculate matches from the raw stored
content, and then delegate presentation changes to the mounted
`LibraryMediaViewer`. Missing mounted widgets during navigation or teardown are
safe no-ops; they do not trigger a screen-level fallback recompose.

### 2. Search chrome becomes a scoped child widget

Extract the search input, optional status line, and optional Previous/Next
toolbar into a small `LibraryMediaContentSearchControls` child. It receives a
complete state consisting of:

- whether the media is Markdown (for placeholder copy),
- the submitted query,
- the ordered matching source-line indices,
- the current wrapped match index.

Its public `sync_state(...)` replaces the complete snapshot and recomposes only
that child. This preserves the existing structural behavior: status and
navigation widgets do not exist for an empty query. The Input is recreated on
Enter, matching current behavior; `LibraryScreen` restores its focus after the
scoped update.

The child exposes the exact signature
`sync_state(*, is_markdown: bool, query: str, matches: tuple[int, ...],`
`match_index: int) -> None`.

### 3. Content body owns lazy, persistent view instances

Extract the scrollable content body into `LibraryMediaContentBody`.

- Non-Markdown media mounts only the Raw `Static`.
- Markdown media initially mounts only the selected mode. The default Raw view
  therefore does not pay a Markdown parse during initial viewer load.
- The first switch to an unmounted mode dynamically mounts that mode's widget.
- Once mounted, Raw and Rendered instances remain children of the content body;
  subsequent toggles change `display` rather than remounting either instance.
- The Rendered widget always receives the original stored content and is not
  updated for match navigation.
- The Raw widget receives a Rich `Text` renderable built from raw slices. Query
  submission, clearing, and match navigation update that `Static` in place,
  including while Raw is hidden, so returning from Rendered never exposes stale
  highlighting.

The content body exposes narrow public methods rather than allowing the screen
to reach into child implementation details:

- `async sync_mode(mode: str) -> None` ensures the target instance exists and
  updates visibility.
- `sync_search(query: str, match_index: int) -> None` updates only the Raw
  renderable.

`LibraryMediaViewer` exposes corresponding delegation methods with those same
names and signatures. The async mode method does not return until a first-use
target has mounted and visibility is correct.

### 4. Library handlers perform narrow synchronization

Search submission:

1. Stop the event, trim the submitted query, and no-op if it equals the current
   canonical query.
2. Store the query and reset the canonical match index to zero.
3. Synchronize the search-controls child and Raw body renderable.
4. Restore search-input focus after the scoped controls update.
5. Scroll to the first matching source line when one exists.

Next/Previous navigation:

1. Rebuild the ordered source-line match tuple from canonical raw content and
   query.
2. No-op when there are no matches.
3. Wrap the canonical index, synchronize the status and Raw renderable, and
   scroll to the selected source line.
4. Never call `LibraryScreen.refresh(recompose=True)`, viewer recompose, or
   `Markdown.update()`.

Mode switching:

1. Preserve the current no-op guard when the requested mode is already active.
2. Store the canonical mode.
3. Update toggle labels/classes in place and ask the content body to show the
   target mode.
4. Pay at most one necessary Markdown parse, on the first transition to an
   unmounted Rendered view. Later toggles reuse the same widget identity.

### 5. Legacy panel search uses a guarded 250 ms debounce

`MediaViewerPanel` uses a named `MEDIA_CONTENT_SEARCH_DEBOUNCE_SECONDS = 0.25`,
matching the Library prompt and STTS profile search convention.

On non-empty `Input.Changed`:

1. Cancel the prior content-search timer.
2. Increment a search generation.
3. Capture the generation, current media identity, and exact query.
4. Arm one 250 ms timer.

When the timer fires, it applies the search only if all captured values still
match current state. It then finds matches, updates match/status state, and
calls the content display update exactly once.

Clearing the input cancels the timer, increments the generation, clears search
state immediately, and renders the unhighlighted document once. Loading another
media item and unmounting the panel also cancel/invalidate pending work. A stale
callback is a no-op.

`update_content_display()` removes the preliminary `Markdown.update("")` and
performs only `Markdown.update(content)`.

## Error and Lifecycle Handling

- Narrow sync helpers catch only Textual absence/query exceptions expected
  during navigation and teardown. Unexpected exceptions remain visible.
- A search or navigation action with no open media detail or no matches is a
  no-op, preserving current behavior.
- Lazy mounting is guarded against duplicate concurrent mounts. Repeated mode
  presses cannot create two content widgets with the same ID.
- Debounce state is invalidated before programmatic input clearing during media
  replacement, preventing an old query from repainting a new item.
- Search matching remains case-insensitive and operates on raw stored content
  in both Raw and Rendered modes.

## Testing and Evidence

### Library viewer behavior and identity

Mounted Textual tests will prove:

- Enter-to-search remains the only Library query-application path.
- Search submission preserves `LibraryScreen`, `LibraryMediaViewer`, and an
  already-mounted Markdown instance.
- Next/Previous preserve those identities, wrap correctly, update the status,
  preserve Raw current-match emphasis, and scroll to the selected source line.
- `Markdown.update()` is not called during Library match navigation.
- Empty submission removes status/navigation through only the scoped controls
  update.
- Opening Markdown media in Raw mode does not mount/parse Markdown; the first
  Rendered switch mounts it once; Raw then Rendered reuses the same Markdown
  instance.
- A query submitted while Rendered updates the hidden Raw renderable before a
  switch back.

### Legacy panel debounce

Tests with the real mounted panel and a narrow Markdown update spy will prove:

- Multiple changes within 250 ms produce no early search render and exactly one
  final render after the window.
- The render payload is never the empty-string cache-busting update.
- Clearing, loading different media, and unmounting invalidate pending work.
- A single stable query still preserves match highlighting and status behavior.

### Performance evidence

Use one deterministic long Markdown document and the identical sequence of
search submission plus repeated next/previous clicks before and after the
change. Record:

- median click latency,
- screen/viewer/Markdown identity,
- Markdown update/construct counts.

The automated latency probe records measurements but uses no wall-clock pass
threshold. Behavioral identity and parse-count assertions are the stable
regression gate.

## Documentation and Task Completion

Update TASK-15458 with the implementation plan, completed acceptance criteria,
measured before/after evidence, verification commands, and concise
implementation notes. Update user documentation only if implementation exposes
an observable interaction change; the approved design intentionally preserves
the current interaction and copy.

## ADR Check

ADR required: no  
ADR path: N/A  
Reason: this applies the existing Library screen/canvas/widget ownership model
and Textual timer patterns. It changes neither persistent state nor service,
security, dependency, or cross-module architecture boundaries.
