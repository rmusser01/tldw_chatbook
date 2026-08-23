# Library Media NetNewsWire Reader Design

Date: 2026-08-23

Status: Approved

ADR: [ADR-070](../../../backlog/decisions/070-library-media-reader-ia.md)

Related: [Watchlists NetNewsWire reader design](2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md), [ADR-042](../../../backlog/decisions/042-watchlists-reader-first-ia.md)

## Summary

Redesign the Library Media Item view as a terminal-native, NetNewsWire-shaped reader with three
spatial roles:

- **Library** — the existing destination rail, independently collapsible.
- **Items** — a contextual media list, independently collapsible.
- **Reader** — the permanent reading surface and width priority.

The redesign keeps the existing media-reading scope service, list state, viewer state, and media
records authoritative. It changes the information architecture and interaction model rather than
creating a parallel media backend. Manual pane choices persist; responsive collapses are temporary.
The Reader defaults to complete stored text and may add a richer preview when the current runtime
can render one safely.

Library Media and Watchlists deliberately share an interaction grammar—permanent centre content,
full-height five-column ASCII grips, preferred-versus-effective layout, and minimum-width
resolution—but remain independently implemented. A shared split-pane framework is deferred until
both concrete consumers exist and their common contract can be extracted from evidence.

## Goals

- Make scanning and reading media a continuous three-pane workflow instead of a list-to-viewer
  takeover.
- Keep the selected item's readable content continuously available while users traverse the list.
- Let users reclaim width by collapsing Library and Items independently.
- Preserve deliberate pane and custom-width preferences without allowing narrow terminals to
  corrupt those preferences.
- Keep search scope, source provenance, pending loads, and delete recovery explicit.
- Retain all valuable current capabilities: Read Later, reading progress, Analysis, Highlights,
  metadata, content search, Use in Console, and soft delete.
- Remain usable from 80×24 through wide desktop terminals, with representative verification at
  80×24, 100×30, 120×35, and 160×50.

## Non-goals

- Adding unread, read/unread, or starred state to Library media.
- Replacing the existing media database, reading-scope service, or backend policy enforcement.
- Making rich preview the authoritative or only representation of an item.
- Building an embedded general-purpose browser or media player.
- Redesigning other Library destinations in this slice.
- Sharing implementation code with Watchlists before both designs have shipped.
- Making Reader collapsible.

## Current state

Library Media currently loads up to 50 records into the Library screen, then replaces the media
canvas with a full in-canvas viewer when an item opens. The Library rail remains, but the list and
viewer do not coexist. The viewer already supports metadata editing, Read Later, reading progress,
Markdown/raw content, Analysis, Highlights, Find, Use in Console, deletion, and an escape route to
the legacy media manager.

The existing service layer already provides:

- paginated media listing;
- offset/limit media search with a total count;
- local and server item detail normalization;
- soft delete and restore on both supported backends;
- authoritative stored content and metadata used by Console handoff.

The redesign exposes these capabilities in a stable reader shell instead of duplicating them.

## Information architecture

The Media destination has three left-to-right regions:

| Region | Purpose | Collapse behavior |
| --- | --- | --- |
| Library | Existing Library destination navigation and its existing search affordance | Independently collapsible |
| Items | Media filter, result count, and two-line item rows | Independently collapsible |
| Reader | Selected item's content, modes, status, and actions | Permanent |

Reader remains mounted when no item is selected and shows: **“Select a media item to read it
here.”** It is never a layout toggle target and receives all columns reclaimed from the other two
regions.

Opening Media restores the last valid selected item for the current session when practical. If the
item is unavailable, selection falls back deterministically to the first visible item. An empty
result set leaves Reader mounted with an explanatory empty state.

### Relationship to other Library destinations

The three-pane shell is owned by the Media destination. Other Library destinations keep their
existing canvases and behavior. Collapsing Library is a screen-level preference and may be reused
by those destinations only through the existing Library rail contract; Items exists only while
Media is active.

## Pane grips

Library and Items each have a five-column, full-height grip: four ASCII label columns plus the
divider. The visible label is horizontal and centred vertically.

Direction describes what activating the grip will do:

| Pane | Collapsed grip | Expanded inside-edge grip |
| --- | --- | --- |
| Library | `--->` expands right | `<---` collapses left |
| Items | `--->` expands right | `<---` collapses left |

Each grip is clickable, focusable, and keyboard-operable. It has a plain-language tooltip and
accessibility label naming the pane and action. Its focused style remains visible without changing
geometry. Reader has no grip.

The Media implementation should use a Library-local grip widget or the smallest existing neutral
primitive that already satisfies the contract. It must not couple its state or widget lifecycle to
Watchlists.

## Preferred, responsive, and effective layout

Layout has three explicit layers:

1. **Preferred layout** — the user's manual open/collapsed choices and, when enabled, custom pane
   widths. Manual grip actions update and persist this state.
2. **Responsive override** — collapses additionally required by the current available shell width.
   It is recomputed and never persisted.
3. **Effective layout** — the derived layout rendered for the current width.

The default is fixed target widths. Settings may opt into custom pane widths and reset them to
defaults. Custom widths are clamped to declared minimums and a reasonable maximum before they
participate in resolution. Derived responsive collapses are never written back as preferences.

### Width calculation

The resolver uses the width available to the Media reader shell after application chrome—not raw
terminal width—and declared target/minimum component widths:

| Component | Target / minimum width |
| --- | --- |
| Library | 28 / 24 columns |
| Items | 40 / 32 columns |
| Reader | flexible / 44 comfort columns |
| Each mounted grip | 5 fixed columns |

Starting from the preferred layout, the resolver collapses side panes until Reader comfort, all
mounted grips, and expanded-pane minimums fit. Normal collapse priority is Library first, then
Items. Reader is never a candidate.

The 44-column Reader value is a comfort threshold, not a hard CSS minimum. When both panes are
collapsed, both grips remain and Reader consumes the remaining width with `min-width: 0`; no
horizontal overflow is introduced. The supported live-verification floor is 60 shell columns: two
grips consume 10 and Reader receives 50. Below that floor the content may truncate, but the shell
must remain reachable and must not raise a compositor exception.

If the user explicitly opens a pane hidden by the responsive override, that pane becomes the
temporary priority target for the current width. Its preferred state is persisted as open and the
resolver may collapse the other side pane to honor the action. A small hysteresis band around
collapse/expand thresholds prevents layout thrash during resize. Repeated shrink/expand cycles are
idempotent and restore the preferred wide layout when space returns.

Representative whole-terminal captures should normally produce:

| Terminal | Expected outcome |
| --- | --- |
| 160×50 | Library + Items + Reader |
| 120×35 | Items + Reader, Library responsively collapsed |
| 100×30 | Items + Reader when minima fit; otherwise Reader-only |
| 80×24 | Reader-only with both grips available |

These captures validate the resolver; they do not define breakpoints. Actual decisions derive from
available shell width.

## Items pane

Items is a reading list, not an operations table. Each row uses a balanced two-line treatment:

- line one: item title, with selected and loaded status expressed textually as needed;
- line two: media type, author/source when available, and compact age/date.

Decorative icons may supplement but never replace labels or state. Read Later can appear as a
compact textual marker only if it does not make rows wrap unpredictably.

The pane header contains **Filter media**, an honest query over media records through the existing
`search_media` scope-service seam. It is not merely a client filter over the first 50 rows. Results
use the returned total and offset/limit contract. The ordinary unfiltered path continues to use the
paginated list seam. Both paths support incremental page loading and expose the result count.

Pagination must preserve selection by stable backend-qualified media id. Appending a page may not
reselect the first row or replace the Reader. Duplicate ids are ignored defensively. When a filter
changes, selection moves to the first matching row only after the new result snapshot arrives; if
the query is cleared, the prior unfiltered selection is restored when still available.

The pane distinguishes:

- **Selected · loading preview** — keyboard/pointer selection has moved but detail is pending;
- **Loaded in Reader** — Reader is showing that row's detail.

This prevents a blue highlight from falsely claiming that adjacent stale content belongs to it.

## Selection and loading model

Pointer and keyboard selection update the Items highlight immediately. Detail loading begins after
a short traversal settle window so holding an arrow key does not launch a request per intermediate
row. Enter bypasses the settle window and loads immediately.

`LibraryMediaReaderSessionState` keeps at least:

- selected backend-qualified media id;
- loaded backend-qualified media id;
- pending/loading/error state;
- monotonically increasing request generation;
- active Reader mode;
- preferred pane choices and responsive priority target;
- session-only selection anchors needed to restore filtered/unfiltered context.

Selected and loaded ids are intentionally separate. While item B loads, Reader may keep item A
visible, but it must show an explicit banner such as:

> Loading preview for “B”… showing “A” until ready.

Every detail response carries the requested id and generation. A response updates Reader only when
both still match current pending state. Late success or failure from A cannot overwrite B. On
failure, Items remains usable and Reader offers **Retry** and **Open original** when the latter is
available.

## Reader composition

Reader is reading-first. Its chrome is deliberately quieter than its content:

1. a compact identity line for media type/source and date;
2. title and author/source;
3. a focused action toolbar;
4. one active mode surface;
5. optional transient loading/error/undo banners.

### Modes

Exactly one mode is visible:

- **Read** — default on first entry; complete stored text or Markdown.
- **Analysis** — existing analysis content.
- **Highlights** — existing highlight management.
- **Info** — metadata, provenance, representation status, and edit affordances.

The active mode persists while moving between items during the screen session. Missing mode content
shows an item-specific empty state rather than silently switching back to Read.

### Toolbar

The primary Reader toolbar contains:

- **Find** — find within the currently loaded item;
- **Read Later** — toggle the existing media state;
- **Use in Console** — hand off the representation identified in Info;
- **More** — item-level secondary actions such as edit metadata, open original, copy/export when
  already supported, and move to trash.

Pane controls are not in More; they are screen-level grips. At constrained widths actions collapse
by priority into More without truncating ambiguous labels. Find and Read Later remain visible as
long as practical; no action may disappear without remaining reachable.

### Search vocabulary

Search labels declare scope and never silently change meaning:

| Label | Scope |
| --- | --- |
| Search Library | Existing Library navigation search scope; its copy must name the destinations it actually searches |
| Filter media | Backend media result query in Items |
| Find in item | Loaded Reader representation only |

If the existing Library rail search does not search media or all Library destinations, it may not
claim to be Library-wide. The redesign must either narrow its label/help copy or extend its service
semantics in a separately accepted task.

## Stored content, rich preview, and provenance

Complete stored text/Markdown is always the authoritative readable fallback and is never replaced
or truncated merely because richer rendering is available. Rich preview is optional enhancement:

- mount it only when the record advertises a supported representation and the current Textual
  runtime can render it safely;
- keep the complete text representation reachable in the same item session;
- label preview failure without turning the item into an overall load failure;
- never require rich rendering in headless tests.

Info identifies:

- active backend and backend-qualified id;
- original source/file/URL when available;
- stored representation and its completeness;
- rich-preview capability and status;
- the exact representation **Use in Console** will send.

Console handoff uses the existing authoritative payload contract. The Reader must not imply that a
decorative preview, partial excerpt, or rendered image is what Console receives when it is not.

## Read Later and reading progress

Read Later remains the existing cross-session media state and updates optimistically only when the
current service contract can reconcile failure. A failed mutation restores the prior visible state
and reports the error.

Reading progress belongs to the loaded item, not merely the highlighted row. Progress restoration
occurs after its content is mounted. Stale loads may not write progress under a newer selected id.
Mode changes preserve per-item Read scroll position during the screen session.

## Delete and Undo

Move to trash is available through More and requires title-specific confirmation. After successful
soft delete:

1. remove the row from the current list snapshot;
2. select the next adjacent row, preferring the following row and then the previous row;
3. load that row into Reader, or show the empty state;
4. show a bounded **Undo** action.

Undo calls the existing `restore_media_item` scope-service seam for the same backend-qualified id,
reinserts the restored item according to current sort/filter rules, and reselects it when it still
matches the active result scope. If it no longer matches, Undo still succeeds and explains that the
item was restored outside the current filter. Permanent delete is not introduced here.

## Focus, keyboard, and escape behavior

The focus order follows the visual hierarchy: Library, Library grip, Items controls/list, Items
grip, Reader toolbar/modes/content. Hidden panes are skipped while their grips remain reachable.

Escape graduates outward:

1. close transient state such as More, Find, confirmation, or a tooltip/popover;
2. move focus from Reader to Items when Items is effectively open;
3. move focus from Items to Library when Library is effectively open;
4. from Library, use the screen's normal back behavior.

If an intermediate pane is responsively collapsed, Escape skips it. Existing global and terminal-
convention bindings remain unshadowed. Footer hints advertise only actions implemented for the
current focus and state.

## Component ownership

### `LibraryMediaReaderShell`

Owns layout composition only:

- mounts Library, grips, Items, and permanent Reader according to effective layout;
- applies resolved widths;
- emits pane-toggle and width-adjustment messages;
- performs no persistence, service calls, or item-selection policy.

### `LibraryMediaReaderSessionState`

A pure state/resolver module owns:

- selected versus loaded identity;
- pending generation and stale-response acceptance;
- active Reader mode;
- preferred pane state and responsive priority target;
- custom-width normalization;
- effective-layout calculation and hysteresis.

It does not own database records or duplicate existing `library_media_state` and
`library_media_viewer_state` derivations.

### `LibraryScreen`

Remains the orchestrator:

- owns query/page/result snapshots and invokes `media_reading_scope_service`;
- maps existing media list/viewer state into the new widgets;
- sequences selection, detail loading, mutations, Reader updates, and focus restoration;
- persists only preferred user layout values through the canonical Settings/config path;
- keeps service and policy errors visible without destroying usable list state.

Existing media list and viewer state remain authoritative. This design does not create a second
backend abstraction or a general-purpose pane framework.

## Error, loading, and empty states

- Initial Media load shows Items skeleton/status and the permanent Reader empty state.
- Page/filter failure preserves the previous stable Items snapshot and offers Retry.
- Detail failure affects Reader only; Items remains navigable.
- Rich-preview failure falls back to complete stored content and is labelled locally.
- Backend switch clears incompatible request generations and backend-qualified selection anchors.
- Empty filter copy includes the active query and a clear-filter action.
- No state relies on color, icon, or focus styling alone.

## Configuration

The canonical Settings screen exposes a compact Library Media layout group:

- remember Library pane open/collapsed preference;
- remember Items pane open/collapsed preference;
- fixed default widths or opt-in custom widths;
- custom Library and Items widths when enabled;
- reset layout to defaults.

Defaults are Library open, Items open, fixed target widths. Responsive state, pending requests,
selection, focus, and Reader mode are session state and are not persisted as layout preference.
No settings are added to deprecated settings surfaces.

## Testing and verification

Tests should be focused by contract instead of accumulated in one large shell test module.

### Pure state and resolver tests

- preferred, responsive, and effective state remain distinct;
- collapse priority is Library then Items;
- explicit open prioritizes the requested pane without losing preferred state;
- custom widths clamp correctly;
- hysteresis prevents threshold thrash;
- shrink/expand cycles are idempotent;
- Reader and both grips remain mounted at the supported floor;
- request generations reject stale success and stale failure;
- selected and loaded ids can differ only with explicit pending state.

### Items and service integration tests

- unfiltered pagination loads beyond the first 50 records;
- Filter media calls the search seam and respects total/offset instead of filtering only loaded rows;
- append preserves selection and ignores duplicate ids;
- filter clear restores the prior unfiltered selection when possible;
- local and server backend-qualified ids cannot collide;
- delete selects the correct adjacent row;
- Undo restores and reselects, including the restored-outside-filter message.

### Reader tests

- first entry defaults to Read and mode persists across items;
- pending banner identifies both selected and loaded titles;
- Find searches only loaded item content;
- Console handoff matches the representation declared in Info;
- reading progress writes only for the loaded identity;
- rich-preview capability-on uses a fake renderer;
- mandatory capability-off coverage proves complete text fallback;
- preview failure does not become item failure.

### Interaction and visual tests

- both grips work by pointer and keyboard in expanded and collapsed states;
- focus and Escape graduate through only effective panes;
- toolbar overflow keeps every action reachable;
- search labels and help text pass the scope-honesty gate;
- geometry assertions verify pane/grip/Reader widths at representative sizes;
- condition-based waits replace arbitrary sleeps;
- focused mutation tests guard stale-response and responsive-state branches where practical;
- screenshots at 160×50, 120×35, 100×30, and 80×24 confirm information hierarchy, text state,
  no overlap, and no horizontal overflow.

Headless tests do not assert image rendering. Live verification should exercise a real local media
record and, when configured, a real server record, following the repository's evidence guidance.

## Delivery boundaries

Implementation planning should split work into atomic Backlog tasks that leave the Media
destination usable at each step. The plan must read the applicable testing/live-verification
lessons, link ADR-070, and avoid marking any task Done without focused automated evidence and the
required task notes.

## Alternatives considered

### Keep the current list-to-viewer takeover

Rejected because it interrupts list traversal, hides context, and cannot provide the NetNewsWire
scan/read loop the redesign targets.

### Build a new independent Media reader screen

Rejected because it would duplicate Library navigation, backend state, and current media viewer
capabilities while making cross-destination behavior harder to reason about.

### Share Watchlists' layout implementation immediately

Rejected because the two consumers have different pane counts, lifecycle, state, and content
contracts. Shared grammar is valuable now; shared code should follow evidence from two shipped
implementations.

### Let every pane, including Reader, collapse

Rejected because Reader is the screen's purpose and must remain the stable spatial anchor.

### Client-filter the initial 50 media rows

Rejected because it produces incomplete, misleading results. The existing search and pagination
seams already support an honest result scope.
