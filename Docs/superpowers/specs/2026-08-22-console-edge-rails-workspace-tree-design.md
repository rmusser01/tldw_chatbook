# Console Edge Rails and Workspace Tree Design

**Date:** 2026-08-22
**Status:** Approved for planning
**Surface mode:** Operate
**Target:** Native Textual Console screen

## Goal

Make the Console read as one edge-to-edge work surface, give every expanded
Context section a predictable reading viewport across terminal emulators, and
make workspace ownership immediately understandable by nesting workspace-owned
conversations under their workspaces.

The change must preserve the Console's terminal-native density, existing
conversation and workspace persistence, responsive rail behavior, and
keyboard-first operation.

## Evidence and problem statement

The current production layout has five related problems:

1. The Console workspace grid is framed and horizontally inset beneath the
   full-width header. Context and Inspect therefore read as panels embedded
   inside another panel rather than the application's side edges.
2. Context's content-aware allocator divides the currently visible height
   among open sections whenever their headers fit. A terminal emulator that
   reports fewer rows at the same physical window size therefore gives each
   section a smaller viewport. This reproduces in iTerm2 while a Windows
   Terminal window with more reported rows can reach the intended ceilings.
3. The Workspaces section displays only active-workspace identity and actions,
   while the Conversations section owns the grouped all-workspaces browser.
   This contradicts the user's ownership model.
4. Default/unassigned conversations are mixed into the same grouped browser as
   workspace-owned conversations, increasing scan cost.
5. Character art is not given enough vertical room for roleplay use and must
   never be stretched or cropped.

## Approved product decisions

- Retain the current full-width Console header above the three-column work
  surface.
- Remove the inset workspace frame and horizontal gutters. Context touches the
  application's left edge and Inspect touches the right edge below the header.
- Use one divider between Context and the transcript and one divider between
  the transcript and Inspect. Do not retain doubled or nested borders.
- Give open Context sections their natural height through a section-specific
  content ceiling. Do not shrink them merely to keep every later header in the
  initial viewport.
- Let the entire Context rail scroll when the sum of expanded sections exceeds
  its available height.
- Use a native Textual `Tree` for named workspaces and their conversations.
- Keep default-workspace and unassigned conversations in a separate flat
  Conversations browser.
- Treat starring as a conversation property, not as a third display location.
  Starred conversations sort first inside their one owning projection and are
  never duplicated into a cross-owner Starred aggregate.
- Give Workspaces and Conversations independent search controls.
- Render Character images with aspect-ratio-preserving contain behavior and a
  35-row section ceiling.
- Keep Inspector's existing 20-row bounded-section contract unchanged.

## Shell layout

The global Console header and control rows remain full width. Immediately below
them, the workspace has three direct visual regions:

1. Context at the left application edge.
2. Transcript/composer in the center.
3. Inspect at the right application edge.

The workspace owner may retain the top and bottom boundary needed to separate
it from global chrome, but it must not draw a surrounding left/right frame or
add an outer horizontal inset. Each rail owns its application edge; the center
column owns neither rail border. Exactly one stable divider marks each
rail/transcript boundary.

Collapsed Context and Inspect handles occupy the same corresponding application
edge. Existing responsive width rules and focus handoff remain authoritative.

Border ownership includes the workspace grid, both rail roots, transcript,
collapsed handles, and focus states. Removing the grid inset alone is
insufficient: no descendant may retain a second full frame or rounded border
that recreates an inlaid center or rail. Focus uses paint-only emphasis on the
existing divider/header and never adds border cells or changes measured
geometry.

## Context height policy

### Content ceilings

| Section | Maximum visible content rows before local scrolling |
| --- | ---: |
| Sessions | 15 |
| Workspaces | 20 |
| Conversations | 20 |
| Model | 15 |
| Agent | 15 |
| Details | 15 |
| Character | 35 |

Each open section hugs content shorter than its ceiling. Content beyond the
ceiling scrolls within that section. An overflowing section reserves one
additional non-focusable hint row with the existing exact local copy,
`▼ more — scroll`.

The shared bounded-section primitive accepts a per-instance content ceiling for
Context instead of enforcing one global 20-row constant. Context passes the
15-, 20-, or 35-row ceiling from the table above. Inspector continues to pass
20 and retains its current behavior; this change must not silently make
Inspector's ceiling configurable through user or transient runtime state.

The rail no longer distributes a shared content-row budget among open sections.
Every section receives `min(desired_content_rows, section_ceiling)` plus its
hint when required. When complete expanded sections exceed the rail viewport,
the Context outer body scrolls and uses the existing distinct outer cue,
`▼ more sections — scroll`.

This makes allocation depend on terminal cells reported by Textual, not on the
terminal emulator or physical pixel size. A 15-row section is therefore 15
terminal content rows in iTerm2, Windows Terminal, and other supported
emulators.

## Workspaces Tree

### Ownership

The Workspaces section owns one native Tree with an always-expanded,
non-selectable synthetic root and `show_root=False`. Every currently
visible/selectable, non-retired named workspace is a top-level node beneath
that hidden root. Each workspace's associated conversations are direct child
nodes. Default-workspace and unassigned conversations are excluded so no
conversation appears in both sections.

This is a display projection over existing workspace IDs and conversation
records. It does not create a new database relation, migrate records, or change
the meaning of the built-in Default workspace.

### Interaction

- Top-level workspace nodes expand and collapse with Textual Tree's native pointer
  behavior and a narrowly scoped Console key adapter. This is a subclass or
  handler over Textual's shipped Tree, not a replacement Tree implementation.
- Workspace selection and disclosure are separate actions. Selecting a
  workspace label with pointer or Enter switches to that workspace without
  changing its disclosure state. Textual's label-selection auto-expand is
  disabled so one gesture cannot perform both actions.
- Pointer disclosure and Space retain Textual's native toggle behavior. The
  adapter adds conventional plain Left/Right behavior only while the Tree owns
  focus: Left collapses an expanded branch, otherwise moves to its visible
  parent; Left on a collapsed top-level workspace is a no-op rather than moving
  the cursor to the hidden root. Right expands a collapsed branch, otherwise
  moves to its first selectable child; Right on a leaf or empty workspace is a
  no-op. It does not intercept keys while either section search input or any
  other editable control owns focus.
- Existing workspace disclosure preferences seed and retain expansion state.
- The active workspace and active conversation remain textually and visually
  identified without relying on color alone.
- Selecting a workspace retains the existing switch behavior.
- Selecting a conversation retains the existing resume/switch behavior,
  recovery messages, run markers, and session-state semantics.
- The Tree must preserve a stable focus target across keyed updates whenever
  the same workspace or conversation still exists.
- Removing a focused conversation follows the existing bounded-rail recovery
  order rather than moving focus to an unrelated rail section.
- The Tree uses a compact two-cell guide depth. `Load more…`, loading, empty,
  error, and Retry entries are explicit non-expandable action/status node types
  and never receive workspace-switch semantics.
- Native shifted ancestry/sibling commands are retained only while their target
  is a visible node. At the hidden-root boundary, Shift+Left and Shift+Up on the
  first top-level workspace are no-ops; no shifted command may move the cursor
  to the hidden root, clear the visible cursor, or wrap across the boundary.
- Tree disclosure icons and guide lines use the application's existing glyph
  fallback policy. Unicode mode may use Textual's disclosure/guide vocabulary;
  ASCII mode supplies explicit readable `ICON_NODE`, `ICON_NODE_EXPANDED`, and
  guide-line equivalents rather than relying on hard-coded Unicode glyphs.

### Preserved workspace controls

The Workspaces section retains the current active-workspace identity plus a
compact pinned Switch/New/RAG Scope action strip above its search and Tree.
They remain part of the section's measured 20-row body rather than becoming
global chrome. The strip is one physical row at supported rail widths and does
not wrap merely because labels or recovery state change.

The active-workspace identity is also exactly one ellipsized physical row. Its
full literal value remains available through the existing tooltip/help pattern;
a long user-controlled workspace name cannot consume additional Tree rows.

- Switch remains the authoritative route to every workspace, including the
  built-in Default workspace that is intentionally omitted from the Tree.
- New retains its existing workspace-creation behavior and recovery state.
- RAG Scope continues to reflect and act on the active workspace, including
  its existing enabled/disabled and recovery semantics.
- Switching to Default leaves the Workspaces Tree visible for later switching
  while the flat Conversations section becomes the owner of Default records.
- When a conversation leaf owns the Tree cursor, starring remains available as
  a compact contextual `Star`/`Unstar` action and an `s` accelerator; the Tree
  label exposes the same state textually. This action may use a second compact
  pinned row only while needed. It does not turn Tree rows into embedded widget
  containers or create a duplicate Starred projection. The `s` accelerator is
  active only while the Tree itself owns focus and never intercepts input from
  either search field or another editable control.

### Search

Workspaces owns its own search input. A query matches workspace names and
nested conversation titles. A matching child keeps its parent workspace visible
even when the parent name does not match. Clearing the query restores the
user’s pre-search expansion preferences.

Expansion and collapse gestures while a query is active are temporary search
presentation state. They are never persisted and are discarded when search is
cleared, at which point the exact pre-search expansion snapshot is restored.
Search does not mutate workspace membership, active selection, or persisted
disclosure preferences.

Search is evaluated against the workspace/conversation service rather than
only the children currently materialized in the Tree, so an unloaded matching
conversation can still reveal its parent and result row.

Workspaces search owns an independent query value, debounce timer, attempt
generation, result/error snapshot, cache namespace, worker group, and
search-specific Retry target. Conversations search owns a separate set.
Typing, clearing, canceling, retrying, or completing one search must not cancel,
overwrite, or restore presentation state in the other section.

Every asynchronous Workspaces search commit validates its search-attempt
generation and projection/mount identity. Paging owns a separate monotonically
increasing attempt generation per workspace and validates that generation,
workspace ID, membership snapshot, requested cursor/offset, and mounted Tree
identity before mutating nodes. Page state resets when its membership snapshot
changes. A late search or page from a previous query, Retry attempt, collapsed
or removed workspace, moved conversation, or unmounted Tree is discarded
without moving focus or changing disclosure state. Overlapping requests for the
same cursor commit at most once.

### Update strategy

The Tree uses stable keys derived from existing workspace and conversation IDs.
Ordinary status, marker, title, or selection changes update existing nodes
incrementally. A full Tree reconstruction is reserved for structural changes
such as workspace/conversation addition, removal, or membership movement.

The ownership projection includes every associated conversation, but it need
not materialize every child at mount. Each expanded workspace loads records in
the service's existing bounded pages and exposes an explicit, keyboard-
reachable `Load more…` child while more records remain. Search queries the full
service scope independently of loaded pages. This removes the current
12-visible-row group ceiling without an unbounded synchronous mount.

Loading replaces `Load more…` with one non-selectable loading row. A partial
page failure preserves already loaded conversations and exposes an actionable
Retry row with concise recovery copy. Retrying the same cursor is idempotent;
duplicate records are ignored by stable conversation ID. Clearing search or
collapsing a workspace never converts temporary search expansion into a saved
preference.

Tree labels are always constructed as literal Rich `Text` (or equivalently
escaped before reaching Textual); user-provided workspace names and
conversation titles are never parsed as markup. Tests cover brackets,
markup-looking text, CJK, emoji, and mixed-width labels.

No new dependency is required; the exact-pinned Textual version already ships
the Tree widget and refreshes visible node lines for node-level updates.

### Scroll ownership

Textual Tree is itself a scroll view. It is therefore the Workspaces section's
single local scroll owner and must not be mounted inside a second
`VerticalScroll`. The active identity/actions and search are fixed measured
chrome within the section's 20-row body; the Tree receives the remaining rows.
The fixed chrome may consume at most 12 rows at every supported rail width;
service and recovery copy is truncated to one row with the full text available
through the existing tooltip/help pattern rather than consuming the Tree. The
Tree receives `min(desired_tree_rows, available_tree_rows)`, so a one-node Tree
still hugs its content while demand of eight or more rows is guaranteed at
least eight visible rows inside the 20-row body. The Tree remains
keyboard-focusable whenever it contains an enabled selectable workspace or
conversation node, even when its content does not overflow. Overflow alone
controls the local hint and wheel handoff, not whether the interactive Tree
enters the Tab order.

The shared bounded-section integration reads the Tree's desired/virtual height,
controls its allocated viewport height, and renders the standard local hint as
a sibling outside the Tree. Wheel handoff at the Tree boundary bubbles to the
Context outer rail.

## Conversations browser

Conversations remains a flat browser and displays only:

- conversations assigned to the built-in Default workspace; and
- conversations with no workspace association.

It owns an independent search input that matches only this flat projection.
Current row selection, title wrapping, status details, run markers, empty/error
copy, and resume behavior remain unless a direct conflict with the new
ownership filter is proven. Star and unstar remain per-row actions, but the
cross-owner Starred aggregate is removed. Starred rows sort before unstarred
rows within this flat projection; recency remains the secondary ordering key.
The same starred-first then recency ordering applies within each workspace's
Tree children. A conversation appears exactly once in search and ordinary
projections regardless of star state.

The empty state explains that conversations belonging to named workspaces are
available under Workspaces. It must not imply deletion or missing data.

## Character section

Character receives a 35-content-row ceiling for the complete local body,
including the image, character name, reaction state, reaction action, and any
layout spacing between them. Its image:

- preserves the complete source image;
- preserves its aspect ratio;
- scales down only when needed to fit the available rail width and the rows
  remaining after the mounted non-image controls are measured, and never
  enlarges a smaller source merely to consume the available box;
- is centered in unused horizontal or vertical space; and
- is never cropped or stretched.

When a valid image is present, the section reserves the measured rows required
by the name, reaction state, reaction action, and their real margins before
fitting the image. The image may grow only until the complete body reaches 35
rows, so those controls remain inside the initial 35-row viewport. The fit is
recomputed when rail width, terminal height, image, character metadata, or
control geometry changes; it must not assume a hard-coded control-row count.

Image fitting uses a bounded, equality-guarded two-phase settle. First measure
the mounted non-image controls and stable content width. Then compute one
scale-down-only contained cell size from `35 - measured_non_image_rows`, update
or remount the image only when that size changed, and permit at most one
follow-up section/rail reconciliation for the resulting geometry. Scrollbar appearance or
disappearance must settle to the same size on the next pass rather than
oscillating. Both terminal-graphics and mosaic fallback paths reuse the
existing image-fit/contain utilities; no parallel aspect-ratio algorithm is
introduced.

Letterboxing is acceptable and preferred to information loss or distortion.
Character metadata and reaction controls remain below the image in the same
local scroll owner. A missing image leaves the short textual body at its natural
height rather than claiming 35 empty rows. Portrait, landscape, square, very
large, missing, corrupt, and unsupported images must all leave controls
reachable and expose existing textual recovery behavior.

## Focus, scrolling, and accessibility

- Passive local scroll viewports enter the Tab order only while overflowing,
  as today. The Workspaces Tree is the deliberate exception because it is an
  interactive content owner: it enters the Tab order whenever it has an
  enabled selectable node, whether or not it overflows.
- Wheel input scrolls the local section under the pointer until it reaches its
  boundary, then naturally hands off to the Context outer rail.
- Arrow, Page Up/Down, Home, and End retain the shared bounded-section
  behavior.
- Tree navigation retains Textual-native Up/Down, Space, Enter, and shifted
  ancestry/sibling semantics except for the hidden-root boundary guard defined
  above; the scoped Left/Right adapter follows the exact branch and leaf
  behavior without intercepting editable search keys.
- Outer Context scrolling must not write section-open preferences or workspace
  disclosure preferences.
- Ordinary title, marker, page, membership, image, and status updates preserve
  the Context outer offset. Reveal occurs only after deliberate selection,
  disclosure, or focus navigation; a reduced maximum clamps the existing
  offset without snapping to the active section.
- Responsive rail hiding still hands focus to the visible reveal control.
- Focus, active ownership, blocked states, and image recovery remain readable
  without color.

## Responsive behavior

Existing ADR-043 width floors, explicit rail intent, single-pane fallbacks, and
transcript minimums remain unchanged. This design changes vertical allocation
and rail framing, not responsive width ownership.

At short heights, Context outer scrolling is normal rather than an exceptional
header-only fallback. Every expanded section keeps its content ceiling or
natural shorter height, and every section/header remains reachable through the
outer viewport.

Inspector retains its existing independent 20-row local sections, outer hint,
navigation, and ownership classifier.

Phone/touch/hover/soft-keyboard behavior remains owned by TASK-18911 and is not
expanded by this terminal redesign.

## Performance requirements

Before implementation, capture current settled projection/render, search, and
selection timings. Repeat the same protocol after the Tree implementation.

The benchmark matrix is deterministic:

| Dataset | Named workspaces | Conversations per named workspace | Default/unassigned conversations | Expansion state | Search hit ratio |
| --- | ---: | ---: | ---: | --- | ---: |
| Small | 3 | 4 | 4 | active workspace expanded | 25% |
| Representative | 12 | 12 | 20 | active + 2 others expanded | 10% |
| Stress | 50 | 75 | 75 | active + 9 others expanded | 10% |

For each dataset and implementation, run three unreported warm-ups followed by
20 measured iterations of initial projection/mount, a non-structural marker
update affecting 5% of conversations, search apply/clear, and active-row
selection. Use the same Python/Textual version, terminal dimensions, generated
records, and machine. Report median and p95 wall time from `time.perf_counter`
plus reconcile/recompose counts. The report, fixtures, and raw measurements are
committed with the task evidence.

The completed design must:

- avoid a repeated recompose or reconcile loop;
- use incremental keyed node updates for non-structural changes;
- avoid querying or rebuilding conversations outside the changed projection;
- schedule at most one settled UI reconciliation for one logical keyed update;
  and
- document any stress-size trade-off rather than claiming an unmeasured speed
  improvement.

The benchmark report records both total service records and materialized Tree
nodes for each case so paging does not make two differently sized projections
look equivalent.

Because the new Tree exposes more rows than the old 12-row-per-group cap, timing
results are report-only rather than a brittle hard pass/fail comparison. A
representative median more than 20% slower must be investigated and either
corrected or explicitly accepted in the task notes before completion. No speed
claim may be made unless the measurements support it.

Performance is not a reason to change storage or introduce a custom virtualized
Tree.

## Error and empty states

- Workspace service unavailable: retain named recovery copy and disable only
  actions that cannot succeed.
- Search failure: preserve the last settled ordinary projection, show a
  section-local Retry target tied to the failed search generation, and never
  replace or clear the other section's search result/error state.
- Page load failure: preserve loaded children and replace the failed loading
  row with one keyboard-reachable Retry action; stale or duplicate page
  completions produce no visible mutation.
- No named workspaces: show a compact Workspaces empty state with the existing
  New workspace recovery path.
- Workspace with no conversations: keep the workspace visible and show an
  explicit empty child state only when expanded.
- No Default/unassigned conversations: show the flat Conversations empty state
  and point to Workspaces for named-workspace conversations.
- Search with no results: scope the message to the section whose search is
  active.
- Conversation moved between workspace owners: remove it from the old
  projection and add it to the new one atomically, preserving focus when the
  selected record remains reachable.

## Testing and evidence

Automated coverage must include:

1. Shell compositor containment and hit testing with both rails open, closed,
   and responsive across representative widths and heights.
2. No inset left/right workspace gutter and exactly one rail/transcript divider.
3. Exact natural/ceiling/overflow transitions for every 15-, 20-, and 35-row
   class, including separate local hints.
4. Context outer scrolling with many sections open, including reachability and
   offset clamping after collapse, mutation, and resize.
5. Workspaces Tree ownership, order, expansion persistence, temporary search
   disclosure, independent full-scope search, active markers, literal-safe
   labels, bounded paging/`Load more…`, keyboard/pointer selection, exact
   Left/Right branch/leaf behavior, editable-input exclusion, structural
   mutation, and focus recovery. This includes a fitting non-overflow Tree that
   remains Tab-reachable, hidden-root/top-level Left behavior, compact guide
   depth, partial-page Retry, duplicate/stale-page rejection, and cross-search
   interleavings. Search coverage includes settled-result preservation on
   failure, Retry generation replacement, stale failed-attempt rejection, and
   independence from page Retry state. Shifted ancestry/sibling navigation must
   also preserve a visible cursor at the hidden-root boundary.
6. Preservation of active-workspace identity, Switch/New/RAG Scope actions,
   and a working route to the built-in Default workspace when it has no
   conversation row. A narrow-width worst case combines a long active identity,
   the action strip, search, contextual Star row, and recovery copy while
   using Tree demand of at least eight rows and asserting a body no taller than
   20 rows and at least eight visible Tree rows; a one-node variant must hug its
   natural height. Typing `s` in either search must not trigger the Tree Star
   action.
7. Conversations filtering for Default/unassigned records and exclusion of
   named-workspace records, with starred-first ordering and no Starred
   aggregate or duplicate search result.
8. Atomic membership movement between the two projections with no duplicate or
   missing record.
9. Character contain geometry and control reachability for portrait,
   landscape, square, missing, and corrupt images, including equality-bounded
   reconciliation when scrollbar width changes.
10. Preservation of Inspector geometry/navigation and ADR-043 responsive width
   behavior.
11. Mutation-sensitive performance evidence for incremental node updates under
    the documented benchmark matrix.
12. Unicode, ASCII-glyph, and no-color Tree rendering plus stable compositor
    geometry for focused/unfocused rails, collapsed handles, transcript, and
    the single divider owners.

Live verification must include iTerm2 and Windows Terminal using equivalent
terminal row/column dimensions. Physical window pixel size alone is not an
acceptable comparison. The implementation plan must name the Windows Terminal
operator/environment before work begins; lack of that environment is an
explicit closeout blocker rather than a reason to waive the comparison.

## Architecture decision record

ADR required: **yes**.

Create a new ADR that supersedes only the affected clauses of:

- ADR-017's decision not to reorganize the left-rail sections and its blanket
  ban on rail glyphs. The exception is limited to Textual Tree disclosure and
  guide glyphs inside Workspaces; all other rail headers and controls remain
  text-labeled under ADR-017; and
- ADR-077's shared-budget Context allocator and its rejection of full-height
  expanded sections.

The new ADR must preserve ADR-077's local hint, bounded body, focus, ownership,
and Inspector contracts, and preserve ADR-043's responsive-width and preference
contracts. It must also record the thin Left/Right Tree adapter and the rule
that Tree, not a nested `VerticalScroll`, owns Workspaces' local scrolling.
The ADR explicitly carves the interactive Tree out of ADR-077's
overflow-only-focus rule while retaining that rule for passive scroll
viewports. It also records that Context alone accepts per-section 15/20/35
ceilings while Inspector remains fixed at 20.

## Scope boundaries

In scope:

- Console shell framing and edge ownership;
- Context section-specific height ceilings and outer scrolling;
- Workspace Tree projection and independent search;
- Default/unassigned flat Conversations projection and independent search;
- Character image contain behavior and 35-row ceiling;
- relevant documentation, ADR, tests, and generated CSS.

Out of scope:

- database schema or membership migration;
- moving staged Sources out of Inspector;
- Inspector section reordering or new Inspector ownership;
- responsive width-floor changes;
- phone/touch work owned by TASK-18911;
- a replacement/custom-rendered Tree or new dependency (the narrow subclass or
  key handler over Textual's native Tree is explicitly in scope); and
- Ctrl+K session-switcher activity ranking.

## Follow-up: Ctrl+K active conversation view

File a separate Backlog task after this design task. It should add an activity-
focused sort/filter or group to the Ctrl+K session switcher while retaining
historical-date browsing. Before implementation it must define whether
"active" means an open tab, currently running work, recent activity, or a
ranked combination. That decision affects session state and ordering but not
the rail projections in this design.
