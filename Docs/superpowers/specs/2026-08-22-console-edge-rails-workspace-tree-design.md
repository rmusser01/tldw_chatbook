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

The Workspaces section displays every named, non-default workspace as a root
node. Each workspace's associated conversations are direct child nodes.
Default-workspace and unassigned conversations are excluded so no conversation
appears in both sections.

This is a display projection over existing workspace IDs and conversation
records. It does not create a new database relation, migrate records, or change
the meaning of the built-in Default workspace.

### Interaction

- Workspace roots expand and collapse with Textual Tree's native pointer
  behavior and a narrowly scoped Console key adapter. This is a subclass or
  handler over Textual's shipped Tree, not a replacement Tree implementation.
- Workspace selection and disclosure are separate actions. Selecting a
  workspace label with pointer or Enter switches to that workspace without
  changing its disclosure state. Textual's label-selection auto-expand is
  disabled so one gesture cannot perform both actions.
- Pointer disclosure and Space retain Textual's native toggle behavior. The
  adapter adds conventional plain Left/Right behavior only while the Tree owns
  focus: Left collapses an expanded branch, otherwise moves to its parent;
  Right expands a collapsed branch, otherwise moves to its first child; Right
  on a leaf is a no-op. It does not intercept keys while either section search
  input or any other editable control owns focus.
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

### Preserved workspace controls

The Workspaces section retains the current active-workspace identity plus
Switch, New, and RAG Scope actions above its search and Tree. They remain part
of the section's measured 20-row body rather than becoming global chrome.

- Switch remains the authoritative route to every workspace, including the
  built-in Default workspace that is intentionally omitted from the Tree.
- New retains its existing workspace-creation behavior and recovery state.
- RAG Scope continues to reflect and act on the active workspace, including
  its existing enabled/disabled and recovery semantics.
- Switching to Default leaves the Workspaces Tree visible for later switching
  while the flat Conversations section becomes the owner of Default records.

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
The shared bounded-section integration reads the Tree's desired/virtual height,
controls its allocated viewport height, and renders the standard local hint as
a sibling outside the Tree. Wheel handoff at the Tree boundary bubbles to the
Context outer rail.

## Conversations browser

Conversations remains a flat browser and displays only:

- conversations assigned to the built-in Default workspace; and
- conversations with no workspace association.

It owns an independent search input that matches only this flat projection.
Current row selection, title wrapping, status details, run markers, starring,
result limits, empty/error copy, and resume behavior remain unless a direct
conflict with the new ownership filter is proven.

The empty state explains that conversations belonging to named workspaces are
available under Workspaces. It must not imply deletion or missing data.

## Character section

Character receives a 35-content-row ceiling for the complete local body,
including the image, character name, reaction state, reaction action, and any
layout spacing between them. Its image:

- preserves the complete source image;
- preserves its aspect ratio;
- scales to the largest size that fits the available rail width and the rows
  remaining after the mounted non-image controls are measured;
- is centered in unused horizontal or vertical space; and
- is never cropped or stretched.

When a valid image is present, the section reserves the measured rows required
by the name, reaction state, reaction action, and their real margins before
fitting the image. The image may grow only until the complete body reaches 35
rows, so those controls remain inside the initial 35-row viewport. The fit is
recomputed when rail width, terminal height, image, character metadata, or
control geometry changes; it must not assume a hard-coded control-row count.

Letterboxing is acceptable and preferred to information loss or distortion.
Character metadata and reaction controls remain below the image in the same
local scroll owner. A missing image leaves the short textual body at its natural
height rather than claiming 35 empty rows. Portrait, landscape, square, very
large, missing, corrupt, and unsupported images must all leave controls
reachable and expose existing textual recovery behavior.

## Focus, scrolling, and accessibility

- Only overflowing local viewports enter the Tab order, as today.
- Wheel input scrolls the local section under the pointer until it reaches its
  boundary, then naturally hands off to the Context outer rail.
- Arrow, Page Up/Down, Home, and End retain the shared bounded-section
  behavior.
- Tree navigation retains Textual-native Up/Down, Space, Enter, and shifted
  ancestry semantics; the scoped Left/Right adapter follows the exact branch
  and leaf behavior defined above without intercepting editable search keys.
- Outer Context scrolling must not write section-open preferences or workspace
  disclosure preferences.
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
   mutation, and focus recovery.
6. Preservation of active-workspace identity, Switch/New/RAG Scope actions,
   and a working route to the built-in Default workspace when it has no
   conversation row.
7. Conversations filtering for Default/unassigned records and exclusion of
   named-workspace records.
8. Atomic membership movement between the two projections with no duplicate or
   missing record.
9. Character contain geometry and control reachability for portrait,
   landscape, square, missing, and corrupt images.
10. Preservation of Inspector geometry/navigation and ADR-043 responsive width
   behavior.
11. Mutation-sensitive performance evidence for incremental node updates under
    the documented benchmark matrix.

Live verification must include iTerm2 and Windows Terminal using equivalent
terminal row/column dimensions. Physical window pixel size alone is not an
acceptable comparison.

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
