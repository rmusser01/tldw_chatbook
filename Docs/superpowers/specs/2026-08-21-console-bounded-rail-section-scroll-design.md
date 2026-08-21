# Console Bounded Rail-Section Scrolling Design

**Task:** TASK-19428

**Decision:** ADR-077

**Status:** Approved by the user on 2026-08-21

## Summary

Bound every direct named section body in the Console Context and Inspector rails to
20 rendered content lines. Short sections use their natural height. Longer sections
scroll internally and expose a separate one-line `▼ more — scroll` affordance while
content remains below. Context adapts the ceiling downward when necessary to keep all
of its section headers visible; Inspector retains its outer scroll and gains a separate
outer fold hint for entire sections below the viewport.

The primary usage target is a maximized, browser-like terminal. The design therefore
optimizes 235x52 and 160x45 without abandoning the established 120x30 and 80x24
responsive contracts.

## Users and outcomes

The design is evaluated for four personas:

1. A first-time technical user must recognize that a dense source, run, or settings
   section continues without inferring it from clipped text.
2. A first-time non-technical user must see a literal scrolling instruction and keep
   every Context category discoverable.
3. A regular technical power user must move quickly among independently scrollable
   sections without losing keyboard access, focus, or run state.
4. A regular non-technical user must retain predictable section heights and avoid one
   verbose section displacing the rest of the rail.

Success means all four personas can scan section headers, read up to 20 content lines
without an unnecessary scroll, discover additional content, and continue through a
rail without becoming trapped in a nested viewport.

## Protected contracts

This change preserves the following accepted behavior:

- Staged Sources remain first in Inspector and above Source Readiness.
- Scope remains a compact sibling row, not a new section or source picker.
- Context direct section headers remain visible at supported terminal heights, as
  required by TASK-15110.
- Context and Inspector rail labels, priorities, stored preferences, 70/74-column
  explicit-open floors, responsive focus handoff, and exact-100 geometry remain as
  defined by ADR-043 and TASK-19427.
- Collapse/reopen semantics, rail badges, action routing, row IDs, selected-message
  state, and in-place Inspector updates remain intact.
- Phone, hover, soft-keyboard, and served-browser ownership remains outside this
  terminal-specific task.

## Scope

### Context direct sections

The bounded-body contract applies to these direct Context sections:

- Sessions
- Workspaces
- Conversations
- Model
- Agent
- Details
- Character, when present

Nested components inside a direct body, such as the Agent fleet subsection, remain
ordinary content of that body. They do not become another nested scroll owner.

### Inspector direct sections

The contract applies to these named Inspector sections when present:

- Sources
- Changed Files
- Run
- Source Readiness
- Tools
- Approvals
- Artifacts
- Selected Conversation
- Session Defaults
- Selected Message
- Chat Dictionaries
- World Books
- Session Settings
- Live Work / live-work Source Readiness

The run-status summary and Scope row remain compact non-section rows. Disabled controls
with zero rendered height do not consume a content line.

## Non-goals

- Reordering Inspector content or changing Sources ownership.
- Renaming Context, Inspector, Scope, RAG, Sources, or other product vocabulary.
- Adding a Console-native source picker.
- Changing section open-state persistence.
- Persisting scroll offsets across application sessions.
- Redesigning the rail visual language, ASCII collapse controls, badges, or colors.
- Adding more nested scroll owners below a direct rail section.

## Visual and sizing contract

`MAX_SECTION_CONTENT_LINES` is 20.

For a direct section with desired rendered content height `D` and allocated content
height `A`:

- `D == 0`: the body occupies zero content rows and has no hint.
- `1 <= D <= A`: the body occupies `D` rows and has no scrollbar or hint.
- `D > A`: the body occupies `A` scrollable content rows and reserves one additional
  hint row below it.
- A header is always outside `A` and outside the scroll viewport.
- The hint never overlays, covers, or replaces a content row.
- Wrapped terminal rows count physically. A logical item that wraps to three screen
  rows consumes three of the 20 lines.

Inspector direct sections normally receive `A = min(D, 20)`. Context uses the
adaptive allocation below. A section never reserves blank rows to reach 20.

The hint copy is exactly:

```text
▼ more — scroll
```

It uses existing semantic surface/text tokens and the established one-row fold-hint
grammar. Color is not the only continuation signal.

## Shared bounded-section body

A reusable Console widget owns the common behavior. Its public inputs are:

- a stable `section_id` used to derive body and hint IDs;
- already-built content children;
- `max_content_lines`, fixed at 20 by the rail contract;
- an optional current allocation below 20 for Context;
- the outer scroll owner used for boundary handoff.

It renders:

1. a `VerticalScroll` content viewport;
2. a non-focusable, always-mounted fold-hint `Static` beneath it.

The component has no database, provider, task, conversation, or preference access.
Existing widgets keep ownership of headers, data, controls, actions, ordering, and
sync timing. The bounded body owns only layout and transient scroll presentation.

Fold state is derived from laid-out geometry:

```text
has_more_below = max_scroll_y > 0 and scroll_y < max_scroll_y
```

The component schedules one post-refresh reconciliation after mount, resize, content
structure changes, open/close, and allocation changes. Scroll-position changes update
the hint without recomposing content. A missing body or hint during recompose fails
closed and is retried on the next scheduled reconciliation.

## Context height allocation

Context must preserve all direct headers while using available expanded-terminal
height efficiently. Allocation is a pure deterministic function of:

- the Context body viewport height;
- measured visible header and inter-section chrome height;
- each direct section's open state;
- each open section's desired rendered content height, capped at 20.

The allocator follows these rules:

1. Reserve every visible direct header and required fixed inter-section chrome.
2. Closed or empty section bodies receive zero rows and no hint.
3. Give short sections only the rows they request.
4. Distribute remaining content rows among longer sections by progressive water
   filling, never exceeding 20 content rows for a section.
5. An allocation smaller than a section's desired height also reserves one hint row.
6. Re-run the distribution when hint-row requirements change, until the allocation is
   stable.
7. At supported heights, every open non-empty section receives at least one content
   row plus its hint before any longer section receives additional rows.
8. Deterministic DOM order breaks allocation ties.

This replaces the fixed `max-height: 20%` rule. Short open sections return unused rows
to the pool, allowing another open section to approach the 20-line ceiling. At smaller
heights, sections scroll sooner so later headers remain visible.

## Inspector structure

The Inspector outer rail remains:

1. its fixed collapse header;
2. the scrollable Inspector body containing direct sections in existing order;
3. a separate pinned outer fold hint.

The outer hint uses the outer body's own `scroll_y` and `max_scroll_y`. It communicates
that complete rows or sections remain below the rail viewport and is independent of
all per-section hints.

`ConsoleRunInspector` currently emits flat headings followed by rows and actions. It
will group each existing `_ROW_GROUPS` entry into a stable heading plus bounded body.
Dictionary and World Book groups use the same structure. The structural fingerprint
continues to be based on the same row IDs and action definitions, and in-place updates
continue to query the same row IDs; grouping must not force recompose for text/status
only changes.

Specialized top-level widgets such as Sources, Changed Files, Session Settings, and
live-work cards retain their header and business-state seams. Their content regions
adopt the shared bounded body without moving or duplicating their header.

## Interaction contract

### Focus

- A section viewport joins focus order only while it overflows.
- A short, empty, closed, or absent body adds no keyboard stop.
- The hint is never focusable.
- Focus styling uses the existing Console focus tokens without changing dimensions.
- If a focused section stops overflowing, focus moves to that section's visible
  header or nearest valid rail control rather than disappearing.
- Responsive hiding continues to hand focus to the appropriate rail reveal control.

### Keyboard scrolling

While an overflowing body itself has focus:

- Up/Down scroll one rendered line.
- Page Up/Page Down scroll one viewport.
- Home/End scroll to that section's first/last line.

Existing child controls keep their established input behavior. No terminal-convention
Ctrl binding or app-global binding is added or shadowed.

### Pointer scrolling and boundary handoff

Pointer-wheel input over an overflowing body scrolls that body while movement remains
in the requested direction. At its top or bottom boundary, continued movement is
offered to the outer rail so nested scrolling cannot trap the user. Pointer events
over a short body go directly to the outer rail.

### State continuity

Section scroll offset survives in-place state synchronization and rail
collapse/reopen. When content shrinks or allocation decreases, Textual clamps the
offset to the new valid maximum and fold state reconciles afterward. No scroll offset
is persisted to disk or included in rail preference saves.

## Responsive behavior

The expanded primary states are 235x52 and 160x45. At these sizes, short sections hug
content and longer sections use as much of the 20-line allowance as the rail's
allocation contract permits.

At 120x30 and 80x24, existing width-responsive rail behavior still decides whether a
rail is visible. When a rail is visible, bounded sections adapt to the actual height.
No width breakpoint, rail priority, handle label, stored intent, or exact-100 minimum
waiver changes in this task.

## Error and race handling

- Layout-dependent state is never computed from zero-size pre-layout regions.
- Recompose-time missing selectors cause a no-op, not an exception.
- Multiple mount/resize/sync requests coalesce into bounded post-refresh work.
- Content shrink clamps scroll offsets before hint visibility is asserted.
- An allocator receiving no usable body rows prioritizes headers and returns a
  deterministic zero allocation rather than negative dimensions.
- A section becoming hidden or collapsed cannot leave an invisible focused viewport.
- No background worker or database read is introduced for layout reconciliation.

## Testing strategy

### Pure tests

- Context allocator: empty, short, long, mixed, all-open, collapsed, tie, and
  insufficient-budget cases.
- Exact 20/21 rendered-line boundary.
- Deterministic redistribution and hint-row accounting.

### Isolated widget tests

- Natural height with 0, 1, and 20 content rows.
- Exactly 20 rows: no scroll or hint.
- 21 rows: 20-row viewport plus one separate hint row.
- Hint visible before the end, hidden at the end, and visible again after scrolling up.
- Content shrink clamps the offset and removes obsolete overflow.
- Focusability toggles with overflow and recovers when overflow disappears.
- Keyboard scrolling and pointer boundary handoff.
- Recompose and in-place update paths preserve stable child and row IDs.

### Production-CSS compositor tests

At 235x52, 160x45, 120x30, and 80x24:

- every visible Context header remains within the Context viewport;
- Context short sections do not waste height and long sections receive redistributed
  rows up to 20;
- Inspector and Context section bodies never exceed their allocation;
- the 20 content rows and separate hint row do not overlap siblings;
- the Inspector outer hint remains pinned and tracks outer overflow;
- collapse/reopen, focus handoff, rail badges, stored preference call counts, and
  responsive geometry remain unchanged.

### Regression scope

Run the existing focused Context/Inspector rail, section, compact-access, resize,
geometry, changed-files, staged-context, run-inspector, and CSS-bundle suites. Do not
substitute the repository-wide suite for these behavior-specific tests.

## Migration and documentation

- Replace only the fixed Context `20%` body-cap CSS after the allocator is proven red
  and green.
- Keep old stable IDs or provide explicit compatibility assertions wherever a wrapper
  changes nesting.
- Update the Console user guide to explain the 20-line section ceiling, local hints,
  and outer Inspector hint.
- Record the task-ID collision correction: the Console Phase 0 stream moves from the
  conflicting TASK-18912/TASK-18913/TASK-18915 IDs to
  TASK-19426/TASK-19427/TASK-19428.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/077-console-bounded-rail-section-scrolling.md`

Reason: the change establishes a long-lived cross-rail layout, scroll-ownership,
focus, pointer, and keyboard interaction contract and replaces the fixed Context body
allocation rule.
