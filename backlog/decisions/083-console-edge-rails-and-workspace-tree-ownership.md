# ADR-083: Console edge rails and workspace-owned conversation Tree

Status: Accepted
Date: 2026-08-22
Related Task: [TASK-20937](../tasks/task-20937%20-%20Make-Console-rails-edge-native-and-organize-conversations-by-workspace.md)
Related Spec: [Console edge rails and workspace Tree design](../../Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md)
Amends: ADR-017, ADR-077
Preserves: ADR-043

## Decision

The Console keeps its full-width global header and renders the work surface
immediately beneath it as three edge-owned regions: Context at the application
left edge, transcript/composer in the center, and Inspect at the right edge.
The workspace grid no longer adds horizontal inset or a surrounding side frame.
Each rail/transcript boundary has one dimensionally stable divider owner;
focused and collapsed states paint that existing geometry rather than adding a
second border.

Context sections use per-section natural-height ceilings measured in rendered
terminal content rows: Sessions, Model, Agent, and Details use 15; Workspaces
and Conversations use 20; Character uses 35. Overflow receives the existing
separate `▼ more — scroll` hint row. Context no longer divides one shared
content budget to keep all headers simultaneously visible and no longer exposes
`· no room` or `[>]` reprioritization for ordinary height pressure. Complete
expanded sections retain their natural or capped height, and the Context outer
body scrolls with `▼ more sections — scroll` when their sum exceeds the rail.
Inspector remains fixed at ADR-077's independent 20-row contract.

Workspaces owns one native Textual `Tree` with a hidden, non-selectable
synthetic root and top-level nodes for every visible/selectable, non-retired
named workspace. Associated conversations are child nodes. The built-in
Default workspace is intentionally absent; Default and unassigned conversations
live in the separate flat Conversations section. A conversation appears in
exactly one ordinary or search projection. Starred is a property, marker, and
action—not a third location—and sorts first within the conversation's one owner
before recency ordering.

The Tree remains a native Textual scroll owner, uses stable workspace and
conversation IDs, compact two-cell guides, literal Rich `Text` labels, explicit
ASCII glyph fallbacks, bounded service pages, and keyed incremental updates.
It is not nested inside another local `VerticalScroll`. A thin Console adapter
disables label auto-expand, separates selection from disclosure, adds guarded
plain Left/Right branch navigation, and prevents native shifted navigation from
moving onto the hidden root. Search-time disclosure is temporary and restores
the exact persisted pre-search snapshot on clear.

The Workspaces Tree is the deliberate exception to ADR-077's
overflow-only-focus rule for passive scroll viewports: whenever it contains an
enabled selectable node it is keyboard-focusable even when all nodes fit.
Overflow still controls its local hint and wheel handoff. Switch, New, and RAG
Scope remain a compact pinned strip, and Switch remains the route to Default.

Workspaces and Conversations searches own separate query, debounce, generation,
cache, result/error, Retry, and worker state. Paging owns an independent attempt
generation per workspace. Every async commit validates its projection,
generation, membership/cursor snapshot, and mounted owner. Late or duplicate
search/page results cannot overwrite the other projection, move focus, or
persist temporary disclosure.

Character's 35 rows cover the complete local body, including image, name,
reaction state/action, and real spacing. The image uses aspect-ratio-preserving
contain behavior over the rows left after mounted controls are measured. An
equality-guarded two-phase settle reuses the existing graphics and mosaic fit
utilities and allows at most one geometry follow-up, preventing scrollbar/image
resize oscillation.

Existing workspace/conversation storage, Default-workspace meaning, rail-open
preferences, responsive widths, explicit-toggle authority, focus handoff, and
session-local scroll-offset rules remain unchanged.

## Amendment (2026-08-24): deliberate activation, global layout, and pinned authority

UAT of the production-shaped Context rail found that activating a section could
move the Tree between mouse-down and Textual's final coordinate-derived click,
retargeting a workspace press to a neighboring conversation. The same review
found that per-workspace disclosure layouts silently changed Context's spatial
arrangement during navigation and that Inspect buried next-send authority below
duplicated telemetry.

Tree selection and activation are now deliberately separate. A single pointer
click selects a row and expands a collapsed workspace without activating it. A
rapid double-click, using Textual's native click chain, or Enter activates the
selected workspace or conversation. Space and disclosure-glyph gestures toggle
workspace disclosure; Left and Right retain branch navigation. The node's stable
key at press time owns the entire gesture, so rail reconciliation cannot retarget
it. Both clicks in an activation chain must resolve that same still-selected
key; a coordinate that moves over another row cancels activation. Full-label
tooltips exist only for actually truncated rows and are cleared or recomputed
after reflow. A focused Tree context row plus contextual F1 help exposes the
complete selected label and activation grammar without pointer hover.

Rail disclosure layout is global by default. Console Behavior offers an
explicit per-workspace mode; existing workspace-scoped records remain stored and
become authoritative again when that mode is selected. A missing global record
is seeded once from the active workspace's effective saved layout, otherwise
product defaults apply. Scope changes never persist transient scroll, focus,
search disclosure, Tree selection, or tooltip state and never delete the
inactive scope's records. Responsive compact-collapse remains a rendering
override rather than a preference mutation.

The shared record uses the collision-safe reserved key
`console_rail_state:global:shared-layout-v1`; it does not reuse the current
Default/unscoped `console_rail_state:global:layout` key. A workspace without a
per-workspace record seeds once from the effective shared layout, or product
defaults when none exists. Existing Default, named-workspace, legacy, and
shared records remain lossless and independent. Preference pruning retains the
reserved `shared-layout-v1` scope alongside the existing `layout` and legacy
`global` scopes.

Inspect pins a compact `What happens if I send now?` authority summary above its
outer scroll owner. One existing atomic display snapshot supplies workspace and
conversation identity, next-send scope, run state, staged-source count, and
pending-approval state/count in one heading plus five single-line rows. Lower
groups do not repeat the same facts. Empty Tools, Approvals, and Artifacts groups
live under one More boundary and promote in fixed Tools/Approvals/Artifacts order
whenever actionable or nonzero. More defaults collapsed, follows the selected
layout scope, supports click/Enter/Space plus Left/Right, and owns deterministic
focus recovery when a focused group demotes: preserve a still-mounted focusable
descendant, otherwise use the visible demoted header, then More's disclosure
control.

This amendment changes the earlier clause that preserved the existing
per-workspace rail-layout preference behavior and supersedes the original Tree
interaction statement that pointer label selection or Enter immediately
switched workspaces. Edge ownership, section ceilings, workspace/conversation
data ownership, responsive width authority, and session-local scroll-offset
rules remain unchanged.

## Context

The current Console visually nests both rails inside a framed workspace, so
Context and Inspect read as inset panels rather than application edges. Its
Context allocator also shrinks open bodies according to the terminal's reported
height; iTerm2 can therefore show fewer rows per section than Windows Terminal
at a similar physical window size. Workspaces currently shows only active
identity/actions while Conversations owns a grouped all-workspaces browser,
which contradicts the user's ownership model and duplicates starred records.
Character art is width-sized independently of the controls and cannot use the
larger roleplay viewport safely.

ADR-017 deliberately kept the earlier rail text-only and structurally stable;
this design now needs a workspace Tree and a different ownership hierarchy.
ADR-077 deliberately kept all Context headers visible by distributing a shared
height budget; the product decision now prefers full natural section viewports
plus ordinary outer scrolling. Those clauses are superseded narrowly. The
remaining ADR-017 text labels and ADR-077 local hints, physical-row measurement,
focus recovery, offset continuity, mutation reconciliation, Inspector ownership,
and nested-scroll handoff continue to govern.

## Alternatives Considered

| Alternative | Why rejected |
| --- | --- |
| Keep Context's shared height allocator | Reproduces emulator-dependent undersized bodies and conflicts with the approved per-section reading ceilings. |
| Put the workspace Tree inside a second bounded `VerticalScroll` | Creates competing local scroll owners, ambiguous wheel handoff, and duplicate focus behavior. |
| Build a custom Tree or add a dependency | Textual 8.2.8 already provides the required node, scrolling, and keyed-update primitives; only a narrow behavior adapter is needed. |
| Keep a global Starred group | Duplicates records across owners and makes the new workspace mental model false. |
| Put Default in the workspace Tree | Hides the requested scratch/default conversation list inside a hierarchy and weakens the separate Conversations purpose. |
| Load every conversation synchronously | Large workspace sets would make mount/search latency and memory scale without a bound. |
| Crop Character images to fill the box | Loses user-provided roleplay art and violates the complete-image requirement. |
| Make every bounded viewport focusable | Adds passive, non-overflow scroll containers to Tab order; only the interactive Tree needs the exception. |

## Consequences

- Context can be much taller than the terminal when many sections are open;
  outer scrolling is the normal reachability path, not an exceptional fallback.
- The shared bounded-section primitive accepts Context ceilings of 15, 20, or
  35 while Inspector remains fixed at 20 and regression-tested separately.
- The old Context allocation/no-room policy is retired rather than extended.
- Tree focus, hidden-root boundaries, temporary search disclosure, page Retry,
  and ASCII glyphs become explicit Console contracts.
- Star actions need a contextual Tree affordance because native Tree labels do
  not embed the current per-row Button.
- Search and paging state become more explicit, but stale async results can no
  longer cross projections or membership generations.
- Character layout requires measured post-mount geometry but is bounded to one
  equality-guarded follow-up and introduces no new image fitter.
- Production verification must compare equal row/column dimensions in iTerm2
  and Windows Terminal. Missing Windows Terminal access blocks closeout rather
  than weakening the evidence requirement.
- Source TCSS and the generated CSS bundle must remain canonical and identical.

## Amendment (2026-08-22, TASK-20937.4 — identity-preserving Tree moves)

Textual 8.2.8 exposes public node addition and removal but no public node move
or reparent operation. Public `TreeNode.remove()` recursively deletes the node
from `Tree._tree_nodes`; adding it again creates a different `TreeNode`, so a
public remove/add reorder cannot preserve node object identity, cursor identity,
or the incremental keyed-update contract above.

TASK-20937.4 therefore permits one isolated, version-pinned private helper in
the thin Console Tree adapter. The helper must fail closed unless the runtime is
exactly Textual 8.2.8 and the source and destination nodes expose the expected
`_children` and `_parent` attributes while the Tree exposes `_tree_nodes` and
`_invalidate`. It detaches the existing node from its current parent's
`_children`, inserts that same object into the destination parent's
`_children`, updates only that node's `_parent`, retains the existing
`_tree_nodes` registration, and calls `_invalidate()` exactly once after a
successful mutation. It does not clear/reset the Tree, recreate the node, or
become a general Tree/rendering framework. Public removal remains the only path
for true deletion.

Exact-version, private-shape, same-parent reorder, cross-parent move,
`_tree_nodes`, cursor/data/object-identity, and single-invalidation tests pin
this exception. A Textual upgrade must either provide a public identity-
preserving move API or explicitly re-audit and update this compatibility check;
until then structural moves fail closed rather than silently corrupting native
Tree state.

The rejected alternative is public remove/add: it uses supported APIs but
necessarily loses node identity and cursor continuity in Textual 8.2.8. A
custom Tree or generalized private rendering layer remains rejected.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md)
- [ADR-017: Console left-rail usability redesign](017-console-left-rail-usability.md)
- [ADR-043: Console rail compact collapse yields to explicit toggles](043-console-rail-compact-collapse-yields-to-explicit-toggle.md)
- [ADR-077: Bound Console rail sections and expose hidden overflow](077-console-bounded-rail-section-scrolling.md)
- [TASK-20937](../tasks/task-20937%20-%20Make-Console-rails-edge-native-and-organize-conversations-by-workspace.md)
