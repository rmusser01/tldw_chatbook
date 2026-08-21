# Console Bounded Rail-Section Scrolling Design

**Task:** TASK-19428

**Decision:** ADR-077

**Status:** Approved by the user on 2026-08-21

## Summary

Bound every direct named section body in the Console Context and Inspector rails to
20 rendered content lines. Short sections use their natural height. Longer sections
scroll internally and expose a separate one-line `▼ more — scroll` affordance while
content remains below. Context adapts the ceiling downward while its headers fit; at
heights where the headers themselves cannot fit, its outer body becomes the fallback
scroll owner. Inspector retains its outer scroll and gains a distinct outer fold hint
for content below the viewport.

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
- Context direct section headers remain simultaneously visible whenever their fixed
  chrome fits the rail body, preserving TASK-15110. When it cannot physically fit,
  every header remains reachable through an explicitly signaled outer Context scroll.
- Context and Inspector rail labels, priorities, stored preferences, 70/74-column
  explicit-open floors, responsive focus handoff, and exact-100 geometry remain as
  defined by ADR-043 and TASK-19639.
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
- Live Work or Live work sources (the mutually exclusive final card)
- Changes, when the `Review Changes` action is visible

The run-status summary and Scope row remain compact non-section rows. Disabled controls
with zero rendered height do not consume a content line.

## Non-goals

- Reordering Inspector content or changing Sources ownership.
- Renaming Context, Inspector, Scope, RAG, Sources, or other product vocabulary.
- Adding a Console-native source picker.
- Changing section open-state persistence.
- Persisting scroll offsets across application sessions.
- Redesigning the rail visual language, badges, colors, or ordinary collapse controls;
  the constrained Context `[>]` state is the sole added ASCII control state.
- Adding more nested scroll owners below a direct rail section.
- Exposing unowned Inspector content through a generic user-facing catch-all section.

## Visual and sizing contract

`MAX_SECTION_CONTENT_LINES` is 20.

For a direct section with uncapped desired rendered content height `D` and allocated
content viewport height `A`:

- `D == 0`: the body occupies zero content rows and has no hint.
- `1 <= D <= A`: the body occupies `D` rows and has no scrollbar or hint.
- `D > A > 0`: the body occupies `A` scrollable content rows and reserves one
  additional hint row below it.
- `A == 0`: the body and hint occupy zero rows even when `D > 0`; normal Context mode
  marks its visible header `· no room` and exposes transient `[>]` reprioritization.
- A header is always outside `A` and outside the scroll viewport.
- The hint never overlays, covers, or replaces a content row.
- Wrapped terminal rows count physically. A logical item that wraps to three screen
  rows consumes three of the 20 lines.

`A` is the laid-out `VerticalScroll.content_region.height`. `D` is the corresponding
uncapped physical content height before viewport clipping. Vertical child margins and
padding laid out inside that viewport count in `D`; the direct header, the hint slot,
and margins outside the viewport are fixed chrome and do not. The migration removes
the existing Context body bottom padding and Inspector heading top margin from the
scrollable body box, so those legacy decorations cannot consume an invisible 21st
content row.

While `D > A`, the hint slot remains exactly one row high. Its text is visible while
more content remains below and becomes blank/visually hidden at scroll end without
collapsing the slot, so reaching the end cannot shift following content. When `D <= A`
there is no overflow and the entire slot is removed from layout.

Inspector direct sections normally receive `A = min(D, 20)`. Context uses the
adaptive allocation below. A section never reserves blank rows to reach 20.

The local section hint copy is exactly:

```text
▼ more — scroll
```

It uses existing semantic surface/text tokens and the established one-row fold-hint
grammar. Color is not the only continuation signal.

The pinned outer-rail hint copy is exactly:

```text
▼ more sections — scroll
```

The distinct noun differentiates rail scrolling from local section scrolling while
remaining one line inside the 26-cell narrow Context content width. The outer cue is
used by Inspector whenever its body overflows and by Context only in the short-height
fallback mode. When a local viewport or one of its descendants owns focus, its header
and viewport receive the same dimensionally stable active treatment: existing focus
background plus underlined header text. When the outer body owns focus, the rail title
is underlined instead. These non-color signals make the active scroll owner visible.

Outer overflow is measured counterfactually so the hint cannot create its own overflow.
Let `R` be the outer body viewport height if the hint slot were absent, and `D_outer`
the laid-out outer virtual content height. The slot exists iff `D_outer > R`; when it
exists, the actual outer viewport is `R - 1`. Measurement never uses the already reduced
`R - 1` viewport to decide whether the slot should remain. Thus a 10-row viewport follows
`10 rows -> no slot`, `11 rows -> slot`, `10 rows -> slot removed` without a sticky
feedback state.

## Shared bounded-section body

A reusable Console widget owns the common behavior. Its public inputs are:

- a stable `section_id` used to derive body and hint IDs;
- already-built content children;
- `max_content_lines`, fixed at 20 by the rail contract;
- an optional current allocation below 20 for Context;
- the outer scroll owner used for boundary handoff;
- an owner-supplied focus-recovery callback.

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

The component exposes an idempotent `request_reconcile()` seam that coalesces requests
into one post-refresh reconciliation. It runs after mount, resize, content structure
changes, open/close, and allocation changes. Scroll-position changes update the hint
without recomposing content. A missing body or hint during recompose fails closed and
is retried on the next scheduled reconciliation.

Context allocation is coordinated above the shared bodies. `ConsoleLeftRail` owns an
idempotent `request_allocation_reconcile()` that coalesces same-tick invalidations.
After refresh, it snapshots the outer content-region height, every mounted direct
header/chrome height, all open states, every uncapped desired height `D`, and transient
active-section state. It runs the pure allocator once and applies the complete set of
body allocations atomically; an individual Context body never applies a sibling-local
allocation from a partial snapshot. Each Context body `request_reconcile()` invalidates
this coordinator after updating its own desired geometry.

Textual descendant Mount, Unmount, and Resize events do not bubble, so implicit
ancestor resize observation is not sufficient. Each owning mutation path must call
`request_reconcile()` after its existing state update:

- Context workspace trays, Conversations, Model/settings rows, Agent rows/actions,
  Details, Character remounts, pinned Agent fleet-summary visibility, outer-body
  resize, and section toggles;
- Inspector Sources, Changed Files, Run groups/actions, dictionaries, World Books,
  Session Settings, and live-work card swaps.

Every Inspector section reconciliation and every Inspector outer-body/rail resize also
invalidates the outer Inspector fold state through the supplied outer owner. This
preserves one coalesced refresh boundary even when multiple sections update in the same
synchronization tick or terminal resize.

## Context height allocation

Context preserves all direct headers simultaneously when their fixed chrome fits the
available body height. Allocation is a pure deterministic function of:

- the Context body viewport height;
- measured visible header and inter-section chrome height;
- each direct section's open state;
- each open section's uncapped desired rendered content height `D`;
- the session-local most recently activated section ID.

### Normal header-fit mode

When the complete header chrome fits the Context body, its outer body does not scroll.
The allocator follows these rules:

1. Reserve every visible direct header and required fixed inter-section chrome.
2. Closed or empty section bodies receive zero rows and no hint.
3. Consider the most recently activated open section first, then the remaining open
   sections in DOM order. Fund a base allocation only while the budget can represent
   it honestly: one row for `D == 1`, or one content row plus one hint row for `D > 1`.
4. After the base pass, distribute remaining rows by progressive water filling, never
   exceeding 20 content rows. The activated section wins the first tie; DOM order
   breaks all remaining ties.
5. Derive hint cost from the uncapped predicate `D > A`, not from a value already
   capped at 20. Thus `D == 20, A == 20` has no hint, while `D == 21, A == 20` does.
6. When increasing `A` makes `A == D <= 20`, release that section's hint row back to
   the pool and re-run distribution until stable.
7. An open non-empty section that cannot receive its honest base allocation gets
   `A = 0`, no hint, the header suffix `· no room`, and the constrained toggle `[>]`.
   Activating `[>]` changes only the transient allocation priority, immediately
   recomputes, and does not close the section or alter its persisted open preference.
   Once funded, the ordinary open-state toggle returns. This makes zero allocation a
   visible, recoverable constraint rather than an empty-body failure.
8. The stable DOM order is Sessions, Workspaces, Conversations, Model, Agent, Details,
   Character.

### Short-height outer-scroll fallback

When fixed header chrome alone exceeds the Context body viewport, normal mode is
mathematically impossible. Context switches to a height-only outer-scroll fallback:

1. The Context outer body becomes scrollable and reserves the distinct pinned
   `▼ more sections — scroll` slot while outer content remains below.
2. Every open non-empty section receives an honest base allocation: one row for
   `D == 1`, or one content row plus its local hint for `D > 1`. No open section gets a
   zero-height body in this mode.
3. The most recently activated section's **total** content allocation is
   `A = min(D, 20, max(1, H - 3))`, where `H` is the outer Context body viewport height
   and three rows reserve its two-row header plus a possible local hint. Other sections
   remain at their base allocation so one active section is readable without making
   the short rail unbounded. If `D <= A`, the unused hint row is released normally.
4. The outer scroll offset brings the activated section header and at least its first
   content row into view. Switching modes preserves open preferences and clamps both
   outer and inner offsets; it does not persist the active priority.
5. Width-responsive visibility, explicit-open width floors, and rail priority remain
   unchanged. This fallback is selected by measured height, not by a new width
   breakpoint.

### Activated-section transitions

The transient active Context section follows one explicit state table:

| Event | Result |
| --- | --- |
| First ChatScreen/Console mount | `None`; ordinary DOM order breaks allocator ties. |
| Open a closed section | Set that section active before allocation, then save only the open preference through the existing path. |
| Activate constrained `[>]` | Set that already-open section active and reallocate; no preference write. |
| Keyboard focus enters a section header control, overflowing viewport, or body descendant | Set that section active and request allocation reconciliation; no preference write. |
| Pointer presses a section header control, viewport, or body descendant | Set that section active before handling the control action; pointer-wheel movement alone does not change active priority. |
| Close the active section | Choose the nearest preceding open non-empty section in DOM order, otherwise the first following one, otherwise `None`; the close preference uses the existing save path. |
| Content makes the active section empty/absent | Apply the same preceding-then-following fallback without a preference write. |
| Enter/leave short-height fallback | Retain the active ID if it is still open and non-empty; otherwise apply the same fallback. |
| Collapse/reopen a rail on the same mounted screen | Retain active ID. |
| Unmount and later remount Console | Reset to `None`; active priority is never restored from preferences. |

Allocation application is equality-guarded. A resize caused by applying a new
allocation may request another reconciliation, but an identical complete allocation
set is a no-op, preventing a post-refresh loop.

This replaces the fixed `max-height: 20%` rule. Short open sections return unused rows
to the pool, allowing another open section to approach the 20-line ceiling. At smaller
feasible heights, sections scroll sooner so headers remain simultaneous; below that
physical limit, the outer fallback keeps every header and body reachable.

## Inspector structure

The Inspector outer rail remains:

1. its fixed collapse header;
2. the scrollable Inspector body containing direct sections in existing order;
3. a separate pinned outer fold hint.

The outer hint uses the exact `▼ more sections — scroll` copy. Its slot is controlled
by the counterfactual `D_outer > R` predicate above, while visible text still uses the
actual outer body's `scroll_y < max_scroll_y`. It communicates that any outer content
remains below, including the remainder of a partially visible section, and is
independent of local `▼ more — scroll` hints. The slot remains one row high while
counterfactual overflow exists, becomes blank at scroll end, and leaves layout as soon
as `D_outer <= R`. Slot removal/growth clamps outer scroll before fold state is painted.

`ConsoleRunInspector` currently emits flat headings followed by rows and actions. It
will group each existing `_ROW_GROUPS` entry into a stable heading plus bounded body.
Dictionary and World Book groups use the same structure. The structural fingerprint
continues to be based on the same row IDs and action definitions, and in-place updates
continue to query the same row IDs; grouping must not force recompose for text/status
only changes.

Specialized top-level widgets such as Sources, Changed Files, Session Settings, and
live-work cards retain their header and business-state seams. Their content regions
adopt the shared bounded body without moving or duplicating their header. Existing
child-owned vertical caps are retired: this includes Sources' inline 6/10-row cap and
CSS 6-row cap, Session Settings' CSS 9-row minimum and inline 9-row maximum, and
equivalent specialized-section constraints. Content-specific product limits that
deliberately summarize data, such as
Changed Files' `MAX_VISIBLE_ROWS` plus its honest remainder row, remain data contracts;
the bounded viewport owns only the rendered-height ceiling.

### Definitive Inspector ownership map

Every currently emitted row, action, or specialized card belongs to exactly one
direct section boundary:

| Direct boundary | Owned content |
| --- | --- |
| Sources | `ConsoleStagedContextTray`: summary, staged-source rows, empty state, and recovery. |
| Scope (compact row, not a section) | `ConsoleRetrievalScopeRow`. |
| Changed Files | `ConsoleChangedFilesSection` header/body/tails. |
| Run status (compact row, not a section) | `console-inspector-run-status-summary`. |
| Run | Run recipe, Live work, Setup, Send blocked, Recovery action, Blocked impact, Next action, Provider. |
| Source Readiness | Sources, RAG/source, Evidence, Authority. |
| Tools | Tools, MCP. |
| Approvals | Approvals and `Review approval`. |
| Artifacts | Artifacts and `Save as Chatbook`. |
| Selected Conversation | Selected conversation, Conversation source, Workspace, Resume state, both Prefill rows. |
| Session Defaults | Session provider, model, endpoint, sampling, persona. |
| Selected Message | Selected message, Message actions, Keyboard, Variants, Excerpt, Delete confirmation. |
| Changes | The `Review Changes` action. It retains the existing tail position after Selected Message and before Chat Dictionaries and mounts only while the action has nonzero rendered height. |
| Chat Dictionaries | Dictionary rows and dictionary actions. |
| World Books | World Book rows and World Book actions. |
| Session Settings | `ConsoleSettingsSummary` rows and Open Settings action. |
| Live Work / Live work sources | The mutually exclusive pending-launch status card or no-launch source-readiness card, including its RAG controls and source rows. |

This table gives `_ACTION_GROUPS["Changes"]` its missing named boundary at the same
position. It also assigns `Send blocked`, `Recovery action`, and `RAG/source`, which
currently fall through `_ROW_GROUPS`. Known row/action IDs and their relative order
within each boundary remain unchanged. Only the new body and hint nodes receive
derived IDs; compatibility wrappers retain the old section and header IDs.

The ownership classifier is exhaustive and receives an explicit injected
`InspectorOwnershipPolicy`:

- `STRICT` raises `UnownedInspectorContentError` for an unknown row label or action ID
  before mounting a new tree. Unit/component tests use `STRICT`; developer launches opt
  into the same policy with `TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP=1`.
- `RESILIENT` is passed explicitly by ordinary production composition. It renders all
  known sections, omits unknown children, and never mounts a generic `Other` section.
  It logs each unknown structural fingerprint once using only stable identifiers in
  the form `row:<label>` or `action:<widget_id>`—never row text, action copy, or user
  content—and changes the compact summary to
  `Status: Inspector data incomplete`.

The policy is constructor-injected rather than inferred from `__debug__`, packaging,
or test discovery. A later valid state clears the incomplete flag and updates the
summary in place when the known structural fingerprint is unchanged; it must not force
an unnecessary recompose. A changed known structure follows the existing structural
recompose path.

## Interaction contract

### Focus

- A section viewport joins focus order only while it overflows.
- A short, empty, closed, or absent body adds no keyboard stop.
- The hint is never focusable.
- Focus traversal preserves the complete existing DOM order:
  `enabled section-header controls -> overflowing section viewport -> enabled body
  descendants -> next section's header controls`. Shift+Tab reverses the path. A
  boundary with no focusable header or body control contributes only its overflowing
  viewport; a non-overflowing static boundary adds no stop. The outer rail body remains
  its own stop for outer scrolling and section navigation.
- Focusing a descendant scrolls it fully into its local viewport before fold state is
  reconciled. It does not reset the section's stored session-local offset.
- While a viewport or descendant owns focus, its header is underlined and its viewport
  uses the existing Console focus background. The styles are preallocated and do not
  change geometry. When the outer body owns focus, the rail title is underlined
  instead, distinguishing the active scroll owner without color alone.
- If a focused section stops overflowing, the shared body invokes its owner-supplied
  recovery callback. If a focused descendant disappears, recovery chooses the next
  enabled visible control in DOM order within that section, then the previous one,
  then an enabled section-header control. Context next targets its section toggle;
  Inspector next targets the outer body and finally its collapse button. This
  next-then-previous rule is the sole `nearest` tie-breaker.
- Responsive hiding continues to hand focus to the appropriate rail reveal control.

### Keyboard scrolling

While an overflowing body itself has focus:

- Up/Down scroll one rendered line.
- Page Up/Page Down scroll one viewport.
- Home/End scroll to that section's first/last line.

Existing child controls keep their established input behavior. No terminal-convention
Ctrl binding or app-global binding is added or shadowed.

### Inspector section navigation

The Inspector adds rail-local `n` (next section) and `p` (previous section) commands.
"Inspector active" means the focused widget is `#console-right-rail` or one of its
descendants. The commands are inactive inside an editable input that owns printable
keys. Their anchor rules are:

- inside a direct boundary, `n/p` targets its next/previous mounted boundary;
- on Scope, run status, or another non-boundary descendant, `n` targets the first
  following boundary and `p` the last preceding boundary;
- on the rail collapse button, outer body itself, or another node without a positional
  boundary anchor, `n` targets the first mounted boundary and `p` the last;
- at the first/last anchored boundary, the outward command is a no-op; navigation does
  not wrap.

For a resolved target the commands:

1. find the next/previous mounted direct boundary without wrapping;
2. scroll its header fully into the outer viewport;
3. focus its overflowing viewport, otherwise its first enabled control, otherwise the
   outer Inspector body;
4. preserve local scroll offsets and section open state.

The footer refreshes on focus entering or leaving `#console-right-rail` and advertises
`n/p Sections` only while Inspector is active. F1 help evaluates the same active-focus
predicate at invocation time instead of relying on a static startup snapshot. The
screen does not bind `n` or `p` globally, so transcript selection, composer typing, and
other panes retain their existing behavior.

### Pointer scrolling and boundary handoff

Pointer-wheel input over an overflowing body scrolls that body while movement remains
in the requested direction. At its top or bottom boundary, continued movement is
offered to the outer rail so nested scrolling cannot trap the user. Pointer events
over a short body go directly to the outer rail.

### State continuity

Section scroll offset survives in-place state synchronization and rail
collapse/reopen. When content shrinks or allocation decreases, Textual clamps the
offset to the new valid maximum and fold state reconciles afterward. No scroll offset
or activated-section allocation priority is persisted to disk or included in rail
preference saves. Reopening the Console starts with ordinary DOM-order ties until the
user activates a Context section.

## Responsive behavior

The expanded primary states are 235x52 and 160x45. At these sizes, short sections hug
content and longer sections use as much of the 20-line allowance as the rail's
allocation contract permits.

At 120x30 and 80x24, existing width-responsive rail behavior still decides whether a
rail is visible. When a rail is visible, bounded sections adapt to the actual height.
At 120x30 the normal header-fit allocator remains active when all header chrome fits.
At 80x24 an explicitly opened all-section Context state uses the short-height outer
fallback because the header chrome does not fit. No width breakpoint, rail priority,
handle label, stored intent, or exact-100 minimum waiver changes in this task.

## Error and race handling

- Layout-dependent state is never computed from zero-size pre-layout regions.
- Recompose-time missing selectors cause a no-op, not an exception.
- Multiple mount/resize/sync requests coalesce into bounded post-refresh work.
- Each named descendant mutation seam explicitly requests reconciliation; no contract
  depends on non-bubbling descendant layout events reaching an ancestor.
- Content shrink clamps scroll offsets before hint visibility is asserted.
- Outer hints use counterfactual no-slot viewport height and reconcile on outer-body
  resize, preventing their reserved row from sustaining false overflow.
- A normal-mode allocator receiving too few usable body rows marks unfunded sections
  `· no room` and supports transient reprioritization rather than silently appearing
  empty. If header chrome itself does not fit, it switches to the outer fallback.
- A section becoming hidden or collapsed cannot leave an invisible focused viewport.
- Unknown Inspector row/action ownership raises in test/development and produces a
  compact incomplete-data status plus a stable diagnostic in production.
- No background worker or database read is introduced for layout reconciliation.

## Testing strategy

### Pure tests

- Context allocator: empty, short, long, mixed, all-open, collapsed, tie, and
  insufficient-budget cases, activated-section priority, `· no room`, and the exact
  transition into/out of short-height outer-scroll fallback.
- Every activated-section transition in the state table, including active-section
  removal, rail collapse/reopen, Console remount reset, and zero extra preference
  writes for transient priority changes.
- Exact 20/21 rendered-line boundary.
- Deterministic redistribution and hint-row accounting.
- Exhaustive Inspector row/action ownership under `STRICT`; known Changes content
  passes, and unknown labels/action IDs raise with no `Other` section.
- `RESILIENT` ownership keeps known sections, omits unknown children, logs exactly one
  stable identifier per unknown structural fingerprint with no user content, sets the
  incomplete summary, then clears it on a valid state through the in-place path when
  known structure is unchanged.

### Isolated widget tests

- Natural height with 0, 1, and 20 content rows.
- Session Settings with one content row occupies one row under the production CSS;
  neither its former 9-row minimum nor maximum remains.
- Exactly 20 rows: no scroll or hint.
- 21 rows: 20-row viewport plus one separate hint row.
- Hint visible before the end, hidden at the end, and visible again after scrolling up.
- Content shrink clamps the offset and removes obsolete overflow.
- Focusability toggles with overflow and recovers when overflow disappears.
- Keyboard scrolling and pointer boundary handoff.
- Tab/Shift+Tab traversal across header controls, viewport, and body descendants in
  exact DOM order; descendant auto-reveal, active-owner non-color styling, and the
  next-then-previous focused-descendant removal fallback.
- Coalesced reconciliation from every named Context/Inspector mutation owner, including
  invalidation of the outer Inspector hint and direct Inspector outer-body resize.
- Counterfactual outer-hint sequence at an available height of 10: 10 content rows have
  no slot, 11 add the slot, and shrink back to 10 removes it and clamps scroll.
- Terminal grow/shrink with fixed-height children recomputes Context/Inspector outer
  overflow even when no child emits Resize, removing and restoring the slot correctly.
- Atomic `ConsoleLeftRail` allocation reconciliation from same-tick multi-section
  updates, outer resize, section toggle, Character remount, and fleet-summary
  visibility; no mixed old/new measurement set and no equality-loop.
- Inspector-local `n/p` next/previous section navigation, no-wrap boundaries, editable
  input exclusion, every non-boundary anchor rule, focus target selection, footer
  refresh on Inspector focus transitions, and F1 help evaluated at invocation.
- Recompose and in-place update paths preserve stable child and row IDs.

### Production-CSS compositor tests

At 235x52 and 160x45, exercise both rails expanded with every Context section open.
At 120x30, exercise the default responsive state and an explicitly opened, all-open
Context rail. At 80x24, assert the default hidden-rail state and separately exercise
an explicitly opened, all-open Context rail to prove the outer-scroll fallback keeps
every header and body reachable. Across those states:

- normal header-fit states contain every Context header simultaneously; the 80x24
  fallback reaches each complete header and its base body through outer scrolling;
- Context short sections do not waste height and long sections receive redistributed
  rows up to 20;
- Inspector and Context section bodies never exceed their allocation;
- the 20 content rows and separate hint row do not overlap siblings;
- local hints use `▼ more — scroll`; Context/Inspector outer hints use the distinct
  `▼ more sections — scroll`, remain pinned, and track their own overflow;
- normal-mode `· no room` sections are visibly constrained and become funded when
  reprioritized; 80x24 never silently loses a header or open non-empty body;
- Inspector `n/p` navigation reaches every mounted direct boundary in order;
- collapse/reopen, focus handoff, rail badges, stored preference call counts, and
  responsive geometry remain unchanged.

### Regression scope

Run the existing focused Context/Inspector rail, section, compact-access, resize,
geometry, changed-files, staged-context, run-inspector, and CSS-bundle suites. Do not
substitute the repository-wide suite for these behavior-specific tests.

## Migration and documentation

- Replace the fixed Context `20%` body-cap CSS and the specialized Inspector vertical
  caps named above only after the bounded component and allocator are proven red and
  green.
- Keep old stable IDs or provide explicit compatibility assertions wherever a wrapper
  changes nesting.
- Update the Console user guide to explain the 20-line section ceiling, local hints,
  distinct outer hints, constrained Context reprioritization, short-height outer
  fallback, focus traversal, and Inspector `n/p` section navigation.
- Record the task-ID collision correction: the Console Phase 0 stream moves from the
  conflicting TASK-18912/TASK-18913/TASK-18915 IDs to
  TASK-19638/TASK-19639/TASK-19428.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/077-console-bounded-rail-section-scrolling.md`

Reason: the change establishes a long-lived cross-rail layout, scroll-ownership,
focus, pointer, and keyboard interaction contract and replaces the fixed Context body
allocation rule.
