# ADR-077: Bound Console rail sections and expose hidden overflow

Status: Accepted
Date: 2026-08-21
Related Task: [TASK-19428](../tasks/task-19428%20-%20Bound-Console-Context-and-Inspector-sections-with-20-line-scroll-limits.md)
Related Spec: [Console bounded rail-section scrolling design](../../Docs/superpowers/specs/2026-08-21-console-bounded-rail-section-scroll-design.md)
Amends: N/A
Supersedes: N/A

## Decision

Direct named section bodies in the Console Context and Inspector rails will grow
naturally through 20 rendered terminal content lines. Content beyond 20 lines will
scroll inside a 20-line viewport. The section header is outside that viewport. An
overflowing section reserves one additional, non-focusable row beneath the viewport
for the exact hint `▼ more — scroll`; the hint is visible only while content remains
below the section's current scroll position.

The 20-line value is a ceiling, not a fixed height. Empty and short bodies hug their
content and do not reserve blank rows, a scrollbar, or a hint. Physical rendered rows,
including wrapped content and visible controls, count toward the ceiling; logical
record count does not.

The Context rail preserves TASK-15110's outcome that every direct section header is
simultaneously visible whenever the fixed header chrome fits. A deterministic,
content-aware allocator reserves those headers first, prioritizes the most recently
activated section, then progressively distributes remaining rows up to the 20-line
ceiling. An unfunded open section is visibly marked `· no room`; its constrained `[>]`
control reprioritizes it without changing persisted open preferences.

When the headers themselves cannot fit, Context switches to a height-only outer-scroll
fallback. Every open non-empty section receives at least one honest content row plus a
local hint when needed, the activated section may receive additional rows, and the
outer body exposes a pinned `▼ more sections — scroll` cue. This is the explicit 80x24
path; it replaces the impossible promise that fourteen rows of header chrome fit a
ten-row viewport.

Inspector sections use the full 20-line ceiling independently and the Inspector's
outer body remains scrollable so later sections can be reached. Outer rails use the
distinct pinned `▼ more sections — scroll` cue while local sections retain the exact
`▼ more — scroll` copy. Scope remains a compact row rather than a named section.
Nested subsections do not create additional scroll owners: only direct section bodies
of each rail receive this contract.

Outer-hint existence is derived counterfactually from content height versus the outer
viewport height with no hint slot. The actual one-row slot never participates in its
own existence predicate, preventing sticky overflow when content shrinks or the
terminal grows. Section changes and outer-body resize both invalidate this measurement.

Only overflowing section viewports enter keyboard focus order. Traversal proceeds from
the viewport through its enabled descendants in DOM order, reverses with Shift+Tab,
and fully reveals focused descendants. Arrow keys scroll by line; Page Up and Page
Down scroll by viewport; Home and End move within the focused section. Pointer
scrolling targets the section under the pointer and naturally bubbles to the outer rail
at a boundary. Inspector-local `n/p` commands move to the next/previous direct boundary
without becoming screen-global bindings.

Section scroll offsets and activated-section priority are session-local presentation
state. Offsets survive in-place content updates and rail collapse/reopen, clamp when
content shrinks, and are not written to Console rail preferences. Every descendant
mutation owner explicitly requests a coalesced post-refresh reconciliation because
Textual descendant layout events do not bubble; Inspector reconciliation also
invalidates the outer hint. `ConsoleLeftRail` is the single allocation coordinator: it
measures every sibling from one post-refresh snapshot and applies the whole allocation
set atomically. Existing responsive rail priority, explicit-open width floors, focus
transfer, ordering, badges, and data semantics remain unchanged.

Desired section height is measured before the 20-line cap, so exactly 20 physical
content rows do not overflow and 21 do. A hint slot remains one row high while the
section overflows, becoming blank at scroll end rather than shifting later content.
Existing specialized child constraints that compete with this rule are retired,
including Sources' 6/10-row caps and Session Settings' CSS 9-row minimum plus inline
9-row maximum; product-level summarization such as Changed Files' honest remainder row
remains.

Every known Inspector row and action is assigned to the direct section named in the
approved design. `Send blocked` and `Recovery action` belong to Run, `RAG/source`
belongs to Source Readiness, and `Review Changes` receives a named Changes boundary at
its existing tail position. There is no user-facing Other section. Unknown ownership
uses a constructor-injected policy: STRICT raises in tests and opt-in developer
launches; RESILIENT production retains known content, omits unknown children, logs only
deduplicated stable identifiers, and exposes `Status: Inspector data incomplete` until
a valid state clears it through the in-place path where structure permits.

## Context

The Console is primarily used in an expanded, browser-like terminal, but both rails
contain dynamic sections whose content can range from one status row to long source,
agent, settings, or selected-message detail. The Inspector previously exposed one
outer scrollbar with no standard below-fold cue. Context used a fixed percentage cap
that kept headers visible but could waste available height or constrain a short set of
open sections before the intended reading limit.

First-time users need an explicit signal that content continues. Regular users need
bounded regions that do not let one verbose section displace every later section.
Technical and non-technical users both benefit when short sections remain fully
visible and long sections behave consistently.

## Alternatives Considered

| Alternative | Why rejected |
| --- | --- |
| Keep one outer scrollbar per rail | A verbose section can push later content far below the fold, and the section containing hidden content has no local continuation cue. |
| Apply independent CSS caps to every existing widget | Several Inspector groups are not containers today, and duplicated hint/scroll logic would drift across dynamic update paths. |
| Replace overflow with `View all` actions | It adds a navigation step and does not meet the requirement that each section itself remain scrollable. |
| Give every Context section exactly 20 rows | It would push later Context headers off-screen and reverse TASK-15110's accepted discoverability outcome. |
| Compact every Context header to one row at 80x24 | It changes the established header grammar and still leaves too little honest body space; a rare height fallback is clearer and less invasive. |
| Disallow explicit Context opening at 80x24 | It removes user control even though the rail can remain usable through outer scrolling. |
| Use identical local and outer fold copy | Simultaneous hints would not identify which scroll owner moves next. |
| Render unknown Inspector content under Other | It exposes an implementation ownership failure as permanent user-facing information architecture. |
| Persist section scroll offsets | Scroll position is transient presentation state; persistence would add stale state and a new preference contract without user value. |

## Consequences

- Context and Inspector share one bounded-section interaction model while retaining
  their different outer-rail behavior.
- The Context rail gains a pure height-allocation policy and no longer relies on the
  fixed `20%` CSS cap.
- `ConsoleLeftRail` owns atomic allocation reconciliation; individual section bodies
  cannot independently commit sibling allocations.
- At constrained explicit-open heights, some open Context bodies may receive zero
  rows in normal mode; they are visibly marked and can be reprioritized. When headers
  cannot fit, the Context outer body scrolls and every open body receives a base row.
- Run Inspector's flat groups need stable section body containers, but their row IDs,
  order, actions, and in-place update contract remain unchanged.
- Nested scrolling becomes intentional and must include boundary handoff, conditional
  focusability, visible scroll ownership, explicit reconciliation, and production-CSS
  compositor coverage.
- Every new direct Console rail section must use the shared bounded body instead of
  inventing a local cap or hint.
- No database, provider, preference, or cross-session state contract changes.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-21-console-bounded-rail-section-scroll-design.md)
- [TASK-19428](../tasks/task-19428%20-%20Bound-Console-Context-and-Inspector-sections-with-20-line-scroll-limits.md)
- [ADR-043: Console rail compact collapse yields to explicit toggle](043-console-rail-compact-collapse-yields-to-explicit-toggle.md)
