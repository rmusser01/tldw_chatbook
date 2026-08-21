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
visible at supported terminal heights. A deterministic, content-aware allocator will
reserve the headers first, give short open sections only the rows they need, and
redistribute remaining rows among longer open sections up to the 20-line ceiling.
When the available shared height is insufficient, a Context section scrolls before 20
lines. This replaces the fixed `20%` body cap while retaining the reason for that cap.

Inspector sections use the full 20-line ceiling independently and the Inspector's
outer body remains scrollable so later sections can be reached. The outer Inspector
also reserves its own pinned `▼ more — scroll` hint while complete sections remain
below the rail viewport. Scope remains a compact row rather than a named section.
Nested subsections do not create additional scroll owners: only direct section bodies
of each rail receive this contract.

Only overflowing section viewports enter keyboard focus order. Arrow keys scroll by
line; Page Up and Page Down scroll by viewport; Home and End move within the focused
section. Pointer scrolling targets the section under the pointer and hands continued
motion to the outer rail at the section's top or bottom boundary. Existing interactive
child controls keep their current input behavior.

Section scroll offsets are session-local presentation state. They survive in-place
content updates and rail collapse/reopen, clamp when content shrinks, and are not
written to Console rail preferences. Existing responsive rail priority, explicit-open
floors, focus transfer, ordering, badges, and data semantics remain unchanged.

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
| Persist section scroll offsets | Scroll position is transient presentation state; persistence would add stale state and a new preference contract without user value. |

## Consequences

- Context and Inspector share one bounded-section interaction model while retaining
  their different outer-rail behavior.
- The Context rail gains a pure height-allocation policy and no longer relies on the
  fixed `20%` CSS cap.
- Run Inspector's flat groups need stable section body containers, but their row IDs,
  order, actions, and in-place update contract remain unchanged.
- Nested scrolling becomes intentional and must include boundary handoff, conditional
  focusability, and production-CSS compositor coverage.
- Every new direct Console rail section must use the shared bounded body instead of
  inventing a local cap or hint.
- No database, provider, preference, or cross-session state contract changes.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-21-console-bounded-rail-section-scroll-design.md)
- [TASK-19428](../tasks/task-19428%20-%20Bound-Console-Context-and-Inspector-sections-with-20-line-scroll-limits.md)
- [ADR-043: Console rail compact collapse yields to explicit toggle](043-console-rail-compact-collapse-yields-to-explicit-toggle.md)
