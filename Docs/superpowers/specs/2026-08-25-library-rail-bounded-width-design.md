# Library Rail Bounded-Width Design

**Date:** 2026-08-25  
**Task:** TASK-22301  
**Status:** Approved direction; pending implementation plan

## Problem

The persistent Library navigation rail currently declares `width: 3fr` and a
24-cell minimum but no upper bound. It therefore grows well beyond the useful
Collections-reference width on wide terminals. Because every Library mode uses
the same rail, the layout should express one stable sizing contract rather than
allowing the rail to consume an unbounded share of the content workbench.

The requested behavior is deliberately still fluid: keep the existing `3fr`
share, but bound the rendered rail around the approximately 31-cell reference
seen in Collections.

## Goals

- Keep the persistent Library rail on the existing `3fr` sizing model.
- Apply an exact 24-cell minimum and 34-cell maximum from one shared rail owner.
- Preserve the same contract across Media, Chats, Notes, Prompts, Skills,
  Collections, Search / RAG, Import, Export, Study handoffs, and the landing
  canvas.
- Preserve current rail search, section disclosure, scrolling, focus, row-label
  fitting, and selection behavior.
- Protect the content canvas and surrounding Library shell during initial
  mount, navigation, recompose, and live resize.

## Non-Goals

- Making the rail a fixed 31-cell column.
- Adding a user-resizable splitter or persisting a custom width.
- Changing Library information architecture, labels, counts, or section state.
- Changing inner canvas list/detail splits such as the Collections workbench.
- Redesigning compact-terminal behavior outside the width bound needed here.

## Design

### Single sizing owner

`LibraryRail` remains the authority for its inline Textual geometry. Its current
`3fr` width and 24-cell minimum remain. A 34-cell maximum is added at the same
construction boundary, so every newly mounted or recomposed rail receives the
same three-part contract:

```text
width: 3fr
min-width: 24
max-width: 34
```

Keeping all three declarations together avoids a split-brain contract between
Python inline styles and the bundled stylesheet. The app-tier `#library-rail`
rule continues to own presentation and overflow styling, not competing width
values.

### Behavior across Library modes

The selected Library row may replace canvas content, start a loader, or trigger
a screen recompose, but it does not create a mode-specific rail. The bounded
contract is therefore applied once to the shared `LibraryRail`; no per-mode
width branches, copied style assignments, or post-navigation corrections are
introduced.

At ordinary widths, the rail receives its `3fr` share until it reaches 34
cells. At compact widths, it may shrink with the layout until it reaches the
existing 24-cell readable floor. Remaining horizontal space belongs to
`#library-canvas`, subject to its existing minimum and containment rules.

### Responsive and failure behavior

No persisted preference or transient state is added. Resizing recomputes normal
Textual fractional layout from the same bounds. If a terminal is narrower than
the combined existing minimums of the Library shell, this change does not invent
a new collapse policy; verification must instead prove that the new maximum does
not introduce any additional overlap or clipping compared with the current
compact baseline.

## Accessibility and HCI

- A 24-cell floor preserves the current search placeholder and width-aware row
  fitting behavior at compact widths.
- A 34-cell ceiling prevents navigation chrome from expanding into dead space
  that should support reading and editing in the primary canvas.
- The rail remains in the same location and focus order, avoiding mode-switch
  movement that burdens both first-time and habitual users.
- Keyboard, pointer, section disclosure, scrollbar, tooltip, and F6 behavior
  remain unchanged.

## Verification

Implementation follows test-driven development.

1. Add a production-styled geometry test that fails against the current
   unbounded `3fr` rail at a wide viewport.
2. Assert the rail's declared `3fr` width, 24-cell minimum, and 34-cell maximum
   from the mounted production widget.
3. At 235, 170, 120, 100, and 80 columns, verify the rendered rail remains in
   the 24–34-cell range and the canvas remains contained without rail/canvas or
   footer overlap.
4. At 60 columns, compare against the current compact baseline and prove the
   new maximum introduces no new overlap or hidden shell region.
5. Navigate through every Library canvas family and verify the rail geometry
   contract and widget identity remain stable.
6. Resize wide-to-compact-to-wide and verify the fractional rail returns to the
   bounded wide result without persistence or stale inline geometry.
7. Run Library shell, row fitting, focus, destination geometry, CSS integrity,
   Ruff, compilation, and diff-hygiene gates.
8. Perform isolated PTY UAT at representative wide, standard, and compact
   terminal sizes; native terminal-emulator-specific acceptance is reported
   separately and is not inferred from Pilot tests.

## ADR Check

**ADR required:** no  
**ADR path:** N/A  
**Reason:** This is a reversible presentation constraint within the existing
Library rail ownership and responsive layout boundaries. It changes no storage,
schema, sync policy, service contract, data ownership, dependency, security, or
long-lived application structure.

