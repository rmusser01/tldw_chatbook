# Console Row Collapse Controls

**Date:** 2026-08-12
**Status:** Approved for implementation planning

## Goal

Reduce visual noise in the Console by letting the user collapse the status-chip
row above the composer, and make restore controls predictable by placing them at
the far left of their collapsed rows.

## Interaction Design

### Status row

- Expanded: render `Status ▾` at the far left, followed by the existing status
  chips in their current order and with their current behavior.
- Collapsed: render `Status ▴` at the far left, followed by muted `Status hidden`
  copy.
- Activating `Status ▾` collapses the row without discarding status state.
- After collapse, keyboard focus moves to `Status ▴` so the same control remains
  immediately reversible.
- Activating `Status ▴` expands the row and restores keyboard focus to the
  collapse control.

The collapsed status presentation remains one terminal row tall. This removes
the busy chip presentation while preserving an obvious, row-local restore
control; it does not attempt to reclaim the row vertically.

### Composer row

- Keep the expanded composer unchanged.
- In the collapsed composer presentation, move `Expand ▴` from the far right to
  the far left.
- Render the existing `Composer hidden` status copy after the restore control.
- Keep the conditional `Stop` action at the right edge while generation is
  active.

## State and Ownership

- `ChatScreen` owns the status-row collapsed boolean, matching its ownership of
  the composer collapsed boolean.
- The state defaults to expanded for each newly created Console screen.
- The state is not written to configuration, a database, or any cross-screen
  store.
- `ConsoleStatusChips` keeps expanded and collapsed presentations mounted and
  switches their display in place, preserving current chip state without a
  recompose.
- The widget exposes a narrow `set_collapsed(bool)` behavior; button activation
  is routed through the screen so screen-owned state remains authoritative.

## Accessibility and Layout

- Collapse and restore controls use readable text plus directional glyphs;
  glyphs are not the sole carrier of meaning.
- Both controls remain keyboard-focusable and expose descriptive tooltips.
- Visual order and focus order agree: the restore control is the first visible
  control in each collapsed row.
- Existing horizontal overflow behavior for the expanded status chips remains
  unchanged.
- No new keyboard shortcut or footer hint is added.

## Verification

- A widget test proves the status row switches between expanded and collapsed
  presentations without remounting or losing updated chip state.
- A screen-level test proves the status collapse/expand buttons update the
  screen-owned state and restore focus appropriately.
- Geometry assertions prove `Status ▴` and composer `Expand ▴` are left of their
  respective status copy, and that conditional composer `Stop` remains visible
  at the right when active.
- Existing composer geometry expectations are updated from the former
  right-aligned restore order to the approved left-aligned order.
- Existing composer-collapse, Console workbench, chip-interaction, and compact
  viewport tests remain green.
- A live Textual render is inspected at representative narrow and wide terminal
  sizes.

## Scope Boundaries

- No persistence across Console screen recreation or app restarts.
- No combined control for collapsing both rows.
- No status-chip redesign, reordering, or copy changes beyond the new row label
  and collapsed status copy.
- No new settings, dependencies, or architectural interfaces.

## ADR Check

ADR required: no

ADR path: N/A

Reason: This is a small, screen-local UI behavior change that follows an
existing ownership and presentation pattern. It does not change storage,
security, runtime boundaries, service contracts, or long-lived application
structure.
