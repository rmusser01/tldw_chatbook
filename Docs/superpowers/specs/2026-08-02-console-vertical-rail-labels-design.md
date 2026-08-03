# Console Vertical Rail Labels Design

## Summary

Render the collapsed Console `Context` and `Inspector` rail labels from top to
bottom, approximating CSS `writing-mode: vertical-lr` within Textual's terminal
cell model. The change makes collapsed rails narrower while keeping their
meaning, keyboard behavior, and state cues intact.

## Scope

- Apply the vertical treatment only to the collapsed handles in Console.
- Preserve the expanded `Console context` and `Inspector` headers as horizontal
  text.
- Preserve horizontal, descriptive tooltips for both open-handle buttons.
- Keep Personas and other consumers of the shared rail-handle widget unchanged.
- Do not change rail persistence, open/close behavior, responsive thresholds, or
  focus order.

## Interaction and Presentation

Textual does not implement web CSS writing modes or terminal glyph rotation.
The handle therefore renders one label character per row, in source order:

```text
C      I
o      n
n      s
t      p
e      e
x      c
t      t
       o
       r
```

The directional glyphs used by the horizontal state labels are omitted from
the stacked display. Direction remains clear from the handle's edge placement
and its `Open Context rail` or `Open Inspector rail` tooltip.

Both Console handles use the same narrow, fixed width and full available rail
height. Any state badge remains visible below the button, uses the same stacked
terminal-cell treatment when needed to fit the narrow handle, and retains the
full badge text in its tooltip. Width must not change when badge state changes.

## Implementation Boundary

Add an explicit opt-in presentation flag to the shared `ConsoleRailHandle`.
Console passes the flag for its two collapsed handles; existing callers inherit
the current horizontal behavior. The widget owns label normalization, stacking,
and narrow child sizing. Console layout owns the corresponding stable handle
width. Component TCSS remains the source of truth and the bundled stylesheet is
regenerated from it.

## Verification

- Widget-level tests pin horizontal default behavior and vertical opt-in label
  rendering.
- Mounted Console tests verify both collapsed labels, stable narrow widths,
  tooltips, focusability, and open behavior.
- Existing Personas rail tests guard the unchanged default presentation.
- The targeted Console rail and stylesheet parity tests pass.

## Architecture Decision Record

ADR required: no  
ADR path: N/A  
Reason: This is a small, reversible presentation refinement that preserves all
existing component boundaries, interaction contracts, and persisted state.
