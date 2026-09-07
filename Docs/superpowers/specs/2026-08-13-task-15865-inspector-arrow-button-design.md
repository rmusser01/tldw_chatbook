# TASK-15865 — Inspector Arrow Button Design

## Goal

Make the collapsed horizontal Inspector affordance read as one obvious control by changing its visible button label from `Inspect` to `Inspect->`. The full nine-cell string is the existing button; users can click anywhere on it to open the Inspector rail.

## Approved Direction

Use the fixed literal `Inspect->` for the canonical horizontal right-side Console handle. It occupies exactly the current nine-cell content width:

```text
Inspect->
```

Do not widen the eleven-column outer rail, introduce a second arrow widget, or calculate a variable number of dashes. The current `Button` already owns the full content region. Clear Textual's default one-cell line padding inline on the existing horizontal right Button so the nine-cell literal uses that full region and remains one clickable row.

## Alternatives Considered

### `Inspect-->`

This is ten terminal cells (`Inspect` is seven and `-->` is three), so it cannot fit the fixed nine-cell content width on one row. Runtime compositor evidence showed it wrapping even after Textual's line padding was cleared.

### Width-dependent arrow fill

This could calculate the dash count from the live button width. It adds layout coupling and more test surface without user value because this rail intentionally has a fixed nine-cell content width.

## Scope and Behavior

- Change only the horizontal right-side `ConsoleRailHandle` display copy for the canonical Inspector label.
- Clear `line_pad` inline only on the existing horizontal right Button. Textual 8 defaults to `line_pad=1`, reserving one cell on each side; the repository sets zero inline because TCSS rejects `line-pad: 0`.
- Preserve the single existing `Button`; do not add nested controls or a separate arrow hit target.
- Preserve the fixed `Open Inspector rail` tooltip.
- Preserve the optional badge as a separate row beneath the button, including `1 appr` / `3 appr` abbreviations and containment.
- Preserve the vertical presentation as stacked `Inspector` without an arrow.
- Preserve the left Context handle, shared `DestinationRailHandle`, eleven-column outer width, nine-column content width, full-height filled treatment, open rail, and compact-width behavior.

## Verification

Focused regressions will prove:

1. The horizontal canonical Inspector display label is exactly `Inspect->`.
2. The mounted button paints that string on one row within its existing content region.
3. Clicking the body of the button opens the Inspector rail; there is no separate arrow control.
4. Badged and unbadged geometry remain contained.
5. Vertical Inspector and left Context copy remain unchanged.
6. Representative real-Console viewport states retain one-row label, full-height parity, and positive transcript width.

Verification remains limited to directly modified and related functionality, following the user's standing instruction not to run the full repository suite.

## Architecture Decision Record

ADR required: no

ADR path: N/A

Reason: this is a reversible copy-only refinement inside the existing Console rail component and its established fixed-width interaction contract. It does not change application structure, persistence, service boundaries, dependencies, security policy, or long-lived UX ownership.
