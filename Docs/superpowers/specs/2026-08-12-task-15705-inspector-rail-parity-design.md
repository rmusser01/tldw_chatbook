# TASK-15705: Collapsed Inspector rail parity design

## Goal

Make the collapsed Console Inspector rail read as the right-side counterpart
to the collapsed Context rail: the rail surface fills the workspace height and
the Inspector label is centered vertically. Preserve the current width,
tooltip, badge vocabulary, and open/collapse behavior.

## Current behavior and root cause

`DestinationRailHandle` gives collapsed handles a full-height, bordered,
panel-background base treatment. Two right-side overrides undo that treatment:

- `.console-rail-handle-right` changes the container to an auto-height,
  three-to-six-row transparent block with no border.
- `DestinationRailHandle.compose()` and
  `.console-rail-handle-button-right` fix the right-side button at three rows.

The open Inspector rail is not involved. The mismatch belongs entirely to the
collapsed handle.

## Approaches considered

### A. Full-height parity (selected)

Restore the right-side handle's full-height panel and border treatment, then
let its button consume the available vertical space. When a badge exists, it
keeps its natural row at the bottom and the label centers within the remaining
button area.

This directly meets the request and reuses the existing handle structure.

### B. Filled wrapper with the compact button retained

Fill the column background but keep the three-row button at the top. This is a
smaller geometry change, but the label remains top-weighted and does not meet
the requested vertical centering.

### C. Narrow vertically stacked label

Use the opt-in vertical-label presentation developed for TASK-1335. This
recovers horizontal space, but changes reading direction and is a different
interaction/design decision from visual parity with the Context rail.

## Design

The existing `ConsoleRailHandle` and `DestinationRailHandle` boundaries stay
unchanged. The right-side modifier continues to own right-specific geometry,
but it will no longer opt out of the base handle's full-height background and
border. The right-side button will use flexible remaining height instead of a
fixed three-row height so an optional badge can remain visible without
overflow.

No ids, messages, state builders, persistence keys, rail widths, responsive
thresholds, labels, tooltips, or badge abbreviations change. The open Inspector
rail and Context rail are untouched.

The component stylesheet remains the source of truth. The generated modular
stylesheet must be rebuilt with the repository CSS generator.

## Verification

Add a mounted handle regression that fails against the current implementation
and proves:

- left and right collapsed handles occupy the full harness height;
- the right label's button consumes the available height and centers content;
- the right handle has a non-transparent panel background and visible border;
- an optional Inspector badge remains mounted within the handle bounds.

Then run the focused destination/Console rail tests, CSS build integrity check,
and a live Console render at representative terminal widths. The visual check
must inspect both an unbadged handle and a state with a badge if the harness can
produce one deterministically.

## ADR check

ADR required: no

ADR path: N/A

Reason: This is a reversible presentation refinement inside an existing
widget and stylesheet boundary. It changes no storage, state ownership,
service contract, security boundary, dependency, or long-lived application
structure. ADR-017's text-only Console rail visual language remains in force.
