# MCP Hub Narrow-Width Triad — Design

Date: 2026-09-11
Status: Draft (awaiting review)
Review basis: MCP screen UX review 2026-09-11 (Finding F6 / Low-15), evidence screenshot `07` (100 cols)
ADR: backlog/decisions/148-mcp-hub-rail-ia-and-responsive-triad.md
Related: F-057 (width-aware table columns + rail truncation), `_COMPACT_WIDTH = 120`, fix/console-rail-ux precedent

## Problem

Below ~120 terminal columns the rail/canvas/inspector triad squeezes the
inspector to `2fr / min-width 20`: words break mid-token (`Overvi|ew`),
JSON payloads truncate mid-key (`external_se`), the rail state legend
wraps to three lines, and row labels ellipsize aggressively. F-057 fixed
the *table* for narrow widths; the inspector band did not get an
equivalent policy. A meaningful share of terminal users run at ≤120 cols
where the third pane is effectively unreadable.

## Goals

- At widths below `_COMPACT_WIDTH`, all three surfaces remain readable
  without horizontal scrolling or mid-word breaks.
- No behavior change above the threshold (wide layouts byte-identical).
- The inspector stays reachable in every width (it hosts the actions:
  Connect, Edit config, permission explanations, audit drill-through).

## Non-goals

- No mobile/phone-sized support below ~80 cols; the floor for this design
  is 100 cols × 30 rows (the QA viewport the suite already pins).
- No change to per-mode canvas content.

## Approaches considered

**(a) Stack the inspector below the canvas at compact widths (recommended).**
When `#mcp-hub-grid` gets `.mcp-compact`, the grid switches from
`Horizontal` to a two-row layout: top = rail + canvas (rail shrinks to a
fixed 24), bottom = a bounded inspector band (`max-height ~12`, scrolls
internally). Trade-off: the inspector shares vertical space with the
canvas (tables already cap at 70% height, so this fits); geometry is
deterministic and testable; no new toggle machinery.

**(b) Hide the inspector behind a Detail toggle.** Compact widths hide the
inspector entirely; a per-mode `Detail ▸` chip reveals it as an overlay.
Trade-off: preserves canvas height but hides the action surface behind a
mode-conditional control — the approval-action discoverability cost is
exactly what the review flagged in Finding 1; rejected as primary, keep
as fallback if (a) proves cramped at 100×30.

**(c) Squeeze harder.** Keep three columns; fix only word-breaking
(`no-wrap` + ellipsis overflow on inspector Statics, Select min-widths,
legend abbreviation). Trade-off: cheapest, and worth doing regardless —
but at 2fr the inspector is ~20 cols, which no wrapping policy makes
usable. Adopted as a baseline layer under (a), not as the fix.

## Design (approach a + c)

**Grid switch** (`mcp_workbench.BUNDLED_CSS` + `_agentic_terminal.tcss`
lockstep copy): `.mcp-compact #mcp-hub-grid` becomes a vertical layout
(`grid-rows: 1fr auto` or an equivalent nested Horizontal/Vertical
restructure if Textual's layout model requires it — implementation
detail: prefer restructuring `compose()` into
`Vertical[Horizontal[rail, canvas], inspector]` and toggling classes, so
the CSS stays declarative). `on_resize`/`_sync_compact_class` already
toggle the class; no new triggers.

**Inspector band**: `max-height: 12; overflow-y: auto; min-height: 4`.
The Advanced collapsible renders collapsed inside the band regardless of
`advanced_visible` at compact widths (it is the least-narrow-friendly
content; one-line note replaces it).

**Word hygiene (always on, not compact-only)**:
- Inspector Statics: `overflow: ellipsis`-equivalent truncation helpers
  for one-line statuses; never mid-word CSS breaks.
- Selects in the hub: explicit `min-width` so prompts never clip
  mid-token ("Overvi|ew" class).
- Rail legend (`mcp_rail.py`): abbreviation map below a width budget —
  `needs setup → setup`, `needs attention → attention`,
  `off (opt-in) → off`, `no tools → ∅` — derived from `STATE_LABELS`
  through one function so the full and short forms cannot drift.
- Tables keep F-057 column dropping.

**Testing**

- Extend the reachability test matrix: 100×30 assertions for every mode
  (servers overview + detail, tools, permissions, audit) that no text
  region renders a mid-word break (assert on rendered line content via
  the existing harness helpers rather than pixel inspection).
- Legend abbreviation unit tests (budget boundaries).
- Inspector-band geometry test at 100×30 (bounded height, scrollable).

## Open questions (for reviewer)

1. Is 12 rows the right band cap at 30-row terminals, or should it scale
   (`min(12, terminal_rows * 0.4)`)?
2. Should the compact rail keep the Source select at fixed 24 cols, or
   collapse to a single-line `[Local ▾]` chip?
