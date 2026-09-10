# Watchlists Feed Items Width Design

## Goal

Give the Feed Items pane immediately left of the Reader 10 more terminal columns
without changing the other Watchlists panes or the existing responsive layout
policy.

## Design

Increase the Read-mode Feed Items pane's minimum width from 32 to 42 columns and
its preferred/maximum width from 40 to 50 columns. Keep the pane flexible between
those bounds. In production CSS, change the main rule to `width: 50fr`,
`min-width: 42`, and `max-width: 50`, and change the expanded-side-pane override
to the same 42-column minimum. Update the pure responsive-layout minimum to 42
columns so Watchlists collapses side panes before squeezing Feed Items below its
new minimum.

This shifts only Read-mode collapse and reopening thresholds that include Feed
Items. Management-tab geometry remains unchanged. The Reader remains the permanent
centre, and the existing Inspector, Navigation, then Feed Items collapse priority
is preserved.

## Verification

- Update every Feed-Items-dependent Read-mode boundary by exactly 10 columns:
  nominal thresholds `145/115/91` become `155/125/101`, with corresponding reopen
  thresholds moving by 10. Management-tab thresholds remain unchanged.
- Update the focused pure, mounted, hysteresis, and scoped-rebuild layout coverage,
  including `test_watchlists_responsive_layout.py`,
  `test_watchlists_workbench.py`, `test_watchlists_layout_hysteresis_probe.py`, and
  `test_watchlists_scoped_rebuilds.py` wherever those Read thresholds are encoded.
- Verify both production CSS declarations carry the new 42-column minimum and the
  main rule carries the 50-column preferred/maximum width.
- Regenerate `tldw_cli_modular.tcss`, run the focused Watchlists layout tests, and
  run the CSS bundle integrity checks.

## Architecture Decision

ADR required: no

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: ADR-042 already owns the permanent Reader, side-pane minimums, and
responsive-collapse policy. This is a small presentation refinement within that
accepted boundary.
