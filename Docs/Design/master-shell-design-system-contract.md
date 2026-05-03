# Master Shell Design System Contract

Date: 2026-05-03

## Purpose

This contract maps the agentic terminal design system onto the master-shell implementation slices. It prevents Home, Console, and destination wrappers from inventing screen-local visual language while the shell migration is in progress.

## Source Documents

- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md`
- `Docs/Design/agentic-terminal-visual-system.md`
- `Docs/Design/master-shell-route-inventory.md`

The New_UI concept images are design references, not required runtime or review assets. If the image files are unavailable on a branch, the extracted visual rules in the design-system spec remain authoritative.

## Visual Source Of Truth

Runtime shell implementation must not introduce new visual patterns, token names, or state treatments that are not represented in `Docs/Design/agentic-terminal-visual-system.md`.

If a runtime slice needs a new visual pattern, update the visual-system contract before changing Python widgets or TCSS.

## Required Shared Classes

| Class | Purpose | First master-shell use |
| --- | --- | --- |
| `.ds-destination-header` | Page title, purpose, local scope/status, primary action | Home and all wrappers |
| `.ds-panel` | Bordered purposeful content region | Home dashboard and wrappers |
| `.ds-inspector` | Selected item detail, readiness, permissions, recovery | Console and wrappers |
| `.ds-status-badge` | Readable semantic status label | Home, Console, wrappers |
| `.ds-recovery-callout` | Owner/problem/next action recovery copy | Home, Console, wrappers |
| `.ds-source-role` | Context/evidence/editable-target/output-seed chips | Console staged context |
| `.ds-approval-card` | Approve/reject decision surface | Console, Home summary |
| `.ds-event-row` | Tool/run/audit event row | Console, Home active work |
| `.ds-field-row` | Label/control/help/validation row | Settings, builders |
| `.ds-toolbar` | Local action group | Destination wrappers |
| `.ds-shortcut-bar` | Current-context shortcuts/status | Shell bottom bar |

## Required State Classes

`.is-active`, `.is-disabled`, `.is-blocked`, `.is-running`, `.is-paused`, `.is-unsaved`, `.is-stale`, `.is-conflict`, `.needs-approval`, `.source-local`, `.source-server`, `.source-workspace`, `.source-remote-only`, `.source-dry-run`

## Density Contract

All new shell surfaces must support `.density-compact` and `.density-comfortable` without separate widget implementations.

## Top Navigation Contract

- Top navigation is global primary destination navigation only.
- `Home` and `Console` remain visible at all supported widths.
- Compact labels such as `W+C` are allowed in the top bar only when tooltip, command palette, and destination header expose the full label.
- Overflow must be discoverable and keyboard-reachable; destinations must not disappear silently at narrow widths.
- Destination context, workspace, backend, readiness, and recovery state stay inside destination headers or local panels.

## Shortcut Bar Contract

- The shell footer renders an explicit active shortcut context from the current destination or focused workflow.
- The footer does not infer shortcuts by scraping widgets.
- Detailed readiness and recovery state belongs in page headers, panels, inspectors, and recovery callouts.
- If no shortcut context is registered, the footer falls back to global command palette, help, and quit actions.

## Token Mapping Contract

- Design discussion may use dotted token names such as `status.warning`.
- TCSS variables use hyphenated names such as `$ds-status-warning`, `$ds-authority-workspace`, and `$ds-source-role-evidence`.
- Core palette values belong in Textual `Theme` fields where possible.
- Design-system-specific semantic values belong in `Theme.variables` and modular TCSS classes.
- Widgets should apply shared classes and state classes rather than raw color values.
- Font and glyph usage must provide plain-text or ASCII fallbacks.

## Required Readable Status Labels

`Ready`, `Running`, `Paused`, `Blocked`, `Unavailable`, `Approval required`, `Unsaved`, `Recovered`

## Testing Rules

- Assert stable IDs or classes for primary actions, status badges, source authority, source roles, approval cards, shortcut bars, next-best actions, and open/follow-in-Console controls.
- Assert readable status text. Do not assert raw color values.
- Assert destination context appears inside page headers, not the global top navigation.
- Assert top navigation keeps `Home` and `Console` reachable and exposes full names through tooltips or command palette when compact labels are used.
- Assert the footer renders current shortcut context and does not retain stale shortcuts after navigation.
- Treat `#console-pending-launch-card`, `*-follow-in-console`, `*-launch-in-console`, and `*-attach-to-console` IDs as behavioral testing hooks, not styling hooks.

## Implementation Status

- Home and all primary destination wrappers use `.ds-destination-header` and `.ds-panel`.
- Console renders pending live-work launch context with `.ds-panel` and a readable source/title label.
- Static source/artifact/persona/skill actions stage `ChatHandoffPayload` context; live work from W+C, Schedules, Workflows, and ACP uses `open_console_for_live_work()`.
- `tools_settings` resolves to MCP; global preferences are owned by Settings.

## Stop Conditions

- Stop before shell implementation if the design-system spec or route inventory is unavailable on the implementation branch.
- Stop before shell implementation if the shared design-system TCSS classes, density classes, state classes, and `agentic_terminal` theme are missing and need a separate design-system PR.
- Stop before shell implementation if the generated stylesheet loaded by `TldwCli.CSS_PATH` does not contain the design-system classes and semantic tokens.
