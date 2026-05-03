# Master Shell Design System Contract

Date: 2026-05-03

## Purpose

This contract maps the agentic terminal design system onto the master-shell implementation slices. It prevents Home, Console, and destination wrappers from inventing screen-local visual language while the shell migration is in progress.

## Source Documents

- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Docs/Design/master-shell-route-inventory.md`

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

## Required Readable Status Labels

`Ready`, `Running`, `Paused`, `Blocked`, `Unavailable`, `Approval required`, `Unsaved`, `Recovered`

## Testing Rules

- Assert stable IDs or classes for primary actions, status badges, source authority, source roles, approval cards, shortcut bars, next-best actions, and open/follow-in-Console controls.
- Assert readable status text. Do not assert raw color values.
- Assert destination context appears inside page headers, not the global top navigation.
- Treat `#console-pending-launch-card`, `*-follow-in-console`, `*-launch-in-console`, and `*-attach-to-console` IDs as behavioral testing hooks, not styling hooks.

## Implementation Status

- Home and all primary destination wrappers use `.ds-destination-header` and `.ds-panel`.
- Console renders pending live-work launch context with `.ds-panel` and a readable source/title label.
- Static source/artifact/persona/skill actions stage `ChatHandoffPayload` context; live work from Watchlists+Collections, Schedules, Workflows, and ACP uses `open_console_for_live_work()`.
- `tools_settings` resolves to MCP; global preferences are owned by Settings.

## Stop Conditions

- Stop before shell implementation if the design-system spec or route inventory is unavailable on the implementation branch.
- Stop before shell implementation if the shared design-system TCSS classes, density classes, state classes, and `agentic_terminal` theme are missing and need a separate design-system PR.
- Stop before shell implementation if the generated stylesheet loaded by `TldwCli.CSS_PATH` does not contain the design-system classes and semantic tokens.
