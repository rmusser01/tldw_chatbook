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

## Shared Flat-Button Vocabulary

`.console-action-primary`, `.console-action-secondary`, `.console-action-subdued`, and `.console-action-disabled` are the shared cross-screen flat-button vocabulary for destination action toolbars. They render `border: none`, one-row height, and `$ds-surface-raised` (secondary/subdued), `$ds-action-focus`-alpha (primary), or `$ds-surface-panel` (disabled) backgrounds; the exact declarations are verified by `Tests/UI/test_non_obscuring_focus_contract.py`. New destination screens apply these classes to toolbar actions instead of defining screen-local button rules.

## Required State Classes

`.is-active`, `.is-disabled`, `.is-blocked`, `.is-running`, `.is-paused`, `.is-unsaved`, `.is-stale`, `.is-conflict`, `.needs-approval`, `.source-local`, `.source-server`, `.source-workspace`, `.source-remote-only`, `.source-dry-run`

## Density Contract

All new shell surfaces must support `.density-compact` and `.density-comfortable` without separate widget implementations.

## Top Navigation Contract

- Top navigation is global primary destination navigation only.
- `Home` and `Console` remain visible at all supported widths.
- Compact legacy labels such as `W+C` are allowed in the top bar only when tooltip, command palette, and destination header expose the full label and current ownership. New Collections workflows are Library-owned.
- Runtime destination metadata uses `ShellDestination.label` for the visible top-nav label and `ShellDestination.full_label`/`accessible_label` for full names exposed through help, tooltips, and command-palette search.
- `Home` and `Console` are the first priority destinations; `Library` and the watchlist control surface follow as primary product-model destinations rather than local page tabs.
- Overflow must be discoverable and keyboard-reachable; destinations must not disappear silently at narrow widths.
- Destination context, workspace, backend, readiness, and recovery state stay inside destination headers or local panels.

## Shortcut Bar Contract

- The shell footer renders an explicit active shortcut context from the current destination or focused workflow.
- The footer does not infer shortcuts by scraping widgets.
- Runtime shortcut state is represented by `ShortcutContext` and `ShortcutAction` in `tldw_chatbook/UI/Navigation/shortcut_context.py`.
- `AppFooterStatus.set_shortcut_context()` replaces the current footer shortcut display, and `clear_shortcut_context()` restores the global fallback.
- Detailed readiness and recovery state belongs in page headers, panels, inspectors, and recovery callouts.
- If no shortcut context is registered, the footer falls back to global actions; the verified default is command palette and quit.

## Token Mapping Contract

- Design discussion may use dotted token names such as `status.warning`.
- TCSS variables use hyphenated names such as `$ds-status-warning`, `$ds-authority-workspace`, and `$ds-source-role-evidence`.
- Dotted `ds.*` token names must not appear in TCSS variable references; contract tests reject `$ds.*` syntax.
- Core palette values belong in Textual `Theme` fields where possible.
- Design-system-specific semantic values belong in `Theme.variables` and modular TCSS classes; every required semantic token must appear in the `agentic_terminal` theme variables and generated modular stylesheet.
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
- Assert `BaseAppScreen` mounts exactly one `MainNavigationBar` while app-owned shell chrome remains deferred.
- Treat `#console-pending-launch-card`, `*-follow-in-console`, `*-launch-in-console`, and `*-attach-to-console` IDs as behavioral testing hooks, not styling hooks.

## Implementation Status

- Home and all primary destination wrappers use `.ds-destination-header` and `.ds-panel`.
- Console renders pending live-work launch context with `.ds-panel` and a readable source/title label.
- Top navigation metadata now separates compact visible labels from full accessible/help labels.
- Command-palette tab navigation searches destination help text so hidden product-model terms such as Workspaces, Library Collections, flashcards, quizzes, and watchlists remain discoverable.
- Footer shortcut context has a typed source-of-truth API and stale-context regression coverage.
- Semantic token contracts verify hyphenated TCSS names, theme-variable parity, and generated stylesheet coverage.
- Shell chrome ownership is guarded by tests, but full app-owned chrome migration is deferred; `BaseAppScreen` remains the current wrapper seam.
- Static source/artifact/persona/skill actions stage `ChatHandoffPayload` context; live work from W+C, Schedules, Workflows, and ACP uses `open_console_for_live_work()`.
- `tools_settings` resolves to MCP; global preferences are owned by Settings.

This status does not mean every destination screen has been redesigned. Home and Console shell contracts are hardened; Library, Workspaces, Personas, Flashcards, Quizzes, Search, Media, Notes, Handoffs, and the remaining primary destinations remain product-model commitments unless separately verified.

## Stop Conditions

- Stop before shell implementation if the design-system spec or route inventory is unavailable on the implementation branch.
- Stop before shell implementation if the shared design-system TCSS classes, density classes, state classes, and `agentic_terminal` theme are missing and need a separate design-system PR.
- Stop before shell implementation if the generated stylesheet loaded by `TldwCli.CSS_PATH` does not contain the design-system classes and semantic tokens.
