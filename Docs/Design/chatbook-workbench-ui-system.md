# Chatbook Workbench UI System

Date: 2026-06-29

## Purpose

The Workbench UI System is the shared Textual-native frame for Chatbook destinations. It carries the useful engineering practices from Posting into Chatbook: stable composition, explicit state snapshots, typed widget messages, owned async work, contextual help, and verification gates. It does not copy Posting's exact visual language.

Chatbook's frame stays terminal-native, local-first, keyboard-first, and recoverable. Console is the reference implementation because it owns the hardest live workflow: provider readiness, model state, staged context, streaming, tool state, transcript actions, and recovery.

## Target Frame

```text
+----------------------------------------------------------------------------+
| Console                    Mode: Chat/RAG/Follow       Provider: ready      |
| [New tab] [Settings] [Attach] [Run Library RAG] [Save] [Send] [Stop] [Help] |
+-------------+------------------------------------------+-------------------+
| Context     | Transcript / Event Stream                | Inspector         |
| Workspace   |                                          | Provider/model    |
| Staged      | Empty/loading/error/recovery/messages    | Tools/approvals   |
| Sources     |                                          | Run evidence      |
+-------------+------------------------------------------+-------------------+
| Composer: ask, command, paste task...                   [Send] [Stop]       |
+----------------------------------------------------------------------------+
| Footer: F6 next pane | F1 help | Enter send | Ctrl+P palette                |
+----------------------------------------------------------------------------+
```

## Principles

- Stable composition: compose regions once, then sync state into mounted widgets.
- Visible workflows: core actions must be reachable without the command palette.
- Explicit state: headers, modes, panes, recovery, and actions are snapshots.
- Domain ownership: shared widgets do not query databases, providers, or global services.
- Typed messages: widgets emit actions upward instead of mutating domain state directly.
- Responsiveness gates: route migrations must track heartbeat lag, worker counts, timer counts, and mount churn.
- Density by class: normal and compact layouts are class changes, not alternate widget trees.

## Required Visible Actions

Console must visibly expose provider/model settings, send/stop, attach context, Library RAG, inspector review, save Chatbook, help, destructive confirmations, and recovery actions.

The command palette may duplicate visible workflows and expose global navigation, diagnostics, density toggles, and power-user utilities. It must not be the only route to provider/model settings, send/stop, attach context, Library RAG, inspector review, save Chatbook, help, destructive confirmations, or blocked-state recovery.

## State Contract

Workbench screens adapt domain state into immutable snapshots:

- Header: title, subtitle, readiness, density.
- Modes: active work mode and per-mode status.
- Actions: visible commands with disabled, primary, and tooltip state.
- Panes: stable region IDs, titles, status, and collapsed state.
- Recovery: owner, problem, impact, and next action.

Shared Workbench widgets render those snapshots and emit typed messages such as `WorkbenchActionRequested`. They must not call provider services, run database queries, update global app state, or reach into domain widgets.

## Focus And Help

Workbench destinations expose keyboard-first focus and help:

- `F6` cycles visible panes only.
- `F1` opens contextual help for visible actions and shortcuts.
- The footer publishes route-owned shortcut context.
- Focus and hover changes must not remount major regions; density changes may adjust spacing, but must not remount or reflow major regions.

## Responsiveness Gates

Every major route migration must include evidence for:

- event-loop heartbeat lag.
- active worker count before and after route changes.
- active timer count before and after route changes.
- mount churn for major Workbench regions.
- route-switch soak with zero route and focus failures.

The baseline artifacts live under `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/`.

## Migration Rule

Every route in `Constants.py`, `shell_destinations.py`, and `screen_registry.py` must have a migration owner in `tldw_chatbook/UI/Workbench/route_inventory.py`.

Console is the first migrated reference. Later destination migrations should keep their current domain ownership and use separate tasks before retiring legacy screen widgets.
