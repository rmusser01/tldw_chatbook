# ADR-011: Chatbook Workbench UI System

Status: Accepted
Date: 2026-06-29
Related Task: [backlog/tasks/task-141 - Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md](../tasks/task-141%20-%20Implement-Chatbook-Workbench-UI-foundation-and-Console-reference.md)
Supersedes: N/A

## Decision

Chatbook will adopt a shared Textual-native Workbench UI System for major destinations, using stable composition, explicit state snapshots, visible workflow controls, contextual help, responsiveness instrumentation, and route-owner migration gates before retiring legacy screen widgets.

## Context

Chatbook has a mature product direction: local-first agentic knowledge work with Console as the live work surface, Library and Notes as source/research surfaces, and Settings/Personas/Workflows/Watchlists/MCP/ACP/Skills as preparation, configuration, or operational destinations.

The current UI implementation still mixes legacy screens, destination-native screens, direct widget manipulation, broad recomposition paths, route aliases, dynamic mount/remove regions, inconsistent controls, and unclear focus/command ownership. The user also reports random freezes or lockups after differing amounts of time, which makes UI responsiveness and worker/timer ownership part of the architecture decision rather than only a polish concern.

The approved design spec studies Darren Burns' Posting project for Textual engineering practices: stable compose trees, reactive class toggles, lazy heavy panes, typed widget messages, contextual command providers, explicit async workers/timeouts, and snapshot tests. Chatbook should apply those concepts while keeping its own terminal-native, cyberpunk-cozy, local-first visual language.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Continue screen-by-screen visual polish without shared primitives | This would preserve inconsistent controls and likely keep direct widget manipulation, broad remounting, and hidden state behaviors. |
| Copy Posting's exact UI style and jump overlay | The useful part of Posting is the engineering discipline, not the exact visual language. The user explicitly rejected the jump overlay and requested ASCII-only design work. |
| Put most secondary and advanced actions in the command palette | The user raised a valid discoverability concern. Core workflows must remain completable through visible controls, recovery callouts, footer hints, and contextual help. |
| Rewrite all routes in one PR | Too risky for a large Textual app with existing legacy behavior and reported freeze symptoms. Each migration owner needs parity and responsiveness gates before legacy removal. |
| Start with broad visual component work before instrumentation | Without event-loop, worker, timer, and mount-churn baselines, the redesign could make freezes harder to diagnose or accidentally worse. |
| Create a separate product model for every legacy route | This would fragment the app and undermine the approved destination ownership model. Legacy routes should map to durable migration owners and remain compatible until consolidation is reviewed. |

## Consequences

Implementation planning must start from the canonical route inventory in `Constants.py`, `shell_destinations.py`, and `screen_registry.py`. Each current route or alias gets a destination-specific replacement or an explicit reviewed consolidation decision under its migration owner.

Shared UI primitives should be built around clear boundaries:

- frame primitives such as destination headers, mode strips, workbench panes, command strips, and footer context.
- list primitives with stable row identity and reconciliation.
- work-surface primitives for transcript, editor, form, preview, and output workflows.
- state primitives for empty, loading, blocked, error, recovery, authority, and readiness.

Domain screens remain responsible for domain state and services. Shared widgets accept explicit state snapshots and emit Textual messages. Shared primitives must not query DBs, call provider services, or mutate global app state directly.

Responsiveness becomes a gate for UI migration. Before major screen replacement, implementation must capture event-loop heartbeat, worker backlog, timer registry, mount/remove churn, and route-switch or streaming soak evidence. A replacement cannot retire its legacy path if it increases stalls, worker buildup, timer leaks, or unbounded mount churn.

The command palette remains valuable, but it is not the hiding place for core workflows. Primary actions, blocked-state recovery, destructive confirmations, current mode switches, and source/workspace/provider readiness must be visible in the workbench, header, mode strip, inspector, footer, or contextual help. Command palette entries may duplicate visible workflows and expose global navigation, fuzzy search, diagnostics, layout/density toggles, and power-user utilities.

Console is the reference implementation because it exercises the hardest interaction surface: provider readiness, staged context, streaming, transcript reconciliation, message actions, tool calls, workspace context, and recovery. Later migrations follow the approved owner map and receive their own scoped tasks/plans.

## Links

- [Posting-inspired Chatbook UI redesign spec](../../Docs/superpowers/specs/2026-06-29-posting-inspired-chatbook-ui-redesign-design.md)
- [Destination Layout And IA Contracts Design](../../Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md)
- [Workspace operating context handoff PRD](../../Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md)
- [ADR-005: Console Workspace Server-Readiness Boundary](005-console-workspace-server-readiness.md)
- [ADR-007: Personas Workbench Route Consolidation](007-personas-workbench-route-consolidation.md)
- [ADR-010: Console Conversation Local Marks](010-console-conversation-local-marks.md)
