---
id: TASK-3605
title: Enable fail-closed MCP Hub execution for local agent tools
status: To Do
assignee: []
created_date: '2026-08-08 19:02'
updated_date: '2026-08-30 17:24'
labels:
  - mcp
  - agents
  - hub
  - security
dependencies:
  - TASK-2838
references:
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
  - Docs/superpowers/specs/2026-07-13-mcp-hub-redesign-design.md
  - Docs/superpowers/specs/2026-08-30-mcp-hub-local-tool-execution-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP Hub lists local workspace tools and manages their shared permission state but intentionally marks them non-executable. Add the missing operator-initiated Test Tool path through a fresh, descriptor-filtered `LocalToolProvider` so users can exercise eligible `local:__local__` tools without opening a raw `tools/call` bypass or weakening workspace confinement, current-definition permission checks, execution ownership, or auditability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Hub Test Tool is available only for catalogued `local:__local__` tools whose code-owned descriptor permits shared Console/external-MCP exposure; catalogued Console-only tools remain visible but non-executable, and session-owned tools remain absent
- [ ] #2 Every Hub Test Tool Ask verdict uses the explicit one-click "Approve & run once" action without a separate armed-confirm state; click intent is bound to an immutable rendered preview, so a fresh Ask reached from rendered Allow or any definition/root change blocks and refreshes instead of executing
- [ ] #3 A one-time Ask approval is bound to the rendered full tool identity, current definition hash, canonical exact arguments, strict canonical root plus directory-identity chain, service-issued single-use panel nonce, and invocation; it is consumed at most once, never persists, and never authorizes a later or changed run
- [ ] #4 The complete local admission and invocation pipeline runs off the Textual UI loop under a service-owned in-flight registry, honors each tool's code-owned timeout override and execution policy, and cannot admit a duplicate or report cancellation/timeout while a definitive mutation may still commit
- [ ] #5 No raw MCP `tools/call` route is opened, `todo_*` and other Console-only tools remain unavailable, and all path-taking handlers remain confined to the freshly resolved workspace root
- [ ] #6 The control-plane service owns preview issuance/revocation/atomic consumption and one typed local execution outcome carrying final gate, approval consumption, refusal category, dispatch-started state, and terminal; only its coordinator may synthesize timeout or detached-cancellation, and late worker completion cannot replace or re-audit a sealed outcome; the service attempts at most one best-effort terminal audit row per admitted test without matching refusal text, while display and audit derive from the same root-redacted result and expose no absolute workspace paths or secrets
- [ ] #7 Automated tests cover executable projection, generic one-click Ask UX, Allow-to-Ask and Ask-to-Off races, definition/root/ancestor-identity preview mismatch, revoked/expired/reused preview nonces and concurrent double-click, exact argument binding, local Allow/Ask/Off, gate failure, disabled configuration, provider/root failure, confinement, typed detailed-provider outcomes and ordinary-invoke compatibility, coordinator-owned timeout/detached cancellation and late-worker cleanup, timeout precedence, remount/duplicate/cancellation ownership, non-persistence, at-most-one audit finalization and audit-write failure, kill-switch-independent diagnostics, and the unchanged raw-call refusal
<!-- AC:END -->

## ADR Check

ADR required: yes, by amendment of an existing decision

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: ADR-032 already owns the synthetic `local:__local__` principal, descriptor exposure, shared permission store, definition-hash checks, confinement, approval discipline, and post-dispatch execution policy. TASK-3605 amends it to make the operator-only Hub carve-out explicit: configured Off blocks, Ask is a rendered one-click approval, and the chat/runtime kill switch does not block an in-app diagnostic. The MCP Hub design is corrected to match this existing fail-closed behavior.

## Design

See `Docs/superpowers/specs/2026-08-30-mcp-hub-local-tool-execution-design.md`. The implementation plan will be added after the design is approved and the task is moved to In Progress.
