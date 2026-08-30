---
id: TASK-3605
title: Enable fail-closed MCP Hub execution for local agent tools
status: To Do
assignee: []
created_date: '2026-08-08 19:02'
updated_date: '2026-08-30 16:32'
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
- [ ] #1 Hub Test Tool is available only for catalogued `local:__local__` tools whose code-owned descriptor permits shared Console/external-MCP exposure, while Console-only and session-owned tools remain visible but non-executable
- [ ] #2 Every run resolves a fresh provider, workspace authority, tool definition, and shared Off/Ask/Allow verdict immediately before dispatch; Allow runs directly, Ask offers one-click "Approve & run once", and Off or unresolved state cannot dispatch
- [ ] #3 A one-time Ask approval is bound to the selected tool, current definition, arguments, and invocation, is revalidated on click, and never persists or authorizes a later run
- [ ] #4 Local Hub execution runs off the Textual UI loop, honors each tool's code-owned execution policy after dispatch, and cannot report cancellation or timeout while a definitive mutation may still commit
- [ ] #5 No raw MCP `tools/call` route is opened, `todo_*` and other Console-only tools remain unavailable, and all path-taking handlers remain confined to the freshly resolved workspace root
- [ ] #6 Results, failures, denials, and approval outcomes are bounded, redacted, and recorded in the existing MCP execution audit trail without exposing absolute workspace paths or secrets
- [ ] #7 Automated tests cover executable projection, Allow, one-click Ask, Off, gate failure, definition change, disabled configuration, provider/root failure, confinement, cancellation ownership, non-persistence, audit records, and the unchanged raw-call refusal
<!-- AC:END -->

## ADR Check

ADR required: no new ADR

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: ADR-032 already owns the synthetic `local:__local__` principal, descriptor exposure, shared permission store, definition-hash checks, confinement, approval discipline, and post-dispatch execution policy. The accepted MCP Hub design already defines operator-initiated Test Tool execution and its audit trail. This task joins those existing boundaries without adding storage, a new principal, a new transport, or a new authorization policy.

## Design

See `Docs/superpowers/specs/2026-08-30-mcp-hub-local-tool-execution-design.md`. The implementation plan will be added after the design is approved and the task is moved to In Progress.
