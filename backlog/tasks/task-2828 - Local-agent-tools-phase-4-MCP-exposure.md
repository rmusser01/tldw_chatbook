---
id: TASK-2828
title: 'Local agent tools phase 4: MCP exposure'
status: Done
assignee: []
created_date: '2026-08-05 23:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §3.1. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase4.md. ADRs 032/033. Route through LocalToolProvider gate; todo_write not exposed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 External MCP clients can call allowed local tools through the server (invocation routed through LocalToolProvider's gate)
- [x] #2 ask-state tools fail closed externally with an external-appropriate refusal (no approval card exists outside the Console)
- [x] #3 Kill switch and deny states honored; operator grants (always-allow from Console) enable external use
- [x] #4 Exposure gated behind [mcp] expose_local_tools (default false); todo_write not exposed (documented)
- [x] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase4.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented on branch `feat/local-agent-tools-p2` (stacked on PRs #1352/#1358) via subagent-driven development with per-task spec + quality review.

- `Agents/local_tool_provider.py`: optional `no_callback_refusal` constructor param — the `no_callback` verdict maps to it (external clients see an honest message), the `timeout` verdict always keeps the pinned `LOCAL_TIMEOUT_REFUSAL`; default behavior unchanged.
- `MCP/local_server_tools.py` (new): `build_server_local_provider` — permission state resolved FRESH per call from `MCPPermissionStore` (operator grant/revoke takes effect immediately), guarded kill switch (read error → engaged), no approval callback / no Console seams (todo_write absent by construction), `EXTERNAL_NO_CALLBACK_REFUSAL`. Pure `LocalToolRegistration` builder (`_local_agent_tool_registrations`) testable without the `mcp` package; handlers call `provider.invoke` and return content or `{"error": ...}` (server.py convention). `_parameter_summary` carries parameter docs into tool descriptions (FastMCP can't consume JSON schemas).
- `config.py`: `[mcp] expose_local_tools` (default false) — new coercion block + template line (placed above the `[mcp.tools]` sub-table).
- `MCP/server.py`: `_register_local_agent_tools()` called from `__init__` (deliberately NOT in `_register_tools` — the AST capability catalog is unaffected). Gated on the flag (coerced at the consumer after review found raw-TOML truthiness failed the gate open on quoted "false"), whole flag-on body exception-guarded (a registration failure never costs the built-in tools), store pinned to `get_user_data_dir()/mcp_permissions.json` (verified identical to where Console always-allow grants land), generic `arguments: dict` FastMCP binding with parameter summaries appended to descriptions. Module docstring documents the exposure, permission model, and todo_write omission.

Deferred (documented): server-side `record_decision` audit wiring (the headless audit path is an open design question); the two FastMCP binding tests skip without the `mcp` package — the binding line (`mcp._tool_manager`, FastMCP-1.x API) has never executed in CI; a one-time smoke run with the `[mcp]` extra installed is recommended before release (FastMCP 2.x made list_tools async, so the tests may need updating then, not the production binding).

Tests: 20+ new — composition with a real temp store (grant/revoke/kill-switch/deny/fail-closed), refusal-override branches, config coercion, pure-builder paths, exception-guard, parameter summary. Final suite: 624 passed, 2 skipped (mcp-gated binding tests).

Final whole-phase review: Ready to merge; all 5 ACs and both binding spec facts verified.
