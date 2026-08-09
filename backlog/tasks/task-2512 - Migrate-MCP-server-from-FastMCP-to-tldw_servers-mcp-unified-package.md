---
id: TASK-2512
title: Migrate MCP server from FastMCP to tldw_server's mcp-unified package
status: To Do
assignee: []
created_date: '2026-08-06 07:18'
labels:
  - mcp
  - fastmcp
  - migration
dependencies: []
references:
  - backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md
  - backlog/decisions/030-local-library-agent-tool-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP server (tldw_chatbook/MCP/server.py) currently uses the official SDK's FastMCP 1.x (optional mcp[cli] extra). It should instead use the standalone mcp-unified package from tldw_server (apps/mcp-unified, on PyPI, GPL-3.0-only — fine as a dependency for this AGPLv3+ project). Decisions (user, 2026-08-05): FULL migration — all 10 built-in tools, resources, prompts, and the phase-4 local agent tools (fs_*/git_*/web_*/fs_patch, permission-gated via LocalToolProvider) move to mcp-unified as modules served via its stdio gateway; dependency via PyPI optional extra. Research pointers: gateway in apps/mcp-unified/src/mcp_unified/gateway/{jsonrpc,fastapi,stdio}.py; modules subclass BaseModule (get_tools/execute_tool) loaded via ModuleRegistry (tldw_Server_API/app/core/MCP_unified/modules/base.py, registry.py); profiles/permission_rules.py + policy_grants/ for the permission layer. KEY UNKNOWN to resolve first: whether mcp-unified supports MCP resources and prompts (not just tools) — chatbook's server exposes both; if unsupported, the spec must decide the fallback. Also: protocol version compat with our hand-rolled client (2025-03-26), and whether serving stdio is programmatic or CLI-only. Refs: ADR-032/033, re-plan spec Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md, task-2511 (FastMCP smoke — may become moot if this lands first).

Update (2026-08-08, via task-1337 close-out / PR #1435): (1) LANDMINE — `mcp[cli]>=1.0.0` now resolves to mcp 2.0.0, which REMOVES `mcp.server.fastmcp`; a fresh install therefore breaks the legacy standalone server (`TldwMCPServer`, `MCP/__main__.py`) today, and the current dev venv ships with NO `mcp` installed (the in-app surface is unaffected). This makes the migration time-sensitive rather than cosmetic. (2) Scope reduction — the 18 descriptor-backed `library_*` tools (task-1337) are already FastMCP-free: they ride the capability manifest (`MCP/server.py::_describe_local_library_tools`) plus the direct runtime delegate (`LocalMCPRuntimeDelegate.execute_tool` via `asyncio.to_thread`) with the shared-service factory `build_local_library_tool_service`; they need NO migration. Remaining scope is the 10 legacy built-ins + resources + prompts + the phase-4 local agent tools. (3) Contract to preserve — the in-app direct runtime now refuses raw protocol `tools/call` for every tool with typed `RawToolCallRefusedError` (execution only via the gated, logged Execute Local Tool action); the migrated server must keep an equivalent policy-gated path, and `Tests/MCP/test_library_tools.py` (legacy-name/shape pinning) must stay green throughout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mcp-unified API surface verified (programmatic stdio serving, module registration, resources/prompts support, permission hook),Spec written covering full migration incl. resources/prompts decision and local-tool gating,Server migrated: all built-in tools + local tools served via mcp-unified, mcp[cli] extra replaced by mcp-unified extra,Existing MCP tests updated; new tests cover the module/gateway mapping,Backwards-compat story for MCP client configs documented
<!-- AC:END -->
