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
  - backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md
  - Docs/superpowers/specs/2026-08-09-mcp-unified-standalone-server-migration-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP server (tldw_chatbook/MCP/server.py) currently uses the official SDK's FastMCP 1.x (optional mcp[cli] extra). It should instead use the standalone mcp-unified package from tldw_server (apps/mcp-unified, on PyPI, GPL-3.0-only — fine as a dependency for this AGPLv3+ project). Decisions (user, 2026-08-05): FULL migration — all 10 built-in tools, resources, prompts, and the phase-4 local agent tools (fs_*/git_*/web_*/fs_patch, permission-gated via LocalToolProvider) move to mcp-unified as modules served via its stdio gateway; dependency via PyPI optional extra. Research pointers: gateway in apps/mcp-unified/src/mcp_unified/gateway/{jsonrpc,fastapi,stdio}.py; modules subclass BaseModule (get_tools/execute_tool) loaded via ModuleRegistry (tldw_Server_API/app/core/MCP_unified/modules/base.py, registry.py); profiles/permission_rules.py + policy_grants/ for the permission layer. KEY UNKNOWN to resolve first: whether mcp-unified supports MCP resources and prompts (not just tools) — chatbook's server exposes both; if unsupported, the spec must decide the fallback. Also: protocol version compat with our hand-rolled client (2025-03-26), and whether serving stdio is programmatic or CLI-only. Refs: ADR-032/033, re-plan spec Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md, task-2511 (FastMCP smoke — may become moot if this lands first).

Update (2026-08-08, via task-1337 close-out / PR #1435): (1) LANDMINE — `mcp[cli]>=1.0.0` now resolves to mcp 2.0.0, which REMOVES `mcp.server.fastmcp`; a fresh install therefore breaks the legacy standalone server (`TldwMCPServer`, `MCP/__main__.py`) today, and the current dev venv ships with NO `mcp` installed (the in-app surface is unaffected). This makes the migration time-sensitive rather than cosmetic. (2) Scope reduction — the 18 descriptor-backed `library_*` tools (task-1337) are already FastMCP-free: they ride the capability manifest (`MCP/server.py::_describe_local_library_tools`) plus the direct runtime delegate (`LocalMCPRuntimeDelegate.execute_tool` via `asyncio.to_thread`) with the shared-service factory `build_local_library_tool_service`; they need NO migration. Remaining scope is the 10 legacy built-ins + resources + prompts + the phase-4 local agent tools. (3) Contract to preserve — the in-app direct runtime now refuses raw protocol `tools/call` for every tool with typed `RawToolCallRefusedError` (execution only via the gated, logged Execute Local Tool action); the migrated server must keep an equivalent policy-gated path, and `Tests/MCP/test_library_tools.py` (legacy-name/shape pinning) must stay green throughout.

Resolved design (2026-08-09): public `mcp-unified==0.2.1` supplies the required programmatic stdio, resources, resource templates, prompts, typed errors, and all required protocol revisions. ADR-053 and the linked specification select a thin `GatewayCoreRuntime` adapter, explicitly exclude the 18 in-app-only Library tools from standalone stdio, define canonical tool/resource/prompt mappings, bound long resources with continuation metadata, preserve the phase-4 permission gate, and retain the existing client command.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The public `mcp-unified==0.2.1` programmatic stdio, tool, resource, resource-template, prompt, typed-error, and required protocol-revision surfaces are verified against the real package.
- [ ] #2 ADR-053, the reviewed migration specification, and the implementation plan define the standalone runtime, canonical mappings, privacy boundary, resource continuation, and local-tool permission behavior.
- [ ] #3 The standalone server exposes all ten legacy built-in tools, five resource templates plus the dynamic resource catalog, five prompts, and all explicitly enabled phase-4 local-agent tools through `mcp-unified`.
- [ ] #4 The eighteen in-app-only `library_*` tools remain available through their descriptor-backed direct runtime, remain absent from standalone stdio, and retain raw in-app `tools/call` refusal.
- [ ] #5 Tool values, fixed/redacted typed local-tool failures, namespaced resource dictionaries, long resource chunks, non-empty prompt messages, and prompt arguments map to bounded canonical MCP results with regression coverage.
- [ ] #6 Both `mcp[cli]` declarations and every live availability/dependency surface are replaced by the exact `mcp-unified==0.2.1` optional dependency, with no production FastMCP or official-SDK import remaining.
- [ ] #7 Chatbook's hand-written client negotiates `2025-03-26`, the adapter passes real `2025-11-25` and current-profile flows with batching limited to `2025-03-26`, catalog pagination fails closed at malformed or over-bound cursors, resource `_meta` is preserved exactly, and the existing subprocess configuration command remains valid.
- [ ] #8 Wheel and sdist `[mcp]` installs independently pass a site-packages-isolated standalone protocol smoke confined to temporary configuration/data/workspace paths, and the relevant MCP, Library, packaging, documentation, static-analysis, and security gates are green or compared against an identical clean-dev baseline.
- [ ] #9 User and developer documentation explain installation, supported protocol behavior, continuation, the standalone-versus-in-app Library boundary, and the privacy risk of exposing local data to an external MCP client or cloud model.
<!-- AC:END -->
