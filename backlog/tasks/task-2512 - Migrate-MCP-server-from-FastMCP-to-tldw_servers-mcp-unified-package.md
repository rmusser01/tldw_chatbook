---
id: TASK-2512
title: Migrate MCP server from FastMCP to tldw_server's mcp-unified package
status: To Do
assignee: []
created_date: '2026-08-06 07:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP server (tldw_chatbook/MCP/server.py) currently uses the official SDK's FastMCP 1.x (optional mcp[cli] extra). It should instead use the standalone mcp-unified package from tldw_server (apps/mcp-unified, on PyPI, GPL-3.0-only — fine as a dependency for this AGPLv3+ project). Decisions (user, 2026-08-05): FULL migration — all 10 built-in tools, resources, prompts, and the phase-4 local agent tools (fs_*/git_*/web_*/fs_patch, permission-gated via LocalToolProvider) move to mcp-unified as modules served via its stdio gateway; dependency via PyPI optional extra. Research pointers: gateway in apps/mcp-unified/src/mcp_unified/gateway/{jsonrpc,fastapi,stdio}.py; modules subclass BaseModule (get_tools/execute_tool) loaded via ModuleRegistry (tldw_Server_API/app/core/MCP_unified/modules/base.py, registry.py); profiles/permission_rules.py + policy_grants/ for the permission layer. KEY UNKNOWN to resolve first: whether mcp-unified supports MCP resources and prompts (not just tools) — chatbook's server exposes both; if unsupported, the spec must decide the fallback. Also: protocol version compat with our hand-rolled client (2025-03-26), and whether serving stdio is programmatic or CLI-only. Refs: ADR-032/033, re-plan spec Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md, task-2511 (FastMCP smoke — may become moot if this lands first).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mcp-unified API surface verified (programmatic stdio serving, module registration, resources/prompts support, permission hook),Spec written covering full migration incl. resources/prompts decision and local-tool gating,Server migrated: all built-in tools + local tools served via mcp-unified, mcp[cli] extra replaced by mcp-unified extra,Existing MCP tests updated; new tests cover the module/gateway mapping,Backwards-compat story for MCP client configs documented
<!-- AC:END -->
