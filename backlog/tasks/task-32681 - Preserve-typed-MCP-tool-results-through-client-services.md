---
id: TASK-32681
title: Preserve typed MCP tool results through client services
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:26'
updated_date: '2026-09-17 18:51'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retain the protocol information needed to distinguish tool errors from successful structured results while keeping ordinary display compatible.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stdio/client/service tool results retain content, structuredContent, isError and metadata before any display projection, with separate transport-failure identity.
- [ ] #2 Malformed error flags, serialization overflow and tool errors cannot turn into successful hook effects.
- [ ] #3 Existing non-hook consumers retain compatible presentation through an explicit projection while typed consumers receive the complete result.
- [ ] #4 Production client and service tests cover structured-only, text-only, mirrored, error-bearing and oversized results without executing third-party plugins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/111-mcp-remote-transport-and-client-dependency.md
Reason: Direct implementation of the accepted typed MCP result boundary, preserving the existing dependency-free client and separate display projection. No new transport or permission owner.

1. Read task14 brief, actual TASK32681, the accepted typed-result sections and final H1/R41 raw-provenance requirements. Recheck current MCP/client.py, local_control_service.py and Agents/mcp_tool_provider.py, plus any prior task changes, before code. Use existing Docs/superpowers/specs/2026-09-15-managed-plugins-design.md, Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md and backlog/decisions/163-expanded-console-hook-runtime.md for typed-result/provenance notes; ADR111 transport ownership is reconciled under the accepted R4 in M2, not changed in M1. Notify controller before extra paths or an unsettled contract.
2. Establish behavioral RED in Tests/MCP/test_typed_tool_results.py around the current client/service raw result loss with a valid same-entry success control; distinguish missing import scaffolding. Cover structured-only, mirrored text and error-with-pass payloads through actual controlled stdio/client/service entries, without third-party execution.
3. Add MCP/tool_results.py with the frozen complete typed result, strict boolean isError, bounded data ownership and explicit legacy display projection. Retain content, structuredContent and metadata before normalization; distinguish sanitized transport failure from tool-declared error and empty success. Preserve original duplicate-rejecting raw decoding/framing evidence rather than claiming serialization restores it.
4. Correct the first loss in _StdioJSONRPCConnection.call_tool and the outer MCPClient call path, then wire typed client/service/provider use. Keep the existing dependency-free custom stdio client; do not add an SDK to satisfy obsolete plan wording. Preserve supported actual raw/model representations at their real boundary.
5. Enforce the brief's exact protocol framing/result/depth/catalog limits independently of the hook-specific result cap. Complete typed data, including metadata, reaches typed consumers before any presentation truncation. Reject malformed flags, nonfinite/deep/oversized/unserializable content and preserve provenance without logging remote payloads, credentials or arbitrary exception text.
6. Keep existing non-hook rendering/tool-log shapes compatible through project_tool_result and legacy call_tool delegation. Cover resource/image content, text-only, missing/false flags, structured-only, mirrored, tool-error, transport-error and oversized metadata with successful same-entry controls. No error sentinel may turn into a successful hook effect.
7. Run the exact three brief files: Tests/MCP/test_typed_tool_results.py, Tests/MCP/test_client_catalog_pagination.py and Tests/MCP/test_control_plane_tool_execute.py, plus only controller-recorded affected neighbors. Use isolated HOME/XDG/config profiles, null keyring, network refusal and verified worktree package/data-root provenance; preserve frozen hashes through final covering runs. No full sweep.
8. Record final public API, complete result provenance and separate projection contracts, static baseline parity, exact commands/output and final frozen hashes in task-14-report.md and the existing documentation paths named in step1. State H6 and HTTP-transport handoffs precisely. Controller owns exact staging/commits, independent spec/quality review, all four ACs, final notes and Done after gates pass.

Verification handoff from H2 review F3: any child bootstrap/network/profile fixture needed by committed tests must be repository-owned and use the actual pytest temporary root. A private implementation runner may isolate the parent, but cannot be a hidden prerequisite for the ordinary exact test command. Reuse the final reviewed H2 fixture where appropriate; controlled HTTP tests must explicitly scope their own loopback allowance rather than disable network checks globally. Record extra helper paths before edits.

9. Activation precheck while P1 runs confirms the first-loss/raw-decode seams still exist at _StdioJSONRPCConnection.call_tool, MCPClient.call_tool and the stdio json.loads reader. Because M1 changes the actual provider invocation path, include unchanged Tests/Agents/test_mcp_tool_provider.py as an affected regression. Include the exact existing legacy-client node Tests/MCP/test_mcp_unified_stdio.py::test_real_legacy_client_command_discovers_calls_continues_and_stops_child, not its unrelated standalone-server suite. That module has an explicit mcp_unified.gateway optional import; report a missing/incompatible dependency before treating a skip as evidence or changing fixtures. The new controlled dependency-free stdio/client/service tests remain mandatory. These are read-only regression additions, not permission to edit predecessor fixtures or expand transport scope. Record any necessary extra fixture/source path before that change.

10. Apply controller R58 from m1-typed-provenance-ruling.md before production edits. Extend the existing MCP/unified_control_plane_service.py execution owner with an explicit typed external-result entry and shared legacy display projection; no second permission, timeout, cancellation or audit owner. Preserve bounded exact result-value bytes plus private host decode/dispatch provenance, and a per-call observation across cancelled awaits. Unknown/mutable/copied/remote-authored provenance cannot qualify hook effects; unacknowledged cancellation cannot prove not_started. Record the contract in the existing expanded-hook spec/ADR163 and managed-plugin spec before code. Tests/Agents/test_mcp_tool_provider.py may receive only necessary typed fixtures/sanitized diagnostic expectation updates while preserving all scenarios; also run unchanged Tests/MCP/test_control_plane_lifecycle.py and Tests/MCP/test_execution_log.py for the shared service/audit change. Keep the missing optional mcp-unified exact-node check explicitly unqualified, without installing dependencies or changing its fixture. Controlled dependency-free actual client/service/provider coverage remains mandatory. Preserve BASE snapshots of all added paths.
<!-- SECTION:PLAN:END -->
