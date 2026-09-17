---
id: TASK-32681
title: Preserve typed MCP tool results through client services
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:26'
updated_date: '2026-09-17 20:00'
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
- [x] #1 Stdio/client/service tool results retain content, structuredContent, isError and metadata before any display projection, with separate transport-failure identity.
- [x] #2 Malformed error flags, serialization overflow and tool errors cannot turn into successful hook effects.
- [x] #3 Existing non-hook consumers retain compatible presentation through an explicit projection while typed consumers receive the complete result.
- [x] #4 Production client and service tests cover structured-only, text-only, mirrored, error-bearing and oversized results without executing third-party plugins.
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

11. Fix the independently reproduced review I1 audit-publication race under existing R58: coordinate provider bridge fallback and unified-service audit publication per invocation so timeout/cancellation races cannot append contradictory duplicate rows. Preserve truthful uncertain versus not-started state, sanitized failures, failed-submission/pre-start audit coverage and direct service behavior, without a new permission/runtime/storage owner. Runtime edits are restricted to the existing MCP/tool_results.py, MCP/unified_control_plane_service.py and Agents/mcp_tool_provider.py; Tests/MCP/test_typed_tool_results.py carries actual provider/service race regressions with successful controls. Narrow necessary expectation updates in Tests/Agents/test_mcp_tool_provider.py and contract clarification in the already authorized specs/ADR163 are allowed; report additional paths before edits. Capture Fix1 BASE 4f3a170674aae6d4bdeffa827e9a525cd0679d51 snapshots before code. Run the typed-result, provider, control-plane execution, lifecycle and execution-log files under the existing isolated runner, preserve baseline static parity and final frozen hashes, and append exact RED/final evidence to the existing task report. Controller commits and dispatches a fresh scoped re-review before closeout.

12. Apply controller R59 from m1-audit-publication-ruling.md before Fix1 runtime edits. Keep one atomic per-invocation audit publication claim and cap existing unified-service bridge fallback at one in-flight nonqueued metadata-only background append; do not join or hold locks across I/O. Best-effort saturation/start/write failures never change execution authority, duplicate audit publication, fabricate blocked certainty or trigger replay. Preserve actual service/direct/legacy contracts and failed-submit coverage. Qualify both audit-publication races, stalled/failing fallback, saturation and capacity reuse in the already authorized typed-result tests, with owned thread cleanup. Root records the contract in the existing specs/ADR163 before code; no additional source or schema paths.

13. Fix1 covering run exposed the prior expectation in Tests/MCP/test_control_plane_tool_execute.py::test_agent_bridge_outer_timeout_has_one_bridge_owned_audit_row: it reads the best-effort append synchronously and labels an already dispatched uncertain call blocked. Under recorded R59, authorize only this scenario to capture and join its actual owned fallback writer before reading the log, preserve the existing builtin/service/provider timeout path and one-row/initiator/decision assertions, and assert ToolResult uncertainty plus truthful error/completion_uncertain audit metadata. No scenario removal or fixture-wide rewriting. This path is already in the 18-file Fix1 BASE/freeze; include its changed-file static parity. Preserve failed covering/static output, correct the three new broad best-effort exception handlers with justified diagnostics policy, refreeze and rerun the same five covering files.

14. After accepted independent Fix1 review, append the factual I1 Future-publication race incident to backlog/docs/lessons-testing-evidence.md, with the actual duplicate-row probe, committed controls and best-effort limitation. This evidence-only documentation closeout adds no runtime/test behavior. Complete all four ACs and Implementation Notes after preserving the five downstream qualification boundaries and deferred bootstrap-noise observation; controller verifies and commits only the task and lesson metadata.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved complete typed MCP results through the existing dependency-free stdio/client/local/unified/provider path, with strict raw JSON decoding, bounded exact original result bytes, private host dispatch provenance and explicit legacy display projection. Tool/transport errors remain separate from successful structured effects. R58 extends the existing owner; R59 coordinates bridge/service audit publication once and bounds fallback logging to one nonqueued metadata writer per service. A stalled writer may persist and best-effort rows may be lost; uncertainty never authorizes replay.

Validation: initial six-file coverage399 passed15.28s; final Fix1 five-file coverage266 passed14.06s, exit0, no skips. Counts overlap. Eighteen final hashes/BASE snapshots and unchanged runner verified; eight Python static comparisons show no added Ruff/Black issues, new module/test clean. Independent initial review found I1 audit race; fresh scoped Fix1 review accepted I1 with zero new Critical, Important or Minor findings. Initial implementation4f3a170674, Fix19de0ab3ef2.

ADR162/163 apply; R58/R59 contracts are recorded in both specs and ADR163. Optional mcp_unified.gateway legacy-server node remains unqualified because its dependency is absent. HTTP/authentication, H6 effect acceptance, real H3/MCP combined integration and other-platform qualification remain the explicit later-task gates. Bootstrap output noise remains final-review triage.

The factual race/test-qualification incident is recorded in backlog/docs/lessons-testing-evidence.md. No runtime or test edits followed the final freeze/review.
<!-- SECTION:NOTES:END -->
