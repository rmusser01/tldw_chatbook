---
id: TASK-3605
title: Enable fail-closed MCP Hub execution for local agent tools
status: Done
assignee: []
created_date: '2026-08-08 19:02'
updated_date: '2026-08-31 19:35'
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
- [x] #1 Hub Test Tool is available only for catalogued `local:__local__` tools whose code-owned descriptor permits shared Console/external-MCP exposure; catalogued Console-only tools remain visible but non-executable, and session-owned tools remain absent
- [x] #2 Every Hub Test Tool Ask verdict uses the explicit one-click "Approve & run once" action without a separate armed-confirm state; click intent is bound to an immutable rendered preview, so a fresh Ask reached from rendered Allow or any definition/root change blocks and refreshes instead of executing
- [x] #3 A one-time Ask approval is bound to the rendered full tool identity, current definition hash, canonical exact arguments, strict canonical root plus directory-identity chain, service-issued single-use panel nonce, and invocation; it is consumed at most once, never persists, and never authorizes a later or changed run
- [x] #4 The complete local admission and invocation pipeline runs off the Textual UI loop under a service-owned in-flight registry, honors each tool's code-owned timeout override and execution policy, and cannot admit a duplicate or report cancellation/timeout while a definitive mutation may still commit
- [x] #5 No raw MCP `tools/call` route is opened, `todo_*` and other Console-only tools remain unavailable, and all path-taking handlers remain confined to the freshly resolved workspace root
- [x] #6 The control-plane service owns preview issuance/revocation/atomic consumption and one typed local execution outcome carrying final gate, approval consumption, refusal category, dispatch-started state, and terminal; only its coordinator may synthesize timeout or detached-cancellation, and late worker completion cannot replace or re-audit a sealed outcome; the service attempts at most one best-effort terminal audit row per admitted test without matching refusal text, while display and audit derive from the same root-redacted result and expose no absolute workspace paths or secrets
- [x] #7 Automated tests cover executable projection, generic one-click Ask UX, Allow-to-Ask and Ask-to-Off races, definition/root/ancestor-identity preview mismatch, revoked/expired/reused preview nonces and concurrent double-click, exact argument binding, local Allow/Ask/Off, gate failure, disabled configuration, provider/root failure, confinement, typed detailed-provider outcomes and ordinary-invoke compatibility, coordinator-owned timeout/detached cancellation and late-worker cleanup, timeout precedence, remount/duplicate/cancellation ownership, non-persistence, at-most-one audit finalization and audit-write failure, kill-switch-independent diagnostics, and the unchanged raw-call refusal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, by amendment of ADR-032.

Detailed plan: `Docs/superpowers/plans/2026-08-30-mcp-hub-local-tool-execution.md`.

1. Add a structured `LocalToolProvider.invoke_detailed()` seam while preserving ordinary `invoke()` behavior.
2. Build a closable descriptor-filtered Hub-local provider and project only shared descriptors as executable.
3. Add exact argument canonicalization, `DirectoryChain` authority binding, and a service-owned single-use preview registry.
4. Put every Hub Test Tool click behind immutable prepared admission and explicit click intent.
5. Execute local tests under service-owned timeout, cancellation, definitive-after-start, cleanup, redaction, and audit ownership.
6. Replace the Workbench's armed-confirm state with preview-backed one-click Ask behavior.
7. Run the focused Agent/MCP/UI security and lifecycle matrix, static analysis, compilation, diagnostics, diff hygiene, and independent review.
8. Check every acceptance criterion from evidence, add implementation notes, mark the task Done, and re-run exact-head gates before PR/merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fail-closed MCP Hub execution for descriptor-approved local tools. A closable Hub-local provider projects only `console_and_external_mcp` descriptors; `LocalToolProvider.invoke_detailed()` preserves the ordinary `invoke()` result contract while exposing typed gate, approval, dispatch, refusal, and terminal facts to the control plane. Service-issued previews bind full identity, definition hash, canonical exact arguments, the strict `DirectoryChain` authority fingerprint, click intent, and a bounded single-use nonce. The service owns atomic admission, off-UI-loop execution, timeout/cancellation policy, definitive-after-start completion, cleanup, and the one best-effort terminal audit. The Workbench now offers one-click **Approve & run once** for Ask and derives display/audit from the same root- and secret-redacted result.

Acceptance evidence:

- AC #1: descriptor projection and absence/visibility contracts are covered by `test_hub_local_factory_filters_shared_descriptors_and_wires_runtime_seams`, `test_console_only_watchlists_tools_are_never_externally_registered`, `test_hub_tools_omits_all_task_tools_without_a_todo_store`, and the mounted Hub local-group projection tests. Static review confirmed there is no bare-name eligibility allowlist.
- AC #2: `test_test_tool_preview_ask_is_one_click_approve_once_and_keeps_edits`, `test_test_tool_one_click_ask_dispatches_approve_once_from_first_activation`, the Allow-to-Ask/Ask-to-Off prepared-gate race matrix, and definition-change refresh coverage prove click intent is never silently promoted to approval.
- AC #3: canonical-argument, authority-fingerprint, private-preview-registry, nonce revoke/expire/reuse, concurrent double-click, definition/ancestor-identity mismatch, one-shot consumption, and non-persistence tests cover every binding component and replay boundary.
- AC #4: `test_local_hub_entire_rebuild_gate_compare_policy_and_invoke_are_off_loop`, timeout-floor and lifecycle-deadline tests, duplicate/remount/cancellation coverage, and `test_local_hub_definitive_after_start_detaches_caller_but_audits_actual_terminal` prove service ownership and execution-policy behavior.
- AC #5: the unchanged raw `tools/call` refusal matrix, absent `todo_*`/Console-only registration tests, admitted-root guard and replacement tests, and fresh click-time configuration/root failure tests prove the transport and confinement boundaries.
- AC #6: detailed-provider compatibility/reason tests, the closed terminal matrix, structured-fact-not-text classification, timeout/detached-cancellation late-worker tests, single-audit and audit-write-failure tests, and shared recursive redaction tests prove typed ownership, at-most-one finalization, and common safe output.
- AC #7: the complete Task 7 Agent/MCP/UI matrix passed **1,110 tests**; the persistent-diagnostic architecture suite passed **67 tests**. Touched-file Ruff check/format, scoped compilation, the diagnostic inventory checker, and diff hygiene were clean. Independent security/static review found no remaining blocking invariant violation after lifecycle, nonce-revocation, unresolved-gate-copy, and path/secret-redaction corrections.

Core production changes are in `tldw_chatbook/Agents/local_tool_provider.py`, `tldw_chatbook/MCP/hub_test_execution.py`, `tldw_chatbook/MCP/local_server_tools.py`, `tldw_chatbook/MCP/unified_control_plane_service.py`, `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`, and `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`; focused coverage is in the corresponding Agent, MCP control-plane/local-runtime, and UI test modules listed by the implementation plan. Review-driven diagnostics required an intentional generated update to `Docs/security/production-diagnostic-inventory.json`.

Plan deviations: review expanded the original focused regression set to cover fail-closed path scrubbing, cancellation-proof preview cleanup, pre-admission nonce revocation, and honest unresolved-gate copy. The generated diagnostic inventory was refreshed after semantic review. No full repository sweep was run, per project policy, and no new lesson was added because the work did not reveal a new generalizable incident beyond the existing testing/backlog guidance.

Governance: [ADR-032](../decisions/032-local-agent-tool-permission-boundary.md), [approved design](../../Docs/superpowers/specs/2026-08-30-mcp-hub-local-tool-execution-design.md), and [implementation plan](../../Docs/superpowers/plans/2026-08-30-mcp-hub-local-tool-execution.md).
<!-- SECTION:NOTES:END -->

## ADR Check

ADR required: yes, by amendment of an existing decision

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: ADR-032 already owns the synthetic `local:__local__` principal, descriptor exposure, shared permission store, definition-hash checks, confinement, approval discipline, and post-dispatch execution policy. TASK-3605 amends it to make the operator-only Hub carve-out explicit: configured Off blocks, Ask is a rendered one-click approval, and the chat/runtime kill switch does not block an in-app diagnostic. The MCP Hub design is corrected to match this existing fail-closed behavior.

## Design

See `Docs/superpowers/specs/2026-08-30-mcp-hub-local-tool-execution-design.md`.
