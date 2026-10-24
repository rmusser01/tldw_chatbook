---
id: TASK-18928
title: 'Approval history mining: "Always allow" suggestions surface'
status: Done
assignee:
- '@Robert'
created_date: 2026-08-19 09:55
updated_date: 2026-10-18 16:30
labels:
- console
- agents
- approvals
- mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `approvals suggest` (2026-08-19 hermes-release review). Chatbook users repeatedly pick "Approve once" for the same safe MCP tools. Mine the user's own approval history into proposed "Always allow" entries, surfaced for explicit review in Settings (MCP permissions area): the user accepts or dismisses each proposal; accepting writes a real Always-allow entry through the existing permission store. Nothing is ever auto-allowed — the feature only drafts proposals, and runtime approval behavior does not change until the user accepts one. Proposals are definition-fingerprint aware, consistent with the existing "(definition changed)" badge semantics, and high-risk tool classes are excluded from suggestion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Settings surface lists suggestions mined from approval history (tool identity + definition fingerprint aware; a tool whose definition changed is not proposed against the stale fingerprint)
- [x] #2 Accepting a suggestion writes an explicit Always-allow entry through the existing permission-store API — the same path the MCP screen uses today
- [x] #3 Dismissing a suggestion is remembered (not re-proposed) and dismissals are reviewable/clearable from the same surface
- [x] #4 Tools in the existing high-risk classes are never suggested; the exclusion set is shared with the approval card's high-risk badge so the two cannot drift
- [x] #5 Tests prove no runtime approval behavior changes merely because suggestions exist (until one is accepted), plus mining, fingerprint, and dismissal-persistence coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reducer: count approved-session decisions alongside approved (execution-log rows are the durable history; mining core permission_prompt_reducer.build_permission_prompt_report already exists and is kept)
2. MCPPermissionStore: optional top-level suggestion_dismissals map (key '<server_key>::<tool_name>' -> dismissed_at ISO), strict validators extended, SCHEMA_VERSION stays 1 (additive); APIs dismiss_suggestion / restore_suggestion / clear_suggestion_dismissals / list_suggestion_dismissals via the guarded _mutate_locked pattern
3. UnifiedControlPlaneService.permission_suggestions(): reducer report minus dismissed keys; fingerprint awareness stays live-resolved (state.config_changed) and high-risk exclusion stays shared via risk_floored/HIGH_RISK_TAGS (drift-guard test)
4. Accept writes through service.set_tool_state(server_key, tool_name, 'allow', tool=<live HubTool>, profile context) — the same path the permissions mode state-cycle uses
5. Surface: MCP Permissions mode Suggestions review subsection (accept/dismiss + dismissed restore/clear), keyboard actions per ADR-031
6. Tests: no-runtime-change proof (resolution unchanged while suggestions exist), dismissal persistence round-trip, mining matrix additions, accept write, UI surface; docs mcp + settings guide
<!-- SECTION:PLAN:END -->
## Deferral note (2026-10-11, dev `9f8533686c`)

DEFERRED from the t2-hermes porting slice for capacity. Two sequencing facts
for the next session:

1. **The approval UX surface is under active rework**: tldw_chatbook PR
   #3036 ("Clarify Console permission approvals and decision feedback",
   draft) is open against the approval-card/decision-feedback test family —
   a Settings suggestions surface that reads approval semantics should land
   after that settles, per the slice's collision rule.
2. **The data-source decision is still open**: verify where Approve
   once/session decisions are durably recorded today (the permission
   store's audit trail vs the approval card flow) before designing the
   mining layer; the task's plan step 1 anticipates adding minimal durable
   history if none exists — that schema decision plus the Settings surface
   (MCP permissions area) plus the high-risk exclusion set shared with the
   approval card's badge is a full session on its own.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented on branch feat/18928-approval-suggestions (worktree .worktrees/tasks-18925-28-34, base dev b07fc04252).

Approach — the mining core already existed (permission_prompt_reducer + service.permission_prompt_recommendations + /fewer command); this task added the missing review surface, dismissal persistence, and widened the mined decisions:
- Reducer: COUNTED_APPROVAL_DECISIONS = {approved, approved-session} (Approve-for-session is as explicit a human approval as Approve-once; the old exclusion test was updated, a new mixing test added).
- Store: optional top-level suggestion_dismissals map (tool-id -> ISO ts) in mcp_permissions.json, SCHEMA_VERSION stays 1 (additive); strict validators extended; invalid dismissal shape is COERCED (dropped) by _normalize_payload_shape instead of tripping the corrupt-reset path (nuisance data, not policy — the servers-coercion precedent). APIs dismiss/restore/clear/list via the guarded _mutate_locked pattern; key omitted while empty.
- Service: dismissal filtering lives at the ONE choke point (_permission_prompt_recommendation_snapshot), so report, apply, and the new surface all agree — a dismissed suggestion is absent everywhere and not appliable. Thin forwards dismiss/restore/clear/list; accept reuses the existing apply_permission_prompt_recommendation (hash-safe: re-verifies against the CURRENT catalog snapshot at apply time). Fingerprint awareness stays live-resolved (state.config_changed); high-risk exclusion stays shared via risk_floored from resolve_effective_state — the same flag the approval card badge renders (drift-guard test pins both consumers).
- Surface: MCP Permissions mode gains a suggestion-review section under the matrix (DataTable + hint; hidden when nothing to review). Keys a/d/r/c act ONLY while the suggestions table holds focus, and Space on the matrix is a no-op while it does — the two key surfaces cannot cross-fire. Workbench applies all four messages through the service (single-writer contract) and resyncs; suggestions refresh on every _sync_permissions_mode pass.
- ADR-032 governs (extends its permission surfaces; no new trust boundary, nothing auto-applies — pinned by test_existing_suggestions_do_not_change_runtime_resolution).

Files: MCP/permission_prompt_reducer.py, MCP/permission_store.py, MCP/unified_control_plane_service.py, UI/MCP_Modules/mcp_permissions_mode.py, UI/MCP_Modules/mcp_workbench.py, Docs/User_Guide/mcp.md. Tests: +1 reducer, +4 store, +5 service-level (incl. no-runtime-change + drift guard), +5 mode-surface.

Verification: 281 passed across the 9-file battery (reducer/store/resolution/service/mode/ownership//fewer); the 2 setup errors in the /fewer file are the pre-existing local RecoveryRequired environment condition (verified identical on pristine via stash). Ruff: only pre-existing findings in touched files (blind-Exception style matches the files' established defensive idiom); the one new-code finding was fixed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
