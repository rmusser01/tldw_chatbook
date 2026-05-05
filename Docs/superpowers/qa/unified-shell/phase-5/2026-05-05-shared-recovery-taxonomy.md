# Phase 5.1 Shared Recovery Taxonomy

Date: 2026-05-05
Task: `TASK-6.1`
Parent Task: `TASK-6`
Branch: `codex/unified-shell-phase5-recovery-taxonomy`
Base: `origin/dev` at `42b2ac1e`

<!-- PHASE_5_1_RECOVERY_TAXONOMY_METADATA:BEGIN -->
```json
{
  "task": "TASK-6.1",
  "parent_task": "TASK-6",
  "decision": "foundation_defined",
  "required_user_fields": [
    "status_label",
    "unavailable_what",
    "why",
    "next_action",
    "recovery_action",
    "authority_owner",
    "stable_selector",
    "disabled_tooltip"
  ],
  "canonical_states": [
    "wrong_source",
    "server_not_configured",
    "server_unreachable",
    "server_auth_required",
    "server_session_invalid",
    "policy_denied",
    "capability_disabled",
    "runtime_not_configured",
    "service_unavailable",
    "dependency_missing",
    "empty_selection"
  ],
  "runtime_policy_reason_codes": [
    "wrong_source",
    "server_not_configured",
    "server_unreachable",
    "server_auth_required",
    "server_session_invalid",
    "authority_denied",
    "capability_disabled"
  ],
  "destination_recovery_sources": [
    "phase-1-destination-action-audit",
    "phase-3-console-live-work-closeout",
    "phase-4-destination-service-adoption-closeout",
    "runtime-policy-domain-edge-contracts"
  ]
}
```
<!-- PHASE_5_1_RECOVERY_TAXONOMY_METADATA:END -->

## Goal

Define one shell-level recovery contract for blocked states so future Phase 5 slices can make dependency, auth, server, runtime, policy, and selection failures understandable without inventing copy and selectors screen by screen.

This taxonomy sits above the existing `runtime_policy` package. It does not replace `PolicyDecision`, `PolicyDeniedError`, domain edge contracts, or service hard stops. It defines how those decisions should appear in the Unified Shell.

## Required User-Facing Fields

Every blocked state that appears in a visible shell or destination control should expose these fields:

| Field | Requirement |
| --- | --- |
| `status_label` | Short state label such as `Server unavailable`, `Runtime not configured`, or `Select an item`. |
| `unavailable_what` | The specific workflow or control that cannot run. |
| `why` | The immediate reason in user language, mapped to a fixed policy or recovery state when possible. |
| `next_action` | The next concrete user step, not generic failure copy. |
| `recovery_action` | The target route, retry action, setup action, or selection action when one exists. |
| `authority_owner` | The owner of the capability: local app, active server, shared backend, runtime, optional dependency, or user selection. |
| `stable_selector` | A stable test selector for the blocked message or disabled control. |
| `disabled_tooltip` | Tooltip copy on disabled controls that repeats the missing prerequisite or next action. |

## Canonical States

| Canonical state | Maps from | Required recovery behavior |
| --- | --- | --- |
| `wrong_source` | `PolicyDecision.reason_code == "wrong_source"` | Name the required source and offer the source switch or existing route when available. |
| `server_not_configured` | `PolicyDecision.reason_code == "server_not_configured"` or `ServerContextFailure.reason_code == "server_not_configured"` | Point to server setup or Settings; do not imply retry will work without configuration. |
| `server_unreachable` | `server_unreachable`, `server_unavailable`, or request-time connectivity classification | Offer retry and preserve local alternatives when they exist. |
| `server_auth_required` | `server_auth_required`, `auth_required`, or credential-missing context | Point to sign-in or credential setup; do not hide the server-owned workflow. |
| `server_session_invalid` | `server_session_invalid`, `stale_authorization`, or `profile_no_longer_authorized` | Tell the user to re-authenticate or reconnect the active server profile. |
| `policy_denied` | `authority_denied`, `permission_denied`, or workspace policy denial | State that policy blocks the action and identify the authority owner where possible. |
| `capability_disabled` | `capability_disabled` or feature flag disabled | State that the feature is disabled and where it must be enabled. |
| `runtime_not_configured` | ACP runtime missing, local model runtime missing, or tool runtime missing | Name the missing runtime and the setup path. |
| `service_unavailable` | Missing app service seam, failed local service lookup, or service exception | Offer retry only if retry can re-run the lookup; otherwise point to setup or later-phase work. |
| `dependency_missing` | Optional dependency unavailable | Name the dependency or feature extra and point to install/setup guidance. |
| `empty_selection` | No selected item, no active run, no artifact, or no local record | Tell users what to select or create before the action can run. |

## Copy Contract

Blocked-state copy should follow this shape:

`<Workflow> is unavailable because <reason>. <Next action>.`

Examples:

| State | Good shell copy |
| --- | --- |
| `wrong_source` | `Server notes are unavailable in local mode. Switch to Server to browse server notes.` |
| `server_not_configured` | `Sharing requires a configured server. Add a server profile in Settings before sharing.` |
| `runtime_not_configured` | `ACP launch is unavailable because no ACP-compatible runtime is configured. Configure an ACP runtime before launch.` |
| `empty_selection` | `Use in Console is unavailable until a Library source is available. Add or select a source first.` |

Avoid these patterns:

| Pattern | Why it fails |
| --- | --- |
| `Unavailable` alone | It omits what is blocked and what to do next. |
| `Try again later` for configuration gaps | Retry cannot fix missing setup. |
| Enabled-looking buttons that only notify failure | They create false affordances unless the click itself is a meaningful retry or setup action. |
| Generic service-error copy for policy denial | Policy denial needs the authority owner, not a retry suggestion. |

## Application Rules

- Disabled controls must carry `disabled_tooltip` copy unless the blocked explanation is directly adjacent and focusable.
- Recovery copy must be visible in the same panel as the disabled action or in a reachable status row.
- A blocked state counts as honest recovery only when the user can identify what is unavailable, why, and what to do next without reading logs.
- Retry controls are valid only for transient service, connectivity, or refresh states.
- Setup controls are preferred for missing configuration, auth, runtime, or dependency states.
- Selection states should preserve power-user speed by enabling immediately when a valid item appears.
- Tests should verify both visible copy and disabled tooltip text for representative controls.

## Source Alignment

This taxonomy is grounded in existing repo contracts:

| Source | Used for |
| --- | --- |
| `runtime_policy.types.PolicyDecision` | Fixed policy reason codes and authority owner fields. |
| `runtime_policy.types.ServerContextFailure` | Server setup, credentials, authorization, and reachability states. |
| `runtime_policy.domain_edge_contracts` | Domain ownership and unsupported action reports. |
| `Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-destination-action-audit.md` | Honest blocked-state baseline and false-affordance examples. |
| `Docs/superpowers/qa/unified-shell/phase-3/2026-05-05-phase-3-console-live-work-closeout.md` | Console source-readiness recovery states. |
| `Docs/superpowers/qa/unified-shell/phase-4/2026-05-05-phase-4-destination-service-adoption-closeout.md` | Destination recovery states for ACP, Schedules, Workflows, and Artifacts. |

## Next Slices

Phase 5.2 should apply this contract to the highest-impact remaining blocker-state family first. The likely order is:

1. Shell destination blocked states: ACP runtime, Schedules empty active run, Workflows empty active run, Artifacts empty Chatbook.
2. Runtime-policy blocked states: wrong source, server not configured, server auth/session, policy denied.
3. Optional dependency states: local model, speech, transcription, embeddings/RAG extras.
4. Selection and empty-data states that still expose generic disabled copy.

## Verification

- Red regression: `python3 -m pytest Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q`
- Red result before this artifact existed: `2 failed`
- Final focused verification: `python3 -m pytest Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Final focused result: `7 passed`

## Closeout Decision: foundation_defined

`TASK-6.1` defines the shared recovery taxonomy and tracking contract. It does not mark Phase 5 verified; `TASK-6` remains open until later slices apply the taxonomy to remaining blockers and running-app QA verifies understandable recovery paths.
