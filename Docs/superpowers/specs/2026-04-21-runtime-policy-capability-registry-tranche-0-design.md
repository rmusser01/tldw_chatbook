# Runtime Policy And Capability Registry Tranche 0 Design

Date: 2026-04-21
Status: Approved for spec review
Scope: Infrastructure-only runtime source state, authoritative capability registry, central policy enforcement, and offline/source authority plumbing

## Summary

The 2026-04-21 parity audit established that Chatbook needs a single runtime-policy foundation before additional parity work can land safely. Today, local versus server behavior is encoded piecemeal in screen state, scope services, and mode-sensitive widgets. That is enough to make specific features work, but not enough to make the product reliably standalone-first, source-separated, and resistant to parity regressions.

Tranche 0 should introduce one authoritative runtime-policy layer that later verticals and the parallel UX work can consume. This slice is infrastructure-only. It does not add the end-user capability-map surface, mixed local/server views, sync behavior, or multi-server switching UX. It defines the policy model that makes those future changes possible without re-deciding authority rules screen by screen.

The central design choice is to create a single authoritative capability registry seeded from the full audited capability set at action-level granularity, pair it with persisted runtime source state, and enforce it in two places:

- early in UI handlers for feedback
- centrally at shared service/client boundaries as the hard stop

## Problem Statement

The current product direction is explicit:

- Chatbook is standalone-first
- local and server surfaces stay separated by default
- local writes remain local until a later sync design exists
- remote-only domains still need discoverability and explicit offline behavior

The codebase does not yet have one place that enforces those rules.

Current issues:

1. Source authority is fragmented. Screen-local fields such as `runtime_backend` and domain-specific mode checks exist, but there is no app-wide authoritative source record.
2. Capability rules are implicit. Whether a domain is local-first, dual-backend, remote-only, or offline-unavailable depends on scattered conditionals rather than one authoritative registry.
3. Blocking is inconsistent. Some screens already hide or disable controls when the wrong backend is active, but old and parallel call paths can still bypass that logic because services do not consistently enforce the same rules.
4. Offline fallback rules are not normalized. The audit defined where local fallback must exist and where the product should show explicit unavailable-offline states, but the codebase has no unified way to answer those questions.
5. The upcoming UX work needs a single source of truth. Without a registry and policy engine, the UX layer will either duplicate policy logic or keep reading ad hoc service-specific heuristics.

## Goals

- Introduce one authoritative runtime source state for `local` versus `server`.
- Persist runtime source selection across restarts.
- Introduce one authoritative action-level capability registry seeded from the full audited capability set.
- Return structured allow/deny policy decisions with fixed machine-readable reason codes.
- Enforce invalid-action blocking centrally at shared service/client boundaries from day one.
- Reuse the same policy decisions in UI preflight checks for early user feedback.
- Keep the design compatible with later server switching without implementing switching UX now.
- Keep the design compatible with later capability-map UX without implementing that UX now.

## Non-Goals

- Build the end-user capability-map screen.
- Implement sync, mirroring, import rules, or conflict resolution.
- Add mixed local/server record views.
- Introduce named multi-server switching UX.
- Finish all per-domain parity work.
- Replace all existing screen-local mode checks in one pass.
- Redesign settings/navigation UX that a parallel agent is already handling.

## User And Product Fit

This slice exists to protect the user-reviewed operating model from regression while later parity work lands.

Primary product promises it must uphold:

- local operations still work without the server
- remote-only actions fail clearly rather than silently implying authority
- Chatbook-owned local surfaces such as local notifications and local MCP runtime remain locally authoritative
- remote-only surfaces such as workflows, sharing, web clipper, and remote MCP governance remain clearly server-owned
- future parity verticals can add behavior by extending a registry rather than inventing fresh authority logic

This is infrastructure, but it is infrastructure in direct service of user trust: the app should not let users accidentally act against the wrong source or assume offline authority where none exists.

## Current Constraints

- Screen and tab state already carry some runtime metadata, especially in `UI/Screens/chat_screen_state.py`.
- App and config state already exist, but there is no dedicated runtime-policy package.
- Many domains already expose local/server scope services:
  - notes
  - media
  - study
  - evaluations
  - RAG admin
  - characters/personas
- Existing local/server behavior must remain source-separated rather than silently merged.
- Only a single active server slot is required now, but server switching is expected behavior later.

## Proposed Approach

The recommended approach is a central runtime-policy subsystem with code-defined registry data and dual enforcement seams.

### Core Decisions

1. Use a single authoritative capability registry rather than per-screen policy fragments.
2. Seed the registry from the full audited capability set now, including unsupported and remote-only rows.
3. Model capability permissions at action level, not just domain level.
4. Persist authoritative runtime source state across restarts.
5. Enforce policy in both places:
   - UI handlers for early feedback
   - shared service/client boundaries as the hard stop
6. Use fixed machine-readable reason codes now rather than free-form deny messages.
7. Treat the current server model as a single active server slot, but shape runtime state so named switching can be added later without redesigning the policy layer.

### Rejected Alternatives

#### Minimal guard layer first

This would add blocking without a proper registry. It is faster initially, but it guarantees a later migration from hard-coded rules to structured policy data. That would make the UX and later parity verticals inherit unstable contracts.

#### Advisory registry first

This would build a registry but leave enforcement mostly optional. It contradicts the requirement to block invalid actions centrally from day one.

## Architecture

### RuntimeSourceState

`RuntimeSourceState` is the authoritative persisted record of current source and server context.

It should contain:

- `active_source`: `local` or `server`
- `active_server_id`: nullable string
- `server_configured`: bool
- `server_reachable`: bool
- `last_known_server_label`: optional string

Important rules:

- `active_source` is the source of truth for app mode, not screen-local copies.
- `active_server_id` exists now even though only one active slot is supported, so later switching adds orchestration rather than a schema rewrite.
- reachability is runtime state, not a substitute for configuration

### CapabilityRegistry

The registry is the authoritative capability map for the app. It is action-level and code-defined.

Registry entries should be keyed by stable ids such as:

- `chat.create.local`
- `chat.create.server`
- `notes.update.local`
- `workflows.launch.server`
- `watchlists.observe.local`

Each entry should define:

- domain id
- capability label
- action kind:
  - `browse`
  - `detail`
  - `create`
  - `update`
  - `delete`
  - `launch`
  - `observe`
- allowed sources
- offline policy
- authority owner
- optional feature flag or maturity marker
- fixed default deny reasons

The registry is seeded from the full 2026-04-21 audited capability set so unsupported and remote-only rows exist as first-class policy data from the start.

### PolicyEngine

`PolicyEngine` resolves allow/deny decisions from:

- runtime source state
- capability registry entry
- connectivity
- optional context such as workspace scope or server readiness

It returns a structured `PolicyDecision`.

`PolicyDecision` should include:

- `allowed: bool`
- `reason_code: Optional[str]`
- `user_message: str`
- `effective_source: str`
- `authority_owner: str`

### Fixed Reason Codes

The initial fixed reason-code set should cover at least:

- `wrong_source`
- `offline_unavailable`
- `server_not_configured`
- `capability_disabled`
- `authority_denied`

Additional codes are allowed later, but this initial set should be enough for the first tranche of enforcement and UI feedback.

## Enforcement Model

### Service/Client Hard Stop

Shared service/client boundaries are the correctness seam.

Representative initial hard-stop integration points:

- `Notes/notes_scope_service.py`
- `Media/media_reading_scope_service.py`
- `Study_Interop/study_scope_service.py`
- `Evaluations_Interop/evaluation_scope_service.py`
- `RAG_Admin/rag_admin_scope_service.py`
- `Character_Chat/character_persona_scope_service.py`

Rules:

- service calls must reject invalid source/authority operations even if a UI caller forgot to preflight
- service responses should preserve the fixed reason-code contract
- the service layer must not silently coerce source, auto-switch mode, or invent fallback authority

### UI Preflight

UI and widget callers use the same policy engine before dispatching actions.

Initial representative consumers should be existing mode-sensitive callers rather than a full product sweep. The point of Tranche 0 is to establish the seam, not complete every screen migration.

Rules:

- UI may disable, hide, relabel, or warn based on policy
- UI must not define separate policy semantics
- if UI and service disagree, service is the final authority

## File Boundaries

### New Package

Create a dedicated `runtime_policy` package under `tldw_chatbook/`.

#### `tldw_chatbook/runtime_policy/types.py`

Responsibility:

- source/action/authority literals or enums
- policy reason codes
- typed data structures for registry entries and decisions

#### `tldw_chatbook/runtime_policy/registry.py`

Responsibility:

- full audited capability registry
- stable action-level ids
- code-defined registry data only

#### `tldw_chatbook/runtime_policy/source_state.py`

Responsibility:

- authoritative persisted runtime source state
- load/save helpers
- single active server slot model

#### `tldw_chatbook/runtime_policy/engine.py`

Responsibility:

- evaluate policy from runtime state plus capability entry
- return structured `PolicyDecision`

#### `tldw_chatbook/runtime_policy/enforcement.py`

Responsibility:

- common helpers for:
  - service hard-stop checks
  - UI preflight checks
- avoid every caller inventing wrapper logic

#### `tldw_chatbook/runtime_policy/bootstrap.py`

Responsibility:

- startup wiring into app/global state
- minimal glue only

### Existing Files To Modify

#### `tldw_chatbook/state/app_state.py`

Add authoritative app-level runtime source state access rather than letting screens own source authority independently.

#### `tldw_chatbook/app.py`

Bootstrap runtime-policy state and registry during app startup, and expose one authoritative runtime-source accessor.

#### `tldw_chatbook/UI/Screens/chat_screen_state.py`

Retain per-tab metadata but stop treating tab-local `runtime_backend` as the final runtime authority for the app.

#### Existing scope services

Add hard-stop policy checks to the shared scope services listed above.

### Deliberately Not In Scope

- no new end-user capability-map screen
- no new switching UI
- no broad restyling
- no full screen-by-screen policy migration

## Data Flow

### Startup

1. app boots
2. runtime-policy bootstrap loads persisted `RuntimeSourceState`
3. capability registry is initialized
4. app state exposes both to downstream consumers

### UI preflight path

1. user triggers an action
2. UI asks enforcement helper for policy decision
3. if denied:
   - UI shows source-aware feedback using fixed reason code
   - service call is not attempted
4. if allowed:
   - call proceeds to service layer

### service hard-stop path

1. service receives request
2. service resolves required capability action id
3. service asks policy engine for allow/deny
4. if denied:
   - return or raise structured policy failure
5. if allowed:
   - proceed with local or server behavior

## Error Handling

- denied actions must be explicit, not silent no-ops
- service/client hard stops must preserve fixed reason codes
- UI copy may vary, but reason codes must not
- server-unconfigured and server-unreachable are different states and must not collapse into one generic error
- offline behavior must preserve local authority for local-first domains and explicit unavailable state for remote-only domains

## Testing Strategy

### Unit Tests

- registry completeness and stable action ids
- reason-code coverage
- policy engine decisions for:
  - correct source
  - wrong source
  - server not configured
  - server unreachable
  - offline remote-only action
  - capability disabled

### Persistence Tests

- runtime source state persists across restart
- single active server slot shape remains intact

### Service-Level Tests

- invalid actions are blocked centrally at scope-service boundaries
- service failures return fixed reason codes
- services do not silently coerce source

### UI-Level Tests

- representative mode-sensitive screens perform preflight checks
- denied actions produce early feedback
- UI and service agree on the same reason codes

### Regression Goal

Tranche 0 is successful when local-first domains still function offline, remote-only domains fail cleanly with explicit policy reasons, and old service call paths cannot bypass source/authority checks.

## Risks

### Partial adoption risk

If only UI checks are added, older service paths will bypass policy. This is why the service/client boundary is the required hard-stop seam.

### Registry drift risk

If later verticals add behavior without extending the registry, the policy layer will decay. The registry must be treated as the source of truth for future parity work.

### Overreach risk

This slice can sprawl into UX work or sync design if not kept constrained. It must remain infrastructure-only.

## Success Criteria

- one authoritative action-level registry exists
- one authoritative persisted runtime source state exists
- one policy engine returns structured decisions with fixed reason codes
- hard-stop enforcement exists at shared service/client boundaries
- representative UI callers use the same policy decisions for early feedback
- no new capability-map UX is required for this tranche to land
- the design remains compatible with later server switching and later capability-map UX without reworking the core model
