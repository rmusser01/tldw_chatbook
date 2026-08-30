# ADR-079: Workspace assistant defaults

- **Status:** Accepted
- **Date:** 2026-08-29
- **Task:** [TASK-21969](../tasks/task-21969%20-%20Workspace-assistant-defaults-—-personas-policy-rules-permission-profiles.md)
- **Design:** [Workspace Assistant Defaults design](../../Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md)
- **Related:** ADR-069 (console project-instruction bindings and tool authority),
  ADR-074 (actor packs; persona JSON authority), ADR-037 (persona/user-profile
  boundary), ADR-078 (Research Workspace — explicitly out of scope here)
- **Server contracts:** `tldw_server` Workspace Persona Defaults PRD (V1 shipped
  server-side, PR #2286); Persona Tool Administration PRD (draft);
  `workspace_schemas.py` (`WorkspaceAssistantDefaults`,
  `WorkspaceEffectiveAssistantDefault`); `persona.py` (`PersonaPolicyRule`,
  `PersonaScopeRule`)

## Context

The Console has no default-agent concept: the primary agent is automatic per
tab, and personas reach the Console only as one-shot handoffs that seed a
system prompt. Tool-calling posture is global — `[tools]`/`[console]` config
gates plus a single `default` permission profile — with no per-workspace or
per-persona narrowing.

tldw_server has already shipped Workspace Assistant Defaults V1: workspace
`assistant_defaults`, a permission-filtered `effective_assistant_default`,
four-tier startup precedence, and session independence after creation. The
Chatbook needs the same capability locally, shaped so the two stay mirrorable.

The decision spans three stores — WorkspaceDB (schema v2), the persona JSON
store, and the MCP permission store's dormant `profiles` dict — and layers a
security-adjacent permission model on top of existing gates and floors
(ADR-069 binding authority, kill switch, rug-pull definition-hash check,
high-risk floors, ephemeral restrictions).

## Decision

### 1. Adopt the server's workspace `assistant_defaults` contract

Each explicit workspace stores a nullable, reference-backed
`assistant_defaults` JSON object on its WorkspaceDB record (schema v6 → v7)
mirroring the server's `WorkspaceAssistantDefaults` shape:
`assistant_kind` (`"persona"` only in V1), `assistant_id` (a persona profile
id), `persona_memory_mode` (`read_only | read_write`), and reserved
`voice`/`style`/`tool_policy_profile_id` fields. No persona content is
snapshotted — name, prompt, avatar, policy rules, and tool permissions stay on
the persona; workspaces recommend, they do not own personas.

A local resolver produces the server's `WorkspaceEffectiveAssistantDefault`
shape: `available | unavailable | none` status with the server's degraded
reason codes (`persona_deleted`, `persona_unavailable`,
`persona_feature_disabled`, `permission_denied`, `invalid_default`,
`unsupported_assistant_kind`). Malformed stored JSON degrades to null with a
logged warning, never a crash, and degradation never blocks a session.

Startup precedence mirrors the server's four tiers: (1) existing session
metadata, (2) an explicit assistant choice before first send (handoff,
explicit persona selection, or "start plain"), (3) the workspace's
`effective_assistant_default` (the server's global tier is reserved-null
locally), (4) plain fallback. After creation, sessions are independent: later
default edits never mutate existing sessions.

Identity and posture are separate mechanics: the persona identity is
conversation-scoped and overridable; the permission profile is
workspace-scoped, re-resolved from the session's current workspace per run.

### 2. Persona-local policy rules narrow only

Persona profile records gain a `policy_rules` list mirroring the server's
`PersonaPolicyRule` (`rule_kind: "mcp_tool" | "skill"`, bounded-wildcard
`rule_name`, `allowed`, `require_confirmation`, `max_calls_per_turn`). Locally
`mcp_tool` covers every non-skill catalog tool — builtin, local fs/web/git,
library, MCP — because the `ToolCatalogRegistry` is the unified tool surface.

Semantics are narrowing-only, applied after all existing gates in a fixed
order: config gates → ephemeral restrictions → ADR-069 binding access → kill
switch → profile grant resolution → persona policy floor → call caps. With
rules present for a kind, advertising is deny-by-default; `allowed=false`
removes the tool from the advertised set; `require_confirmation=true` floors
the tool to "ask" at invoke time regardless of profile grants (a persisted
`always_allow` grant does not bypass it); `max_calls_per_turn` caps
invocations per run with a pinned refusal message. A rule can never re-enable
anything a gate or floor disabled. Absent or empty rules mean a prompt-only
persona — exactly today's behavior. The server's persona-level confirmation
mode and `PersonaScopeRule` stay reserved locally, unevaluated in V1.

### 3. Named permission profiles, referenced by id

The permission store's dormant `profiles` dict goes live. Every mutator and
resolver threads a `profile_id` defaulting to `"default"`, so existing call
sites are unchanged. A workspace run with `tool_policy_profile_id = P`
resolves permission states against `profiles.P`, inheriting per-key from
`profiles.default` — a fresh empty profile therefore behaves exactly like
today. "Always allow" grants made during a workspace-scoped run write to the
referenced profile and persist even if the reference is later cleared or
rebound; they simply stop applying to that workspace. The `ws-` profile-id
prefix is reserved for auto-created workspace profiles. Rug-pull
definition-hash and high-risk floors apply after profile resolution
regardless of which profile supplied the grant. The store shape change is
additive and verified compatible with `load()` normalization, so
`SCHEMA_VERSION` stays 1 — bumping it would trigger the corrupt-file `.bak`
policy and destroy existing permissions.

### 4. `tool_policy_profile_id` accepted locally ahead of the server

This is a deliberate single-field local-ahead-of-server divergence: the
server validator locks `tool_policy_profile_id` (with `voice` and `style`) to
null because its Persona Tool Administration lifecycle is still draft, while
the Chatbook's permission-profile substrate already exists. Locally the field
accepts a profile-id string in V1; `voice` and `style` stay null-locked. When
the server unlocks the field, the shapes already match. Relatedly, local
saves skip the server patch contract's optimistic-locking `version` field
(single-user, single-process local store).

### 5. Convenience auto-create is a local extension

For explicit workspace creation, and once via an idempotent startup backfill,
the app creates a normal persona `"{name} Agent"` and an empty permission
profile `ws-<workspace_id>`, then stores references to both — the stored
contract stays reference-backed and server-conformant. Convenience failure is
non-fatal: if persona or profile creation fails, the workspace is still
created with `assistant_defaults = null` plus a logged warning, and the
backfill retries next launch. Backfill runs through the app's personas
service instance (never raw JSON writes), records completion in WorkspaceDB,
and skips archived workspaces and the built-in Default workspace. Rebind and
clear live in workspace settings; a NULL default falls back to today's
behavior. Scope is explicit user-created workspaces only — the Default
workspace and global-scope sessions keep today's behavior exactly.

### 6. `read_write` memory mode is gated by explicit confirmation

`persona_memory_mode` defaults to `read_only`. Saving `read_write` requires
an explicit user confirmation step in the workspace settings surface — the
local analog of the server's backend-required
`confirm_read_write_assistant_default=true` flag, kept as a UI-level gate.

### 7. ADR-069 authority and every existing floor are unchanged

Project-instruction bodies (AGENTS.md / AGENTS.override.md) never grant
permission or influence posture; read-only ADR-069 bindings still strip
mutating fs tools even if persona rules allow them. The kill switch, rug-pull
hash, high-risk floors, and in-memory session approvals keep their current
semantics. Runtime narrowing-only semantics make imported rules safe by
construction; actor-pack import (ADR-074) displays policy rules in its review
UI, and `persona_memory_mode` is binding-level so it never travels in packs.

## Alternatives considered

### Snapshot persona content into the workspace default

Rejected. References keep the persona as the single editable structure (one
persona, two management surfaces) and match the server contract, which stores
only ids and modes.

### Mirror the server's null-lock on `tool_policy_profile_id`

Rejected. The local permission-profile substrate already exists and delivers
per-workspace posture now; locking the field locally would block the feature
on a draft server PRD. Accepting the single field early keeps shapes aligned
for the eventual server unlock.

### Store tool posture on the workspace instead of the persona

Rejected. Identity-scoped policy rules must travel with the conversation's
persona (including persona reuse across workspaces and actor packs); the
server shape also places rules on the persona. The workspace contributes only
the profile reference.

### Bump the permission store `SCHEMA_VERSION` for profiles

Rejected. The `profiles` addition is additive and survives `load()`
normalization; a version bump would misclassify every existing store as
corrupt and trigger the `.bak` reset policy, destroying live permissions.

### Let a persisted `always_allow` grant satisfy `require_confirmation`

Rejected. The persona floor would be silently defeatable by an earlier grant.
The ask must fire; relaxing it means editing the rule.

### Persona scope rules and voice/style defaults in V1

Reserved, not adopted. The server schema is understood (`PersonaScopeRule`)
and the fields exist, but local evaluation and defaults wait for server-side
lifecycle decisions.

### Research Workspace adoption

Out of scope per ADR-078: its chat has no agent loop; adoption is a later
server-defined stage.

## Consequences

### Benefits

- Explicit workspaces get a default agent persona and a per-workspace tool
  posture with server-mirrorable shapes, reason codes, and precedence.
- Narrowing-only rules are safe by construction — no persona rule, profile,
  or workspace default can widen tool access beyond today's gates.
- Existing call sites and stores keep working: the profile id defaults to
  `default`, empty profiles inherit it, and the permission-store shape
  change is additive without a schema bump.
- Session independence prevents default edits from rewriting running or saved
  conversations; degraded references degrade gracefully instead of blocking.

### Costs and constraints

- WorkspaceDB migrates v6 → v7 (nullable column plus a small backfill-flag
  table); migration failure must roll back cleanly to v2.
- Three stores participate (WorkspaceDB, persona JSON, permission store), so
  resolution, degradation, backfill idempotency, and inheritance each need
  targeted tests, including parity tests against the mirrored server schemas.
- The local-ahead `tool_policy_profile_id` divergence must be tracked until
  the server's Tool Administration PRD lands, then reconciled.
- Auto-create can leave an orphaned persona or profile on partial failure;
  they persist as normal entities and the workspace runs plain.
- Server sync of bindings, personas, or profiles is deferred to a later
  stage; V1 is local-authority only.
- The persona `require_confirmation` floor is applied at the MCP provider's
  and local provider's invoke-time gates, but the `BuiltinToolGate` persona
  floor remains a V1 non-goal (deferred with the review's other minor
  items, not tracked as a separate follow-up task).
