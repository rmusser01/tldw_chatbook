# Workspace Assistant Defaults — Design

Status: Approved in brainstorming (2026-08-29); pending spec review.

Date: 2026-08-29

Owner: Console / Personas / Workspaces integration

Related:
- Server PRD (unification target): `tldw_server` `origin/dev` `Docs/Product/Workspace_Persona_Defaults_PRD.md` (V1 implemented server-side; PR #2286, feature commit "apply workspace persona default to chat workspace")
- Server tool-policy direction (Draft): `tldw_server` `Docs/Product/Persona_Tool_Administration_PRD.md`
- Implemented server schemas: `tldw_server` `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py` (`WorkspaceAssistantDefaults`, `WorkspaceEffectiveAssistantDefault`), `.../schemas/persona.py` (`PersonaPolicyRule`, `PersonaScopeRule`)
- Local prior art: ADR-069 (project-instruction bindings as tool authority), ADR-074 (actor packs; persona JSON authority), ADR-037 (persona/user-profile boundary), ADR-078 (research workspace — explicitly out of scope here)
- Local seams: `Workspaces/registry_service.py`, `Workspaces/models.py`, `DB/Workspace_DB.py` (schema v2), `MCP/permission_store.py` (dormant `profiles` dict), `Chat/console_agent_bridge.py` (`_compose_run_registry_and_allowed`), `Chat/console_chat_controller.py` (`resolve_turn_execution_context`, `_compose_local_provider`), `Character_Chat/local_character_persona_service.py` (persona JSON store)

ADR required: yes
ADR path: `backlog/decisions/<NNN>-workspace-assistant-defaults.md` — next free number, verified at filing (ADR-076/077 collision renumbering is live on this branch; see TASK-19610)
Reason: data ownership spanning three stores (WorkspaceDB, persona JSON, permission store), a cross-module interface (workspace × persona × tool runtime), and security-adjacent permission layering.

## Summary

Give each explicit Console workspace a default agent persona and a per-workspace tool-calling posture, unified with tldw_server's Workspace Assistant Defaults contract. Workspaces recommend; they do not own personas. The workspace stores a reference-backed `assistant_defaults` object (server shape); the persona carries server-shaped local policy rules that can only narrow tool access; and the MCP permission store's dormant `profiles` dict becomes real named permission profiles that workspaces reference by id. New workspace-scoped conversations start as that persona's sessions; existing sessions are never mutated by later default edits.

## Problem

- The Console has no default-agent concept: the primary agent is automatic per tab, and personas reach the Console only as one-shot handoffs that seed a system prompt.
- Tool-calling posture is global: `[tools]`/`[console]` config gates, one `default` permission profile, no per-workspace or per-persona narrowing.
- tldw_server has already shipped Workspace Assistant Defaults V1 (workspace `assistant_defaults`, permission-filtered `effective_assistant_default`, four-tier precedence, session independence). The Chatbook needs the same capability locally, shaped so the two stay mirrorable.

## Goals

- Mirror the server's `assistant_defaults` contract (stored vs effective, precedence, degraded reason codes, session independence).
- Per-persona tool policy rules (server `PersonaPolicyRule` shape) that only narrow: remove from advertised set, floor to "ask", cap calls per turn.
- Per-workspace permission profiles in the local permission store, referenced by id, inheriting unset keys from the global `default` profile.
- Manage the workspace's agent from the workspace surfaces while editing the same persona structure from the Personas workbench.
- Keep every existing security floor intact (kill switch, rug-pull hash, high-risk floors, ADR-069 binding authority, ephemeral restrictions).

## Non-goals

- No Research Workspace adoption (ADR-078: its chat has no agent loop; adoption is a later server-defined stage).
- No server sync of bindings, personas, or profiles in this task (V1 is local-authority; `WorkspaceAuthority` server workspaces get a later mapping stage).
- No voice, style, or global chat-startup persona defaults (server reserves those; local equivalents stay null).
- No persona scope rules in V1 (server `PersonaScopeRule` over conversation/character/media/note ids is understood but not evaluated locally; schema reserved).
- No new grant authority: nothing in this design can widen tool access beyond what today's gates allow.

## Decisions (from brainstorming)

1. Substrate: persona profiles (Personas workbench entity) — one persona structure, two management surfaces.
2. Permissions: persona carries posture; permission store gains real per-workspace profiles; persona narrows only.
3. Lifecycle: convenience auto-create (workspace creation and one-time backfill create a normal persona `"{name} Agent"` and a dedicated permission profile, then store references) — contract stays reference-backed and server-conformant; binding is rebindable and clearable; NULL falls back to today's behavior.
4. Scope: explicit user-created workspaces only. The built-in Default workspace and global-scope sessions (the Console folds these into one identity) keep today's behavior exactly.
5. Mechanics: identity + posture. New conversations in a workspace start with the default persona as assistant identity (`assistant_kind="persona"`); identity is conversation-scoped and overridable.
6. Posture is two components (refined against the server subject model): persona policy rules are identity-scoped (travel with the conversation's persona); the permission profile is workspace-scoped (follows the session's current workspace, re-resolved per run).

## Data Model

### Workspace `assistant_defaults` (WorkspaceDB schema v2 → v3)

Nullable `assistant_defaults TEXT` column (JSON) on `workspace_records`; `WorkspaceRecord` gains the parsed field. Shape mirrors the server's `WorkspaceAssistantDefaults` exactly:

```json
{
  "assistant_kind": "persona",
  "assistant_id": "local-persona-<uuid>",
  "persona_memory_mode": "read_only",
  "voice": null,
  "style": null,
  "tool_policy_profile_id": "ws-<workspace_id>"
}
```

- `assistant_kind` is `"persona"` only in V1 (server `WorkspaceAssistantKind = Literal["persona"]`).
- `assistant_id` references a persona profile id from the local persona JSON store (`LocalCharacterPersonaService`; ids like `local-persona-<uuid>`).
- `persona_memory_mode` is `read_only | read_write`, default `read_only`. Saving `read_write` requires an explicit user confirmation step in the workspace settings surface (local analog of the server's backend-required `confirm_read_write_assistant_default=true`).
- `voice`, `style`, `tool_policy_profile_id` are reserved fields. This is a deliberate, single-field local-ahead-of-server divergence: the server validator locks all three to null because its Persona Tool Administration lifecycle is still draft, while the Chatbook's permission-profile substrate already exists — so locally `tool_policy_profile_id` accepts a profile-id string in V1 and `voice`/`style` stay null-locked. When the server unlocks the field, the shapes already match.
- No persona content is snapshotted: name, prompt, avatar, policy, tool permissions stay on the persona.
- Malformed stored JSON is treated as null with a logged warning (degraded `invalid_default`), never a crash.

### Effective default resolution

A local resolver produces the server's `WorkspaceEffectiveAssistantDefault` shape: `status: available | unavailable | none`, `source: workspace | none`, `assistant_kind`, `assistant_id`, `label`, `persona_memory_mode`, `degraded_reason` with the server's reason codes (`persona_deleted`, `persona_unavailable`, `persona_feature_disabled`, `permission_denied`, `invalid_default`, `unsupported_assistant_kind`). Single-user locally, so `permission_denied` exists for parity and is not produced in V1. Resolution rules:

- `assistant_defaults` null → `status: "none"`.
- Referenced persona missing or deleted → `persona_deleted`; malformed persona record → `persona_unavailable`; malformed defaults JSON → `invalid_default`; any non-persona kind → `unsupported_assistant_kind`.
- Degradation never blocks: sessions proceed plain, workspace surfaces show status + fix affordance.

### Persona policy rules (persona JSON store)

Persona profile records gain a `policy_rules` list mirroring server `PersonaPolicyRule`:

```json
"policy_rules": [
  {"rule_kind": "mcp_tool", "rule_name": "fs_write", "allowed": false, "require_confirmation": false, "max_calls_per_turn": null},
  {"rule_kind": "mcp_tool", "rule_name": "web_*", "allowed": true, "require_confirmation": true, "max_calls_per_turn": 5},
  {"rule_kind": "skill", "rule_name": "deep-research", "allowed": false, "require_confirmation": false, "max_calls_per_turn": null}
]
```

- `rule_kind`: `"mcp_tool" | "skill"` (server enum). Locally `mcp_tool` covers every non-skill catalog tool — builtin, local fs/web/git, library, MCP — because the Chatbook's `ToolCatalogRegistry` is the unified tool surface the server routes through MCP-unified; the name is kept for parity.
- `rule_name`: tool or skill name with bounded wildcard matching (`prefix*`), mirroring the server evaluator's bounded wildcards. No unbounded `*`.
- Semantics: deny-by-default when rules are present for a kind — a tool matching no `allowed=true` rule is not advertised. `allowed=false` removes the tool from the advertised set (consistent with the MCP provider's existing deny-drop). `require_confirmation=true` floors the tool to "ask" at invoke time regardless of profile grants. `max_calls_per_turn` caps invocations per run with a pinned refusal message when exceeded (new counter, style of `LOCAL_ROOT_CHANGED_REFUSAL`).
- Absent `policy_rules` or empty list → prompt-only persona, exactly today's behavior.
- Validation at the persona-service boundary; malformed individual rules are dropped with a warning, not fatal.
- The server's persona-level `PersonaConfirmationMode` (`always | destructive_only | never`) stays reserved locally; per-rule `require_confirmation` covers V1 needs.

### Named permission profiles (`MCP/permission_store.py`)

The store's dormant `profiles` dict goes live:

- Every mutator and resolver threads a `profile_id` parameter defaulting to `"default"` — existing call sites are unchanged.
- A workspace run with `tool_policy_profile_id = P` resolves permission states against `profiles.P`, falling through to `profiles.default` for keys absent from P (inheritance: a fresh empty profile behaves exactly like today).
- "Always allow" grants made during a workspace-scoped run write to the referenced profile (or `default` when none is referenced).
- Kill switch and `schema_version` stay global. The store shape change is additive (new keys under `profiles`), so **SCHEMA_VERSION stays 1** — bumping it would trigger the corrupt-file `.bak` policy and destroy existing permissions. The load-time normalization must be verified (and fixed if needed) so non-default profile keys survive `load()`.
- Rug-pull definition-hash and high-risk floors apply after profile resolution regardless of which profile supplied the grant.

## Runtime Resolution

One new seam, then narrowing:

- `resolve_turn_execution_context()` gains two keys: `persona_policy_rules` (resolved from the conversation's assistant persona identity; absent for plain sessions) and `tool_policy_profile_id` (resolved from the session's current workspace's `assistant_defaults`; `"default"` when unset, Default workspace, or global scope).
- `_compose_run_registry_and_allowed()` and `_compose_local_provider()` apply persona rules **after all existing gates**, in this order: `[tools]`/`[console]` config gates → ephemeral restrictions → ADR-069 binding access (read-only bindings still strip `fs_write`/`fs_edit`/`fs_patch` even if persona rules allow them) → kill switch → profile grant resolution (referenced profile, inheriting from `default`) → persona policy floor (`require_confirmation`, deny-by-default advertising) → `max_calls_per_turn` counter.
- A persona rule can never re-enable anything a gate or floor disabled; it only narrows.
- Sub-agents inherit the run's resolved posture run-scoped (same pattern as the ADR-069 activation ledger); `AgentDefinition.tool_allowlist` keeps narrowing within it.
- Session approvals remain in-memory per app run (unchanged semantics).
- AGENTS.md / AGENTS.override.md bodies never influence posture (ADR-069 unchanged).

## Session Mechanics (V1 target surface: Console)

- New conversation in explicit workspace W with no explicit assistant → if `effective_assistant_default.status == "available"`, the session is created with `assistant_kind="persona"`, `assistant_id`, `persona_memory_mode` persisted (columns already exist on `conversations`), and a system prompt composed via the existing persona composer (the same seam the Personas workbench preview uses).
- Precedence, mirroring the server: (1) existing session metadata, (2) explicit assistant choice before first send — handoff attachments, explicit persona selection, or a "start plain" escape hatch, (3) workspace `effective_assistant_default`, (4) plain fallback. The global-default tier the server holds at level 3 is reserved-null locally.
- After creation, the conversation is independent: edits to W's defaults never mutate existing sessions (server's startup-hint rule).
- Switching a session's workspace mid-conversation re-resolves the permission profile from the new workspace on the next run; the assistant identity is unchanged and history is never rewritten.

## Lifecycle

### Workspace creation (convenience auto-create)

For explicit workspace creation: (1) create persona `"{name} Agent"` through the normal personas-store APIs with a workspace-context seed prompt; (2) create an empty permission profile `ws-<workspace_id>`; (3) write `assistant_defaults` referencing both (`tool_policy_profile_id: "ws-<workspace_id>"`). **Convenience failure is non-fatal**: if persona or profile creation fails, the workspace is still created with `assistant_defaults = null` plus a logged warning. `WorkspaceCreateModal` gains a post-create affordance to open the agent in settings / the persona editor; nothing in-modal beyond that in V1.

### Backfill (one-time, idempotent)

A migration-guarded pass over existing explicit non-Default workspaces with null `assistant_defaults`, running the same create-and-reference routine through the personas service APIs (not raw JSON writes). Completion flag persisted in WorkspaceDB; safe to re-run; never touches the Default workspace.

### Rebind, clear, archive, deletion

- Rebind/clear live in workspace settings (the column is just a reference).
- Archiving W archives the binding with the workspace; persona and profile persist as normal entities.
- Persona deleted while referenced → `effective` degrades (`persona_deleted`), settings offer repair, sessions run plain.
- Profile key deleted → grants fall back to `default`, settings surface `invalid_default`.

## UX Surfaces

- **Settings → Workspaces**: per-workspace "Default assistant" section — persona picker (existing + inline create), `persona_memory_mode` selector with a `read_write` confirmation modal, permission-profile picker, clear button; shows `effective_assistant_default` status + degraded reason with a fix affordance.
- **Console workspace switcher**: shows the bound persona label, informational only.
- **Personas workbench**: persona inspector gains the policy-rules editor (kind, name, allowed, require_confirmation, max_calls_per_turn) — the same structure edited from either surface.

## Trust & Security

- Runtime narrowing-only semantics make imported policy rules safe by construction: a rule can never broaden beyond gates/floors. Actor-pack import (ADR-074, review-first) displays policy rules in the review UI so the user sees what they are accepting; no sanitizer needed.
- `persona_memory_mode` is binding-level, so it never travels in packs.
- Project-instruction content never grants permission or influences posture (ADR-069).
- Persona rules are user-authored local config (trusted, unlike AGENTS.md content), but the permission store's floors still apply after them.

## Error Handling

- Persona missing/deleted → degraded effective status; plain session; repair affordance.
- Malformed persona rules → rules dropped with warning; persona stays prompt-only.
- Malformed `assistant_defaults` JSON → treated as null + `invalid_default`.
- Missing permission profile key → inherit from `default`; surface degradation in settings.
- Auto-create partial failure → `assistant_defaults = null` + log; backfill retries next launch.
- Migration failure → standard WorkspaceDB migration rollback (v2 remains valid).

## Testing

- Unit: rule parsing/validation with **parity tests against the mirrored server schemas** (enum values, field shapes); evaluator semantics (deny-by-default-when-rules-present, bounded wildcards, confirmation floor, call caps, pinned refusals); profile resolution precedence + inheritance + floors; effective-default reason codes; read_write confirmation gating; normalization preserving non-default profiles.
- Integration: workspace creation orchestration + non-fatal failure modes; session startup application + four-tier precedence; post-creation independence; backfill idempotency; WorkspaceDB v2→v3 migration.
- Per repo policy: targeted runs only, no full sweeps without opt-in.

## Implementation Notes for the Plan

- Update the stale `tldw_api/character_persona_schemas.py` mirror to the current server shapes (`PersonaPolicyRule`, `PersonaScopeRule`) and add workspace `assistant_defaults` schemas to the mirror — implementation work, not design work.
- File the ADR before implementation begins and link it from the Backlog task, the Superpowers plan, and final implementation notes.

## Future Stages (contract-defined, not V1 blockers)

- Server-workspace authority mapping (server personas/profiles for `WorkspaceAuthority` server workspaces).
- Persona scope rules evaluation (resource-context narrowing).
- Voice/style defaults once the server unlocks those fields.
- Research Workspace / other surface adoption per the server PRD's staged gates (each must persist resolved assistant metadata on its own records first).
