---
id: TASK-32874
title: Preset and provider routing for agent-created chats (fork_chat/new_chat)
status: In Progress
assignee:
  - '@robert'
created_date: '2026-09-20 16:47'
updated_date: '2026-09-20 18:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The deferred integration from ADR-150 decision 7, now unblocked: the agent provider-routing work (ADR-147) landed on dev. Extend fork_chat/new_chat with optional provider/model selection for the CREATED chat, reusing the routing vocabulary and its security model. Spec anchor: Docs/superpowers/specs/2026-09-11-agent-chat-fork-spawn-design.md section Provider/preset integration (deferred). Design directions locked by that spec + ADR-147: (1) optional string args provider/model on both tools, additive to the v1 schemas; (2) ad-hoc args are untrusted model output — gate on [agents] spawn_override_enabled AND spawn_override_allowlist exactly like spawn_subagent overrides (Agents/agent_routing.py SpawnRoutingConfig), with the final-provider guard applying when any ad-hoc arg is present; (3) a named AgentDefinition preset may be referenced instead (its provider/model ride the preset, user-authored so unrestricted); (4) the confirm card must SHOW the requested provider/model — model-chosen routing is always visible before allow; (5) the executor resolves the target via the routing resolver and builds the new session's ConsoleSessionSettings with build_default_console_session_settings for that provider/model (fork keeps everything else verbatim); (6) errors reuse the RoutingError vocabulary (override_disabled, provider_not_allowlisted, unknown_provider, unknown_endpoint_slug, provider_not_ready, no_model_resolved) surfaced as tool error results; (7) advertised-equals-usable: consider omitting the routing args from the schemas when spawn_override_enabled is false, mirroring spawn's dynamic schema — requires making the two ToolSchema constants builder functions like build_spawn_schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 fork_chat and new_chat accept optional provider/model (and preset name) args per the ADR-147 security model,Ad-hoc routing args are gated on spawn_override_enabled plus allowlist with the final-provider guard,Confirmation card displays the requested provider/model before allow,The created chat's session settings resolve through the routing resolver for the requested target,All RoutingError kinds surface as tool error results without creating anything,Tests cover the gating matrix resolver integration card rendering and schema advertisement,User docs updated for the routing args
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. tool_catalog: build_fork_chat_schema/build_new_chat_schema builders (identity when no routed presets and override off; preset enum when definitions carry routing; provider/model args + enumerated targets when spawn_override_enabled — mirroring build_spawn_schema). 2. agent_service: plan builder appends the BUILT schemas (it already receives agent_definitions + spawn_override_enabled/targets); _chat_create_runtime_schemas keeps static pins (disclosure flows through the plan). 3. bridge closures: parse provider/model/preset (strict strings) into the payload. 4. card: render requested provider/model/preset line. 5. executor: when any routing arg present, load [agents] routing config + the named AgentDefinition (agent_runs db loader the service uses), resolve via resolve_spawn_target with parent = the source session's provider/model, then build the new chat's ConsoleSessionSettings via build_default_console_session_settings and pass settings through the completion; RoutingError.code surfaces as the outcome kind, nothing created. 6. tests: builder identity/matrix, closure passthrough, executor gating matrix incl. final-provider guard + custom-ep slug + readiness, preset path, settings applied, card rendering. 7. docs: user guide routing args.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented 2026-09-13 per plan. `build_chat_create_schema` (tool_catalog)
mirrors `build_spawn_schema`'s ADR-147 pattern: identity when no routed
presets and the override gate is closed; optional `preset` enum+roster for
definitions carrying provider/model; gated `provider`/`model` args with
allowlisted targets enumerated (identity only). The first-request plan
appends the BUILT schemas (it already receives agent_definitions +
spawn_override flags); the static constants remain the identity base and
the `_run_one` defense pin. Bridge closures parse the three routing args
as strict strings. The card renders a "Runs on:" line (preset and/or
provider/model) before allow. The executor resolves -- BEFORE creating
anything -- via `resolve_spawn_target` with parent = the source session's
provider/model, with the [agents] sub-agent default level uniformly
cleared (chat creation is not a spawn; levels are ad-hoc -> preset ->
parent-fill), builds the new chat's `ConsoleSessionSettings` via
`build_default_console_session_settings`, and threads it through the
completion's restore. `RoutingError.code` surfaces as the outcome kind;
two chat-specific kinds added: `unknown_preset`, `preset_unrouted`.
Tests: builder identity/roster/override matrix, executor gating matrix
incl. the final-provider guard on model-only overrides, preset ride,
settings-build, card rendering. Local full-suite note: ~150 tests on this
machine fail with `RecoveryRequired: raw_source_selection_changed` from
the backup-recovery bootstrap -- verified IDENTICAL on clean origin/dev
(stash check); CI runners are clean (see task-32873).
