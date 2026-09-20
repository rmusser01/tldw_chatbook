---
id: TASK-32874
title: Preset and provider routing for agent-created chats (fork_chat/new_chat)
status: To Do
assignee: []
created_date: '2026-09-20 16:47'
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
