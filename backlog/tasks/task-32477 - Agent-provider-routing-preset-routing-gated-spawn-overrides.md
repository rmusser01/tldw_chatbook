---
id: TASK-32477
title: 'Agent provider routing: preset routing + gated spawn overrides'
status: To Do
assignee: []
created_date: '2026-09-12 00:57'
updated_date: '2026-09-12 00:59'
labels:
  - agents
  - console
  - llm-routing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a Console master/control agent spawn sub-agents onto a specific provider+model (incl. custom-ep: registry endpoints) or a user-configured sub-agent default, instead of always inheriting the parent's selection. Presets (AgentDefinition) carry provider/model/params routing; spawn_subagent gains provider/model ad-hoc args gated by [agents] spawn_override_enabled + spawn_override_allowlist with a final-provider guard; children never inherit parent sampling/API params (six-layer stack: preset > model profile > console.provider_defaults > registry entry params > chat_defaults > api_settings scalars); resolved target snapshot persisted on the run row for stable resume. Spec: Docs/superpowers/specs/2026-09-11-agent-provider-routing-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Master can spawn a child onto a different provider+model via named preset; ad-hoc args honored only when enabled and allowlisted, else loud tool error
- [ ] #2 Sub-agent default provider/model setting with inherit-parent fallback
- [ ] #3 Children never inherit parent sampling params; preset/endpoint params precedence verified by tests
- [ ] #4 Resolved target snapshot persisted; resume reuses snapshot
- [ ] #5 Registry entries support optional params table (amends ADR-146)
- [ ] #6 No silent cross-provider fallback; RoutingError variants surface as spawn tool errors
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR: backlog/decisions/147-agent-provider-routing.md (amends ADR-146). Spec: Docs/superpowers/specs/2026-09-11-agent-provider-routing-design.md. Plan: Docs/superpowers/plans/2026-09-11-agent-provider-routing.md
<!-- SECTION:NOTES:END -->
