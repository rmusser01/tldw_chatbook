---
id: TASK-18931
title: 'Council (MoA) presets as selectable virtual models'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - agents
  - models
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's Mixture-of-Agents-as-models concept (2026-08-19 hermes-release review) onto chatbook's existing fleet + named-agent substrate (TASK-13154 programme: named agent definitions, parallel fleet, per-child panels). A "council" preset is a named ensemble — a set of member agent definitions (or provider:model pairs) plus an aggregator model. Councils appear in the model picker (Alt+M popover and Console Settings) as virtual models under a dedicated pseudo-provider; picking one routes the next send through the ensemble: members run in parallel within existing fleet caps (`[agents] max_live_subagents`), each member's output renders as a labelled block in the transcript before the aggregator synthesizes the final streamed reply. Usage accounting follows the existing sub-agent token semantics (unpriced member spend line + priced primary where provider-reported). Councils are user-defined; nothing is auto-selected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Council presets are user-definable (name, members as named agent definitions or provider:model pairs, aggregator model) and stored durably via a schema decision made and documented in the ADR
- [ ] #2 A council appears as a selectable virtual model in the Alt+M popover and Console Settings; selecting it is session-scoped exactly like any model choice, and switching away behaves like any model switch
- [ ] #3 A send through a council runs members concurrently under existing fleet caps, renders one labelled transcript block per member, then streams the aggregator's synthesized reply
- [ ] #4 Council members are sub-agents under all existing rules: approvals, risk floors, token ceilings, child wall-clock, fleet panel rows — no rule is relaxed for a council member
- [ ] #5 Usage accounting reuses the cost chip's sub-agent semantics (combined member spend shown unpriced; provider-reported usage folded into the assistant message's usage row per the existing fleet rules)
- [ ] #6 Tests cover preset definition/storage, picker surfacing, fan-out + aggregation flow, accounting, and rule application; non-goals pinned (no auto-selection, no cross-conversation persistence beyond normal session model persistence)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/068-council-moa-presets-as-virtual-models.md (to be drafted before implementation).
Reason: new virtual-provider/model-system boundary and a cross-module interface (model picker ↔ agent runtime/fleet); long-lived UX structure. The ADR must decide storage (config vs DB), the pseudo-provider identity, and failure semantics when a member's provider is unavailable.

1. Draft ADR-068 (storage, picker identity, member-failure semantics, accounting)
2. Preset definition + storage; virtual-model registration in the picker seam
3. Runtime: fan-out via existing fleet spawn, labelled member blocks, aggregator turn
4. Accounting integration + fleet-panel/display polish
5. Tests + docs (console.md model selection, agent-runs fleet sections)
<!-- SECTION:PLAN:END -->
