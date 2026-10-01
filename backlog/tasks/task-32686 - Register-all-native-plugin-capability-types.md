---
id: TASK-32686
title: Register all native plugin capability types
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:29'
updated_date: '2026-10-01 09:28'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32672
  - TASK-32685
  - TASK-32684
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose commands, rules, agents, hooks and complete skill metadata consistently through Chatbook runtime services.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Namespaced manual commands, always/manual rules, validated agent presets and owned hooks join the same immutable run snapshot with explicit component selection.
- [x] #2 Instruction content remains attributed untrusted context through inline/forked skills and agents; required context overflow refuses whole selected material.
- [x] #3 Tool inheritance, empty allowlists, model mappings, alias collisions and dependency closure preserve their declared semantics across composer/catalog/agent paths.
- [x] #4 New enablement never adds capability midway through an existing run, and disabled or missing requirements remain visible without degrading unrelated selected components.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements the accepted immutable native capability/context and existing agent/hook/MCP-owner contracts.
1. Trace actual published/admitted selection, composer substitutions, child presets, MCP wiring and command custody; preserve existing authority owners.
2. Establish missing EMPTY/inherit and actual selected-component behavior through targeted tests.
3. Project selected commands/rules/agents/hooks from the same snapshot, keep attributed untrusted content and explicit mapping/namespace/dependency semantics; bind owned MCP/hooks to the actual host owners before effects.
4. Qualify real Console/agent/native command flows, revocation and whole-block overflow, run targeted/static checks, record platform limits and commit only I1-owned paths.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented selected commands/rules/agents/hooks/MCP through one immutable admission and the actual Console/AgentService owners. Reviewed builtin/owned-MCP/model references preserve EMPTY, parent routing and dependency semantics; live typed context remains attributed user material and whole overflow refuses. Native hooks reserve F2 process/root custody before H2 effects and retain it through revocation/terminal settlement. Existing ADR-162/163 apply and were updated; no separate authority/runtime owner. Qualification: 164 native/Console/MCP/admission checks, 138 isolated agent checks, 85 hook/lifecycle neighbors and 3 latest readiness/EMPTY-approval controls passed without skips. Authored Ruff lint/format, shared baseline lint ratchet, changed AST and whitespace checks pass. Evidence and limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md; API/authoring: Plugins/README.md. Skill model overrides remain unsupported. No OS Keychain/OAuth/vendor-host/cross-platform/full-suite/distribution certification.
<!-- SECTION:NOTES:END -->
