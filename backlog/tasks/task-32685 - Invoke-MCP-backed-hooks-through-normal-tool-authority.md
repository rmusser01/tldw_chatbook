---
id: TASK-32685
title: Invoke MCP-backed hooks through normal tool authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:29'
updated_date: '2026-10-01 23:07'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32680
  - TASK-32684
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete hook interoperability with MCP handlers that preserve recursion, initialization and permission boundaries.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP hooks use explicit typed input templates, current owned definitions and normal schema/profile/approval checks on already-connected eligible servers.
- [x] #2 Error-first result normalization accepts only the specified structured, single-text, exact mirrored or empty forms and rejects ambiguous fallback, extra blocks and oversized metadata.
- [x] #3 Provisional initialization resolves independently eligible dependencies and rejects static/dynamic cycles without dispatch or guard bypass.
- [x] #4 Approval observations cannot recurse into prompts; teardown cannot connect or request approval; nested suspension respects resource tickets and causal depth four with cancellation controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements the accepted normal MCP hook invocation, same-owner checkpoint/context/custody and bounded provisional/teardown contracts.
1. Trace reviewed H6 increment d98450ee01..cdeb1687b6 through actual current Console/MCP/AgentService owners and R70-R73 constraints.
2. Establish strict original-wire normalization RED after M1 positive controls.
3. Integrate normal MCP hook execution, same checkpoint dependencies/nested context, causal/budget/approval/teardown limits; preserve native worker admission, H3 review metadata and current Console projections.
4. Qualify real stdio/controlled HTTP, owned A/B cancellation and exact request custody, actual Console first-input/approval/teardown and normal neighboring paths; run targeted/static checks, record limits, and commit only H6-owned files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated H6 through actual current Console/MCP/checkpoint/native custody owners. Preserved H3 approval provenance through exact capture transfer; strict wire normalization, causal/dependency/context staging and bounded no-connect/no-prompt teardown are qualified. Fixed repeated catalog polling by reusing fresh exact normal definition checks at dispatch and acceptance, preserving permission checks and one/three-second deadlines. Final 162-node native covering set completed through documented corrections; 247 boundary/normal neighbors pass plus 123 earlier interrupt neighbors. Authored Ruff/format, syntax, whitespace and shared-baseline parity pass. ADR162/163 apply; evidence and platform/provider/GUI/full-suite limits in Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. Native managed graph/application composition remains I1.

PR #2946 current-dev integration preserves original typed-result identity when denial text is unchanged. Connected initializer approval reproduced the regression; all initializer branches, normal MCP provider and strict hook execution pass 140 checks. Existing ADR-162/163 apply; evidence and bounded denial projection remain intact. See the 2026-10-01 CI-repair section of Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.
<!-- SECTION:NOTES:END -->
