---
id: TASK-32273
title: >-
  Integrate template-aware local reasoning history with canonical Console
  thinking
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 19:32'
updated_date: '2026-09-10 20:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile PR 2575 with current dev so local Gemma and Qwen thinking is retained and replayed according to the active template without duplicating canonical thinking ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default-on display and canonical transcript persistence remain intact; optional replay never modifies saved thinking.
- [x] #2 Automatic, Current exchange, All available and Off local replay modes and endpoint/model overrides compose with conversation Auto/Include/Exclude and required hosted continuation.
- [x] #3 Supported llama.cpp, vLLM and Ollama adapters capture explicit structured reasoning and native tool calls; replay preserves exact response ownership and does not merge aggregate tool thoughts into final answers.
- [x] #4 Policy projection, token accounting and trace provenance agree, including current-exchange tools and runtime guidance; children inherit policy without parent thinking.
- [x] #5 Targeted tests and rendered-template/live available-server checks pass; all actionable PR review findings are addressed before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md. Reason: extend canonical provider/history contracts without parallel storage. Execute Docs/superpowers/plans/2026-09-10-console-reasoning-dev-integration.md: canonical policy/serialization, per-call agent ownership, canonical settings, gateway integration, targeted verification, Qodo review and merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated template-aware local replay with ADR-090 canonical ownership and per-call thinking. Live Gemma4 on localhost:9099 verified Auto/All/Off and native calculator replay; exact reviewed Gemma/Qwen templates are retained. Addressed all 11 initial Qodo findings: strict sanitized config validation and environment precedence, legacy Off migration, shared settings validation, Local-LLM template forwarding, bounded metadata cache/backoff, policy-aligned profile capacity, API documentation/types and corrected ADR task link. Review-fix verification: gateway regressions 488 passed with 2 sandbox listener skips; config/settings 120 passed; import boundary 14 passed; profile/capture 38 passed; gateway cache focused 36 passed. Changed-line Ruff and diff checks passed. Reviewed the sole new diagnostic as a fixed field-name-only warning and regenerated its inventory. Initial GitHub fast lane and derived artifacts passed; final-head Qodo review and CI remain pending before merge.

Qodo verified e3e8431816 with zero bugs and zero rule violations; all 11 findings resolved and all 10 inline threads answered. Post-rebase focused config, gateway and mounted Settings checks: 62 passed. Implementation and review are complete; merge remains gated on final GitHub checks.

Final merge preparation: refreshed again onto dev 41d14d1f74 after it advanced during CI. Range-diff confirmed unchanged implementation; Qodo reported zero findings on the rebased head. The diagnostic checker found only the aggregate total 7659 -> 7660: Git had coalesced identical +1 total edits from this PR and dev while retaining both correct per-file rows. Regenerated the aggregate; no additional diagnostic statements or sink changes were introduced.
<!-- SECTION:NOTES:END -->
