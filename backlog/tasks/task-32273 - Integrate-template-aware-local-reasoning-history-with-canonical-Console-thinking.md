---
id: TASK-32273
title: >-
  Integrate template-aware local reasoning history with canonical Console
  thinking
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-10 19:32'
updated_date: '2026-09-10 20:05'
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
- [ ] #5 Targeted tests and rendered-template/live available-server checks pass; all actionable PR review findings are addressed before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md. Reason: extend canonical provider/history contracts without parallel storage. Execute Docs/superpowers/plans/2026-09-10-console-reasoning-dev-integration.md: canonical policy/serialization, per-call agent ownership, canonical settings, gateway integration, targeted verification, Qodo review and merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated local replay with ADR-090 canonical ownership, typed local adapter events, exact per-call agent sidecars, paired projection/accounting/provenance and canonical Settings. Added frozen endpoint/model policy, explicit keyless credential handling and scoped native tools; preserved child-final thinking and bounded retained bodies. Tested reviewed Gemma/Qwen templates and live Gemma4 on localhost:9099 including native calculator replay. Targeted gateway 508 pass, final feature matrix75 pass, agent122 pass, history150 pass; known unchanged-dev stop/delete/layout failures reproduced independently. Qodo review and required CI pending before merge.
<!-- SECTION:NOTES:END -->
