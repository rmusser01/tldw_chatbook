---
id: TASK-28236
title: LLM guardian pass for risky tool calls (smart approvals)
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - agents
  - safety
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row A9, unblocked by TASK-25905: the deterministic hardline floor now exists as the backstop, making an advisory LLM guardian layer meaningful (it can only ADD refusals/questions above the floor, never replace it). Assess risky tool calls with a cheap model pass and downgrade auto-approvals to ask-with-rationale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An opt-in guardian evaluates gated tool calls and can escalate allow->ask with a one-line rationale on the approval card
- [ ] #2 The guardian can never downgrade ask->allow or bypass the hardline floor
- [ ] #3 Guardian failure or timeout falls back to today's behavior (fail-open to the existing gate, never auto-approve)
- [ ] #4 Off by default; per-workspace opt-in
<!-- AC:END -->
