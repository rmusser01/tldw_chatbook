---
id: TASK-231
title: Tool-call flow review — identify enhancements and efficiency gains
status: Done
assignee: []
created_date: '2026-07-14 21:12'
updated_date: '2026-07-17 15:22'
labels:
  - agents
  - console
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent runtime's tool-calling flows (fence-first text protocol, per-turn system-prompt re-render, find_tools/load_tools disclosure round-trips, spawn/skill executor dispatch, step budgets, per-chunk StreamGate scanning, AgentRunsDB step persistence) grew across three sub-projects (#623/#629/skills). Review the end-to-end flow for enhancements and efficiency gains — e.g. round-trips per disclosure cycle, budget accounting vs real turn costs (the Skills Phase-2 gate hit step exhaustion on a successful discovery run), protocol token overhead, native tool-call support where providers offer it, payload rebuild cost per turn, and marker/persistence write amplification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A written review documents the current tool-call flow end-to-end with measured/estimated costs per stage,Concrete enhancement opportunities are identified and prioritized with effort estimates,At least the top-3 opportunities have follow-up tasks filed,Budget accounting recommendations reflect real multi-step run shapes (discovery + skill + answer)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #644 (merged to dev): Docs/superpowers/reviews/2026-07-16-tool-call-flow-review.md — end-to-end tool-call flow review with measured/estimated per-stage costs, prioritized enhancement opportunities with effort estimates, and budget-accounting recommendations for real multi-step run shapes. Top-3 opportunities filed as tasks 243/244/245 (commit 5bbfecb4) — all three have since shipped (PRs #648/#652, #655, #658). This status edit fixes drift only — the review itself merged 2026-07-16.
<!-- SECTION:NOTES:END -->
