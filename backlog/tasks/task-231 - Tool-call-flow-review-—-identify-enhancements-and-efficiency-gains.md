---
id: TASK-228
title: Tool-call flow review — identify enhancements and efficiency gains
status: To Do
assignee: []
created_date: '2026-07-14 21:12'
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
- [ ] #1 A written review documents the current tool-call flow end-to-end with measured/estimated costs per stage,Concrete enhancement opportunities are identified and prioritized with effort estimates,At least the top-3 opportunities have follow-up tasks filed,Budget accounting recommendations reflect real multi-step run shapes (discovery + skill + answer)
<!-- AC:END -->
