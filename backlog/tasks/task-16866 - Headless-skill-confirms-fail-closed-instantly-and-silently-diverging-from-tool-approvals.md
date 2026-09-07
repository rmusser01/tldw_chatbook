---
id: TASK-16866
title: >-
  Headless skill confirms fail closed instantly and silently, diverging from
  tool approvals
status: To Do
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - console
  - agents
  - safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #1752 (task-15860 Task 5) made a headless MCP tool approval WAIT and announce itself app-wide: a wake turn firing with Console closed now raises a toast on whatever screen the user is on, and the card is mounted the moment Console opens. Skill install/script confirms were deliberately left untouched by that landing and still take the opposite path: with no view attached their pending-slot is None, which their read site treats as fail-closed-at-once, so the confirm denies immediately and NOTHING surfaces to the user.

That divergence is now a user-visible inconsistency in the same feature: one class of gated action asks the user wherever they are, the other refuses silently. It needs an owner ruling rather than a default, because the two options differ in posture: make skill confirms wait-and-announce like tool approvals (consistent, but skill scripts execute immediately after confirm with no cancel checkpoint - see the tool-approval security stream), or keep them fail-closed and instead SURFACE the refusal so the user learns the skill did not run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The behaviour of a headless skill install confirm and a headless skill script confirm is decided deliberately and recorded, not left as an artefact of a None slot
- [ ] #2 Whichever posture is chosen, the user learns the outcome: either a card they can answer from any screen, or a visible notice that the action was refused
- [ ] #3 The chosen behaviour is pinned by a test driven through the production path with Console genuinely unmounted
- [ ] #4 Docs/User_Guide/console/agent-runs-and-tools.md states the skill-confirm behaviour alongside the tool-approval behaviour
<!-- AC:END -->
